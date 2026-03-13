use anyhow::{Context, Result, ensure};
use dotenv::dotenv;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tracing::{debug, error, info, warn};

mod agents;
mod cli;
mod config;
mod pipeline;
mod prompts;
mod skills;
mod utils;

use agents::{
    Agent, BlueprinterAgent, ExecutorAgent, SurgeonAgent, Ticket, TicketStatus, TicketTokenUsage,
    VerifierAgent,
};
use cli::StartMigrationConfig;
use config::{DockerSandboxConfig, ModelRouter, TaskKind, ZeroClawClient};
use skills::{ASTParsingSkill, FileIOSkill, FileWriteSkill, SandboxSkill, ShadowTestSkill, Skill};
use utils::state::{load_state, save_state};

const MAX_SURGERY_RETRIES: u8 = 3;
const STATE_FILE: &str = ".migration_state.db";
const MAX_CONCURRENT_TICKETS_ENV: &str = "MAX_CONCURRENT_TICKETS";
const DEFAULT_MAX_CONCURRENT_TICKETS: usize = 3;
const PIPELINE_TARGET_FRAMEWORK_ENV: &str = "PIPELINE_TARGET_FRAMEWORK";
const DEFAULT_TARGET_FRAMEWORK: &str = "TypeScript on Node.js LTS";

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    let startup_root =
        std::env::current_dir().context("failed to resolve startup working directory")?;
    let Some(StartMigrationConfig {
        legacy_root,
        modern_root,
    }) = cli::run_cli(&startup_root)?
    else {
        return Ok(());
    };

    run_pipeline(legacy_root, modern_root).await
}

async fn run_pipeline(legacy_root: PathBuf, modern_root: PathBuf) -> Result<()> {
    let target_framework = target_framework_from_env();
    let telemetry_guard = utils::setup_telemetry()?;
    info!("Starting AI-Powered Legacy Code Migration Pipeline");
    info!(
        legacy_root = %legacy_root.display(),
        modern_root = %modern_root.display(),
        target_framework = target_framework.as_str(),
        "Resolved migration configuration"
    );

    let ast_parsing_skill: Arc<dyn Skill> = Arc::new(ASTParsingSkill);
    let file_io_skill: Arc<dyn Skill> = Arc::new(FileIOSkill);
    let file_write_skill: Arc<dyn Skill> = Arc::new(FileWriteSkill);
    let docker_sandbox_config = DockerSandboxConfig::from_env()?;
    let sandbox_skill: Arc<dyn Skill> = Arc::new(SandboxSkill::new(docker_sandbox_config.clone()));
    let shadow_test_skill: Arc<dyn Skill> = Arc::new(ShadowTestSkill::new(docker_sandbox_config));
    let ast_parsing_name = ast_parsing_skill.name().to_owned();
    let file_io_name = file_io_skill.name().to_owned();
    let file_write_name = file_write_skill.name().to_owned();
    let sandbox_name = sandbox_skill.name().to_owned();
    let shadow_test_name = shadow_test_skill.name().to_owned();
    let task_configs = vec![
        ModelRouter::from_env(TaskKind::Blueprinter, BlueprinterAgent::default_model())?,
        ModelRouter::from_env(TaskKind::Executor, ExecutorAgent::default_model())?,
        ModelRouter::from_env(TaskKind::Verifier, VerifierAgent::default_model())?,
        ModelRouter::from_env(TaskKind::Surgeon, SurgeonAgent::default_model())?,
    ];
    let llm_client = Arc::new(ZeroClawClient::new(task_configs)?);
    let blueprinter =
        BlueprinterAgent::new(Arc::clone(&ast_parsing_skill), Arc::clone(&llm_client));
    let executor = Arc::new(ExecutorAgent::new(
        legacy_root.clone(),
        Arc::clone(&file_io_skill),
    ));
    let verifier = Arc::new(VerifierAgent::new(
        legacy_root.clone(),
        modern_root.clone(),
        Arc::clone(&ast_parsing_skill),
        Arc::clone(&sandbox_skill),
        Arc::clone(&shadow_test_skill),
        Arc::clone(&llm_client),
    ));
    let surgeon = Arc::new(SurgeonAgent::new(
        legacy_root.clone(),
        modern_root.clone(),
        Arc::clone(&file_io_skill),
    ));

    info!(
        agent = blueprinter.name(),
        provider = blueprinter.provider(),
        model = blueprinter.model(),
        ast_parsing = ast_parsing_name.as_str(),
        "Initialized phase-1 blueprinter pipeline"
    );
    info!(
        agent = executor.name(),
        provider = executor.provider(),
        model = executor.model(),
        file_io = file_io_name.as_str(),
        file_write = file_write_name.as_str(),
        output_root = %modern_root.display(),
        "Initialized phase-2 executor pipeline"
    );
    info!(
        agent = verifier.name(),
        provider = verifier.provider(),
        model = verifier.model(),
        ast_parsing = ast_parsing_name.as_str(),
        sandbox = sandbox_name.as_str(),
        shadow_test = shadow_test_name.as_str(),
        output_root = %modern_root.display(),
        "Initialized phase-3 verifier pipeline with shadow testing"
    );
    info!(
        agent = surgeon.name(),
        provider = surgeon.provider(),
        model = surgeon.model(),
        file_io = file_io_name.as_str(),
        file_write = file_write_name.as_str(),
        "Initialized phase-4 surgeon pipeline"
    );

    let tickets = match load_state_blocking(STATE_FILE.to_owned()).await? {
        Some(tickets) => {
            info!(
                state_file = STATE_FILE,
                ticket_count = tickets.len(),
                "Loaded migration state from disk and skipping blueprinter"
            );
            tickets
        }
        None => {
            let tickets = blueprinter
                .generate_blueprint(
                    legacy_root.to_string_lossy().as_ref(),
                    target_framework.as_str(),
                )
                .await?;
            debug!(tickets = ?tickets, "Generated migration blueprint payload");
            info!(
                state_file = STATE_FILE,
                ticket_count = tickets.len(),
                "Generated migration blueprint"
            );
            save_tickets_blocking(STATE_FILE.to_owned(), tickets.clone()).await?;
            info!(
                state_file = STATE_FILE,
                ticket_count = tickets.len(),
                "Saved initial migration blueprint state to SQLite"
            );
            tickets
        }
    };

    let (tx, rx) = mpsc::channel(100);
    let state_writer = tokio::spawn(run_state_writer(STATE_FILE.to_owned(), tickets.clone(), rx));

    let max_concurrent_tickets = max_concurrent_tickets_from_env()?;
    let concurrency_limit = Arc::new(Semaphore::new(max_concurrent_tickets));
    info!(
        max_concurrent_tickets,
        "Initialized pipeline concurrency limit"
    );
    let mut join_set = JoinSet::new();

    for ticket in tickets {
        let permit = Arc::clone(&concurrency_limit)
            .acquire_owned()
            .await
            .map_err(|error| {
                anyhow::anyhow!("failed to acquire executor semaphore permit: {error}")
            })?;
        let executor = Arc::clone(&executor);
        let verifier = Arc::clone(&verifier);
        let surgeon = Arc::clone(&surgeon);
        let tx = tx.clone();

        join_set.spawn(async move {
            let _permit = permit;
            run_ticket_feedback_loop(ticket, executor, verifier, surgeon, tx).await
        });
    }

    drop(tx);

    while let Some(join_result) = join_set.join_next().await {
        match join_result {
            Ok(Ok(ticket)) => match &ticket.status {
                TicketStatus::Verified => {
                    info!(ticket_id = ticket.id.as_str(), "Pipeline verified ticket")
                }
                TicketStatus::Failed(error_message) => error!(
                    ticket_id = ticket.id.as_str(),
                    error = error_message.as_str(),
                    "Pipeline failed ticket"
                ),
                TicketStatus::InProgress => {
                    info!(
                        ticket_id = ticket.id.as_str(),
                        "Pipeline left ticket in progress"
                    )
                }
                TicketStatus::Todo => {
                    info!(
                        ticket_id = ticket.id.as_str(),
                        "Pipeline returned todo ticket"
                    )
                }
            },
            Ok(Err(error)) => error!(error = %error, "Pipeline task returned an unexpected error"),
            Err(error) => error!(error = %error, "Pipeline task panicked or was cancelled"),
        }
    }

    let final_state = state_writer.await.map_err(|error| {
        anyhow::anyhow!("state writer task panicked or was cancelled: {error}")
    })??;
    info!(
        state_file = STATE_FILE,
        ticket_count = final_state.len(),
        "State writer completed migration state persistence"
    );
    let aggregated_usage = summarize_token_usage(&final_state);
    info!(
        llm_calls = aggregated_usage.llm_calls,
        prompt_tokens = aggregated_usage.prompt_tokens,
        completion_tokens = aggregated_usage.completion_tokens,
        total_tokens = aggregated_usage.total_tokens,
        audit_log_directory = "logs",
        state_file = STATE_FILE,
        "Aggregated pipeline token usage"
    );

    let failed_tickets = final_state
        .iter()
        .filter(|ticket| matches!(ticket.status, TicketStatus::Failed(_)))
        .count();
    if failed_tickets > 0 {
        error!(
            failed_tickets,
            total_tickets = final_state.len(),
            state_file = STATE_FILE,
            "Pipeline finished with failed tickets"
        );
        drop(telemetry_guard);
        std::thread::sleep(Duration::from_millis(500));
        std::process::exit(1);
    }

    drop(telemetry_guard);
    Ok(())
}

fn target_framework_from_env() -> String {
    env::var(PIPELINE_TARGET_FRAMEWORK_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_TARGET_FRAMEWORK.to_owned())
}

async fn run_ticket_feedback_loop(
    mut ticket: Ticket,
    executor: Arc<ExecutorAgent>,
    verifier: Arc<VerifierAgent>,
    surgeon: Arc<SurgeonAgent>,
    state_tx: mpsc::Sender<Ticket>,
) -> Result<Ticket> {
    loop {
        match &ticket.status {
            TicketStatus::Todo => {
                info!(
                    ticket_id = ticket.id.as_str(),
                    "Dispatching ticket to executor"
                );
                ticket = executor.process_ticket(&ticket).await?;
                publish_ticket_state(&state_tx, &ticket).await?;
            }
            TicketStatus::InProgress => {
                info!(
                    ticket_id = ticket.id.as_str(),
                    retries = ticket.retries,
                    "Dispatching ticket to verifier"
                );
                ticket = verifier.process_ticket(&ticket).await?;
                publish_ticket_state(&state_tx, &ticket).await?;
            }
            TicketStatus::Failed(reason) if ticket.retries < MAX_SURGERY_RETRIES => {
                if !ticket_has_generated_artifacts(&ticket) {
                    error!(
                        ticket_id = ticket.id.as_str(),
                        retries = ticket.retries,
                        failure = reason.as_str(),
                        "Ticket failed before artifacts were generated; skipping surgeon"
                    );
                    return Ok(ticket);
                }
                warn!(
                    ticket_id = ticket.id.as_str(),
                    retries = ticket.retries,
                    failure = reason.as_str(),
                    "Dispatching failed ticket to surgeon"
                );
                ticket = surgeon.process_ticket(&ticket).await?;
                publish_ticket_state(&state_tx, &ticket).await?;
                info!(
                    ticket_id = ticket.id.as_str(),
                    retries = ticket.retries,
                    "Ticket returned from surgery and will be re-verified"
                );
            }
            TicketStatus::Failed(reason) => {
                error!(
                    ticket_id = ticket.id.as_str(),
                    retries = ticket.retries,
                    failure = reason.as_str(),
                    max_retries = MAX_SURGERY_RETRIES,
                    "Ticket exceeded surgery retry limit and requires human intervention"
                );
                return Ok(ticket);
            }
            TicketStatus::Verified => {
                info!(
                    ticket_id = ticket.id.as_str(),
                    retries = ticket.retries,
                    "Ticket completed verification loop"
                );
                return Ok(ticket);
            }
        }
    }
}

async fn publish_ticket_state(state_tx: &mpsc::Sender<Ticket>, ticket: &Ticket) -> Result<()> {
    state_tx.send(ticket.clone()).await.map_err(|error| {
        anyhow::anyhow!(
            "failed to send state update for ticket {}: {error}",
            ticket.id
        )
    })
}

async fn run_state_writer(
    state_path: String,
    tickets: Vec<Ticket>,
    rx: mpsc::Receiver<Ticket>,
) -> Result<Vec<Ticket>> {
    run_state_writer_with_flush_interval(state_path, tickets, rx, Duration::ZERO).await
}

async fn run_state_writer_with_flush_interval(
    state_path: String,
    mut tickets: Vec<Ticket>,
    mut rx: mpsc::Receiver<Ticket>,
    _flush_interval: Duration,
) -> Result<Vec<Ticket>> {
    while let Some(updated_ticket) = rx.recv().await {
        save_state_blocking(state_path.clone(), updated_ticket.clone()).await?;
        upsert_ticket(&mut tickets, updated_ticket);
    }

    Ok(tickets)
}

async fn load_state_blocking(path: String) -> Result<Option<Vec<Ticket>>> {
    tokio::task::spawn_blocking(move || load_state(&path))
        .await
        .map_err(|error| anyhow::anyhow!("state load task failed to join: {error}"))?
}

async fn save_state_blocking(path: String, ticket: Ticket) -> Result<()> {
    tokio::task::spawn_blocking(move || save_state(&path, &ticket))
        .await
        .map_err(|error| anyhow::anyhow!("state save task failed to join: {error}"))?
}

async fn save_tickets_blocking(path: String, tickets: Vec<Ticket>) -> Result<()> {
    tokio::task::spawn_blocking(move || {
        for ticket in tickets {
            save_state(&path, &ticket)?;
        }
        Ok(())
    })
    .await
    .map_err(|error| anyhow::anyhow!("state save task failed to join: {error}"))?
}

fn upsert_ticket(tickets: &mut Vec<Ticket>, updated_ticket: Ticket) {
    if let Some(existing_ticket) = tickets
        .iter_mut()
        .find(|ticket| ticket.id == updated_ticket.id)
    {
        *existing_ticket = updated_ticket;
    } else {
        tickets.push(updated_ticket);
    }
}

fn summarize_token_usage(tickets: &[Ticket]) -> TicketTokenUsage {
    tickets
        .iter()
        .fold(TicketTokenUsage::default(), |mut totals, ticket| {
            totals.llm_calls = totals
                .llm_calls
                .saturating_add(ticket.token_usage.llm_calls);
            totals.prompt_tokens = totals
                .prompt_tokens
                .saturating_add(ticket.token_usage.prompt_tokens);
            totals.completion_tokens = totals
                .completion_tokens
                .saturating_add(ticket.token_usage.completion_tokens);
            totals.total_tokens = totals
                .total_tokens
                .saturating_add(ticket.token_usage.total_tokens);
            totals
        })
}

fn ticket_has_generated_artifacts(ticket: &Ticket) -> bool {
    !ticket.modern_file_paths.is_empty() || !ticket.test_file_paths.is_empty()
}

fn max_concurrent_tickets_from_env() -> Result<usize> {
    match env::var(MAX_CONCURRENT_TICKETS_ENV) {
        Ok(value) if !value.trim().is_empty() => {
            let parsed = value.parse::<usize>().with_context(|| {
                format!(
                    "`{MAX_CONCURRENT_TICKETS_ENV}` must be a positive integer, received `{value}`"
                )
            })?;
            ensure!(
                parsed > 0,
                "`{MAX_CONCURRENT_TICKETS_ENV}` must be greater than zero"
            );
            Ok(parsed)
        }
        Ok(_) | Err(env::VarError::NotPresent) => Ok(DEFAULT_MAX_CONCURRENT_TICKETS),
        Err(error) => Err(anyhow::anyhow!(
            "failed to read `{MAX_CONCURRENT_TICKETS_ENV}`: {error}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        load_state_blocking, max_concurrent_tickets_from_env, run_state_writer_with_flush_interval,
    };
    use crate::agents::{Ticket, TicketStatus, TicketTokenUsage};
    use std::fs;
    use std::path::PathBuf;
    use std::sync::{LazyLock, Mutex};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};
    use tokio::sync::mpsc;

    static ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    fn make_temp_state_path(prefix: &str) -> (PathBuf, PathBuf) {
        let unique_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be monotonic")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("{prefix}_{unique_id}"));
        fs::create_dir_all(&root).expect("temp directory should be created");
        let state_path = root.join(".migration_state.db");
        (root, state_path)
    }

    fn sample_ticket(id: &str) -> Ticket {
        Ticket {
            id: id.to_owned(),
            description: format!("Ticket {id}"),
            context_files: vec!["src/server.js".to_owned()],
            status: TicketStatus::InProgress,
            legacy_code_snippet: "http.createServer(...)".to_owned(),
            target_framework: "TypeScript".to_owned(),
            dependencies: vec!["express".to_owned()],
            modern_file_paths: vec!["src/server.ts".to_owned()],
            test_file_paths: vec!["tests/src/server.test.ts".to_owned()],
            retries: 0,
            token_usage: TicketTokenUsage::default(),
            last_execution_diff: None,
            last_ast_diff: None,
        }
    }

    #[tokio::test]
    async fn state_writer_persists_updates_immediately() {
        let (root, state_path) = make_temp_state_path("migration_pipeline_state_writer_immediate");
        let initial_tickets = vec![sample_ticket("TICKET-1")];
        let (tx, rx) = mpsc::channel(4);
        let state_path_string = state_path.to_string_lossy().into_owned();
        super::save_tickets_blocking(state_path_string.clone(), initial_tickets.clone())
            .await
            .expect("initial state should save");

        let writer = tokio::spawn(run_state_writer_with_flush_interval(
            state_path_string.clone(),
            initial_tickets,
            rx,
            Duration::from_millis(100),
        ));

        tx.send(Ticket {
            retries: 1,
            ..sample_ticket("TICKET-1")
        })
        .await
        .expect("update should send");

        tokio::time::sleep(Duration::from_millis(20)).await;
        let loaded = load_state_blocking(state_path_string.clone())
            .await
            .expect("state should load")
            .expect("state database should exist after immediate write");
        assert_eq!(loaded[0].retries, 1);

        drop(tx);
        writer
            .await
            .expect("state writer should join")
            .expect("state writer should succeed");

        fs::remove_dir_all(root).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn state_writer_retains_updates_when_channel_closes() {
        let (root, state_path) = make_temp_state_path("migration_pipeline_state_writer_close");
        let initial_tickets = vec![sample_ticket("TICKET-1")];
        let (tx, rx) = mpsc::channel(4);
        let state_path_string = state_path.to_string_lossy().into_owned();
        super::save_tickets_blocking(state_path_string.clone(), initial_tickets.clone())
            .await
            .expect("initial state should save");

        let writer = tokio::spawn(run_state_writer_with_flush_interval(
            state_path_string.clone(),
            initial_tickets,
            rx,
            Duration::from_secs(60),
        ));

        tx.send(Ticket {
            retries: 2,
            ..sample_ticket("TICKET-1")
        })
        .await
        .expect("update should send");
        drop(tx);

        writer
            .await
            .expect("state writer should join")
            .expect("state writer should succeed");

        let loaded = load_state_blocking(state_path_string.clone())
            .await
            .expect("state should load")
            .expect("state database should exist after channel close");
        assert_eq!(loaded[0].retries, 2);

        fs::remove_dir_all(root).expect("temp directory should be removed");
    }

    #[test]
    fn max_concurrent_tickets_defaults_when_env_is_missing() {
        let _guard = ENV_LOCK.lock().expect("env lock should not be poisoned");
        let previous = std::env::var(super::MAX_CONCURRENT_TICKETS_ENV).ok();
        unsafe {
            std::env::remove_var(super::MAX_CONCURRENT_TICKETS_ENV);
        }

        let value = max_concurrent_tickets_from_env().expect("default concurrency should load");

        if let Some(previous) = previous {
            unsafe {
                std::env::set_var(super::MAX_CONCURRENT_TICKETS_ENV, previous);
            }
        }

        assert_eq!(value, super::DEFAULT_MAX_CONCURRENT_TICKETS);
    }

    #[test]
    fn max_concurrent_tickets_rejects_zero() {
        let _guard = ENV_LOCK.lock().expect("env lock should not be poisoned");
        let previous = std::env::var(super::MAX_CONCURRENT_TICKETS_ENV).ok();
        unsafe {
            std::env::set_var(super::MAX_CONCURRENT_TICKETS_ENV, "0");
        }

        let error = max_concurrent_tickets_from_env().expect_err("zero concurrency should fail");

        if let Some(previous) = previous {
            unsafe {
                std::env::set_var(super::MAX_CONCURRENT_TICKETS_ENV, previous);
            }
        } else {
            unsafe {
                std::env::remove_var(super::MAX_CONCURRENT_TICKETS_ENV);
            }
        }

        assert!(error.to_string().contains("greater than zero"));
    }
}
