use anyhow::{Context, Result, ensure};
use dotenv::dotenv;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio::time::{MissedTickBehavior, interval};
use tracing::{debug, error, info, warn};

mod agents;
mod config;
mod pipeline;
mod skills;
mod utils;

use agents::{
    Agent, BlueprinterAgent, ExecutorAgent, SurgeonAgent, Ticket, TicketStatus, TicketTokenUsage,
    VerifierAgent,
};
use config::{DockerSandboxConfig, ModelRouter, TaskKind, ZeroClawClient};
use skills::{ASTParsingSkill, FileIOSkill, FileWriteSkill, ShadowTestSkill, Skill};
use utils::state::{load_state, save_state};

const MAX_SURGERY_RETRIES: u8 = 3;
const STATE_FILE: &str = ".migration_state.json";
const STATE_FLUSH_INTERVAL_SECS: u64 = 3;

#[tokio::main]
async fn main() -> Result<()> {
    let _guard = utils::setup_telemetry()?;
    dotenv().ok();

    info!("Starting AI-Powered Legacy Code Migration Pipeline");

    let startup_root =
        std::env::current_dir().context("failed to resolve startup working directory")?;
    let legacy_root = resolve_project_root(&startup_root, Path::new("./legacy_app"), false)?;
    let modern_root = resolve_project_root(&startup_root, Path::new("./modern_app"), true)?;

    let ast_parsing_skill: Arc<dyn Skill> = Arc::new(ASTParsingSkill);
    let file_io_skill: Arc<dyn Skill> = Arc::new(FileIOSkill);
    let file_write_skill: Arc<dyn Skill> = Arc::new(FileWriteSkill);
    let docker_sandbox_config = DockerSandboxConfig::from_env()?;
    let shadow_test_skill: Arc<dyn Skill> = Arc::new(ShadowTestSkill::new(docker_sandbox_config));
    let ast_parsing_name = ast_parsing_skill.name().to_owned();
    let file_io_name = file_io_skill.name().to_owned();
    let file_write_name = file_write_skill.name().to_owned();
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
        modern_root.clone(),
        Arc::clone(&file_io_skill),
        Arc::clone(&file_write_skill),
        Arc::clone(&llm_client),
    ));
    let verifier = Arc::new(VerifierAgent::new(
        legacy_root.clone(),
        modern_root.clone(),
        Arc::clone(&ast_parsing_skill),
        Arc::clone(&shadow_test_skill),
        Arc::clone(&llm_client),
    ));
    let surgeon = Arc::new(SurgeonAgent::new(
        legacy_root.clone(),
        modern_root.clone(),
        Arc::clone(&file_io_skill),
        Arc::clone(&file_write_skill),
        Arc::clone(&llm_client),
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
        output_root = modern_root.to_string_lossy().as_ref(),
        "Initialized phase-2 executor pipeline"
    );
    info!(
        agent = verifier.name(),
        provider = verifier.provider(),
        model = verifier.model(),
        ast_parsing = ast_parsing_name.as_str(),
        shadow_test = shadow_test_name.as_str(),
        output_root = modern_root.to_string_lossy().as_ref(),
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

    let tickets = match load_state(STATE_FILE)? {
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
                .generate_blueprint(legacy_root.to_string_lossy().as_ref())
                .await?;
            debug!(tickets = ?tickets, "Generated migration blueprint payload");
            info!(
                state_file = STATE_FILE,
                ticket_count = tickets.len(),
                "Generated migration blueprint"
            );
            save_state(STATE_FILE, &tickets)?;
            info!(
                state_file = STATE_FILE,
                ticket_count = tickets.len(),
                "Saved initial migration blueprint state"
            );
            tickets
        }
    };

    let (tx, rx) = mpsc::channel(100);
    let state_writer = tokio::spawn(run_state_writer(STATE_FILE.to_owned(), tickets.clone(), rx));

    let concurrency_limit = Arc::new(Semaphore::new(3));
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
        "State writer flushed final migration state"
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

    Ok(())
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
    run_state_writer_with_flush_interval(
        state_path,
        tickets,
        rx,
        Duration::from_secs(STATE_FLUSH_INTERVAL_SECS),
    )
    .await
}

async fn run_state_writer_with_flush_interval(
    state_path: String,
    mut tickets: Vec<Ticket>,
    mut rx: mpsc::Receiver<Ticket>,
    flush_interval: Duration,
) -> Result<Vec<Ticket>> {
    let mut flush_timer = interval(flush_interval);
    flush_timer.set_missed_tick_behavior(MissedTickBehavior::Delay);
    flush_timer.tick().await;
    let mut has_pending_updates = false;

    loop {
        tokio::select! {
            maybe_updated_ticket = rx.recv() => {
                match maybe_updated_ticket {
                    Some(updated_ticket) => {
                        upsert_ticket(&mut tickets, updated_ticket);
                        has_pending_updates = true;
                    }
                    None => {
                        save_state(&state_path, &tickets)?;
                        return Ok(tickets);
                    }
                }
            }
            _ = flush_timer.tick(), if has_pending_updates => {
                save_state(&state_path, &tickets)?;
                has_pending_updates = false;
            }
        }
    }
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

fn resolve_project_root(
    startup_root: &Path,
    configured_path: &Path,
    create_if_missing: bool,
) -> Result<PathBuf> {
    let absolute_path = if configured_path.is_absolute() {
        configured_path.to_path_buf()
    } else {
        startup_root.join(configured_path)
    };

    if create_if_missing {
        fs::create_dir_all(&absolute_path).with_context(|| {
            format!(
                "failed to create required project directory {}",
                absolute_path.display()
            )
        })?;
    }

    ensure!(
        absolute_path.exists(),
        "required project directory does not exist: {}",
        absolute_path.display()
    );
    ensure!(
        absolute_path.is_dir(),
        "required project path is not a directory: {}",
        absolute_path.display()
    );

    Ok(absolute_path)
}

#[cfg(test)]
mod tests {
    use super::{resolve_project_root, run_state_writer_with_flush_interval};
    use crate::agents::{Ticket, TicketStatus, TicketTokenUsage};
    use crate::utils::state::load_state;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};
    use tokio::sync::mpsc;

    fn make_temp_state_path(prefix: &str) -> (PathBuf, PathBuf) {
        let unique_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be monotonic")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("{prefix}_{unique_id}"));
        fs::create_dir_all(&root).expect("temp directory should be created");
        let state_path = root.join(".migration_state.json");
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
    async fn state_writer_batches_updates_until_flush_interval() {
        let (root, state_path) = make_temp_state_path("migration_pipeline_state_writer_batch");
        let initial_tickets = vec![sample_ticket("TICKET-1")];
        let (tx, rx) = mpsc::channel(4);
        let state_path_string = state_path.to_string_lossy().into_owned();

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
        assert!(
            !state_path.exists(),
            "state writer should not flush immediately for each update"
        );

        tokio::time::sleep(Duration::from_millis(120)).await;
        let loaded = load_state(&state_path_string)
            .expect("state should load")
            .expect("state file should exist after flush");
        assert_eq!(loaded[0].retries, 1);

        drop(tx);
        writer
            .await
            .expect("state writer should join")
            .expect("state writer should succeed");

        fs::remove_dir_all(root).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn state_writer_flushes_pending_updates_when_channel_closes() {
        let (root, state_path) = make_temp_state_path("migration_pipeline_state_writer_close");
        let initial_tickets = vec![sample_ticket("TICKET-1")];
        let (tx, rx) = mpsc::channel(4);
        let state_path_string = state_path.to_string_lossy().into_owned();

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

        let loaded = load_state(&state_path_string)
            .expect("state should load")
            .expect("state file should exist after close flush");
        assert_eq!(loaded[0].retries, 2);

        fs::remove_dir_all(root).expect("temp directory should be removed");
    }

    #[test]
    fn resolve_project_root_uses_startup_directory_once() {
        let (root, _) = make_temp_state_path("migration_pipeline_root_resolution");
        let legacy_root = root.join("legacy_app");
        fs::create_dir_all(&legacy_root).expect("legacy root should be created");

        let resolved = resolve_project_root(&root, Path::new("legacy_app"), false)
            .expect("legacy root should resolve");

        assert_eq!(resolved, legacy_root);

        fs::remove_dir_all(root).expect("temp directory should be removed");
    }
}
