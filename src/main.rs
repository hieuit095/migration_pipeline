use anyhow::Result;
use dotenv::dotenv;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::sync::mpsc;
use tokio::task::JoinSet;
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
use config::{LlmGateway, LlmProvider, ModelRouter, TaskKind};
use skills::{FileIOSkill, FileWriteSkill, SandboxSkill, Skill};
use utils::state::{load_state, save_state};

const MAX_SURGERY_RETRIES: u8 = 3;
const STATE_FILE: &str = ".migration_state.json";

#[tokio::main]
async fn main() -> Result<()> {
    let _guard = utils::setup_telemetry()?;
    dotenv().ok();

    info!("Starting AI-Powered Legacy Code Migration Pipeline");

    let legacy_root = PathBuf::from("./legacy_app");
    let modern_root = PathBuf::from("./modern_app");
    fs::create_dir_all(&modern_root)?;

    let file_io_skill: Arc<dyn Skill> = Arc::new(FileIOSkill);
    let file_write_skill: Arc<dyn Skill> = Arc::new(FileWriteSkill);
    let sandbox_skill: Arc<dyn Skill> = Arc::new(SandboxSkill);
    let file_io_name = file_io_skill.name().to_owned();
    let file_write_name = file_write_skill.name().to_owned();
    let sandbox_name = sandbox_skill.name().to_owned();
    let llm_gateway = LlmGateway::new()?;
    let blueprinter_route = ModelRouter::from_env(
        TaskKind::Blueprinter,
        LlmProvider::OpenRouter,
        BlueprinterAgent::default_model(),
    )?;
    let executor_route = ModelRouter::from_env(
        TaskKind::Executor,
        LlmProvider::OpenRouter,
        ExecutorAgent::default_model(),
    )?;
    let verifier_route = ModelRouter::from_env(
        TaskKind::Verifier,
        LlmProvider::OpenRouter,
        VerifierAgent::default_model(),
    )?;
    let surgeon_route = ModelRouter::from_env(
        TaskKind::Surgeon,
        LlmProvider::OpenRouter,
        SurgeonAgent::default_model(),
    )?;
    let blueprinter = BlueprinterAgent::new(
        Arc::clone(&file_io_skill),
        llm_gateway.clone(),
        blueprinter_route,
    );
    let executor = Arc::new(ExecutorAgent::new(
        legacy_root.clone(),
        modern_root.clone(),
        Arc::clone(&file_io_skill),
        Arc::clone(&file_write_skill),
        llm_gateway,
        executor_route,
    ));
    let verifier = Arc::new(VerifierAgent::new(
        legacy_root.clone(),
        modern_root.clone(),
        Arc::clone(&file_io_skill),
        Arc::clone(&file_write_skill),
        Arc::clone(&sandbox_skill),
        LlmGateway::new()?,
        verifier_route,
    ));
    let surgeon = Arc::new(SurgeonAgent::new(
        legacy_root.clone(),
        modern_root.clone(),
        Arc::clone(&file_io_skill),
        Arc::clone(&file_write_skill),
        LlmGateway::new()?,
        surgeon_route,
    ));

    info!(
        agent = blueprinter.name(),
        provider = blueprinter.provider().as_str(),
        model = blueprinter.model(),
        file_io = file_io_name.as_str(),
        "Initialized phase-1 blueprinter pipeline"
    );
    info!(
        agent = executor.name(),
        provider = executor.provider().as_str(),
        model = executor.model(),
        file_io = file_io_name.as_str(),
        file_write = file_write_name.as_str(),
        output_root = modern_root.to_string_lossy().as_ref(),
        "Initialized phase-2 executor pipeline"
    );
    info!(
        agent = verifier.name(),
        provider = verifier.provider().as_str(),
        model = verifier.model(),
        file_io = file_io_name.as_str(),
        file_write = file_write_name.as_str(),
        sandbox = sandbox_name.as_str(),
        "Initialized phase-3 verifier pipeline with Docker sandboxing"
    );
    info!(
        agent = surgeon.name(),
        provider = surgeon.provider().as_str(),
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
        let executor = Arc::clone(&executor);
        let verifier = Arc::clone(&verifier);
        let surgeon = Arc::clone(&surgeon);
        let concurrency_limit = Arc::clone(&concurrency_limit);
        let tx = tx.clone();

        join_set.spawn(async move {
            let _permit = concurrency_limit
                .acquire_owned()
                .await
                .expect("executor semaphore should remain available");
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
    mut tickets: Vec<Ticket>,
    mut rx: mpsc::Receiver<Ticket>,
) -> Result<Vec<Ticket>> {
    while let Some(updated_ticket) = rx.recv().await {
        upsert_ticket(&mut tickets, updated_ticket);
        save_state(&state_path, &tickets)?;
    }

    save_state(&state_path, &tickets)?;
    Ok(tickets)
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
