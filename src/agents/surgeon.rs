use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use tracing::{info, warn};

use crate::config::{LlmGateway, LlmProvider, TaskModelConfig};
use crate::skills::Skill;
use crate::utils::clean_code_response;

use super::{Agent, Ticket, TicketStatus};

const SURGEON_NAME: &str = "surgeon";
const DEFAULT_SURGEON_MODEL: &str = "anthropic/claude-3.5-sonnet";

pub struct SurgeonAgent {
    legacy_root: PathBuf,
    modern_root: PathBuf,
    file_io_skill: Arc<dyn Skill>,
    file_write_skill: Arc<dyn Skill>,
    route: TaskModelConfig,
    llm_gateway: LlmGateway,
}

impl SurgeonAgent {
    pub fn new(
        legacy_root: impl Into<PathBuf>,
        modern_root: impl Into<PathBuf>,
        file_io_skill: Arc<dyn Skill>,
        file_write_skill: Arc<dyn Skill>,
        llm_gateway: LlmGateway,
        route: TaskModelConfig,
    ) -> Self {
        Self {
            legacy_root: legacy_root.into(),
            modern_root: modern_root.into(),
            file_io_skill,
            file_write_skill,
            route,
            llm_gateway,
        }
    }

    pub fn default_model() -> &'static str {
        DEFAULT_SURGEON_MODEL
    }

    pub fn provider(&self) -> LlmProvider {
        self.route.provider
    }

    async fn repair_ticket(&self, ticket: &mut Ticket) -> Result<Ticket> {
        ensure!(
            !ticket.modern_file_paths.is_empty(),
            "ticket {} does not contain any modern_file_paths for surgery",
            ticket.id
        );
        ensure!(
            ticket.modern_file_paths.len() == 1,
            "surgeon currently supports exactly one generated file per ticket, got {}",
            ticket.modern_file_paths.len()
        );

        let failure_reason = extract_failure_reason(ticket)?;
        let legacy_context = self
            .read_files(&self.legacy_root, &ticket.context_files)
            .with_context(|| format!("failed to read legacy context for ticket {}", ticket.id))?;
        let modern_context = self
            .read_files(&self.modern_root, &ticket.modern_file_paths)
            .with_context(|| {
                format!(
                    "failed to read failing modern code for ticket {}",
                    ticket.id
                )
            })?;

        let response = self
            .llm_gateway
            .chat_completion(
                &self.route,
                &self.system_prompt(ticket),
                &self.user_prompt(ticket, failure_reason, &legacy_context, &modern_context),
            )
            .await
            .with_context(|| format!("surgeon model failed for ticket {}", ticket.id))?;
        let (response, usage) = response;
        info!(
            ticket_id = ticket.id.as_str(),
            model = self.route.model.as_str(),
            prompt_tokens = usage.prompt_tokens,
            completion_tokens = usage.completion_tokens,
            total_tokens = usage.total_tokens,
            "Surgeon token usage"
        );
        ticket.record_llm_usage(&usage);
        let corrected_code = clean_code_response(&response);

        ensure!(
            !corrected_code.trim().is_empty(),
            "surgeon returned empty code for ticket {}",
            ticket.id
        );

        let target_relative_path = ticket
            .modern_file_paths
            .first()
            .context("ticket does not contain a target modern file path")?;
        let target_output_path = self.modern_root.join(target_relative_path);
        self.file_write_skill
            .execute(vec![
                target_output_path.to_string_lossy().into_owned(),
                corrected_code,
            ])
            .with_context(|| format!("failed to overwrite modern file for ticket {}", ticket.id))?;

        ticket.retries = ticket.retries.saturating_add(1);
        ticket.status = TicketStatus::InProgress;

        info!(
            ticket_id = ticket.id.as_str(),
            retries = ticket.retries,
            target_file = target_relative_path.as_str(),
            "Surgeon patched failing code and returned ticket to verification"
        );

        Ok(ticket.clone())
    }

    fn read_files(&self, root: &Path, relative_paths: &[String]) -> Result<String> {
        ensure!(
            !relative_paths.is_empty(),
            "no relative paths provided for file read from {}",
            root.display()
        );

        let mut args = vec![root.to_string_lossy().into_owned()];
        args.extend(relative_paths.iter().cloned());
        self.file_io_skill.execute(args)
    }

    fn system_prompt(&self, ticket: &Ticket) -> String {
        format!(
            concat!(
                "You are an Expert Debugger and Code Surgeon specializing in {framework} migrations.\n",
                "You will receive legacy code, failing modern code, and a verifier failure reason.\n",
                "Analyze the failure, locate the bug in the modern implementation, and output only the fully corrected raw modern code.\n",
                "Rules:\n",
                "1. No markdown fences.\n",
                "2. No explanations.\n",
                "3. No preambles or summaries.\n",
                "4. Output only the corrected contents for the failing modern file.\n",
                "5. Preserve the intended behavior of the legacy implementation.\n",
                "6. Fix the reported syntax and logic issues without changing unrelated behavior.\n"
            ),
            framework = ticket.target_framework
        )
    }

    fn user_prompt(
        &self,
        ticket: &Ticket,
        failure_reason: &str,
        legacy_context: &str,
        modern_context: &str,
    ) -> String {
        format!(
            concat!(
                "Ticket ID: {ticket_id}\n",
                "Description: {description}\n",
                "Target framework: {target_framework}\n",
                "Failing file: {modern_file}\n",
                "Failure reason: {failure_reason}\n",
                "Legacy code:\n",
                "{legacy_context}\n\n",
                "Failing modern code:\n",
                "{modern_context}\n"
            ),
            ticket_id = ticket.id,
            description = ticket.description,
            target_framework = ticket.target_framework,
            modern_file = ticket.modern_file_paths[0],
            failure_reason = failure_reason,
            legacy_context = legacy_context,
            modern_context = modern_context
        )
    }
}

#[async_trait]
impl Agent for SurgeonAgent {
    fn name(&self) -> &str {
        SURGEON_NAME
    }

    fn model(&self) -> &str {
        &self.route.model
    }

    async fn process_ticket(&self, ticket: &Ticket) -> Result<Ticket> {
        let mut updated_ticket = ticket.clone();

        match self.repair_ticket(&mut updated_ticket).await {
            Ok(updated_ticket) => Ok(updated_ticket),
            Err(error) => {
                let error_message = error.to_string();
                warn!(
                    ticket_id = ticket.id.as_str(),
                    retries = ticket.retries.saturating_add(1),
                    error = error_message.as_str(),
                    "Surgeon failed to repair ticket"
                );

                updated_ticket.retries = updated_ticket.retries.saturating_add(1);
                updated_ticket.status = TicketStatus::Failed(error_message);
                Ok(updated_ticket)
            }
        }
    }
}

fn extract_failure_reason(ticket: &Ticket) -> Result<&str> {
    match &ticket.status {
        TicketStatus::Failed(reason) => Ok(reason.as_str()),
        other => Err(anyhow!(
            "surgeon expected TicketStatus::Failed but received {other:?}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{SurgeonAgent, extract_failure_reason};
    use crate::agents::{Agent, Ticket, TicketStatus};
    use crate::config::{LlmGateway, LlmProvider, TaskKind, TaskModelConfig};
    use crate::skills::{FileIOSkill, FileWriteSkill};
    use std::sync::Arc;

    fn build_agent() -> SurgeonAgent {
        SurgeonAgent::new(
            "./legacy_app",
            "./modern_app",
            Arc::new(FileIOSkill),
            Arc::new(FileWriteSkill),
            LlmGateway::new().expect("gateway should build"),
            TaskModelConfig::new(
                TaskKind::Surgeon,
                LlmProvider::OpenRouter,
                SurgeonAgent::default_model(),
            ),
        )
    }

    #[test]
    fn surgeon_uses_requested_default_model() {
        let agent = build_agent();
        assert_eq!(agent.model(), "anthropic/claude-3.5-sonnet");
    }

    #[test]
    fn surgeon_extracts_failure_reason_from_failed_ticket() {
        let ticket = Ticket {
            id: "SURG-1".to_owned(),
            description: "Repair failing output".to_owned(),
            context_files: vec!["src/server.js".to_owned()],
            status: TicketStatus::Failed("syntax error".to_owned()),
            legacy_code_snippet: "http.createServer(...)".to_owned(),
            target_framework: "TypeScript".to_owned(),
            dependencies: vec!["express".to_owned()],
            modern_file_paths: vec!["src/server.ts".to_owned()],
            retries: 1,
            token_usage: crate::agents::TicketTokenUsage::default(),
        };

        assert_eq!(
            extract_failure_reason(&ticket).expect("failure reason should be extracted"),
            "syntax error"
        );
    }
}
