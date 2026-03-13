use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, ensure};
use tokio::fs;
use tokio::process::Command;
use tracing::{info, warn};

use crate::config::ZeroClawClient;
use crate::skills::Skill;

use super::{Agent, PromptContext, Ticket, TicketStatus};

const SURGEON_NAME: &str = "surgeon";
const DEFAULT_SURGEON_MODEL: &str = "anthropic/claude-sonnet-4.5";

pub struct SurgeonAgent {
    legacy_root: PathBuf,
    modern_root: PathBuf,
    file_io_skill: Arc<dyn Skill>,
    #[allow(dead_code)]
    file_write_skill: Arc<dyn Skill>,
    #[allow(dead_code)]
    llm_client: Arc<ZeroClawClient>,
}

impl SurgeonAgent {
    pub fn new(
        legacy_root: impl Into<PathBuf>,
        modern_root: impl Into<PathBuf>,
        file_io_skill: Arc<dyn Skill>,
        file_write_skill: Arc<dyn Skill>,
        llm_client: Arc<ZeroClawClient>,
    ) -> Self {
        Self {
            legacy_root: legacy_root.into(),
            modern_root: modern_root.into(),
            file_io_skill,
            file_write_skill,
            llm_client,
        }
    }

    pub fn default_model() -> &'static str {
        DEFAULT_SURGEON_MODEL
    }

    pub fn provider(&self) -> &str {
        "openhands"
    }

    async fn repair_ticket(&self, ticket: &mut Ticket) -> Result<Ticket> {
        ensure!(
            !ticket.modern_file_paths.is_empty() || !ticket.test_file_paths.is_empty(),
            "ticket {} does not contain any generated artifacts for surgery",
            ticket.id
        );

        let failure_reason = extract_failure_reason(ticket)?;
        let legacy_context = self
            .read_files(&self.legacy_root, &ticket.context_files)
            .await
            .with_context(|| format!("failed to read legacy context for ticket {}", ticket.id))?;
        let source_context = self
            .read_files(&self.modern_root, &ticket.modern_file_paths)
            .await
            .with_context(|| {
                format!(
                    "failed to read failing modern code for ticket {}",
                    ticket.id
                )
            })?;
        let test_context = self
            .read_files(&self.modern_root, &ticket.test_file_paths)
            .await
            .with_context(|| {
                format!(
                    "failed to read failing generated tests for ticket {}",
                    ticket.id
                )
            })?;
        let execution_diff = serde_json::to_string_pretty(&ticket.last_execution_diff)
            .context("failed to serialize execution diff for surgeon context")?;
        let ast_diff = serde_json::to_string_pretty(&ticket.last_ast_diff)
            .context("failed to serialize AST diff for surgeon context")?;
        let system_prompt = self.system_prompt(ticket);
        let user_prompt = self
            .user_prompt(
                ticket,
                failure_reason,
                &legacy_context,
                &source_context,
                &test_context,
                &execution_diff,
                &ast_diff,
            )
            .await
            .with_context(|| format!("failed to build surgeon prompt for ticket {}", ticket.id))?;

        let modern_files_list = ticket.modern_file_paths.join(", ");
        let full_prompt = format!(
            "{system_prompt}\n\n{user_prompt}\n\nThe shadow tests failed. Use the TerminalTool to run tests if necessary. Use the FileEditorTool to directly edit the failing modern files ({modern_files_list}) to fix the behavioral divergence."
        );

        let prompt_file_path = format!(".prompt_surgeon_{}.txt", ticket.id);
        fs::write(&prompt_file_path, &full_prompt)
            .await
            .with_context(|| {
                format!("failed to write temporary prompt file {}", prompt_file_path)
            })?;

        let mut child = Command::new("uv")
            .arg("run")
            .arg("python")
            .arg("openhands_worker.py")
            .arg("--prompt-file")
            .arg(&prompt_file_path)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| "failed to spawn openhands_worker.py process")?;

        let status = child
            .wait()
            .await
            .with_context(|| "failed to wait on openhands_worker.py process")?;

        let _ = fs::remove_file(&prompt_file_path).await;

        ticket.retries = ticket.retries.saturating_add(1);

        if !status.success() {
            ticket.status =
                TicketStatus::Failed(format!("openhands_worker.py failed with status {}", status));
            return Ok(ticket.clone());
        }

        ticket.status = TicketStatus::InProgress;

        info!(
            ticket_id = ticket.id.as_str(),
            retries = ticket.retries,
            source_file_count = ticket.modern_file_paths.len(),
            test_file_count = ticket.test_file_paths.len(),
            "Surgeon patched failing artifacts and returned ticket to verification"
        );

        Ok(ticket.clone())
    }

    async fn read_files(&self, root: &Path, relative_paths: &[String]) -> Result<PromptContext> {
        if relative_paths.is_empty() {
            return Ok(PromptContext::inline(String::new()));
        }

        let mut args = vec![root.to_string_lossy().into_owned()];
        args.extend(relative_paths.iter().cloned());
        let context_path = self.file_io_skill.execute(args).await?;
        Ok(PromptContext::from_temp_path(context_path))
    }

    fn system_prompt(&self, ticket: &Ticket) -> String {
        format!(
            concat!(
                "You are an Expert Debugger and Code Surgeon specializing in {framework} migrations.\n",
                "You will receive legacy code, failing generated source files, failing generated tests, a semantic execution diff, an AST diff, and a verifier failure reason.\n",
                "You must respond by calling the provided tool exactly once.\n",
                "Rules:\n",
                "1. Return corrected versions of the failing generated source files and/or tests only.\n",
                "2. Use the exact same relative file paths as the failing generated artifacts.\n",
                "3. Do not return markdown fences, prose, or explanations.\n",
                "4. Each file `content` must be raw compile-ready source code.\n",
                "5. Preserve the intended behavior of the legacy implementation.\n",
                "6. Fix the minimum set of files necessary for the behavioral shadow tests to match the legacy outputs.\n",
                "7. Use the execution diff and AST diff to repair the exact missing or divergent logic.\n"
            ),
            framework = ticket.target_framework
        )
    }

    #[allow(clippy::too_many_arguments)]
    async fn user_prompt(
        &self,
        ticket: &Ticket,
        failure_reason: &str,
        legacy_context: &PromptContext,
        source_context: &PromptContext,
        test_context: &PromptContext,
        execution_diff: &str,
        ast_diff: &str,
    ) -> Result<String> {
        let total_len_hint = legacy_context.byte_len_hint().await?
            + source_context.byte_len_hint().await?
            + test_context.byte_len_hint().await?;
        let execution_sections_len = execution_diff.len() + ast_diff.len();
        let mut prompt = String::with_capacity(total_len_hint + execution_sections_len + 512);
        prompt.push_str(&format!(
            concat!(
                "Ticket ID: {ticket_id}\n",
                "Description: {description}\n",
                "Target framework: {target_framework}\n",
                "Failing source files: {modern_files}\n",
                "Failing test files: {test_files}\n",
                "Failure reason: {failure_reason}\n",
                "Legacy code:\n"
            ),
            ticket_id = ticket.id,
            description = ticket.description,
            target_framework = ticket.target_framework,
            modern_files = ticket.modern_file_paths.join(", "),
            test_files = ticket.test_file_paths.join(", "),
            failure_reason = failure_reason
        ));
        legacy_context.append_to(&mut prompt).await?;
        prompt.push_str("\n\nGenerated source files:\n");
        source_context.append_to(&mut prompt).await?;
        prompt.push_str("\n\nGenerated tests:\n");
        test_context.append_to(&mut prompt).await?;
        prompt.push_str("\n\nExecution diff JSON:\n");
        prompt.push_str(execution_diff);
        prompt.push_str("\n\nAST diff JSON:\n");
        prompt.push_str(ast_diff);
        prompt.push('\n');
        Ok(prompt)
    }
}

impl Agent for SurgeonAgent {
    fn name(&self) -> &str {
        SURGEON_NAME
    }

    fn model(&self) -> &str {
        "openhands"
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
    use super::extract_failure_reason;
    use crate::agents::{Ticket, TicketStatus};

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
            test_file_paths: vec!["tests/src/server.test.ts".to_owned()],
            retries: 1,
            token_usage: crate::agents::TicketTokenUsage::default(),
            last_execution_diff: None,
            last_ast_diff: None,
        };

        assert_eq!(
            extract_failure_reason(&ticket).expect("failure reason should be extracted"),
            "syntax error"
        );
    }
}
