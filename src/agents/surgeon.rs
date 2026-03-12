use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, ensure};
use serde::Deserialize;
use tracing::{info, warn};
use zeroclaw::tools::ToolSpec;

use crate::config::{TaskKind, ZeroClawClient};
use crate::skills::Skill;
use crate::utils::path::normalize_relative_path as normalize_portable_relative_path;

use super::{Agent, PromptContext, Ticket, TicketStatus};

const SURGEON_NAME: &str = "surgeon";
const DEFAULT_SURGEON_MODEL: &str = "anthropic/claude-sonnet-4.5";

#[derive(Debug, Deserialize)]
struct FixedFilesPayload {
    files: Vec<FixedFile>,
}

#[derive(Debug, Deserialize)]
struct FixedFile {
    path: String,
    content: String,
}

pub struct SurgeonAgent {
    legacy_root: PathBuf,
    modern_root: PathBuf,
    file_io_skill: Arc<dyn Skill>,
    file_write_skill: Arc<dyn Skill>,
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
        self.llm_client.provider_for(TaskKind::Surgeon)
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

        let structured_call = self
            .llm_client
            .chat_with_schema(
                TaskKind::Surgeon,
                &system_prompt,
                &user_prompt,
                self.fixed_files_tool(),
            )
            .await
            .with_context(|| format!("surgeon model failed for ticket {}", ticket.id))?;
        info!(
            ticket_id = ticket.id.as_str(),
            model = self.model(),
            prompt_tokens = structured_call.usage.prompt_tokens,
            completion_tokens = structured_call.usage.completion_tokens,
            total_tokens = structured_call.usage.total_tokens,
            "Surgeon token usage"
        );
        ticket.record_llm_usage(&structured_call.usage);
        let payload: FixedFilesPayload = structured_call.deserialize_arguments()?;

        self.persist_fixed_files(ticket, payload).await?;

        ticket.retries = ticket.retries.saturating_add(1);
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

    async fn persist_fixed_files(&self, ticket: &Ticket, payload: FixedFilesPayload) -> Result<()> {
        ensure!(
            !payload.files.is_empty(),
            "surgeon returned no files for ticket {}",
            ticket.id
        );

        let allowed_paths = ticket
            .modern_file_paths
            .iter()
            .chain(ticket.test_file_paths.iter())
            .map(|path| normalize_relative_path(path))
            .collect::<Result<HashSet<_>>>()?;

        for file in payload.files {
            ensure!(
                !file.content.trim().is_empty(),
                "surgeon returned empty file content for ticket {} at path `{}`",
                ticket.id,
                file.path
            );

            let relative_path = normalize_relative_path(&file.path).with_context(|| {
                format!(
                    "surgeon returned an invalid file path for ticket {}",
                    ticket.id
                )
            })?;
            ensure!(
                allowed_paths.contains(&relative_path),
                "surgeon attempted to modify an unexpected file `{relative_path}` for ticket {}",
                ticket.id
            );

            let target_output_path = self.modern_root.join(&relative_path);
            self.file_write_skill
                .execute(vec![
                    target_output_path.to_string_lossy().into_owned(),
                    file.content,
                ])
                .await
                .with_context(|| {
                    format!(
                        "failed to overwrite modern file `{relative_path}` for ticket {}",
                        ticket.id
                    )
                })?;
        }

        Ok(())
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

    fn fixed_files_tool(&self) -> ToolSpec {
        ToolSpec {
            name: "submit_fixed_files".to_owned(),
            description: "Return the corrected contents for the failing modern files".to_owned(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": { "type": "string" },
                                "content": { "type": "string" }
                            },
                            "required": ["path", "content"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["files"],
                "additionalProperties": false
            }),
        }
    }
}

impl Agent for SurgeonAgent {
    fn name(&self) -> &str {
        SURGEON_NAME
    }

    fn model(&self) -> &str {
        self.llm_client.model_for(TaskKind::Surgeon)
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

fn normalize_relative_path(path: &str) -> Result<String> {
    normalize_portable_relative_path(path, "file path").map(|path| path.into_string())
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
