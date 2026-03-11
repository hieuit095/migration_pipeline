use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use serde::Deserialize;
use tracing::{error, info};

use crate::config::{LlmGateway, LlmProvider, TaskModelConfig};
use crate::skills::{Skill, TerminalCommandOutput};
use crate::utils::clean_json_response;

use super::{Agent, Ticket, TicketStatus};

const VERIFIER_NAME: &str = "verifier";
const DEFAULT_VERIFIER_MODEL: &str = "z-ai/glm-5";

#[derive(Debug, Deserialize)]
struct VerificationReport {
    is_semantically_equivalent: bool,
    syntax_or_logic_issues: Vec<String>,
    generated_test_code: String,
}

pub struct VerifierAgent {
    legacy_root: PathBuf,
    modern_root: PathBuf,
    file_io_skill: Arc<dyn Skill>,
    file_write_skill: Arc<dyn Skill>,
    sandbox_skill: Arc<dyn Skill>,
    route: TaskModelConfig,
    llm_gateway: LlmGateway,
}

impl VerifierAgent {
    pub fn new(
        legacy_root: impl Into<PathBuf>,
        modern_root: impl Into<PathBuf>,
        file_io_skill: Arc<dyn Skill>,
        file_write_skill: Arc<dyn Skill>,
        sandbox_skill: Arc<dyn Skill>,
        llm_gateway: LlmGateway,
        route: TaskModelConfig,
    ) -> Self {
        Self {
            legacy_root: legacy_root.into(),
            modern_root: modern_root.into(),
            file_io_skill,
            file_write_skill,
            sandbox_skill,
            route,
            llm_gateway,
        }
    }

    pub fn default_model() -> &'static str {
        DEFAULT_VERIFIER_MODEL
    }

    pub fn provider(&self) -> LlmProvider {
        self.route.provider
    }

    async fn verify_ticket(&self, ticket: &mut Ticket) -> Result<Ticket> {
        ensure!(
            !ticket.modern_file_paths.is_empty(),
            "ticket {} does not have any generated modern file paths to verify",
            ticket.id
        );

        let legacy_context = self
            .read_files(&self.legacy_root, &ticket.context_files)
            .with_context(|| format!("failed to read legacy context for ticket {}", ticket.id))?;
        let modern_context = self
            .read_files(&self.modern_root, &ticket.modern_file_paths)
            .with_context(|| format!("failed to read modern context for ticket {}", ticket.id))?;

        let response = self
            .llm_gateway
            .chat_completion(
                &self.route,
                &self.system_prompt(ticket),
                &self.user_prompt(ticket, &legacy_context, &modern_context),
            )
            .await
            .with_context(|| format!("verifier model failed for ticket {}", ticket.id))?;
        let (response, usage) = response;
        info!(
            ticket_id = ticket.id.as_str(),
            model = self.route.model.as_str(),
            prompt_tokens = usage.prompt_tokens,
            completion_tokens = usage.completion_tokens,
            total_tokens = usage.total_tokens,
            "Verifier token usage"
        );
        ticket.record_llm_usage(&usage);
        let report = self
            .parse_report(&response)
            .with_context(|| format!("failed to parse verifier report for ticket {}", ticket.id))?;

        if !report.is_semantically_equivalent || !report.syntax_or_logic_issues.is_empty() {
            let failure_reason = if report.syntax_or_logic_issues.is_empty() {
                "semantic equivalence check failed".to_owned()
            } else {
                report.syntax_or_logic_issues.join(", ")
            };

            error!(
                ticket_id = ticket.id.as_str(),
                issues = failure_reason.as_str(),
                "Verifier rejected generated code"
            );

            ticket.status = TicketStatus::Failed(failure_reason);
            return Ok(ticket.clone());
        }

        ensure!(
            !report.generated_test_code.trim().is_empty(),
            "verifier returned empty generated_test_code for ticket {}",
            ticket.id
        );

        let test_relative_path = self.determine_test_relative_path(ticket)?;
        let test_output_path = self.modern_root.join(&test_relative_path);
        self.file_write_skill
            .execute(vec![
                test_output_path.to_string_lossy().into_owned(),
                report.generated_test_code,
            ])
            .with_context(|| {
                format!(
                    "failed to write generated test file for ticket {}",
                    ticket.id
                )
            })?;

        for modern_file in &ticket.modern_file_paths {
            self.run_syntax_check(modern_file)
                .with_context(|| format!("syntax check failed for modern file `{modern_file}`"))?;
        }
        self.run_syntax_check(&test_relative_path)
            .with_context(|| {
                format!("syntax check failed for verifier test `{test_relative_path}`")
            })?;

        info!(
            ticket_id = ticket.id.as_str(),
            test_path = test_relative_path.as_str(),
            "Verifier accepted generated code and test"
        );

        ticket.status = TicketStatus::Verified;
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
                "You are a Staff Verification Engineer specializing in semantic validation for {framework} migrations.\n",
                "Compare the legacy implementation against the generated modern implementation.\n",
                "Return only strict JSON matching this exact schema:\n",
                "{{",
                "\"is_semantically_equivalent\": true,",
                "\"syntax_or_logic_issues\": [\"issue if any\"],",
                "\"generated_test_code\": \"test code as a JSON string\"",
                "}}\n",
                "Rules:\n",
                "1. No markdown fences.\n",
                "2. No prose outside the JSON object.\n",
                "3. `syntax_or_logic_issues` must be an array of concrete findings.\n",
                "4. `generated_test_code` must contain only compile-ready unit test source code.\n",
                "5. If there is any semantic mismatch, set `is_semantically_equivalent` to false.\n"
            ),
            framework = ticket.target_framework
        )
    }

    fn user_prompt(&self, ticket: &Ticket, legacy_context: &str, modern_context: &str) -> String {
        format!(
            concat!(
                "Ticket ID: {ticket_id}\n",
                "Description: {description}\n",
                "Target framework: {target_framework}\n",
                "Legacy files: {legacy_files}\n",
                "Modern files: {modern_files}\n",
                "Legacy code:\n",
                "{legacy_context}\n\n",
                "Modern code:\n",
                "{modern_context}\n"
            ),
            ticket_id = ticket.id,
            description = ticket.description,
            target_framework = ticket.target_framework,
            legacy_files = ticket.context_files.join(", "),
            modern_files = ticket.modern_file_paths.join(", "),
            legacy_context = legacy_context,
            modern_context = modern_context
        )
    }

    fn parse_report(&self, raw_response: &str) -> Result<VerificationReport> {
        let cleaned = clean_json_response(raw_response);
        serde_json::from_str::<VerificationReport>(&cleaned).with_context(|| {
            format!(
                "invalid verifier JSON payload: {}",
                truncate_for_error(&cleaned)
            )
        })
    }

    fn determine_test_relative_path(&self, ticket: &Ticket) -> Result<String> {
        let first_modern_file = ticket
            .modern_file_paths
            .first()
            .context("ticket does not contain modern file paths for test generation")?;
        let modern_path = Path::new(first_modern_file);
        let extension = modern_path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("txt");
        let stem = modern_path
            .file_stem()
            .and_then(|value| value.to_str())
            .filter(|value| !value.is_empty())
            .unwrap_or("generated");
        let parent = modern_path.parent().unwrap_or_else(|| Path::new(""));
        let file_name = if extension == "go" {
            format!("{stem}_test.{extension}")
        } else if extension == "py" {
            format!("test_{stem}.{extension}")
        } else {
            format!("{stem}.test.{extension}")
        };

        let relative_dir = if parent.as_os_str().is_empty() {
            PathBuf::from("tests")
        } else {
            Path::new("tests").join(parent)
        };

        Ok(relative_dir
            .join(file_name)
            .to_string_lossy()
            .replace('\\', "/"))
    }

    fn run_syntax_check(&self, relative_path: &str) -> Result<()> {
        let full_path = self.modern_root.join(relative_path);
        let result = self
            .sandbox_skill
            .execute(vec![
                self.modern_root.to_string_lossy().into_owned(),
                relative_path.to_owned(),
            ])
            .with_context(|| format!("sandbox execution failed for {}", full_path.display()))?;
        let output: TerminalCommandOutput =
            serde_json::from_str(&result).context("failed to parse sandbox output payload")?;

        if output.exit_code != 0 {
            let failure_output = if output.stderr.trim().is_empty() {
                output.stdout
            } else {
                output.stderr
            };
            return Err(anyhow!(
                "syntax check failed for {} using `{}`: {}",
                relative_path,
                output.command,
                failure_output.trim()
            ));
        }

        info!(
            file = relative_path,
            command = output.command.as_str(),
            "Sandboxed syntax check completed"
        );
        Ok(())
    }
}

#[async_trait]
impl Agent for VerifierAgent {
    fn name(&self) -> &str {
        VERIFIER_NAME
    }

    fn model(&self) -> &str {
        &self.route.model
    }

    async fn process_ticket(&self, ticket: &Ticket) -> Result<Ticket> {
        let mut updated_ticket = ticket.clone();

        match self.verify_ticket(&mut updated_ticket).await {
            Ok(result) => Ok(result),
            Err(error) => {
                let error_message = error.to_string();
                error!(
                    ticket_id = ticket.id.as_str(),
                    error = error_message.as_str(),
                    "Verifier encountered an unexpected failure"
                );

                updated_ticket.status = TicketStatus::Failed(error_message);
                Ok(updated_ticket)
            }
        }
    }
}

fn truncate_for_error(value: &str) -> String {
    const MAX_LEN: usize = 500;

    if value.len() <= MAX_LEN {
        value.to_owned()
    } else {
        format!("{}...", &value[..MAX_LEN])
    }
}

#[cfg(test)]
mod tests {
    use super::VerifierAgent;
    use crate::agents::{Ticket, TicketStatus};
    use crate::config::{LlmGateway, LlmProvider, TaskKind, TaskModelConfig};
    use crate::skills::{FileIOSkill, FileWriteSkill, SandboxSkill};
    use std::sync::Arc;

    fn build_agent() -> VerifierAgent {
        VerifierAgent::new(
            "./legacy_app",
            "./modern_app",
            Arc::new(FileIOSkill),
            Arc::new(FileWriteSkill),
            Arc::new(SandboxSkill),
            LlmGateway::new().expect("gateway should build"),
            TaskModelConfig::new(
                TaskKind::Verifier,
                LlmProvider::OpenRouter,
                VerifierAgent::default_model(),
            ),
        )
    }

    #[test]
    fn verifier_parses_valid_json_report() {
        let agent = build_agent();
        let report = agent
            .parse_report(
                r#"```json
                {
                    "is_semantically_equivalent": true,
                    "syntax_or_logic_issues": [],
                    "generated_test_code": "export const ok = true;"
                }
                ```"#,
            )
            .expect("report should parse");

        assert!(report.is_semantically_equivalent);
        assert!(report.syntax_or_logic_issues.is_empty());
    }

    #[test]
    fn verifier_builds_test_path_from_modern_file() {
        let agent = build_agent();
        let ticket = Ticket {
            id: "VERIFY-1".to_owned(),
            description: "Verify translation".to_owned(),
            context_files: vec!["src/server.js".to_owned()],
            status: TicketStatus::InProgress,
            legacy_code_snippet: "http.createServer(...)".to_owned(),
            target_framework: "TypeScript".to_owned(),
            dependencies: vec!["express".to_owned()],
            modern_file_paths: vec!["src/server.ts".to_owned()],
            retries: 0,
            token_usage: crate::agents::TicketTokenUsage::default(),
        };

        let path = agent
            .determine_test_relative_path(&ticket)
            .expect("test path should be derived");

        assert_eq!(path, "tests/src/server.test.ts");
    }
}
