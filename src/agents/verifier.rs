use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};
use zeroclaw::tools::ToolSpec;

use crate::config::{TaskKind, ZeroClawClient};
use crate::skills::{
    AstDiff, ExecutionDiff, ExecutionTarget, ShadowFixture, ShadowTestRequest, Skill,
    TerminalCommandOutput, diff_dependency_graphs, parse_dependency_graph_json,
};
use crate::utils::path::normalize_relative_path as normalize_portable_relative_path;

use super::{Agent, Ticket, TicketStatus};

const VERIFIER_NAME: &str = "verifier";
const DEFAULT_VERIFIER_MODEL: &str = "z-ai/glm-5";
const MAX_SHADOW_FIXTURES: usize = 8;

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ShadowFixturePlan {
    legacy_target: ExecutionTarget,
    modern_target: ExecutionTarget,
    fixtures: Vec<ShadowFixture>,
}

pub struct VerifierAgent {
    legacy_root: PathBuf,
    modern_root: PathBuf,
    ast_parsing_skill: Arc<dyn Skill>,
    sandbox_skill: Arc<dyn Skill>,
    shadow_test_skill: Arc<dyn Skill>,
    llm_client: Arc<ZeroClawClient>,
}

impl VerifierAgent {
    pub fn new(
        legacy_root: impl Into<PathBuf>,
        modern_root: impl Into<PathBuf>,
        ast_parsing_skill: Arc<dyn Skill>,
        sandbox_skill: Arc<dyn Skill>,
        shadow_test_skill: Arc<dyn Skill>,
        llm_client: Arc<ZeroClawClient>,
    ) -> Self {
        Self {
            legacy_root: legacy_root.into(),
            modern_root: modern_root.into(),
            ast_parsing_skill,
            sandbox_skill,
            shadow_test_skill,
            llm_client,
        }
    }

    pub fn default_model() -> &'static str {
        DEFAULT_VERIFIER_MODEL
    }

    pub fn provider(&self) -> &str {
        self.llm_client.provider_for(TaskKind::Verifier)
    }

    async fn verify_ticket(&self, ticket: &mut Ticket) -> Result<Ticket> {
        ensure!(
            !ticket.context_files.is_empty(),
            "ticket {} does not provide legacy context files for semantic verification",
            ticket.id
        );
        ensure!(
            !ticket.modern_file_paths.is_empty(),
            "ticket {} does not have any generated source files to verify",
            ticket.id
        );

        let legacy_graph_json = self
            .build_dependency_graph(&self.legacy_root, &ticket.context_files)
            .await
            .with_context(|| {
                format!("failed to build legacy AST graph for ticket {}", ticket.id)
            })?;
        let legacy_graph = parse_dependency_graph_json(&legacy_graph_json)?;

        let modern_graph_result = self
            .build_dependency_graph(&self.modern_root, &ticket.modern_file_paths)
            .await;
        let (modern_graph_json, ast_diff) = match modern_graph_result {
            Ok(graph_json) => {
                let modern_graph = parse_dependency_graph_json(&graph_json)?;
                (
                    Some(graph_json),
                    diff_dependency_graphs(&legacy_graph, &modern_graph),
                )
            }
            Err(error) => {
                warn!(
                    ticket_id = ticket.id.as_str(),
                    error = %error,
                    "Verifier could not build a modern AST graph; falling back to execution-only diffing"
                );
                (
                    None,
                    AstDiff {
                        unsupported_files: ticket.modern_file_paths.clone(),
                        notes: vec![error.to_string()],
                        ..AstDiff::default()
                    },
                )
            }
        };

        if should_use_sandbox_verification(ticket) {
            info!(
                ticket_id = ticket.id.as_str(),
                "Verifier selected sandbox-only validation because the ticket does not expose a shadow-executable modern target"
            );
            let sandbox_result = self.execute_generated_tests(ticket).await?;
            ticket.last_ast_diff = Some(ast_diff.clone());

            if sandbox_result.exit_code == 0 {
                info!(
                    ticket_id = ticket.id.as_str(),
                    command = sandbox_result.command.as_str(),
                    "Verifier accepted generated code after sandbox test execution"
                );
                ticket.status = TicketStatus::Verified;
                return Ok(ticket.clone());
            }

            let failure_reason = build_sandbox_failure_summary(&sandbox_result, &ast_diff);
            error!(
                ticket_id = ticket.id.as_str(),
                failure = failure_reason.as_str(),
                "Verifier rejected generated artifacts after sandbox test execution"
            );
            ticket.status = TicketStatus::Failed(failure_reason);
            return Ok(ticket.clone());
        }

        let structured_call = self
            .llm_client
            .chat_with_schema(
                TaskKind::Verifier,
                &self.system_prompt(ticket),
                &self.user_prompt(ticket, &legacy_graph_json, modern_graph_json.as_deref()),
                self.generate_shadow_fixtures_tool(),
            )
            .await
            .with_context(|| format!("verifier model failed for ticket {}", ticket.id))?;
        info!(
            ticket_id = ticket.id.as_str(),
            model = self.model(),
            prompt_tokens = structured_call.usage.prompt_tokens,
            completion_tokens = structured_call.usage.completion_tokens,
            total_tokens = structured_call.usage.total_tokens,
            "Verifier token usage"
        );
        ticket.record_llm_usage(&structured_call.usage);

        let plan: ShadowFixturePlan = structured_call.deserialize_arguments()?;
        validate_shadow_plan(ticket, &plan)?;
        let execution_diff = self.execute_shadow_tests(ticket, &plan).await?;

        ticket.last_execution_diff = Some(execution_diff.clone());
        ticket.last_ast_diff = Some(ast_diff.clone());

        if execution_diff.equivalent {
            info!(
                ticket_id = ticket.id.as_str(),
                fixture_count = execution_diff.fixture_diffs.len(),
                "Verifier accepted generated code after shadow testing"
            );
            ticket.status = TicketStatus::Verified;
            return Ok(ticket.clone());
        }

        let failure_reason = build_failure_summary(&execution_diff, &ast_diff);
        error!(
            ticket_id = ticket.id.as_str(),
            failure = failure_reason.as_str(),
            "Verifier rejected generated artifacts after semantic shadow testing"
        );
        ticket.status = TicketStatus::Failed(failure_reason);
        Ok(ticket.clone())
    }

    async fn build_dependency_graph(
        &self,
        root: &Path,
        relative_paths: &[String],
    ) -> Result<String> {
        let mut args = vec![root.to_string_lossy().into_owned()];
        args.extend(relative_paths.iter().cloned());
        self.ast_parsing_skill.execute(args).await
    }

    async fn execute_shadow_tests(
        &self,
        ticket: &Ticket,
        plan: &ShadowFixturePlan,
    ) -> Result<ExecutionDiff> {
        let request = ShadowTestRequest {
            legacy_root: self.legacy_root.to_string_lossy().into_owned(),
            modern_root: self.modern_root.to_string_lossy().into_owned(),
            legacy_target: plan.legacy_target.clone(),
            modern_target: plan.modern_target.clone(),
            fixtures: plan.fixtures.clone(),
        };
        let payload = serde_json::to_string(&request).with_context(|| {
            format!(
                "failed to serialize shadow request for ticket {}",
                ticket.id
            )
        })?;
        let raw_diff = self
            .shadow_test_skill
            .execute(vec![payload])
            .await
            .with_context(|| format!("shadow test execution failed for ticket {}", ticket.id))?;
        serde_json::from_str(&raw_diff).context("failed to deserialize shadow execution diff")
    }

    async fn execute_generated_tests(&self, ticket: &Ticket) -> Result<TerminalCommandOutput> {
        ensure!(
            !ticket.test_file_paths.is_empty(),
            "ticket {} does not have any generated tests to execute",
            ticket.id
        );

        let mut args = vec![self.modern_root.to_string_lossy().into_owned()];
        args.extend(ticket.test_file_paths.iter().cloned());
        let raw_output =
            self.sandbox_skill.execute(args).await.with_context(|| {
                format!("sandbox test execution failed for ticket {}", ticket.id)
            })?;
        serde_json::from_str(&raw_output).context("failed to deserialize sandbox execution output")
    }

    fn system_prompt(&self, ticket: &Ticket) -> String {
        format!(
            concat!(
                "You are an uncompromising semantic equivalence judge for {framework} migrations.\n",
                "Your task is to design shadow-test fixtures that prove whether the modern implementation matches the legacy behavior.\n",
                "You must respond by calling the provided tool exactly once.\n",
                "Rules:\n",
                "1. Select one callable entry point from the supplied legacy files and one callable entry point from the supplied modern files.\n",
                "2. Only choose JavaScript, TypeScript, TSX, Python, or Go source files. Never choose configuration or manifest files such as package.json or tsconfig.json.\n",
                "3. Prefer callables that are exported or otherwise directly reachable from module scope without editing the source files.\n",
                "4. If the legacy module is CommonJS and the callable is a top-level function declaration, you may target that function name directly.\n",
                "5. Generate between 3 and {max_fixtures} JSON fixtures total. Prefer the smallest set that still covers the happy path, not-found behavior, nullability, one malformed input, and one boundary case.\n",
                "6. Every fixture must use positional `args` only.\n",
                "7. If the selected callable expects a callback argument, represent that callback position as {{\"capture\":\"callback\"}}. Do not use null, undefined, functions, or prose for callback placeholders.\n",
                "8. Do not use undefined, NaN, Infinity, symbols, or any non-JSON values anywhere in `args`.\n",
                "9. Prefer fixtures that exercise every branch visible in the legacy AST.\n",
                "10. Use only the file paths explicitly provided in the prompt.\n",
                "11. Do not emit prose, markdown, or explanations.\n"
            ),
            framework = ticket.target_framework,
            max_fixtures = MAX_SHADOW_FIXTURES
        )
    }

    fn user_prompt(
        &self,
        ticket: &Ticket,
        legacy_graph_json: &str,
        modern_graph_json: Option<&str>,
    ) -> String {
        let modern_graph_section = modern_graph_json.unwrap_or(
            "{\"warning\":\"modern AST unavailable; choose from the provided modern files only\"}",
        );
        format!(
            concat!(
                "Ticket ID: {ticket_id}\n",
                "Description: {description}\n",
                "Legacy files: {legacy_files}\n",
                "Modern files: {modern_files}\n",
                "Legacy AST graph JSON:\n",
                "{legacy_graph_json}\n\n",
                "Modern AST graph JSON:\n",
                "{modern_graph_json}\n"
            ),
            ticket_id = ticket.id,
            description = ticket.description,
            legacy_files = ticket.context_files.join(", "),
            modern_files = ticket.modern_file_paths.join(", "),
            legacy_graph_json = legacy_graph_json,
            modern_graph_json = modern_graph_section
        )
    }

    fn generate_shadow_fixtures_tool(&self) -> ToolSpec {
        ToolSpec {
            name: "generate_shadow_fixtures".to_owned(),
            description: "Select matching execution entry points and generate exhaustive JSON fixtures for semantic shadow testing".to_owned(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "legacy_target": execution_target_schema(),
                    "modern_target": execution_target_schema(),
                    "fixtures": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_SHADOW_FIXTURES,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string" },
                                "description": { "type": "string" },
                                "args": {
                                    "type": "array",
                                    "items": {}
                                }
                            },
                            "required": ["id", "description", "args"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["legacy_target", "modern_target", "fixtures"],
                "additionalProperties": false
            }),
        }
    }
}

impl Agent for VerifierAgent {
    fn name(&self) -> &str {
        VERIFIER_NAME
    }

    fn model(&self) -> &str {
        self.llm_client.model_for(TaskKind::Verifier)
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

fn validate_shadow_plan(ticket: &Ticket, plan: &ShadowFixturePlan) -> Result<()> {
    let normalized_legacy_target = normalize_relative_path(&plan.legacy_target.relative_path)
        .with_context(|| "verifier returned an invalid legacy target path")?;
    let normalized_modern_target = normalize_relative_path(&plan.modern_target.relative_path)
        .with_context(|| "verifier returned an invalid modern target path")?;

    ensure!(
        !plan.fixtures.is_empty(),
        "verifier returned no shadow fixtures for ticket {}",
        ticket.id
    );
    ensure!(
        ticket
            .context_files
            .iter()
            .any(|path| normalize_relative_path(path).ok().as_deref()
                == Some(normalized_legacy_target.as_str())),
        "verifier selected legacy target `{}` outside the allowed context files for ticket {}",
        plan.legacy_target.relative_path,
        ticket.id
    );
    ensure!(
        ticket
            .modern_file_paths
            .iter()
            .any(|path| normalize_relative_path(path).ok().as_deref()
                == Some(normalized_modern_target.as_str())),
        "verifier selected modern target `{}` outside the allowed modern files for ticket {}",
        plan.modern_target.relative_path,
        ticket.id
    );
    Ok(())
}

fn execution_target_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "relative_path": { "type": "string" },
            "callable": { "type": "string" }
        },
        "required": ["relative_path", "callable"],
        "additionalProperties": false
    })
}

fn build_failure_summary(execution_diff: &ExecutionDiff, ast_diff: &AstDiff) -> String {
    let mut details = execution_diff
        .fixture_diffs
        .iter()
        .flat_map(|fixture_diff| fixture_diff.differences.iter().cloned())
        .take(5)
        .collect::<Vec<_>>();
    details.extend(
        execution_diff
            .sandbox_execution_errors
            .iter()
            .take(5)
            .cloned(),
    );
    if !ast_diff.is_empty() {
        details.push(format!(
            "ast_diff: missing_functions={}, missing_classes={}, missing_imports={}, missing_branches={}",
            ast_diff.missing_functions.len(),
            ast_diff.missing_classes.len(),
            ast_diff.missing_imports.len(),
            ast_diff.missing_branches.len()
        ));
    }

    if details.is_empty() {
        "semantic shadow testing failed without a detailed diff".to_owned()
    } else {
        details.join(" | ")
    }
}

fn build_sandbox_failure_summary(output: &TerminalCommandOutput, ast_diff: &AstDiff) -> String {
    let mut details = Vec::new();
    details.push(format!(
        "sandbox test command failed with exit code {}",
        output.exit_code
    ));

    if !output.stderr.trim().is_empty() {
        details.push(format!("stderr: {}", truncate_for_summary(&output.stderr)));
    } else if !output.stdout.trim().is_empty() {
        details.push(format!("stdout: {}", truncate_for_summary(&output.stdout)));
    }

    if !ast_diff.is_empty() {
        details.push(format!(
            "ast_diff: missing_functions={}, missing_classes={}, missing_imports={}, missing_branches={}",
            ast_diff.missing_functions.len(),
            ast_diff.missing_classes.len(),
            ast_diff.missing_imports.len(),
            ast_diff.missing_branches.len()
        ));
    }

    details.join(" | ")
}

fn should_use_sandbox_verification(ticket: &Ticket) -> bool {
    !ticket
        .modern_file_paths
        .iter()
        .any(|path| is_shadow_executable_path(path))
}

fn is_shadow_executable_path(path: &str) -> bool {
    matches!(
        Path::new(path)
            .extension()
            .and_then(|value| value.to_str())
            .map(|value| value.to_ascii_lowercase())
            .as_deref(),
        Some("js") | Some("mjs") | Some("cjs") | Some("ts") | Some("tsx") | Some("py") | Some("go")
    )
}

fn truncate_for_summary(value: &str) -> String {
    const MAX_SUMMARY_CHARS: usize = 240;
    let trimmed = value.trim();
    if trimmed.chars().count() <= MAX_SUMMARY_CHARS {
        trimmed.to_owned()
    } else {
        let truncated = trimmed.chars().take(MAX_SUMMARY_CHARS).collect::<String>();
        format!("{truncated}...")
    }
}

fn normalize_relative_path(path: &str) -> Result<String> {
    normalize_portable_relative_path(path, "file path").map(|path| path.into_string())
}

#[cfg(test)]
mod tests {
    use super::{
        ShadowFixturePlan, build_failure_summary, is_shadow_executable_path,
        should_use_sandbox_verification,
    };
    use crate::agents::{Ticket, TicketStatus, TicketTokenUsage};
    use crate::skills::{
        AstDiff, ExecutionDiff, ExecutionTarget, ShadowFixture, TerminalCommandOutput,
    };

    #[test]
    fn shadow_fixture_plan_round_trips() {
        let plan = ShadowFixturePlan {
            legacy_target: ExecutionTarget {
                relative_path: "src/server.js".to_owned(),
                callable: "bootstrap".to_owned(),
            },
            modern_target: ExecutionTarget {
                relative_path: "src/server.ts".to_owned(),
                callable: "bootstrap".to_owned(),
            },
            fixtures: vec![ShadowFixture {
                id: "edge-null".to_owned(),
                description: "null input".to_owned(),
                args: vec![serde_json::json!(null)],
            }],
        };

        let value = serde_json::to_value(&plan).expect("plan should serialize");
        assert_eq!(value["fixtures"][0]["id"], "edge-null");
    }

    #[test]
    fn failure_summary_includes_execution_and_ast_signals() {
        let execution_diff = ExecutionDiff {
            equivalent: false,
            fixture_diffs: Vec::new(),
            sandbox_execution_errors: vec!["fixture `fx-1` return value mismatch".to_owned()],
        };
        let _fixture = ShadowFixture {
            id: "fx-1".to_owned(),
            description: "basic".to_owned(),
            args: Vec::new(),
        };
        let ast_diff = AstDiff {
            missing_branches: vec!["src/server.js::if::port > 0".to_owned()],
            ..AstDiff::default()
        };

        let summary = build_failure_summary(&execution_diff, &ast_diff);
        assert!(summary.contains("return value mismatch"));
        assert!(summary.contains("ast_diff"));
    }

    #[test]
    fn sandbox_verification_detects_non_executable_modern_outputs() {
        let ticket = Ticket {
            id: "CFG-001".to_owned(),
            description: "baseline".to_owned(),
            context_files: vec!["server.js".to_owned()],
            status: TicketStatus::InProgress,
            legacy_code_snippet: "baseline".to_owned(),
            target_framework: "TypeScript".to_owned(),
            dependencies: vec!["typescript".to_owned()],
            modern_file_paths: vec!["tsconfig.json".to_owned(), "package.json".to_owned()],
            test_file_paths: vec!["tests/migration_smoke.test.ts".to_owned()],
            retries: 0,
            token_usage: TicketTokenUsage::default(),
            last_execution_diff: None,
            last_ast_diff: None,
        };

        assert!(should_use_sandbox_verification(&ticket));
        assert!(!is_shadow_executable_path("package.json"));
        assert!(is_shadow_executable_path("tests/migration_smoke.test.ts"));
    }

    #[test]
    fn sandbox_failure_summary_includes_exit_code_and_output() {
        let output = TerminalCommandOutput {
            command: "node --test".to_owned(),
            exit_code: 1,
            stdout: String::new(),
            stderr: "test failed".to_owned(),
        };
        let summary = super::build_sandbox_failure_summary(&output, &AstDiff::default());
        assert!(summary.contains("exit code 1"));
        assert!(summary.contains("test failed"));
    }
}
