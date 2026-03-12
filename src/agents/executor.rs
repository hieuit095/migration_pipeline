use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};

use anyhow::{Context, Result, anyhow, ensure};
use serde::Deserialize;
use tracing::{info, warn};
use zeroclaw::tools::ToolSpec;

use crate::config::{TaskKind, ZeroClawClient};
use crate::skills::Skill;
use crate::utils::path::normalize_relative_path as normalize_portable_relative_path;

use super::{Agent, PromptContext, Ticket, TicketStatus};

const EXECUTOR_NAME: &str = "executor";
const DEFAULT_EXECUTOR_MODEL: &str = "minimax/minimax-m2.5";

static FRAMEWORK_EXTENSIONS: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    HashMap::from([
        ("react", "tsx"),
        ("next", "tsx"),
        ("next.js", "tsx"),
        ("tsx", "tsx"),
        ("typescript", "ts"),
        ("node", "ts"),
        ("node.js", "ts"),
        ("nest", "ts"),
        ("angular", "ts"),
        ("go", "go"),
        ("gin", "go"),
        ("fiber", "go"),
        ("rust", "rs"),
        ("axum", "rs"),
        ("actix", "rs"),
        ("rocket", "rs"),
        ("python", "py"),
        ("django", "py"),
        ("fastapi", "py"),
        ("flask", "py"),
        ("c#", "cs"),
        (".net", "cs"),
        ("asp.net", "cs"),
        ("blazor", "cs"),
        ("java", "java"),
        ("spring", "java"),
        ("kotlin", "kt"),
        ("ktor", "kt"),
        ("php", "php"),
        ("laravel", "php"),
        ("ruby", "rb"),
        ("rails", "rb"),
    ])
});

#[derive(Debug, Deserialize)]
struct GeneratedArtifactsPayload {
    source_files: Vec<GeneratedFile>,
    test_files: Vec<GeneratedFile>,
}

#[derive(Debug, Deserialize)]
struct GeneratedFile {
    path: String,
    content: String,
}

#[derive(Debug)]
struct PersistedArtifacts {
    source_files: Vec<String>,
    test_files: Vec<String>,
}

pub struct ExecutorAgent {
    legacy_root: PathBuf,
    output_root: PathBuf,
    file_io_skill: Arc<dyn Skill>,
    file_write_skill: Arc<dyn Skill>,
    llm_client: Arc<ZeroClawClient>,
}

impl ExecutorAgent {
    pub fn new(
        legacy_root: impl Into<PathBuf>,
        output_root: impl Into<PathBuf>,
        file_io_skill: Arc<dyn Skill>,
        file_write_skill: Arc<dyn Skill>,
        llm_client: Arc<ZeroClawClient>,
    ) -> Self {
        Self {
            legacy_root: legacy_root.into(),
            output_root: output_root.into(),
            file_io_skill,
            file_write_skill,
            llm_client,
        }
    }

    pub fn default_model() -> &'static str {
        DEFAULT_EXECUTOR_MODEL
    }

    pub fn provider(&self) -> &str {
        self.llm_client.provider_for(TaskKind::Executor)
    }

    async fn execute_ticket(&self, ticket: &mut Ticket) -> Result<PersistedArtifacts> {
        let legacy_context = self
            .load_legacy_context(ticket)
            .await
            .with_context(|| format!("failed to load legacy context for ticket {}", ticket.id))?;
        let suggested_source_path = determine_output_relative_path(ticket)?;
        let suggested_test_path = determine_test_output_relative_path(&suggested_source_path)?;
        let system_prompt = self.system_prompt(ticket);
        let user_prompt = self
            .user_prompt(
                ticket,
                &legacy_context,
                &suggested_source_path,
                &suggested_test_path,
            )
            .await
            .with_context(|| format!("failed to build executor prompt for ticket {}", ticket.id))?;
        let structured_call = self
            .llm_client
            .chat_with_schema(
                TaskKind::Executor,
                &system_prompt,
                &user_prompt,
                self.generated_artifacts_tool(),
            )
            .await
            .with_context(|| format!("executor model failed for ticket {}", ticket.id))?;
        info!(
            ticket_id = ticket.id.as_str(),
            model = self.model(),
            prompt_tokens = structured_call.usage.prompt_tokens,
            completion_tokens = structured_call.usage.completion_tokens,
            total_tokens = structured_call.usage.total_tokens,
            "Executor token usage"
        );
        ticket.record_llm_usage(&structured_call.usage);
        let payload: GeneratedArtifactsPayload = structured_call.deserialize_arguments()?;

        self.persist_generated_artifacts(
            ticket,
            payload,
            &suggested_source_path,
            &suggested_test_path,
        )
        .await
    }

    async fn load_legacy_context(&self, ticket: &Ticket) -> Result<PromptContext> {
        if ticket.context_files.is_empty() {
            if ticket.legacy_code_snippet.trim().is_empty() {
                return Err(anyhow!(
                    "ticket {} does not contain context_files or a legacy_code_snippet",
                    ticket.id
                ));
            }

            return Ok(PromptContext::inline(format!(
                "// File: legacy_snippet.txt\n{}",
                ticket.legacy_code_snippet
            )));
        }

        let mut args = vec![self.legacy_root.to_string_lossy().into_owned()];
        args.extend(ticket.context_files.iter().cloned());
        let context_path = self.file_io_skill.execute(args).await?;
        Ok(PromptContext::from_temp_path(context_path))
    }

    async fn persist_generated_artifacts(
        &self,
        ticket: &Ticket,
        payload: GeneratedArtifactsPayload,
        suggested_source_path: &str,
        suggested_test_path: &str,
    ) -> Result<PersistedArtifacts> {
        ensure!(
            !payload.source_files.is_empty(),
            "executor returned no source files for ticket {}",
            ticket.id
        );
        ensure!(
            !payload.test_files.is_empty(),
            "executor returned no test files for ticket {}",
            ticket.id
        );

        let source_files = self
            .persist_files(
                "source",
                ticket,
                payload.source_files,
                suggested_source_path,
            )
            .await?;
        let test_files = self
            .persist_files("test", ticket, payload.test_files, suggested_test_path)
            .await?;

        Ok(PersistedArtifacts {
            source_files,
            test_files,
        })
    }

    async fn persist_files(
        &self,
        file_role: &str,
        ticket: &Ticket,
        files: Vec<GeneratedFile>,
        suggested_primary_path: &str,
    ) -> Result<Vec<String>> {
        let mut persisted_paths = Vec::with_capacity(files.len());
        let mut includes_suggested_path = false;

        for file in files {
            ensure!(
                !file.content.trim().is_empty(),
                "executor returned empty {file_role} content for ticket {} at path `{}`",
                ticket.id,
                file.path
            );

            let relative_path = normalize_relative_path(&file.path).with_context(|| {
                format!(
                    "executor returned an invalid {file_role} file path for ticket {}",
                    ticket.id
                )
            })?;
            if relative_path == suggested_primary_path {
                includes_suggested_path = true;
            }

            let absolute_path = self.output_root.join(&relative_path);
            self.file_write_skill
                .execute(vec![absolute_path.to_string_lossy().into_owned(), file.content])
                .await
                .with_context(|| {
                    format!(
                        "failed to persist generated {file_role} output `{relative_path}` for ticket {}",
                        ticket.id
                    )
                })?;

            if !persisted_paths.contains(&relative_path) {
                persisted_paths.push(relative_path);
            }
        }

        if !includes_suggested_path {
            warn!(
                ticket_id = ticket.id.as_str(),
                file_role,
                suggested_primary_path,
                "Executor generated files did not include the suggested primary path"
            );
        }

        Ok(persisted_paths)
    }

    fn system_prompt(&self, ticket: &Ticket) -> String {
        format!(
            concat!(
                "You are a Senior Developer specializing in {framework} migrations.\n",
                "Translate the provided legacy code into compile-ready source files and matching runnable tests.\n",
                "You must respond by calling the provided tool exactly once.\n",
                "Rules:\n",
                "1. Return both `source_files` and `test_files`.\n",
                "2. Each file `content` must be raw compile-ready source code.\n",
                "3. Tests run in a two-phase Docker sandbox: dependencies may be preinstalled during a warm-up step with temporary network access, but the actual execution phase has no network access.\n",
                "4. Include any required dependency manifests or lockfiles (for example `package.json`, `requirements.txt`, or `go.mod`) when the generated code relies on third-party packages.\n",
                "5. Use relative paths only.\n",
                "6. Preserve semantic behavior, data contracts, and side effects from the legacy code.\n",
                "7. Make the tests deterministic and self-contained.\n"
            ),
            framework = ticket.target_framework
        )
    }

    async fn user_prompt(
        &self,
        ticket: &Ticket,
        legacy_context: &PromptContext,
        suggested_source_path: &str,
        suggested_test_path: &str,
    ) -> Result<String> {
        let dependencies = if ticket.dependencies.is_empty() {
            "None".to_owned()
        } else {
            ticket.dependencies.join(", ")
        };
        let context_files = if ticket.context_files.is_empty() {
            "legacy_snippet.txt".to_owned()
        } else {
            ticket.context_files.join(", ")
        };

        let prefix = format!(
            concat!(
                "Ticket ID: {ticket_id}\n",
                "Task: {description}\n",
                "Target framework: {target_framework}\n",
                "Suggested primary source path: {suggested_source_path}\n",
                "Suggested primary test path: {suggested_test_path}\n",
                "Dependencies: {dependencies}\n",
                "Relevant legacy files: {context_files}\n",
                "Legacy snippet anchor:\n",
                "{legacy_snippet}\n\n",
                "Legacy context:\n"
            ),
            ticket_id = ticket.id,
            description = ticket.description,
            target_framework = ticket.target_framework,
            suggested_source_path = suggested_source_path,
            suggested_test_path = suggested_test_path,
            dependencies = dependencies,
            context_files = context_files,
            legacy_snippet = ticket.legacy_code_snippet
        );
        let mut prompt =
            String::with_capacity(prefix.len() + legacy_context.byte_len_hint().await? + 1);
        prompt.push_str(&prefix);
        legacy_context.append_to(&mut prompt).await?;
        prompt.push('\n');
        Ok(prompt)
    }

    fn generated_artifacts_tool(&self) -> ToolSpec {
        ToolSpec {
            name: "submit_generated_artifacts".to_owned(),
            description: "Return the generated modern source files and runnable tests".to_owned(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "source_files": file_array_schema(),
                    "test_files": file_array_schema()
                },
                "required": ["source_files", "test_files"],
                "additionalProperties": false
            }),
        }
    }
}

impl Agent for ExecutorAgent {
    fn name(&self) -> &str {
        EXECUTOR_NAME
    }

    fn model(&self) -> &str {
        self.llm_client.model_for(TaskKind::Executor)
    }

    async fn process_ticket(&self, ticket: &Ticket) -> Result<Ticket> {
        let mut updated_ticket = ticket.clone();

        match self.execute_ticket(&mut updated_ticket).await {
            Ok(artifacts) => {
                info!(
                    ticket_id = ticket.id.as_str(),
                    source_file_count = artifacts.source_files.len(),
                    test_file_count = artifacts.test_files.len(),
                    "Executor generated source and test artifacts"
                );

                updated_ticket.status = TicketStatus::InProgress;
                updated_ticket.modern_file_paths = artifacts.source_files;
                updated_ticket.test_file_paths = artifacts.test_files;
                Ok(updated_ticket)
            }
            Err(error) => {
                let error_message = error.to_string();
                warn!(
                    ticket_id = ticket.id.as_str(),
                    error = error_message.as_str(),
                    "Executor failed to process ticket"
                );

                updated_ticket.status = TicketStatus::Failed(error_message);
                Ok(updated_ticket)
            }
        }
    }
}

fn normalize_relative_path(path: &str) -> Result<String> {
    normalize_portable_relative_path(path, "generated file path").map(|path| path.into_string())
}

fn determine_output_relative_path(ticket: &Ticket) -> Result<String> {
    let default_filename = sanitize_for_filename(&ticket.id);
    let source_relative_path = ticket
        .context_files
        .first()
        .cloned()
        .unwrap_or_else(|| format!("generated/{default_filename}.txt"));
    let source_path = Path::new(&source_relative_path);
    let extension = target_extension(
        &ticket.target_framework,
        source_path.extension().and_then(|value| value.to_str()),
    );
    let file_stem = source_path
        .file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or(default_filename);

    let parent = source_path.parent().unwrap_or_else(|| Path::new(""));
    let output_file_name = format!("{file_stem}.{extension}");

    let output_path = if parent.as_os_str().is_empty() {
        PathBuf::from(output_file_name)
    } else {
        parent.join(output_file_name)
    };

    normalize_relative_path(output_path.to_string_lossy().as_ref())
}

fn determine_test_output_relative_path(source_relative_path: &str) -> Result<String> {
    let source_path = Path::new(source_relative_path);
    let extension = source_path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("txt");
    let stem = source_path
        .file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("generated");
    let parent = source_path.parent().unwrap_or_else(|| Path::new(""));
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

    normalize_relative_path(relative_dir.join(file_name).to_string_lossy().as_ref())
}

fn target_extension(target_framework: &str, source_extension: Option<&str>) -> String {
    framework_lookup_keys(target_framework)
        .into_iter()
        .find_map(|key| FRAMEWORK_EXTENSIONS.get(key.as_str()).copied())
        .unwrap_or(source_extension.unwrap_or("txt"))
        .to_owned()
}

fn framework_lookup_keys(target_framework: &str) -> Vec<String> {
    let normalized = target_framework.trim().to_ascii_lowercase();
    let mut keys = Vec::new();

    push_framework_key(&mut keys, normalized.clone());
    push_framework_key(&mut keys, normalized.replace(' ', ""));

    for token in normalized
        .split(|character: char| {
            !(character.is_ascii_alphanumeric() || matches!(character, '#' | '+' | '.'))
        })
        .filter(|token| !token.is_empty())
    {
        push_framework_key(&mut keys, token.to_owned());
        if let Some(stripped_js) = token.strip_suffix(".js") {
            push_framework_key(&mut keys, stripped_js.to_owned());
        }
    }

    keys
}

fn push_framework_key(keys: &mut Vec<String>, key: String) {
    if !key.is_empty() && !keys.contains(&key) {
        keys.push(key);
    }
}

fn sanitize_for_filename(value: &str) -> String {
    let sanitized: String = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();

    sanitized.trim_matches('-').to_owned()
}

fn file_array_schema() -> serde_json::Value {
    serde_json::json!({
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
    })
}

#[cfg(test)]
mod tests {
    use super::{
        determine_output_relative_path, determine_test_output_relative_path,
        normalize_relative_path, target_extension,
    };

    #[test]
    fn executor_maps_node_ticket_to_typescript_path() {
        let ticket = crate::agents::Ticket {
            id: "EXEC-1".to_owned(),
            description: "Migrate HTTP service".to_owned(),
            context_files: vec!["src/server.js".to_owned()],
            status: crate::agents::TicketStatus::Todo,
            legacy_code_snippet: "http.createServer(...)".to_owned(),
            target_framework: "TypeScript".to_owned(),
            dependencies: vec!["express".to_owned()],
            modern_file_paths: Vec::new(),
            test_file_paths: Vec::new(),
            retries: 0,
            token_usage: crate::agents::TicketTokenUsage::default(),
            last_execution_diff: None,
            last_ast_diff: None,
        };

        let path = determine_output_relative_path(&ticket).expect("executor path should normalize");
        assert_eq!(path, "src/server.ts");
    }

    #[test]
    fn executor_derives_test_path_from_source_path() {
        let path = determine_test_output_relative_path("src/server.ts")
            .expect("executor test path should normalize");
        assert_eq!(path, "tests/src/server.test.ts");
    }

    #[test]
    fn normalize_relative_path_rejects_parent_traversal() {
        let error = normalize_relative_path("../secret.txt").expect_err("path should fail");
        assert!(error.to_string().contains("traverse"));
    }

    #[test]
    fn target_extension_prefers_framework_alias_lookup() {
        assert_eq!(target_extension("Next.js TypeScript", Some("js")), "tsx");
        assert_eq!(target_extension("ASP.NET Core", Some("txt")), "cs");
    }
}
