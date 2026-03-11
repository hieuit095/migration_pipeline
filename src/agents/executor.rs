use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use tracing::{info, warn};

use crate::config::{LlmGateway, LlmProvider, TaskModelConfig};
use crate::skills::Skill;
use crate::utils::clean_code_response;

use super::{Agent, Ticket, TicketStatus};

const EXECUTOR_NAME: &str = "executor";
const DEFAULT_EXECUTOR_MODEL: &str = "minimax/minimax-m2.5";

pub struct ExecutorAgent {
    legacy_root: PathBuf,
    output_root: PathBuf,
    file_io_skill: Arc<dyn Skill>,
    file_write_skill: Arc<dyn Skill>,
    route: TaskModelConfig,
    llm_gateway: LlmGateway,
}

impl ExecutorAgent {
    pub fn new(
        legacy_root: impl Into<PathBuf>,
        output_root: impl Into<PathBuf>,
        file_io_skill: Arc<dyn Skill>,
        file_write_skill: Arc<dyn Skill>,
        llm_gateway: LlmGateway,
        route: TaskModelConfig,
    ) -> Self {
        Self {
            legacy_root: legacy_root.into(),
            output_root: output_root.into(),
            file_io_skill,
            file_write_skill,
            route,
            llm_gateway,
        }
    }

    pub fn default_model() -> &'static str {
        DEFAULT_EXECUTOR_MODEL
    }

    pub fn provider(&self) -> LlmProvider {
        self.route.provider
    }

    async fn execute_ticket(&self, ticket: &mut Ticket) -> Result<PathBuf> {
        let legacy_context = self
            .load_legacy_context(ticket)
            .with_context(|| format!("failed to load legacy context for ticket {}", ticket.id))?;
        let output_relative_path = self.determine_output_relative_path(ticket);
        let response = self
            .llm_gateway
            .chat_completion(
                &self.route,
                &self.system_prompt(ticket),
                &self.user_prompt(ticket, &legacy_context, &output_relative_path),
            )
            .await
            .with_context(|| format!("executor model failed for ticket {}", ticket.id))?;
        let (response, usage) = response;
        info!(
            ticket_id = ticket.id.as_str(),
            model = self.route.model.as_str(),
            prompt_tokens = usage.prompt_tokens,
            completion_tokens = usage.completion_tokens,
            total_tokens = usage.total_tokens,
            "Executor token usage"
        );
        ticket.record_llm_usage(&usage);
        let generated_code = clean_code_response(&response);

        ensure!(
            !generated_code.trim().is_empty(),
            "executor returned empty code for ticket {}",
            ticket.id
        );

        let output_path = self.output_root.join(&output_relative_path);
        self.file_write_skill
            .execute(vec![
                output_path.to_string_lossy().into_owned(),
                generated_code,
            ])
            .with_context(|| {
                format!(
                    "failed to persist generated output for ticket {}",
                    ticket.id
                )
            })?;

        Ok(output_path)
    }

    fn load_legacy_context(&self, ticket: &Ticket) -> Result<String> {
        if ticket.context_files.is_empty() {
            if ticket.legacy_code_snippet.trim().is_empty() {
                return Err(anyhow!(
                    "ticket {} does not contain context_files or a legacy_code_snippet",
                    ticket.id
                ));
            }

            return Ok(format!(
                "// File: legacy_snippet.txt\n{}",
                ticket.legacy_code_snippet
            ));
        }

        let mut args = vec![self.legacy_root.to_string_lossy().into_owned()];
        args.extend(ticket.context_files.iter().cloned());
        self.file_io_skill.execute(args)
    }

    fn system_prompt(&self, ticket: &Ticket) -> String {
        format!(
            concat!(
                "You are a Senior Developer specializing in {framework} migrations.\n",
                "Translate the provided legacy code into compile-ready {framework} source code.\n",
                "Return raw source code only.\n",
                "Rules:\n",
                "1. No markdown fences.\n",
                "2. No explanatory text.\n",
                "3. No \"Here is the code\" preambles.\n",
                "4. Do not include path headers or file markers in the output.\n",
                "5. Preserve semantic behavior, data contracts, and side effects from the legacy code.\n",
                "6. Generate exactly one source file body suitable for the requested output path.\n",
                "7. Use any listed dependencies only when they are necessary.\n"
            ),
            framework = ticket.target_framework
        )
    }

    fn user_prompt(
        &self,
        ticket: &Ticket,
        legacy_context: &str,
        output_relative_path: &str,
    ) -> String {
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

        format!(
            concat!(
                "Ticket ID: {ticket_id}\n",
                "Task: {description}\n",
                "Target framework: {target_framework}\n",
                "Output path: {output_relative_path}\n",
                "Dependencies: {dependencies}\n",
                "Relevant legacy files: {context_files}\n",
                "Legacy snippet anchor:\n",
                "{legacy_snippet}\n\n",
                "Legacy context:\n",
                "{legacy_context}\n"
            ),
            ticket_id = ticket.id,
            description = ticket.description,
            target_framework = ticket.target_framework,
            output_relative_path = output_relative_path,
            dependencies = dependencies,
            context_files = context_files,
            legacy_snippet = ticket.legacy_code_snippet,
            legacy_context = legacy_context
        )
    }

    fn determine_output_relative_path(&self, ticket: &Ticket) -> String {
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

        if parent.as_os_str().is_empty() {
            output_file_name
        } else {
            parent
                .join(output_file_name)
                .to_string_lossy()
                .replace('\\', "/")
        }
    }
}

#[async_trait]
impl Agent for ExecutorAgent {
    fn name(&self) -> &str {
        EXECUTOR_NAME
    }

    fn model(&self) -> &str {
        &self.route.model
    }

    async fn process_ticket(&self, ticket: &Ticket) -> Result<Ticket> {
        let mut updated_ticket = ticket.clone();

        match self.execute_ticket(&mut updated_ticket).await {
            Ok(output_path) => {
                let output_path = output_path
                    .strip_prefix(&self.output_root)
                    .unwrap_or(&output_path)
                    .to_string_lossy()
                    .replace('\\', "/");
                info!(
                    ticket_id = ticket.id.as_str(),
                    output_path = output_path.as_str(),
                    "Executor generated modern source file"
                );

                updated_ticket.status = TicketStatus::InProgress;
                updated_ticket.modern_file_paths = vec![output_path];
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

fn target_extension(target_framework: &str, source_extension: Option<&str>) -> String {
    let normalized = target_framework.to_ascii_lowercase();

    if normalized.contains("react") || normalized.contains("next") || normalized.contains("tsx") {
        "tsx".to_owned()
    } else if normalized.contains("typescript")
        || normalized.contains("node")
        || normalized.contains("nest")
        || normalized.contains("angular")
    {
        "ts".to_owned()
    } else if normalized.contains("go")
        || normalized.contains("gin")
        || normalized.contains("fiber")
    {
        "go".to_owned()
    } else if normalized.contains("rust")
        || normalized.contains("axum")
        || normalized.contains("actix")
        || normalized.contains("rocket")
    {
        "rs".to_owned()
    } else if normalized.contains("python")
        || normalized.contains("django")
        || normalized.contains("fastapi")
        || normalized.contains("flask")
    {
        "py".to_owned()
    } else if normalized.contains("c#")
        || normalized.contains(".net")
        || normalized.contains("asp.net")
        || normalized.contains("blazor")
    {
        "cs".to_owned()
    } else if normalized.contains("java") || normalized.contains("spring") {
        "java".to_owned()
    } else if normalized.contains("kotlin") || normalized.contains("ktor") {
        "kt".to_owned()
    } else if normalized.contains("php") || normalized.contains("laravel") {
        "php".to_owned()
    } else if normalized.contains("ruby") || normalized.contains("rails") {
        "rb".to_owned()
    } else {
        source_extension.unwrap_or("txt").to_owned()
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

#[cfg(test)]
mod tests {
    use super::ExecutorAgent;
    use crate::config::{LlmGateway, LlmProvider, TaskKind, TaskModelConfig};
    use crate::skills::{FileIOSkill, FileWriteSkill};
    use std::sync::Arc;

    fn build_agent() -> ExecutorAgent {
        ExecutorAgent::new(
            "./legacy_app",
            "./modern_app",
            Arc::new(FileIOSkill),
            Arc::new(FileWriteSkill),
            LlmGateway::new().expect("gateway should build"),
            TaskModelConfig::new(
                TaskKind::Executor,
                LlmProvider::OpenRouter,
                ExecutorAgent::default_model(),
            ),
        )
    }

    #[test]
    fn executor_maps_node_ticket_to_typescript_path() {
        let agent = build_agent();
        let ticket = crate::agents::Ticket {
            id: "EXEC-1".to_owned(),
            description: "Migrate HTTP service".to_owned(),
            context_files: vec!["src/server.js".to_owned()],
            status: crate::agents::TicketStatus::Todo,
            legacy_code_snippet: "http.createServer(...)".to_owned(),
            target_framework: "TypeScript".to_owned(),
            dependencies: vec!["express".to_owned()],
            modern_file_paths: Vec::new(),
            retries: 0,
            token_usage: crate::agents::TicketTokenUsage::default(),
        };

        let path = agent.determine_output_relative_path(&ticket);
        assert_eq!(path, "src/server.ts");
    }

    #[test]
    fn executor_maps_go_ticket_to_go_path() {
        let agent = build_agent();
        let ticket = crate::agents::Ticket {
            id: "EXEC-2".to_owned(),
            description: "Migrate API".to_owned(),
            context_files: vec!["services/api.js".to_owned()],
            status: crate::agents::TicketStatus::Todo,
            legacy_code_snippet: "app.get('/health')".to_owned(),
            target_framework: "Go Fiber".to_owned(),
            dependencies: vec!["fiber".to_owned()],
            modern_file_paths: Vec::new(),
            retries: 0,
            token_usage: crate::agents::TicketTokenUsage::default(),
        };

        let path = agent.determine_output_relative_path(&ticket);
        assert_eq!(path, "services/api.go");
    }
}
