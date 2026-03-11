use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use serde::Deserialize;
use tracing::info;

use crate::config::{LlmGateway, LlmProvider, TaskModelConfig};
use crate::skills::Skill;
use crate::utils::clean_json_response;

use super::{Agent, Ticket};

const BLUEPRINTER_NAME: &str = "blueprinter";
const DEFAULT_BLUEPRINTER_MODEL: &str = "google/gemini-3-flash-preview";

#[derive(Debug, Deserialize)]
struct BlueprintEnvelope {
    tickets: Vec<Ticket>,
}

pub struct BlueprinterAgent {
    file_io_skill: Arc<dyn Skill>,
    route: TaskModelConfig,
    llm_gateway: LlmGateway,
}

impl BlueprinterAgent {
    pub fn new(
        file_io_skill: Arc<dyn Skill>,
        llm_gateway: LlmGateway,
        route: TaskModelConfig,
    ) -> Self {
        Self {
            file_io_skill,
            route,
            llm_gateway,
        }
    }

    pub fn default_model() -> &'static str {
        DEFAULT_BLUEPRINTER_MODEL
    }

    pub fn provider(&self) -> LlmProvider {
        self.route.provider
    }

    pub async fn generate_blueprint(&self, legacy_dir_path: &str) -> Result<Vec<Ticket>> {
        let codebase_snapshot = self
            .file_io_skill
            .execute(vec![legacy_dir_path.to_owned()])
            .with_context(|| {
                format!("failed to build codebase snapshot from legacy directory {legacy_dir_path}")
            })?;

        info!(
            legacy_dir_path,
            bytes = codebase_snapshot.len(),
            "Collected legacy codebase snapshot"
        );

        let response = self
            .llm_gateway
            .chat_completion(
                &self.route,
                &self.system_prompt(),
                &self.user_prompt(legacy_dir_path, &codebase_snapshot),
            )
            .await
            .context("blueprint generation failed")?;
        let (response, usage) = response;
        info!(
            model = self.route.model.as_str(),
            prompt_tokens = usage.prompt_tokens,
            completion_tokens = usage.completion_tokens,
            total_tokens = usage.total_tokens,
            "Blueprinter token usage"
        );

        self.parse_blueprint_response(&response)
    }

    fn system_prompt(&self) -> String {
        format!(concat!(
            "You are a Staff Software Engineer specializing in large-scale legacy migrations.\n",
            "Your task is to analyze a legacy codebase snapshot and produce an industrial-grade migration blueprint.\n",
            "Return only strict JSON. Do not include markdown fences, comments, or explanatory prose.\n",
            "The JSON must either be an object with a top-level `tickets` array or a raw array of ticket objects.\n",
            "Each ticket object must match this schema exactly:\n",
            "{{",
            "\"id\":\"BP-001\",",
            "\"description\":\"Short actionable migration task.\",",
            "\"context_files\":[\"relative/path.ext\"],",
            "\"status\":\"Todo\",",
            "\"legacy_code_snippet\":\"short verbatim snippet from the legacy codebase\",",
            "\"target_framework\":\"recommended modern framework or runtime\",",
            "\"dependencies\":[\"package-or-library\"]",
            "}}\n",
            "Rules:\n",
            "1. Preserve semantic equivalence and operational behavior.\n",
            "2. Break work into independently executable tickets.\n",
            "3. Use only relative file paths that actually exist in the supplied snapshot.\n",
            "4. `legacy_code_snippet` must be copied from the snapshot, shortened if needed.\n",
            "5. `status` must always be `Todo`.\n",
            "6. `target_framework` should be the best-fit modern destination for that ticket.\n",
            "7. `dependencies` should list concrete libraries, frameworks, or runtime dependencies.\n",
            "8. If the snapshot is too small for certainty, make the most conservative inference possible.\n"
        ))
    }

    fn user_prompt(&self, legacy_dir_path: &str, codebase_snapshot: &str) -> String {
        format!(
            "Analyze the legacy codebase below and create a migration blueprint.\n\
Legacy directory: {legacy_dir_path}\n\
Codebase snapshot:\n\
{codebase_snapshot}\n"
        )
    }

    fn parse_blueprint_response(&self, raw_response: &str) -> Result<Vec<Ticket>> {
        let cleaned = clean_json_response(raw_response);

        if let Ok(envelope) = serde_json::from_str::<BlueprintEnvelope>(&cleaned) {
            return Ok(envelope.tickets);
        }

        if let Ok(tickets) = serde_json::from_str::<Vec<Ticket>>(&cleaned) {
            return Ok(tickets);
        }

        Err(anyhow!(
            "failed to parse blueprinter response as JSON tickets: {}",
            truncate_for_error(&cleaned)
        ))
    }
}

#[async_trait]
impl Agent for BlueprinterAgent {
    fn name(&self) -> &str {
        BLUEPRINTER_NAME
    }

    fn model(&self) -> &str {
        &self.route.model
    }

    async fn process_ticket(&self, _ticket: &Ticket) -> Result<Ticket> {
        Err(anyhow!(
            "BlueprinterAgent does not process individual tickets; call generate_blueprint instead"
        ))
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
    use super::BlueprinterAgent;
    use crate::agents::{Ticket, TicketStatus, TicketTokenUsage};
    use crate::config::{LlmGateway, LlmProvider, TaskKind, TaskModelConfig};
    use crate::skills::FileIOSkill;
    use std::sync::Arc;

    fn build_agent() -> BlueprinterAgent {
        BlueprinterAgent::new(
            Arc::new(FileIOSkill),
            LlmGateway::new().expect("gateway should build"),
            TaskModelConfig::new(
                TaskKind::Blueprinter,
                LlmProvider::OpenRouter,
                BlueprinterAgent::default_model(),
            ),
        )
    }

    #[test]
    fn blueprinter_parses_ticket_envelope() {
        let agent = build_agent();
        let raw = r#"```json
        {
          "tickets": [
            {
              "id": "BP-001",
              "description": "Migrate HTTP endpoint",
              "context_files": ["server.js"],
              "status": "Todo",
              "legacy_code_snippet": "http.createServer(...)",
              "target_framework": "Express.js",
              "dependencies": ["express"]
            }
          ]
        }
        ```"#;

        let tickets = agent
            .parse_blueprint_response(raw)
            .expect("response should parse");

        assert_eq!(tickets.len(), 1);
        assert_eq!(tickets[0].status, TicketStatus::Todo);
        assert_eq!(tickets[0].id, "BP-001");
    }

    #[test]
    fn blueprinter_parses_raw_ticket_array() {
        let agent = build_agent();
        let raw = serde_json::to_string(&vec![Ticket {
            id: "BP-002".to_owned(),
            description: "Migrate persistence".to_owned(),
            context_files: vec!["db.js".to_owned()],
            status: TicketStatus::Todo,
            legacy_code_snippet: "function loadCustomer(id) {}".to_owned(),
            target_framework: "Prisma".to_owned(),
            dependencies: vec!["prisma".to_owned()],
            modern_file_paths: Vec::new(),
            retries: 0,
            token_usage: TicketTokenUsage::default(),
        }])
        .expect("tickets should serialize");

        let tickets = agent
            .parse_blueprint_response(&raw)
            .expect("response should parse");

        assert_eq!(tickets.len(), 1);
        assert_eq!(tickets[0].target_framework, "Prisma");
    }
}
