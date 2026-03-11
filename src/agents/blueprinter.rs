use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use tracing::info;
use zeroclaw::tools::ToolSpec;

use crate::config::{TaskKind, ZeroClawClient};
use crate::skills::Skill;

use super::{Agent, Ticket, TicketStatus, TicketTokenUsage};

const BLUEPRINTER_NAME: &str = "blueprinter";
const DEFAULT_BLUEPRINTER_MODEL: &str = "google/gemini-3-flash-preview";

#[derive(Debug, Deserialize)]
struct BlueprintPayload {
    tickets: Vec<BlueprintTicket>,
}

#[derive(Debug, Deserialize)]
struct BlueprintTicket {
    id: String,
    description: String,
    context_files: Vec<String>,
    legacy_code_snippet: String,
    target_framework: String,
    dependencies: Vec<String>,
}

impl From<BlueprintTicket> for Ticket {
    fn from(value: BlueprintTicket) -> Self {
        Self {
            id: value.id,
            description: value.description,
            context_files: value.context_files,
            status: TicketStatus::Todo,
            legacy_code_snippet: value.legacy_code_snippet,
            target_framework: value.target_framework,
            dependencies: value.dependencies,
            modern_file_paths: Vec::new(),
            test_file_paths: Vec::new(),
            retries: 0,
            token_usage: TicketTokenUsage::default(),
            last_execution_diff: None,
            last_ast_diff: None,
        }
    }
}

pub struct BlueprinterAgent {
    ast_parsing_skill: Arc<dyn Skill>,
    llm_client: Arc<ZeroClawClient>,
}

impl BlueprinterAgent {
    pub fn new(ast_parsing_skill: Arc<dyn Skill>, llm_client: Arc<ZeroClawClient>) -> Self {
        Self {
            ast_parsing_skill,
            llm_client,
        }
    }

    pub fn default_model() -> &'static str {
        DEFAULT_BLUEPRINTER_MODEL
    }

    pub fn provider(&self) -> &'static str {
        self.llm_client.provider_name()
    }

    pub async fn generate_blueprint(&self, legacy_dir_path: &str) -> Result<Vec<Ticket>> {
        let dependency_graph = self
            .ast_parsing_skill
            .execute(vec![legacy_dir_path.to_owned()])
            .await
            .with_context(|| {
                format!(
                    "failed to build AST dependency graph from legacy directory {legacy_dir_path}"
                )
            })?;

        info!(
            legacy_dir_path,
            bytes = dependency_graph.len(),
            "Collected legacy codebase dependency graph"
        );

        let structured_call = self
            .llm_client
            .chat_with_schema(
                TaskKind::Blueprinter,
                &self.system_prompt(),
                &self.user_prompt(legacy_dir_path, &dependency_graph),
                self.blueprint_tool(),
            )
            .await
            .context("blueprint generation failed")?;
        info!(
            model = self.model(),
            prompt_tokens = structured_call.usage.prompt_tokens,
            completion_tokens = structured_call.usage.completion_tokens,
            total_tokens = structured_call.usage.total_tokens,
            "Blueprinter token usage"
        );

        let payload: BlueprintPayload = structured_call.deserialize_arguments()?;
        Ok(payload.tickets.into_iter().map(Ticket::from).collect())
    }

    fn system_prompt(&self) -> String {
        concat!(
            "You are a Staff Software Engineer specializing in large-scale legacy migrations.\n",
            "Analyze the provided legacy codebase dependency graph and generate an industrial-grade migration blueprint.\n",
            "You must respond by calling the provided tool exactly once.\n",
            "Rules:\n",
            "1. Preserve semantic equivalence and operational behavior.\n",
            "2. Break work into independently executable tickets.\n",
            "3. Use only relative file paths that actually exist in the supplied dependency graph.\n",
            "4. `legacy_code_snippet` must be a concise structural summary inferred from the graph, not raw source text.\n",
            "5. `target_framework` should be the best-fit modern destination for that ticket.\n",
            "6. `dependencies` should list concrete libraries, frameworks, or runtime dependencies.\n",
            "7. Do not emit markdown fences, plain text, or explanations.\n"
        )
        .to_owned()
    }

    fn user_prompt(&self, legacy_dir_path: &str, dependency_graph: &str) -> String {
        format!(
            "Analyze the legacy codebase dependency graph below and create a migration blueprint.\n\
Legacy directory: {legacy_dir_path}\n\
Dependency graph JSON:\n\
{dependency_graph}\n"
        )
    }

    fn blueprint_tool(&self) -> ToolSpec {
        ToolSpec {
            name: "submit_blueprint".to_owned(),
            description: "Return the migration blueprint tickets".to_owned(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "tickets": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string" },
                                "description": { "type": "string" },
                                "context_files": {
                                    "type": "array",
                                    "items": { "type": "string" }
                                },
                                "legacy_code_snippet": { "type": "string" },
                                "target_framework": { "type": "string" },
                                "dependencies": {
                                    "type": "array",
                                    "items": { "type": "string" }
                                }
                            },
                            "required": [
                                "id",
                                "description",
                                "context_files",
                                "legacy_code_snippet",
                                "target_framework",
                                "dependencies"
                            ],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["tickets"],
                "additionalProperties": false
            }),
        }
    }
}

impl Agent for BlueprinterAgent {
    fn name(&self) -> &str {
        BLUEPRINTER_NAME
    }

    fn model(&self) -> &str {
        self.llm_client.model_for(TaskKind::Blueprinter)
    }

    async fn process_ticket(&self, _ticket: &Ticket) -> Result<Ticket> {
        Err(anyhow!(
            "BlueprinterAgent does not process individual tickets; call generate_blueprint instead"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{BlueprintPayload, BlueprintTicket};
    use crate::agents::{Ticket, TicketStatus};

    #[test]
    fn blueprint_ticket_conversion_defaults_runtime_fields() {
        let payload = BlueprintPayload {
            tickets: vec![BlueprintTicket {
                id: "BP-001".to_owned(),
                description: "Migrate HTTP endpoint".to_owned(),
                context_files: vec!["server.js".to_owned()],
                legacy_code_snippet: "http.createServer(...)".to_owned(),
                target_framework: "Express.js".to_owned(),
                dependencies: vec!["express".to_owned()],
            }],
        };

        let tickets = payload
            .tickets
            .into_iter()
            .map(Ticket::from)
            .collect::<Vec<_>>();

        assert_eq!(tickets.len(), 1);
        assert_eq!(tickets[0].status, TicketStatus::Todo);
        assert!(tickets[0].modern_file_paths.is_empty());
        assert_eq!(tickets[0].retries, 0);
    }
}
