use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::config::ProjectTokenUsage;
use crate::skills::{AstDiff, ExecutionDiff};

mod blueprinter;
mod executor;
mod surgeon;
mod verifier;

pub use blueprinter::BlueprinterAgent;
pub use executor::ExecutorAgent;
pub use surgeon::SurgeonAgent;
pub use verifier::VerifierAgent;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Ticket {
    pub id: String,
    pub description: String,
    pub context_files: Vec<String>,
    pub status: TicketStatus,
    pub legacy_code_snippet: String,
    pub target_framework: String,
    pub dependencies: Vec<String>,
    #[serde(default)]
    pub modern_file_paths: Vec<String>,
    #[serde(default)]
    pub test_file_paths: Vec<String>,
    #[serde(default)]
    pub retries: u8,
    #[serde(default)]
    pub token_usage: TicketTokenUsage,
    #[serde(default)]
    pub last_execution_diff: Option<ExecutionDiff>,
    #[serde(default)]
    pub last_ast_diff: Option<AstDiff>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq, Eq)]
pub struct TicketTokenUsage {
    pub llm_calls: u32,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum TicketStatus {
    Todo,
    InProgress,
    Verified,
    Failed(String),
}

pub trait Agent: Send + Sync {
    fn name(&self) -> &str;
    fn model(&self) -> &str;
    async fn process_ticket(&self, ticket: &Ticket) -> Result<Ticket>;
}

impl Ticket {
    pub fn record_llm_usage(&mut self, usage: &ProjectTokenUsage) {
        self.token_usage.llm_calls = self.token_usage.llm_calls.saturating_add(1);
        self.token_usage.prompt_tokens = self
            .token_usage
            .prompt_tokens
            .saturating_add(usage.prompt_tokens);
        self.token_usage.completion_tokens = self
            .token_usage
            .completion_tokens
            .saturating_add(usage.completion_tokens);
        self.token_usage.total_tokens = self
            .token_usage
            .total_tokens
            .saturating_add(usage.total_tokens);
    }
}
