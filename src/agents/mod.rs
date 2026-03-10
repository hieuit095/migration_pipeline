use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Ticket {
    pub id: String,
    pub description: String,
    pub context_files: Vec<String>,
    pub status: TicketStatus,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum TicketStatus {
    Todo,
    InProgress,
    Verified,
    Failed(String),
}

#[async_trait]
pub trait Agent: Send + Sync {
    fn name(&self) -> &str;
    fn model(&self) -> &str;
    async fn process_ticket(&self, ticket: &Ticket) -> Result<Ticket>;
}

pub struct BlueprinterAgent {
    pub model: String,
}

pub struct ExecutorAgent {
    pub model: String,
}

pub struct VerifierAgent {
    pub model: String,
}

pub struct SurgeonAgent {
    pub model: String,
}
