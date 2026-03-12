use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::PathBuf;
use tempfile::Builder;
use tokio::{
    fs as tokio_fs,
    io::{AsyncReadExt, BufReader},
    task,
};

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

#[derive(Debug)]
pub(crate) enum PromptContext {
    Inline(String),
    TempFile(TemporaryPromptFile),
}

impl PromptContext {
    pub(crate) fn inline(value: impl Into<String>) -> Self {
        Self::Inline(value.into())
    }

    pub(crate) fn from_temp_path(path: impl Into<PathBuf>) -> Self {
        Self::TempFile(TemporaryPromptFile::from_existing(path.into()))
    }

    pub(crate) async fn spill_to_tempfile(prefix: &str, contents: String) -> Result<Self> {
        Ok(Self::TempFile(
            TemporaryPromptFile::from_contents(prefix, contents).await?,
        ))
    }

    pub(crate) async fn byte_len_hint(&self) -> Result<usize> {
        match self {
            Self::Inline(contents) => Ok(contents.len()),
            Self::TempFile(file) => file.byte_len_hint().await,
        }
    }

    pub(crate) async fn append_to(&self, output: &mut String) -> Result<()> {
        match self {
            Self::Inline(contents) => {
                output.push_str(contents);
                Ok(())
            }
            Self::TempFile(file) => file.append_to(output).await,
        }
    }
}

#[derive(Debug)]
pub(crate) struct TemporaryPromptFile {
    path: PathBuf,
}

impl TemporaryPromptFile {
    const STREAM_BUFFER_SIZE: usize = 8 * 1024;

    fn from_existing(path: PathBuf) -> Self {
        Self { path }
    }

    async fn from_contents(prefix: &str, contents: String) -> Result<Self> {
        let prefix = prefix.to_owned();
        task::spawn_blocking(move || {
            let mut temp_file = Builder::new()
                .prefix(&prefix)
                .suffix(".txt")
                .tempfile()
                .context("failed to create temporary prompt file")?;
            temp_file
                .write_all(contents.as_bytes())
                .context("failed to write temporary prompt file")?;
            temp_file
                .flush()
                .context("failed to flush temporary prompt file")?;
            let (_file, path) = temp_file
                .keep()
                .map_err(|error| anyhow!(error.error))
                .context("failed to persist temporary prompt file")?;
            Ok(Self { path })
        })
        .await
        .map_err(|error| anyhow!("blocking task `prompt_tempfile` failed to join: {error}"))?
    }

    async fn byte_len_hint(&self) -> Result<usize> {
        let metadata = tokio_fs::metadata(&self.path).await.with_context(|| {
            format!(
                "failed to read metadata for temporary prompt file {}",
                self.path.display()
            )
        })?;
        Ok(metadata.len().min(usize::MAX as u64) as usize)
    }

    async fn append_to(&self, output: &mut String) -> Result<()> {
        let file = tokio_fs::File::open(&self.path).await.with_context(|| {
            format!(
                "failed to open temporary prompt file {}",
                self.path.display()
            )
        })?;
        let mut reader = BufReader::new(file);
        let mut buffer = [0_u8; Self::STREAM_BUFFER_SIZE];

        loop {
            let bytes_read = reader.read(&mut buffer).await.with_context(|| {
                format!(
                    "failed to read temporary prompt file {}",
                    self.path.display()
                )
            })?;
            if bytes_read == 0 {
                break;
            }

            let chunk = std::str::from_utf8(&buffer[..bytes_read]).with_context(|| {
                format!(
                    "temporary prompt file contained invalid UTF-8: {}",
                    self.path.display()
                )
            })?;
            output.push_str(chunk);
        }
        Ok(())
    }
}

impl Drop for TemporaryPromptFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
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

#[cfg(test)]
mod tests {
    use super::PromptContext;

    #[tokio::test]
    async fn prompt_context_temp_file_is_removed_on_drop() {
        let path = {
            let prompt_context = PromptContext::spill_to_tempfile(
                "migration_pipeline_prompt_context_",
                "example".to_owned(),
            )
            .await
            .expect("prompt temp file should be created");
            let path = match &prompt_context {
                PromptContext::TempFile(file) => file.path.clone(),
                PromptContext::Inline(_) => panic!("expected temp file prompt context"),
            };
            assert!(path.exists());
            path
        };

        assert!(!path.exists());
    }
}
