use anyhow::{Context, Result, anyhow};
use reqwest::{Client, RequestBuilder, StatusCode};
use serde::{Deserialize, Serialize};
use std::env;
use std::fmt::{self, Display};
use std::str::FromStr;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{debug, info, warn};

const OPENROUTER_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
const TOGETHER_URL: &str = "https://api.together.xyz/v1/chat/completions";
const MAX_RETRIES: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmProvider {
    OpenRouter,
    Together,
}

impl LlmProvider {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenRouter => "openrouter",
            Self::Together => "together",
        }
    }

    fn endpoint(self) -> &'static str {
        match self {
            Self::OpenRouter => OPENROUTER_URL,
            Self::Together => TOGETHER_URL,
        }
    }

    fn api_key_env(self) -> &'static str {
        match self {
            Self::OpenRouter => "OPENROUTER_API_KEY",
            Self::Together => "TOGETHER_API_KEY",
        }
    }

    fn decorate_request(self, request: RequestBuilder) -> RequestBuilder {
        match self {
            Self::OpenRouter => request
                .header("HTTP-Referer", "https://github.com/openai/codex")
                .header("X-Title", "migration_pipeline"),
            Self::Together => request,
        }
    }
}

impl Display for LlmProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for LlmProvider {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "openrouter" => Ok(Self::OpenRouter),
            "together" | "together.ai" | "togetherai" => Ok(Self::Together),
            other => Err(anyhow!(
                "unsupported provider `{other}`; expected `openrouter` or `together`"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskKind {
    Blueprinter,
    Executor,
    Verifier,
    Surgeon,
}

impl TaskKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Blueprinter => "blueprinter",
            Self::Executor => "executor",
            Self::Verifier => "verifier",
            Self::Surgeon => "surgeon",
        }
    }

    fn env_prefix(self) -> &'static str {
        match self {
            Self::Blueprinter => "BLUEPRINTER",
            Self::Executor => "EXECUTOR",
            Self::Verifier => "VERIFIER",
            Self::Surgeon => "SURGEON",
        }
    }

    fn provider_env(self) -> String {
        format!("{}_PROVIDER", self.env_prefix())
    }

    fn model_env(self) -> String {
        format!("{}_MODEL", self.env_prefix())
    }
}

impl Display for TaskKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskModelConfig {
    pub task: TaskKind,
    pub provider: LlmProvider,
    pub model: String,
}

impl TaskModelConfig {
    pub fn new(task: TaskKind, provider: LlmProvider, model: impl Into<String>) -> Self {
        Self {
            task,
            provider,
            model: model.into(),
        }
    }
}

pub struct ModelRouter;

impl ModelRouter {
    pub fn from_env(
        task: TaskKind,
        default_provider: LlmProvider,
        default_model: &str,
    ) -> Result<TaskModelConfig> {
        let provider_key = task.provider_env();
        let model_key = task.model_env();

        let provider_value = match env::var(&provider_key) {
            Ok(value) => Some(value),
            Err(env::VarError::NotPresent) => None,
            Err(error) => {
                return Err(anyhow!("failed to read `{provider_key}`: {error}"));
            }
        };
        let model_value = match env::var(&model_key) {
            Ok(value) => Some(value),
            Err(env::VarError::NotPresent) => None,
            Err(error) => {
                return Err(anyhow!("failed to read `{model_key}`: {error}"));
            }
        };

        resolve_task_model(
            task,
            default_provider,
            default_model,
            provider_value,
            model_value,
        )
        .with_context(|| format!("failed to resolve model route for task `{task}`"))
    }
}

#[derive(Clone)]
pub struct LlmGateway {
    client: Client,
}

#[derive(Debug, Deserialize, Clone, Serialize, Default, PartialEq, Eq)]
pub struct TokenUsage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    temperature: f32,
    messages: Vec<ChatMessage>,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    id: Option<String>,
    model: Option<String>,
    choices: Vec<ChatChoice>,
    usage: Option<TokenUsage>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    content: serde_json::Value,
}

impl LlmGateway {
    pub fn new() -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .context("failed to build LLM HTTP client")?;

        Ok(Self { client })
    }

    pub async fn chat_completion(
        &self,
        route: &TaskModelConfig,
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<(String, TokenUsage)> {
        let api_key_var = route.provider.api_key_env();
        let api_key = env::var(api_key_var).with_context(|| {
            format!(
                "{api_key_var} is not configured for task `{}` using provider `{}`",
                route.task, route.provider
            )
        })?;
        let started_at = Instant::now();

        let request = ChatCompletionRequest {
            model: route.model.clone(),
            temperature: 0.1,
            messages: vec![
                ChatMessage {
                    role: "system".to_owned(),
                    content: system_prompt.to_owned(),
                },
                ChatMessage {
                    role: "user".to_owned(),
                    content: user_prompt.to_owned(),
                },
            ],
        };

        debug!(
            target: "audit::llm",
            provider = route.provider.as_str(),
            task = route.task.as_str(),
            model = route.model.as_str(),
            system_prompt = system_prompt,
            user_prompt = user_prompt,
            request_payload = ?&request,
            "LLM request payload"
        );

        info!(
            target: "audit::llm",
            provider = route.provider.as_str(),
            task = route.task.as_str(),
            model = route.model.as_str(),
            system_prompt_chars = system_prompt.len(),
            user_prompt_chars = user_prompt.len(),
            "Dispatching LLM request"
        );

        for attempt in 1..=MAX_RETRIES {
            let request_builder = self
                .client
                .post(route.provider.endpoint())
                .bearer_auth(&api_key)
                .json(&request);
            let response = route
                .provider
                .decorate_request(request_builder)
                .send()
                .await;

            match response {
                Ok(response) if response.status().is_success() => {
                    let payload: ChatCompletionResponse = response
                        .json()
                        .await
                        .context("failed to deserialize chat completion response body")?;
                    debug!(
                        target: "audit::llm",
                        provider = route.provider.as_str(),
                        task = route.task.as_str(),
                        model = route.model.as_str(),
                        response_payload = ?&payload,
                        "LLM raw response payload"
                    );
                    let choice = payload
                        .choices
                        .into_iter()
                        .next()
                        .context("chat completion response contained no choices")?;
                    let content = extract_message_text(choice.message.content)?;
                    let usage = payload.usage.unwrap_or_else(|| {
                        warn!(
                            target: "audit::llm",
                            provider = route.provider.as_str(),
                            task = route.task.as_str(),
                            model = route.model.as_str(),
                            "LLM response did not include usage metadata; defaulting token usage to zero"
                        );
                        TokenUsage::default()
                    });

                    debug!(
                        target: "audit::llm",
                        provider = route.provider.as_str(),
                        task = route.task.as_str(),
                        model = route.model.as_str(),
                        response_content = %content,
                        "LLM response content"
                    );

                    info!(
                        target: "audit::llm",
                        provider = route.provider.as_str(),
                        task = route.task.as_str(),
                        model = route.model.as_str(),
                        response_id = payload.id.as_deref().unwrap_or(""),
                        response_model = payload.model.as_deref().unwrap_or(""),
                        latency_ms = started_at.elapsed().as_millis() as u64,
                        system_prompt_chars = system_prompt.len(),
                        user_prompt_chars = user_prompt.len(),
                        completion_chars = content.len(),
                        prompt_tokens = usage.prompt_tokens,
                        completion_tokens = usage.completion_tokens,
                        total_tokens = usage.total_tokens,
                        "Completed LLM request"
                    );

                    return Ok((content, usage));
                }
                Ok(response) => {
                    let status = response.status();
                    let body = response.text().await.unwrap_or_default();
                    let error = anyhow!(
                        "{} request for task `{}` failed with status {status}: {body}",
                        route.provider,
                        route.task
                    );

                    if attempt == MAX_RETRIES || !is_retryable_status(status) {
                        return Err(error);
                    }

                    warn!(
                        target: "audit::llm",
                        attempt,
                        max_retries = MAX_RETRIES,
                        provider = route.provider.as_str(),
                        task = route.task.as_str(),
                        model = route.model.as_str(),
                        %status,
                        "LLM provider returned a retryable error"
                    );
                }
                Err(error) => {
                    if attempt == MAX_RETRIES {
                        return Err(anyhow!(
                            "{} request for task `{}` failed after retries: {error}",
                            route.provider,
                            route.task
                        ));
                    }

                    warn!(
                        target: "audit::llm",
                        attempt,
                        max_retries = MAX_RETRIES,
                        provider = route.provider.as_str(),
                        task = route.task.as_str(),
                        model = route.model.as_str(),
                        error = %error,
                        "LLM provider request failed, retrying"
                    );
                }
            }

            let backoff = Duration::from_secs(attempt as u64);
            sleep(backoff).await;
        }

        Err(anyhow!(
            "chat completion retries exhausted for task `{}`",
            route.task
        ))
    }
}

fn resolve_task_model(
    task: TaskKind,
    default_provider: LlmProvider,
    default_model: &str,
    provider_value: Option<String>,
    model_value: Option<String>,
) -> Result<TaskModelConfig> {
    let has_provider_override = provider_value
        .as_deref()
        .is_some_and(|value| !value.trim().is_empty());
    let has_model_override = model_value
        .as_deref()
        .is_some_and(|value| !value.trim().is_empty());

    let provider = match provider_value {
        Some(value) if !value.trim().is_empty() => value
            .parse()
            .with_context(|| format!("invalid provider override for task `{task}`"))?,
        _ => default_provider,
    };

    if has_provider_override && provider != default_provider && !has_model_override {
        return Err(anyhow!(
            "task `{task}` overrides the provider to `{provider}` but does not set a matching model; set `{}` alongside `{}`",
            task.model_env(),
            task.provider_env()
        ));
    }

    let model = match model_value {
        Some(value) if !value.trim().is_empty() => value,
        _ => default_model.to_owned(),
    };

    Ok(TaskModelConfig::new(task, provider, model))
}

fn is_retryable_status(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

fn extract_message_text(content: serde_json::Value) -> Result<String> {
    match content {
        serde_json::Value::String(text) => Ok(text),
        serde_json::Value::Array(parts) => {
            let mut aggregated = String::new();
            for part in parts {
                if let Some(text) = part.get("text").and_then(|value| value.as_str()) {
                    aggregated.push_str(text);
                }
            }

            if aggregated.is_empty() {
                Err(anyhow!(
                    "chat completion response did not contain textual message content"
                ))
            } else {
                Ok(aggregated)
            }
        }
        other => Err(anyhow!(
            "chat completion response content had an unsupported shape: {other}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{ChatCompletionResponse, LlmProvider, TaskKind, resolve_task_model};

    #[test]
    fn provider_parser_accepts_supported_values() {
        assert_eq!(
            "openrouter".parse::<LlmProvider>().unwrap(),
            LlmProvider::OpenRouter
        );
        assert_eq!(
            "together".parse::<LlmProvider>().unwrap(),
            LlmProvider::Together
        );
        assert_eq!(
            "together.ai".parse::<LlmProvider>().unwrap(),
            LlmProvider::Together
        );
    }

    #[test]
    fn task_model_resolution_uses_defaults_when_overrides_are_missing() {
        let route = resolve_task_model(
            TaskKind::Blueprinter,
            LlmProvider::OpenRouter,
            "google/gemini-3-flash-preview",
            None,
            None,
        )
        .unwrap();

        assert_eq!(route.provider, LlmProvider::OpenRouter);
        assert_eq!(route.model, "google/gemini-3-flash-preview");
    }

    #[test]
    fn task_model_resolution_applies_provider_and_model_overrides() {
        let route = resolve_task_model(
            TaskKind::Executor,
            LlmProvider::OpenRouter,
            "default-model",
            Some("together".to_owned()),
            Some("openai/gpt-oss-20b".to_owned()),
        )
        .unwrap();

        assert_eq!(route.provider, LlmProvider::Together);
        assert_eq!(route.model, "openai/gpt-oss-20b");
    }

    #[test]
    fn task_model_resolution_requires_model_when_provider_changes() {
        let error = resolve_task_model(
            TaskKind::Blueprinter,
            LlmProvider::OpenRouter,
            "google/gemini-3-flash-preview",
            Some("together".to_owned()),
            None,
        )
        .unwrap_err();

        assert!(
            error.to_string().contains("does not set a matching model"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn chat_completion_response_parses_usage_payload() {
        let payload = serde_json::from_str::<ChatCompletionResponse>(
            r#"{
                "id":"resp_123",
                "model":"google/gemini-3-flash-preview",
                "choices":[{"message":{"content":"ok"}}],
                "usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}
            }"#,
        )
        .expect("response should deserialize");

        let usage = payload.usage.expect("usage should be present");
        assert_eq!(usage.prompt_tokens, 11);
        assert_eq!(usage.completion_tokens, 7);
        assert_eq!(usage.total_tokens, 18);
    }
}
