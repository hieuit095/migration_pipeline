use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::HashMap;
use std::env;
use std::fmt::{self, Display};
use std::future::Future;
use std::hash::Hash;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{debug, info, warn};
use zeroclaw::config::{ModelRouteConfig, ReliabilityConfig};
use zeroclaw::providers::traits::TokenUsage as ZeroClawTokenUsage;
use zeroclaw::providers::{
    ChatMessage, ChatRequest, ChatResponse, Provider, create_routed_provider,
};
use zeroclaw::tools::ToolSpec;

const OPENROUTER_API_KEY_ENV: &str = "OPENROUTER_API_KEY";
const DOCKER_SANDBOX_MEMORY_ENV: &str = "DOCKER_SANDBOX_MEMORY";
const DOCKER_SANDBOX_CPUS_ENV: &str = "DOCKER_SANDBOX_CPUS";
const DEFAULT_TEMPERATURE: f64 = 0.1;
const PROVIDER_NAME: &str = "openrouter";
const MAX_FORMAT_RETRIES: usize = 3;
const DEFAULT_DOCKER_SANDBOX_MEMORY: &str = "256m";
const DEFAULT_DOCKER_SANDBOX_CPUS: &str = "0.5";
const RATE_LIMIT_BACKOFFS: [Duration; 4] = [
    Duration::from_secs(1),
    Duration::from_secs(2),
    Duration::from_secs(4),
    Duration::from_secs(8),
];

type ChatFuture<'a> = Pin<Box<dyn Future<Output = Result<ChatResponse>> + Send + 'a>>;

trait ChatBackend: Send + Sync {
    fn chat<'a>(
        &'a self,
        request: ChatRequest<'a>,
        model: &'a str,
        temperature: f64,
    ) -> ChatFuture<'a>;
}

struct ZeroClawBackend {
    provider: Arc<dyn Provider>,
}

impl ChatBackend for ZeroClawBackend {
    fn chat<'a>(
        &'a self,
        request: ChatRequest<'a>,
        model: &'a str,
        temperature: f64,
    ) -> ChatFuture<'a> {
        Box::pin(async move { self.provider.chat(request, model, temperature).await })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

    fn model_env(self) -> String {
        format!("{}_MODEL", self.env_prefix())
    }

    fn provider_env(self) -> String {
        format!("{}_PROVIDER", self.env_prefix())
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
    pub model: String,
    pub hint: String,
}

impl TaskModelConfig {
    pub fn new(task: TaskKind, model: impl Into<String>) -> Self {
        Self {
            task,
            model: model.into(),
            hint: format!("hint:{}", task.as_str()),
        }
    }

    fn route_name(&self) -> &str {
        self.hint.strip_prefix("hint:").unwrap_or(&self.hint)
    }

    fn as_model_route(&self) -> ModelRouteConfig {
        ModelRouteConfig {
            hint: self.route_name().to_owned(),
            provider: PROVIDER_NAME.to_owned(),
            model: self.model.clone(),
            api_key: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DockerSandboxConfig {
    pub memory_limit: String,
    pub cpu_limit: String,
}

impl Default for DockerSandboxConfig {
    fn default() -> Self {
        Self {
            memory_limit: DEFAULT_DOCKER_SANDBOX_MEMORY.to_owned(),
            cpu_limit: DEFAULT_DOCKER_SANDBOX_CPUS.to_owned(),
        }
    }
}

impl DockerSandboxConfig {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            memory_limit: read_env_with_default(
                DOCKER_SANDBOX_MEMORY_ENV,
                DEFAULT_DOCKER_SANDBOX_MEMORY,
            )?,
            cpu_limit: read_env_with_default(DOCKER_SANDBOX_CPUS_ENV, DEFAULT_DOCKER_SANDBOX_CPUS)?,
        })
    }
}

pub struct ModelRouter;

impl ModelRouter {
    pub fn from_env(task: TaskKind, default_model: &str) -> Result<TaskModelConfig> {
        let provider_env = task.provider_env();
        if env::var(&provider_env)
            .ok()
            .is_some_and(|value| !value.trim().is_empty())
        {
            warn!(
                task = task.as_str(),
                env_var = provider_env.as_str(),
                "Ignoring deprecated provider override; OpenRouter is the only supported provider for this pipeline"
            );
        }

        let model_key = task.model_env();
        let model_value = match env::var(&model_key) {
            Ok(value) if !value.trim().is_empty() => value,
            Ok(_) => default_model.to_owned(),
            Err(env::VarError::NotPresent) => default_model.to_owned(),
            Err(error) => {
                return Err(anyhow!("failed to read `{model_key}`: {error}"));
            }
        };

        Ok(TaskModelConfig::new(task, model_value))
    }
}

fn read_env_with_default(key: &str, default: &str) -> Result<String> {
    match env::var(key) {
        Ok(value) if !value.trim().is_empty() => Ok(value),
        Ok(_) | Err(env::VarError::NotPresent) => Ok(default.to_owned()),
        Err(error) => Err(anyhow!("failed to read `{key}`: {error}")),
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct ProjectTokenUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

impl ProjectTokenUsage {
    fn from_provider_usage(usage: Option<&ZeroClawTokenUsage>) -> Self {
        let prompt_tokens = usage
            .and_then(|value| value.input_tokens)
            .unwrap_or_default();
        let completion_tokens = usage
            .and_then(|value| value.output_tokens)
            .unwrap_or_default();

        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
        }
    }

    fn accumulate(&mut self, other: Self) {
        self.prompt_tokens = self.prompt_tokens.saturating_add(other.prompt_tokens);
        self.completion_tokens = self
            .completion_tokens
            .saturating_add(other.completion_tokens);
        self.total_tokens = self.total_tokens.saturating_add(other.total_tokens);
    }
}

#[derive(Debug, Clone)]
pub struct StructuredCall {
    pub tool_name: String,
    pub arguments: serde_json::Value,
    pub usage: ProjectTokenUsage,
}

impl StructuredCall {
    pub fn deserialize_arguments<T>(&self) -> Result<T>
    where
        T: DeserializeOwned,
    {
        serde_json::from_value(self.arguments.clone()).with_context(|| {
            format!(
                "failed to deserialize structured call `{}` arguments",
                self.tool_name
            )
        })
    }
}

#[derive(Clone)]
pub struct ZeroClawClient {
    backend: Arc<dyn ChatBackend>,
    task_configs: HashMap<TaskKind, TaskModelConfig>,
    rate_limit_backoffs: Vec<Duration>,
    max_format_retries: usize,
}

impl ZeroClawClient {
    pub fn new(task_configs: Vec<TaskModelConfig>) -> Result<Self> {
        let api_key = env::var(OPENROUTER_API_KEY_ENV).with_context(|| {
            format!(
                "{OPENROUTER_API_KEY_ENV} is not configured; OpenRouter is required for the ZeroClaw pipeline"
            )
        })?;
        let default_model = task_configs
            .first()
            .map(|config| config.model.as_str())
            .context("at least one task model configuration is required")?;
        let model_routes = task_configs
            .iter()
            .map(TaskModelConfig::as_model_route)
            .collect::<Vec<_>>();
        let reliability = ReliabilityConfig {
            provider_retries: 1,
            provider_backoff_ms: 0,
            ..Default::default()
        };
        let provider = create_routed_provider(
            PROVIDER_NAME,
            Some(api_key.as_str()),
            None,
            &reliability,
            &model_routes,
            default_model,
        )
        .context("failed to initialize ZeroClaw OpenRouter provider router")?;
        let provider: Arc<dyn Provider> = provider.into();

        Ok(Self {
            backend: Arc::new(ZeroClawBackend { provider }),
            task_configs: build_task_config_map(task_configs)?,
            rate_limit_backoffs: RATE_LIMIT_BACKOFFS.to_vec(),
            max_format_retries: MAX_FORMAT_RETRIES,
        })
    }

    pub fn model_for(&self, task: TaskKind) -> &str {
        self.task_configs
            .get(&task)
            .map(|config| config.model.as_str())
            .unwrap_or("unconfigured")
    }

    pub fn provider_name(&self) -> &'static str {
        PROVIDER_NAME
    }

    #[allow(dead_code)]
    pub async fn chat(
        &self,
        task: TaskKind,
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<ChatResponse> {
        let route = self.task_config(task)?;
        let started_at = Instant::now();
        let response = self
            .dispatch_with_backoff(route, system_prompt, user_prompt, None)
            .await
            .with_context(|| format!("zeroclaw chat failed for task `{task}`"))?;
        let usage = self.extract_usage(task, route, response.usage.as_ref());

        info!(
            target: "audit::llm",
            provider = self.provider_name(),
            task = task.as_str(),
            model = route.model.as_str(),
            latency_ms = started_at.elapsed().as_millis() as u64,
            prompt_tokens = usage.prompt_tokens,
            completion_tokens = usage.completion_tokens,
            total_tokens = usage.total_tokens,
            "Completed ZeroClaw chat request"
        );

        Ok(response)
    }

    pub async fn chat_with_schema(
        &self,
        task: TaskKind,
        system_prompt: &str,
        user_prompt: &str,
        tool: ToolSpec,
    ) -> Result<StructuredCall> {
        let route = self.task_config(task)?;
        let started_at = Instant::now();
        let mut accumulated_usage = ProjectTokenUsage::default();
        let mut last_error = None;

        for format_attempt in 0..=self.max_format_retries {
            let effective_system_prompt =
                build_format_retry_prompt(system_prompt, &tool, format_attempt);
            let response = self
                .dispatch_with_backoff(
                    route,
                    &effective_system_prompt,
                    user_prompt,
                    Some(std::slice::from_ref(&tool)),
                )
                .await
                .with_context(|| format!("zeroclaw structured chat failed for task `{task}`"))?;
            let usage = self.extract_usage(task, route, response.usage.as_ref());
            accumulated_usage.accumulate(usage);

            match parse_structured_call(&response, &tool) {
                Ok((tool_name, arguments)) => {
                    info!(
                        target: "audit::llm",
                        provider = self.provider_name(),
                        task = task.as_str(),
                        model = route.model.as_str(),
                        format_retries = format_attempt,
                        latency_ms = started_at.elapsed().as_millis() as u64,
                        prompt_tokens = accumulated_usage.prompt_tokens,
                        completion_tokens = accumulated_usage.completion_tokens,
                        total_tokens = accumulated_usage.total_tokens,
                        "Completed ZeroClaw structured chat request"
                    );

                    return Ok(StructuredCall {
                        tool_name,
                        arguments,
                        usage: accumulated_usage,
                    });
                }
                Err(error) => {
                    warn!(
                        target: "audit::llm",
                        provider = self.provider_name(),
                        task = task.as_str(),
                        model = route.model.as_str(),
                        format_attempt = format_attempt + 1,
                        error = %error,
                        "LLM response violated the required structured output contract"
                    );
                    last_error = Some(error);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            anyhow!(
                "structured response retries exhausted for task `{}` and tool `{}`",
                task,
                tool.name
            )
        }))
    }

    #[cfg(test)]
    fn with_backend_for_tests(
        task_configs: Vec<TaskModelConfig>,
        backend: Arc<dyn ChatBackend>,
        rate_limit_backoffs: Vec<Duration>,
        max_format_retries: usize,
    ) -> Self {
        Self {
            backend,
            task_configs: build_task_config_map(task_configs).expect("test task configs"),
            rate_limit_backoffs,
            max_format_retries,
        }
    }

    fn task_config(&self, task: TaskKind) -> Result<&TaskModelConfig> {
        self.task_configs
            .get(&task)
            .with_context(|| format!("missing task model configuration for task `{task}`"))
    }

    fn extract_usage(
        &self,
        task: TaskKind,
        route: &TaskModelConfig,
        usage: Option<&ZeroClawTokenUsage>,
    ) -> ProjectTokenUsage {
        if usage.is_none() {
            warn!(
                target: "audit::llm",
                provider = self.provider_name(),
                task = task.as_str(),
                model = route.model.as_str(),
                "LLM response did not include usage metadata; defaulting token usage to zero"
            );
        }

        ProjectTokenUsage::from_provider_usage(usage)
    }

    async fn dispatch_with_backoff(
        &self,
        route: &TaskModelConfig,
        system_prompt: &str,
        user_prompt: &str,
        tools: Option<&[ToolSpec]>,
    ) -> Result<ChatResponse> {
        let messages = vec![
            ChatMessage::system(system_prompt),
            ChatMessage::user(user_prompt),
        ];
        let max_attempts = self.rate_limit_backoffs.len() + 1;

        debug!(
            target: "audit::llm",
            provider = self.provider_name(),
            task = route.task.as_str(),
            model = route.model.as_str(),
            system_prompt = system_prompt,
            user_prompt = user_prompt,
            tool_schemas = ?tools,
            "Dispatching ZeroClaw request payload"
        );

        info!(
            target: "audit::llm",
            provider = self.provider_name(),
            task = route.task.as_str(),
            model = route.model.as_str(),
            system_prompt_chars = system_prompt.len(),
            user_prompt_chars = user_prompt.len(),
            tool_count = tools.map_or(0, <[ToolSpec]>::len),
            "Dispatching ZeroClaw request"
        );

        for attempt in 0..max_attempts {
            let request = ChatRequest {
                messages: &messages,
                tools,
            };
            match self
                .backend
                .chat(request, route.hint.as_str(), DEFAULT_TEMPERATURE)
                .await
            {
                Ok(response) => {
                    debug!(
                        target: "audit::llm",
                        provider = self.provider_name(),
                        task = route.task.as_str(),
                        model = route.model.as_str(),
                        raw_response = ?response,
                        "Received ZeroClaw provider response"
                    );
                    return Ok(response);
                }
                Err(error)
                    if attempt < self.rate_limit_backoffs.len() && is_rate_limit_error(&error) =>
                {
                    let backoff = self.rate_limit_backoffs[attempt];
                    warn!(
                        target: "audit::llm",
                        provider = self.provider_name(),
                        task = route.task.as_str(),
                        model = route.model.as_str(),
                        attempt = attempt + 1,
                        max_attempts,
                        backoff_ms = backoff.as_millis() as u64,
                        error = %error,
                        "OpenRouter rate limited the request; retrying with exponential backoff"
                    );
                    sleep(backoff).await;
                }
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!(
                            "ZeroClaw provider request failed for task `{}` using model `{}`",
                            route.task, route.model
                        )
                    });
                }
            }
        }

        bail!(
            "ZeroClaw rate-limit retries exhausted for task `{}` using model `{}`",
            route.task,
            route.model
        )
    }
}

fn build_task_config_map(
    task_configs: Vec<TaskModelConfig>,
) -> Result<HashMap<TaskKind, TaskModelConfig>> {
    let mut map = HashMap::with_capacity(task_configs.len());

    for config in task_configs {
        if map.insert(config.task, config.clone()).is_some() {
            bail!(
                "duplicate task model configuration provided for task `{}`",
                config.task
            );
        }
    }

    Ok(map)
}

fn build_format_retry_prompt(base_prompt: &str, tool: &ToolSpec, format_attempt: usize) -> String {
    if format_attempt == 0 {
        return base_prompt.to_owned();
    }

    format!(
        concat!(
            "{base_prompt}\n\n",
            "FORMAT CORRECTION REQUIRED #{format_attempt}:\n",
            "Your previous response did not use the required structured output.\n",
            "You must respond by calling exactly one native tool named `{tool_name}`.\n",
            "Do not return plain text, markdown fences, or explanations.\n",
            "Tool arguments must match this JSON schema exactly: {schema}\n"
        ),
        base_prompt = base_prompt,
        format_attempt = format_attempt,
        tool_name = tool.name,
        schema = tool.parameters
    )
}

fn parse_structured_call(
    response: &ChatResponse,
    tool: &ToolSpec,
) -> Result<(String, serde_json::Value)> {
    if response.tool_calls.len() != 1 {
        bail!(
            "expected exactly one tool call named `{}`, received {} tool calls and text payload `{}`",
            tool.name,
            response.tool_calls.len(),
            truncate_for_error(response.text.as_deref().unwrap_or_default())
        );
    }

    let tool_call = &response.tool_calls[0];
    if tool_call.name != tool.name {
        bail!(
            "expected tool call `{}`, received `{}`",
            tool.name,
            tool_call.name
        );
    }

    let arguments =
        serde_json::from_str::<serde_json::Value>(&tool_call.arguments).with_context(|| {
            format!(
                "tool call `{}` did not contain valid JSON arguments",
                tool.name
            )
        })?;

    Ok((tool_call.name.clone(), arguments))
}

fn truncate_for_error(value: &str) -> String {
    const MAX_LEN: usize = 200;

    if value.chars().count() <= MAX_LEN {
        value.to_owned()
    } else {
        let truncated = value.chars().take(MAX_LEN).collect::<String>();
        format!("{truncated}...")
    }
}

fn is_rate_limit_error(error: &anyhow::Error) -> bool {
    let normalized = error.to_string().to_ascii_lowercase();
    normalized.contains("429")
        || normalized.contains("too many requests")
        || normalized.contains("rate limit")
        || normalized.contains("rate-limited")
}

#[cfg(test)]
mod tests {
    use super::{
        ChatBackend, DockerSandboxConfig, ModelRouter, ProjectTokenUsage, StructuredCall, TaskKind,
        TaskModelConfig, ZeroClawClient,
    };
    use anyhow::{Result, anyhow};
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use zeroclaw::providers::traits::TokenUsage;
    use zeroclaw::providers::{ChatRequest, ChatResponse, ToolCall};
    use zeroclaw::tools::ToolSpec;

    #[derive(Default)]
    struct ScriptedBackend {
        responses: Mutex<VecDeque<Result<ChatResponse>>>,
    }

    impl ScriptedBackend {
        fn new(responses: Vec<Result<ChatResponse>>) -> Self {
            Self {
                responses: Mutex::new(VecDeque::from(responses)),
            }
        }
    }

    impl ChatBackend for ScriptedBackend {
        fn chat<'a>(
            &'a self,
            _request: ChatRequest<'a>,
            _model: &'a str,
            _temperature: f64,
        ) -> super::ChatFuture<'a> {
            Box::pin(async move {
                let mut guard = self
                    .responses
                    .lock()
                    .map_err(|_| anyhow!("scripted backend mutex was poisoned"))?;
                guard
                    .pop_front()
                    .unwrap_or_else(|| Err(anyhow!("scripted backend ran out of responses")))
            })
        }
    }

    fn build_task_configs() -> Vec<TaskModelConfig> {
        vec![
            TaskModelConfig::new(TaskKind::Blueprinter, "google/gemini-3-flash-preview"),
            TaskModelConfig::new(TaskKind::Executor, "minimax/minimax-m2.5"),
            TaskModelConfig::new(TaskKind::Verifier, "z-ai/glm-5"),
            TaskModelConfig::new(TaskKind::Surgeon, "anthropic/claude-3.5-sonnet"),
        ]
    }

    fn verification_tool() -> ToolSpec {
        ToolSpec {
            name: "submit_verification_report".to_owned(),
            description: "Return the verification result".to_owned(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "tests_passed": { "type": "boolean" },
                    "sandbox_execution_errors": {
                        "type": "array",
                        "items": { "type": "string" }
                    }
                },
                "required": [
                    "tests_passed",
                    "sandbox_execution_errors"
                ],
                "additionalProperties": false
            }),
        }
    }

    fn response_with_tool_call(
        arguments: serde_json::Value,
        usage: Option<TokenUsage>,
    ) -> ChatResponse {
        ChatResponse {
            text: None,
            tool_calls: vec![ToolCall {
                id: "tool_1".to_owned(),
                name: "submit_verification_report".to_owned(),
                arguments: arguments.to_string(),
            }],
            usage,
            reasoning_content: None,
        }
    }

    #[test]
    fn model_router_uses_default_model_when_env_is_missing() {
        let route = ModelRouter::from_env(TaskKind::Blueprinter, "google/gemini-3-flash-preview")
            .expect("model route should resolve");

        assert_eq!(route.model, "google/gemini-3-flash-preview");
        assert_eq!(route.hint, "hint:blueprinter");
    }

    #[test]
    fn docker_sandbox_config_uses_defaults_when_env_is_missing() {
        let previous_memory = std::env::var(super::DOCKER_SANDBOX_MEMORY_ENV).ok();
        let previous_cpus = std::env::var(super::DOCKER_SANDBOX_CPUS_ENV).ok();
        unsafe {
            std::env::remove_var(super::DOCKER_SANDBOX_MEMORY_ENV);
            std::env::remove_var(super::DOCKER_SANDBOX_CPUS_ENV);
        }

        let config = DockerSandboxConfig::from_env().expect("sandbox config should load");

        if let Some(value) = previous_memory {
            unsafe {
                std::env::set_var(super::DOCKER_SANDBOX_MEMORY_ENV, value);
            }
        }
        if let Some(value) = previous_cpus {
            unsafe {
                std::env::set_var(super::DOCKER_SANDBOX_CPUS_ENV, value);
            }
        }

        assert_eq!(config.memory_limit, "256m");
        assert_eq!(config.cpu_limit, "0.5");
    }

    #[test]
    fn zeroclaw_client_requires_openrouter_api_key() {
        let previous = std::env::var(super::OPENROUTER_API_KEY_ENV).ok();
        unsafe {
            std::env::remove_var(super::OPENROUTER_API_KEY_ENV);
        }

        let error = ZeroClawClient::new(build_task_configs())
            .err()
            .expect("client should fail");

        if let Some(value) = previous {
            unsafe {
                std::env::set_var(super::OPENROUTER_API_KEY_ENV, value);
            }
        }

        assert!(error.to_string().contains(super::OPENROUTER_API_KEY_ENV));
    }

    #[tokio::test]
    async fn chat_with_schema_retries_plain_text_before_accepting_tool_call() {
        let backend = Arc::new(ScriptedBackend::new(vec![
            Ok(ChatResponse {
                text: Some("Here is the report".to_owned()),
                tool_calls: Vec::new(),
                usage: Some(TokenUsage {
                    input_tokens: Some(11),
                    output_tokens: Some(5),
                }),
                reasoning_content: None,
            }),
            Ok(response_with_tool_call(
                serde_json::json!({
                    "tests_passed": true,
                    "sandbox_execution_errors": []
                }),
                Some(TokenUsage {
                    input_tokens: Some(7),
                    output_tokens: Some(3),
                }),
            )),
        ]));
        let client = ZeroClawClient::with_backend_for_tests(
            build_task_configs(),
            backend,
            vec![Duration::ZERO],
            3,
        );

        let call = client
            .chat_with_schema(
                TaskKind::Executor,
                "Use the tool schema",
                "Verify the code",
                verification_tool(),
            )
            .await
            .expect("structured call should succeed");

        assert_eq!(call.tool_name, "submit_verification_report");
        assert_eq!(
            call.usage,
            ProjectTokenUsage {
                prompt_tokens: 18,
                completion_tokens: 8,
                total_tokens: 26,
            }
        );
    }

    #[tokio::test]
    async fn chat_with_schema_retries_rate_limits() {
        let backend = Arc::new(ScriptedBackend::new(vec![
            Err(anyhow!("429 Too Many Requests")),
            Ok(response_with_tool_call(
                serde_json::json!({
                    "tests_passed": true,
                    "sandbox_execution_errors": []
                }),
                None,
            )),
        ]));
        let client = ZeroClawClient::with_backend_for_tests(
            build_task_configs(),
            backend,
            vec![Duration::ZERO],
            0,
        );

        let call = client
            .chat_with_schema(
                TaskKind::Executor,
                "Use the tool schema",
                "Verify the code",
                verification_tool(),
            )
            .await
            .expect("structured call should succeed after rate-limit retry");

        assert_eq!(call.tool_name, "submit_verification_report");
        assert_eq!(call.usage, ProjectTokenUsage::default());
    }

    #[tokio::test]
    async fn chat_with_schema_rejects_wrong_tool_name() {
        let backend = Arc::new(ScriptedBackend::new(vec![Ok(ChatResponse {
            text: None,
            tool_calls: vec![ToolCall {
                id: "tool_1".to_owned(),
                name: "unexpected_tool".to_owned(),
                arguments: "{}".to_owned(),
            }],
            usage: None,
            reasoning_content: None,
        })]));
        let client =
            ZeroClawClient::with_backend_for_tests(build_task_configs(), backend, Vec::new(), 0);

        let error = client
            .chat_with_schema(
                TaskKind::Executor,
                "Use the tool schema",
                "Verify the code",
                verification_tool(),
            )
            .await
            .expect_err("structured call should fail");

        assert!(error.to_string().contains("unexpected_tool"));
    }

    #[test]
    fn structured_call_deserializes_arguments() {
        let call = StructuredCall {
            tool_name: "submit_verification_report".to_owned(),
            arguments: serde_json::json!({
                "tests_passed": true,
                "sandbox_execution_errors": []
            }),
            usage: ProjectTokenUsage::default(),
        };
        let value = call
            .deserialize_arguments::<serde_json::Value>()
            .expect("arguments should deserialize");

        assert_eq!(value["tests_passed"], true);
    }
}
