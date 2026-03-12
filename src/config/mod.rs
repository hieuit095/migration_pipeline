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

const OPENAI_API_KEY_ENV: &str = "OPENAI_API_KEY";
const TOGETHER_API_KEY_ENV: &str = "TOGETHER_API_KEY";
const OPENROUTER_API_KEY_ENV: &str = "OPENROUTER_API_KEY";
const DOCKER_SANDBOX_MEMORY_ENV: &str = "DOCKER_SANDBOX_MEMORY";
const DOCKER_SANDBOX_CPUS_ENV: &str = "DOCKER_SANDBOX_CPUS";
const DEFAULT_TEMPERATURE: f64 = 0.1;
const DEFAULT_TASK_PROVIDER: &str = "openrouter";
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
    providers_by_hint: HashMap<String, String>,
}

#[derive(Debug)]
struct ProviderHttpStatusError {
    provider: String,
    status_code: u16,
    message: String,
}

impl ProviderHttpStatusError {
    fn new(provider: impl Into<String>, status_code: u16, message: String) -> Self {
        Self {
            provider: provider.into(),
            status_code,
            message,
        }
    }
}

impl Display for ProviderHttpStatusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} provider HTTP error ({}): {}",
            self.provider, self.status_code, self.message
        )
    }
}

impl std::error::Error for ProviderHttpStatusError {}

impl ZeroClawBackend {
    fn provider_for_hint(&self, hint: &str) -> &str {
        self.providers_by_hint
            .get(hint)
            .map(String::as_str)
            .unwrap_or(DEFAULT_TASK_PROVIDER)
    }

    fn normalize_error(provider: &str, error: anyhow::Error) -> anyhow::Error {
        if let Some(status_code) = extract_http_status_code(&error) {
            return anyhow!(ProviderHttpStatusError::new(
                provider,
                status_code,
                error.to_string(),
            ));
        }

        error
    }
}

impl ChatBackend for ZeroClawBackend {
    fn chat<'a>(
        &'a self,
        request: ChatRequest<'a>,
        model: &'a str,
        temperature: f64,
    ) -> ChatFuture<'a> {
        let provider = self.provider_for_hint(model).to_owned();
        Box::pin(async move {
            self.provider
                .chat(request, model, temperature)
                .await
                .map_err(|error| ZeroClawBackend::normalize_error(provider.as_str(), error))
        })
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
    pub provider: String,
    pub model: String,
    pub hint: String,
}

impl TaskModelConfig {
    pub fn new(task: TaskKind, provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            task,
            provider: normalize_provider_name(&provider.into()),
            model: model.into(),
            hint: format!("hint:{}", task.as_str()),
        }
    }

    fn route_name(&self) -> &str {
        self.hint.strip_prefix("hint:").unwrap_or(&self.hint)
    }

    fn as_model_route(&self) -> Result<ModelRouteConfig> {
        Ok(ModelRouteConfig {
            hint: self.route_name().to_owned(),
            provider: self.provider.clone(),
            model: self.model.clone(),
            api_key: provider_api_key(&self.provider)?,
        })
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
        let provider_value = match read_optional_env(&provider_env)? {
            Some(value) if !value.trim().is_empty() => normalize_provider_name(&value),
            Some(_) | None => DEFAULT_TASK_PROVIDER.to_owned(),
        };

        let model_key = task.model_env();
        let model_value = match read_optional_env(&model_key)? {
            Some(value) if !value.trim().is_empty() => value,
            Some(_) | None => default_model.to_owned(),
        };

        Ok(TaskModelConfig::new(task, provider_value, model_value))
    }
}

fn read_env_with_default(key: &str, default: &str) -> Result<String> {
    match read_optional_env(key)? {
        Some(value) if !value.trim().is_empty() => Ok(value),
        Some(_) | None => Ok(default.to_owned()),
    }
}

fn read_optional_env(key: &str) -> Result<Option<String>> {
    match env::var(key) {
        Ok(value) => Ok(Some(value)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error).with_context(|| format!("failed to read `{key}`")),
    }
}

fn normalize_provider_name(provider: &str) -> String {
    match provider.trim().to_ascii_lowercase().as_str() {
        "together.ai" | "together-ai" => "together".to_owned(),
        normalized => normalized.to_owned(),
    }
}

fn provider_api_key_env(provider: &str) -> Option<&'static str> {
    match normalize_provider_name(provider).as_str() {
        "openai" => Some(OPENAI_API_KEY_ENV),
        "together" => Some(TOGETHER_API_KEY_ENV),
        "openrouter" => Some(OPENROUTER_API_KEY_ENV),
        _ => None,
    }
}

fn provider_api_key(provider: &str) -> Result<Option<String>> {
    let Some(env_key) = provider_api_key_env(provider) else {
        return Ok(None);
    };

    match read_optional_env(env_key)? {
        Some(value) if !value.trim().is_empty() => Ok(Some(value)),
        Some(_) | None => Ok(None),
    }
}

fn require_provider_api_key(provider: &str) -> Result<String> {
    let env_key = provider_api_key_env(provider).with_context(|| {
        format!(
            "provider `{provider}` is not supported by this pipeline wrapper; supported providers are `openai`, `together`, and `openrouter`"
        )
    })?;
    provider_api_key(provider)?
        .with_context(|| format!("`{env_key}` is not configured for provider `{provider}`"))
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
        let primary_config = task_configs
            .first()
            .context("at least one task model configuration is required")?;
        let mut providers_by_hint = HashMap::with_capacity(task_configs.len());
        for config in &task_configs {
            let expected_env_key =
                provider_api_key_env(&config.provider).unwrap_or("UNKNOWN_API_KEY");
            require_provider_api_key(&config.provider).with_context(|| {
                format!(
                    "missing API key for task `{}` provider `{}`; expected `{expected_env_key}`",
                    config.task, config.provider
                )
            })?;
            providers_by_hint.insert(config.hint.clone(), config.provider.clone());
        }
        let default_model = task_configs
            .first()
            .map(|config| config.model.as_str())
            .context("at least one task model configuration is required")?;
        let model_routes = task_configs
            .iter()
            .map(TaskModelConfig::as_model_route)
            .collect::<Result<Vec<_>>>()?;
        let reliability = ReliabilityConfig {
            provider_retries: 1,
            provider_backoff_ms: 0,
            ..Default::default()
        };
        let provider = create_routed_provider(
            primary_config.provider.as_str(),
            model_routes
                .iter()
                .find(|route| route.provider == primary_config.provider)
                .and_then(|route| route.api_key.as_deref()),
            None,
            &reliability,
            &model_routes,
            default_model,
        )
        .with_context(|| {
            format!(
                "failed to initialize ZeroClaw provider router with primary provider `{}`",
                primary_config.provider
            )
        })?;
        let provider: Arc<dyn Provider> = provider.into();

        Ok(Self {
            backend: Arc::new(ZeroClawBackend {
                provider,
                providers_by_hint,
            }),
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

    pub fn provider_for(&self, task: TaskKind) -> &str {
        self.task_configs
            .get(&task)
            .map(|config| config.provider.as_str())
            .unwrap_or("unconfigured")
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
            provider = route.provider.as_str(),
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
                        provider = route.provider.as_str(),
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
                        provider = route.provider.as_str(),
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
                provider = route.provider.as_str(),
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
            provider = route.provider.as_str(),
            task = route.task.as_str(),
            model = route.model.as_str(),
            system_prompt = system_prompt,
            user_prompt = user_prompt,
            tool_schemas = ?tools,
            "Dispatching ZeroClaw request payload"
        );

        info!(
            target: "audit::llm",
            provider = route.provider.as_str(),
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
                        provider = route.provider.as_str(),
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
                        provider = route.provider.as_str(),
                        task = route.task.as_str(),
                        model = route.model.as_str(),
                        attempt = attempt + 1,
                        max_attempts,
                        backoff_ms = backoff.as_millis() as u64,
                        error = %error,
                        "Provider returned a retryable HTTP status; retrying with exponential backoff"
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

fn extract_http_status_code(error: &anyhow::Error) -> Option<u16> {
    error
        .downcast_ref::<ProviderHttpStatusError>()
        .map(|error| error.status_code)
        .or_else(|| {
            error
                .downcast_ref::<reqwest::Error>()
                .and_then(|error| error.status())
                .map(|status| status.as_u16())
        })
        .or_else(|| extract_provider_api_status_code(error))
}

fn extract_provider_api_status_code(error: &anyhow::Error) -> Option<u16> {
    let message = error.to_string();
    let (_, remainder) = message.split_once(" API error (")?;
    let status_text = remainder.split_once("):")?.0;
    let status_code = status_text.split_whitespace().next()?;
    status_code.parse::<u16>().ok()
}

fn is_rate_limit_error(error: &anyhow::Error) -> bool {
    matches!(extract_http_status_code(error), Some(429 | 503))
}

#[cfg(test)]
mod tests {
    use super::{
        ChatBackend, DockerSandboxConfig, ModelRouter, ProjectTokenUsage, ProviderHttpStatusError,
        StructuredCall, TaskKind, TaskModelConfig, ZeroClawBackend, ZeroClawClient,
        is_rate_limit_error,
    };
    use anyhow::{Result, anyhow};
    use std::collections::VecDeque;
    use std::sync::{Arc, LazyLock, Mutex};
    use std::time::Duration;
    use zeroclaw::providers::traits::TokenUsage;
    use zeroclaw::providers::{ChatRequest, ChatResponse, ToolCall};
    use zeroclaw::tools::ToolSpec;

    static ENV_MUTEX: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    fn lock_env() -> std::sync::MutexGuard<'static, ()> {
        ENV_MUTEX
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    struct EnvVarGuard {
        key: String,
        previous: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: impl Into<String>, value: &str) -> Self {
            let key = key.into();
            let previous = std::env::var(&key).ok();
            unsafe {
                std::env::set_var(&key, value);
            }
            Self { key, previous }
        }

        fn remove(key: impl Into<String>) -> Self {
            let key = key.into();
            let previous = std::env::var(&key).ok();
            unsafe {
                std::env::remove_var(&key);
            }
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => unsafe {
                    std::env::set_var(&self.key, value);
                },
                None => unsafe {
                    std::env::remove_var(&self.key);
                },
            }
        }
    }

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
            TaskModelConfig::new(
                TaskKind::Blueprinter,
                super::DEFAULT_TASK_PROVIDER,
                "google/gemini-3-flash-preview",
            ),
            TaskModelConfig::new(
                TaskKind::Executor,
                super::DEFAULT_TASK_PROVIDER,
                "minimax/minimax-m2.5",
            ),
            TaskModelConfig::new(
                TaskKind::Verifier,
                super::DEFAULT_TASK_PROVIDER,
                "z-ai/glm-5",
            ),
            TaskModelConfig::new(
                TaskKind::Surgeon,
                super::DEFAULT_TASK_PROVIDER,
                "anthropic/claude-sonnet-4.5",
            ),
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
    fn model_router_uses_default_provider_and_model_when_env_is_missing() {
        let _env_lock = lock_env();
        let _provider_env = EnvVarGuard::remove(TaskKind::Blueprinter.provider_env());
        let _model_env = EnvVarGuard::remove(TaskKind::Blueprinter.model_env());
        let route = ModelRouter::from_env(TaskKind::Blueprinter, "google/gemini-3-flash-preview")
            .expect("model route should resolve");

        assert_eq!(route.provider, super::DEFAULT_TASK_PROVIDER);
        assert_eq!(route.model, "google/gemini-3-flash-preview");
        assert_eq!(route.hint, "hint:blueprinter");
    }

    #[test]
    fn model_router_uses_task_specific_provider_override() {
        let _env_lock = lock_env();
        let _provider_env = EnvVarGuard::set(TaskKind::Blueprinter.provider_env(), "Together.ai");
        let _model_env = EnvVarGuard::set(
            TaskKind::Blueprinter.model_env(),
            "meta-llama/Llama-4-Maverick",
        );

        let route = ModelRouter::from_env(TaskKind::Blueprinter, "google/gemini-3-flash-preview")
            .expect("model route should resolve");

        assert_eq!(route.provider, "together");
        assert_eq!(route.model, "meta-llama/Llama-4-Maverick");
    }

    #[test]
    fn docker_sandbox_config_uses_defaults_when_env_is_missing() {
        let _env_lock = lock_env();
        let _memory_env = EnvVarGuard::remove(super::DOCKER_SANDBOX_MEMORY_ENV);
        let _cpu_env = EnvVarGuard::remove(super::DOCKER_SANDBOX_CPUS_ENV);

        let config = DockerSandboxConfig::from_env().expect("sandbox config should load");

        assert_eq!(config.memory_limit, "256m");
        assert_eq!(config.cpu_limit, "0.5");
    }

    #[test]
    fn zeroclaw_client_requires_openrouter_api_key() {
        let _env_lock = lock_env();
        let _openrouter_key = EnvVarGuard::remove(super::OPENROUTER_API_KEY_ENV);

        let error = ZeroClawClient::new(build_task_configs())
            .err()
            .expect("client should fail");

        assert!(error.to_string().contains(super::OPENROUTER_API_KEY_ENV));
    }

    #[test]
    fn zeroclaw_client_requires_openai_api_key_for_openai_routes() {
        let _env_lock = lock_env();
        let _openai_key = EnvVarGuard::remove(super::OPENAI_API_KEY_ENV);

        let error = ZeroClawClient::new(vec![TaskModelConfig::new(
            TaskKind::Blueprinter,
            "openai",
            "gpt-5",
        )])
        .err()
        .expect("client should fail");

        assert!(error.to_string().contains(super::OPENAI_API_KEY_ENV));
    }

    #[test]
    fn task_model_config_injects_provider_specific_api_keys() {
        let _env_lock = lock_env();
        let _openai_key = EnvVarGuard::set(super::OPENAI_API_KEY_ENV, "openai-key");
        let _together_key = EnvVarGuard::set(super::TOGETHER_API_KEY_ENV, "together-key");
        let _openrouter_key = EnvVarGuard::set(super::OPENROUTER_API_KEY_ENV, "openrouter-key");

        let openai_route = TaskModelConfig::new(TaskKind::Blueprinter, "openai", "gpt-5")
            .as_model_route()
            .expect("openai route should resolve");
        let together_route = TaskModelConfig::new(
            TaskKind::Executor,
            "Together.ai",
            "meta-llama/Llama-4-Maverick",
        )
        .as_model_route()
        .expect("together route should resolve");
        let openrouter_route = TaskModelConfig::new(
            TaskKind::Verifier,
            "openrouter",
            "anthropic/claude-sonnet-4.5",
        )
        .as_model_route()
        .expect("openrouter route should resolve");

        assert_eq!(openai_route.provider, "openai");
        assert_eq!(openai_route.api_key.as_deref(), Some("openai-key"));
        assert_eq!(together_route.provider, "together");
        assert_eq!(together_route.api_key.as_deref(), Some("together-key"));
        assert_eq!(openrouter_route.provider, "openrouter");
        assert_eq!(openrouter_route.api_key.as_deref(), Some("openrouter-key"));
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
            Err(anyhow!(ProviderHttpStatusError::new(
                "openrouter",
                429,
                "Too Many Requests".to_owned()
            ))),
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

    #[test]
    fn rate_limit_detection_only_accepts_typed_retryable_status_codes() {
        let too_many_requests = anyhow!(ProviderHttpStatusError::new(
            "openrouter",
            429,
            "Too Many Requests".to_owned()
        ));
        let service_unavailable = anyhow!(ProviderHttpStatusError::new(
            "openrouter",
            503,
            "Service Unavailable".to_owned()
        ));
        let unauthorized = anyhow!(ProviderHttpStatusError::new(
            "openrouter",
            401,
            "Unauthorized".to_owned()
        ));
        let free_form = anyhow!("429 Too Many Requests");

        assert!(is_rate_limit_error(&too_many_requests));
        assert!(is_rate_limit_error(&service_unavailable));
        assert!(!is_rate_limit_error(&unauthorized));
        assert!(!is_rate_limit_error(&free_form));
    }

    #[test]
    fn zero_claw_backend_normalizes_provider_api_status_errors() {
        let normalized = ZeroClawBackend::normalize_error(
            "openrouter",
            anyhow!("OpenRouter API error (429 Too Many Requests): retry later"),
        );
        let status_error = normalized
            .downcast_ref::<ProviderHttpStatusError>()
            .expect("provider status error should be normalized");

        assert_eq!(status_error.provider, "openrouter");
        assert_eq!(status_error.status_code, 429);
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
