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
const DEFAULT_STRUCTURED_FALLBACK_MODEL: &str = "gpt-5.4";
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
                "failed to deserialize structured call `{}` arguments from payload `{}`",
                self.tool_name,
                truncate_for_error(self.arguments.to_string().as_str())
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
        let mut route_configs = task_configs.clone();
        route_configs.extend(supplemental_task_routes(&task_configs)?);

        let mut providers_by_hint = HashMap::with_capacity(route_configs.len());
        for config in &route_configs {
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
        let model_routes = route_configs
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
        match self
            .chat_with_schema_on_route(task, route, system_prompt, user_prompt, tool.clone())
            .await
        {
            Ok(call) => Ok(call),
            Err(primary_error) => {
                let Some(fallback_route) = self.structured_output_fallback_route(task, route)?
                else {
                    return Err(primary_error);
                };

                warn!(
                    target: "audit::llm",
                    primary_provider = route.provider.as_str(),
                    primary_model = route.model.as_str(),
                    fallback_provider = fallback_route.provider.as_str(),
                    fallback_model = fallback_route.model.as_str(),
                    task = task.as_str(),
                    error = %primary_error,
                    "Structured output retries were exhausted; retrying with the fallback provider"
                );

                self.chat_with_schema_on_route(
                    task,
                    &fallback_route,
                    system_prompt,
                    user_prompt,
                    tool,
                )
                .await
                .with_context(|| {
                    format!(
                        "structured output fallback failed after the primary route exhausted retries for task `{task}`"
                    )
                })
            }
        }
    }

    async fn chat_with_schema_on_route(
        &self,
        task: TaskKind,
        route: &TaskModelConfig,
        system_prompt: &str,
        user_prompt: &str,
        tool: ToolSpec,
    ) -> Result<StructuredCall> {
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
                    if task == TaskKind::Executor && route.provider != "openai" {
                        break;
                    }
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

    fn structured_output_fallback_route(
        &self,
        task: TaskKind,
        primary_route: &TaskModelConfig,
    ) -> Result<Option<TaskModelConfig>> {
        if task != TaskKind::Executor || primary_route.provider == "openai" {
            return Ok(None);
        }

        if read_optional_env(OPENAI_API_KEY_ENV)?.is_none() {
            return Ok(None);
        }

        let mut fallback_route =
            TaskModelConfig::new(task, "openai", DEFAULT_STRUCTURED_FALLBACK_MODEL);
        fallback_route.hint = format!("hint:{}:openai", task.as_str());
        Ok(Some(fallback_route))
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

fn supplemental_task_routes(
    primary_task_configs: &[TaskModelConfig],
) -> Result<Vec<TaskModelConfig>> {
    let mut routes = Vec::new();
    let Some(executor_route) = primary_task_configs
        .iter()
        .find(|config| config.task == TaskKind::Executor)
    else {
        return Ok(routes);
    };

    if executor_route.provider != "openai" && read_optional_env(OPENAI_API_KEY_ENV)?.is_some() {
        let mut fallback_route = TaskModelConfig::new(
            TaskKind::Executor,
            "openai",
            DEFAULT_STRUCTURED_FALLBACK_MODEL,
        );
        fallback_route.hint = "hint:executor:openai".to_owned();
        routes.push(fallback_route);
    }

    Ok(routes)
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
    if response.tool_calls.is_empty() {
        if let Some((tool_name, arguments)) = response
            .text
            .as_deref()
            .and_then(|text| parse_textual_structured_call(text, tool))
        {
            return Ok((tool_name, validate_structured_arguments(arguments, tool)?));
        }
    }

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

    let arguments = parse_json_arguments(&tool_call.arguments, tool.name.as_str())?;

    Ok((
        tool_call.name.clone(),
        validate_structured_arguments(arguments, tool)?,
    ))
}

fn parse_textual_structured_call(
    text: &str,
    tool: &ToolSpec,
) -> Option<(String, serde_json::Value)> {
    if let Some(parsed) = parse_kimi_tool_call_payload(text, tool) {
        return Some(parsed);
    }

    for candidate in extract_tool_call_tag_bodies(text)
        .into_iter()
        .chain(extract_markdown_json_blocks(text))
    {
        if let Some(parsed) = parse_textual_structured_call_candidate(candidate, tool) {
            return Some(parsed);
        }
    }

    parse_raw_json_text_candidate(text.trim(), tool)
}

fn parse_textual_structured_call_candidate(
    candidate: &str,
    tool: &ToolSpec,
) -> Option<(String, serde_json::Value)> {
    if let Ok(arguments) = parse_json_arguments(candidate, tool.name.as_str()) {
        return normalize_textual_structured_arguments(arguments, tool);
    }

    if let Some(parsed) = parse_minimax_invoke_payload(candidate, tool) {
        return Some(parsed);
    }

    if let Some(json_snippet) = extract_balanced_json_snippet(candidate) {
        if let Ok(arguments) = parse_json_arguments(json_snippet, tool.name.as_str()) {
            return normalize_textual_structured_arguments(arguments, tool);
        }
    }

    None
}

fn parse_raw_json_text_candidate(
    candidate: &str,
    tool: &ToolSpec,
) -> Option<(String, serde_json::Value)> {
    let arguments = parse_json_arguments(candidate, tool.name.as_str()).ok()?;
    normalize_textual_structured_arguments(arguments, tool)
}

fn parse_kimi_tool_call_payload(
    text: &str,
    tool: &ToolSpec,
) -> Option<(String, serde_json::Value)> {
    const TOOL_CALL_BEGIN: &str = "<|tool_call_begin|>";
    const TOOL_CALL_ARGUMENT_BEGIN: &str = "<|tool_call_argument_begin|>";
    const TOOL_CALL_ARGUMENT_END: &str = "<|tool_call_argument_end|>";
    const TOOL_CALL_END: &str = "<|tool_call_end|>";
    const TOOL_CALLS_SECTION_END: &str = "<|tool_calls_section_end|>";

    let call_begin = text.find(TOOL_CALL_BEGIN)? + TOOL_CALL_BEGIN.len();
    let argument_begin = text.find(TOOL_CALL_ARGUMENT_BEGIN)?;
    let call_header = text[call_begin..argument_begin].trim();
    let invoked_tool_name = parse_kimi_tool_name(call_header)?;
    if invoked_tool_name != tool.name {
        return None;
    }

    let argument_start = argument_begin + TOOL_CALL_ARGUMENT_BEGIN.len();
    let mut argument_end = text.len();
    for marker in [
        TOOL_CALL_ARGUMENT_END,
        TOOL_CALL_END,
        TOOL_CALLS_SECTION_END,
    ] {
        if let Some(relative_end) = text[argument_start..].find(marker) {
            argument_end = argument_end.min(argument_start + relative_end);
        }
    }

    let candidate = text[argument_start..argument_end].trim();
    if let Ok(arguments) = parse_json_arguments(candidate, tool.name.as_str()) {
        return normalize_textual_structured_arguments(arguments, tool);
    }

    extract_balanced_json_snippet(candidate).and_then(|snippet| {
        parse_json_arguments(snippet, tool.name.as_str())
            .ok()
            .and_then(|arguments| normalize_textual_structured_arguments(arguments, tool))
    })
}

fn parse_kimi_tool_name(header: &str) -> Option<&str> {
    let normalized = header.trim();
    let normalized = normalized
        .strip_prefix("functions.")
        .or_else(|| normalized.strip_prefix("function."))
        .unwrap_or(normalized);
    let tool_name_end = normalized
        .find(|ch: char| [':', ' ', '\t', '\r', '\n'].contains(&ch))
        .unwrap_or(normalized.len());
    let tool_name = normalized[..tool_name_end].trim();
    (!tool_name.is_empty()).then_some(tool_name)
}

fn normalize_textual_structured_arguments(
    arguments: serde_json::Value,
    tool: &ToolSpec,
) -> Option<(String, serde_json::Value)> {
    match arguments {
        serde_json::Value::Object(mut map) => {
            if let Some((tool_name, unwrapped_arguments)) =
                unwrap_textual_tool_envelope(&mut map, tool)
            {
                return Some((tool_name, unwrapped_arguments));
            }

            Some((tool.name.clone(), serde_json::Value::Object(map)))
        }
        other => Some((tool.name.clone(), other)),
    }
}

fn unwrap_textual_tool_envelope(
    map: &mut serde_json::Map<String, serde_json::Value>,
    tool: &ToolSpec,
) -> Option<(String, serde_json::Value)> {
    if let Some(function) = map.remove("function") {
        let serde_json::Value::Object(mut function) = function else {
            return None;
        };
        let function_name = function
            .remove("name")
            .and_then(|value| value.as_str().map(ToOwned::to_owned))
            .unwrap_or_else(|| tool.name.clone());
        if function_name != tool.name {
            return None;
        }

        let arguments = function
            .remove("arguments")
            .or_else(|| function.remove("parameters"))
            .and_then(|value| coerce_textual_argument_value(value, tool.name.as_str()))
            .or_else(|| {
                if function.is_empty() {
                    None
                } else {
                    Some(serde_json::Value::Object(function))
                }
            })?;
        return Some((function_name, arguments));
    }

    if let Some(tool_call) = map.remove("tool_call").or_else(|| map.remove("call")) {
        let serde_json::Value::Object(mut nested_call) = tool_call else {
            return None;
        };
        let tool_name = nested_call
            .remove("name")
            .or_else(|| nested_call.remove("tool_name"))
            .or_else(|| nested_call.remove("tool"))
            .and_then(|value| value.as_str().map(ToOwned::to_owned))
            .unwrap_or_else(|| tool.name.clone());
        if tool_name != tool.name {
            return None;
        }

        let arguments = nested_call
            .remove("arguments")
            .or_else(|| nested_call.remove("parameters"))
            .or_else(|| nested_call.remove("params"))
            .and_then(|value| coerce_textual_argument_value(value, tool.name.as_str()))?;
        return Some((tool_name, arguments));
    }

    let tool_name = map
        .get("name")
        .or_else(|| map.get("tool_name"))
        .or_else(|| map.get("tool"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or(tool.name.as_str());
    if tool_name != tool.name {
        return None;
    }

    map.remove("arguments")
        .or_else(|| map.remove("parameters"))
        .or_else(|| map.remove("params"))
        .and_then(|value| coerce_textual_argument_value(value, tool.name.as_str()))
        .map(|arguments| (tool.name.clone(), arguments))
}

fn coerce_textual_argument_value(
    value: serde_json::Value,
    tool_name: &str,
) -> Option<serde_json::Value> {
    match value {
        serde_json::Value::String(raw_json) => parse_json_arguments(&raw_json, tool_name).ok(),
        other => Some(other),
    }
}

fn extract_tool_call_tag_bodies(text: &str) -> Vec<&str> {
    let mut bodies = Vec::new();
    let mut search_start = 0;

    while let Some(relative_open_start) = text[search_start..].find('<') {
        let open_start = search_start + relative_open_start;
        let tag_name_start = open_start + 1;
        let Some(first_tag_char) = text[tag_name_start..].chars().next() else {
            break;
        };
        if first_tag_char == '/' {
            search_start = tag_name_start;
            continue;
        }

        let Some(relative_tag_name_end) =
            text[tag_name_start..].find(|ch: char| ['>', ' ', '\t', '\r', '\n'].contains(&ch))
        else {
            break;
        };
        let tag_name_end = tag_name_start + relative_tag_name_end;
        let tag_name = &text[tag_name_start..tag_name_end];
        if !tag_name.to_ascii_lowercase().contains("tool_call") {
            search_start = tag_name_end;
            continue;
        }

        let Some(relative_open_tag_end) = text[open_start..].find('>') else {
            break;
        };
        let open_tag_end = open_start + relative_open_tag_end;
        let body_start = open_tag_end + 1;
        let close_tag = format!("</{tag_name}>");
        let Some(relative_close_start) = text[body_start..].find(&close_tag) else {
            search_start = body_start;
            continue;
        };
        let close_start = body_start + relative_close_start;
        bodies.push(text[body_start..close_start].trim());
        search_start = close_start + close_tag.len();
    }

    bodies
}

fn extract_markdown_json_blocks(text: &str) -> Vec<&str> {
    let mut blocks = Vec::new();
    let mut search_start = 0;

    while let Some(relative_fence_start) = text[search_start..].find("```") {
        let fence_start = search_start + relative_fence_start;
        let info_start = fence_start + 3;
        let Some(relative_line_end) = text[info_start..].find('\n') else {
            break;
        };
        let info_end = info_start + relative_line_end;
        let info_string = text[info_start..info_end].trim();
        let body_start = info_end + 1;
        let Some(relative_fence_end) = text[body_start..].find("```") else {
            break;
        };
        let body_end = body_start + relative_fence_end;
        if info_string.is_empty() || info_string.eq_ignore_ascii_case("json") {
            blocks.push(text[body_start..body_end].trim());
        }
        search_start = body_end + 3;
    }

    blocks
}

fn extract_balanced_json_snippet(text: &str) -> Option<&str> {
    for (start, ch) in text.char_indices() {
        if matches!(ch, '{' | '[') {
            if let Some(end) = find_balanced_json_end(&text[start..]) {
                return Some(text[start..start + end].trim());
            }
        }
    }

    None
}

fn find_balanced_json_end(text: &str) -> Option<usize> {
    let mut stack = Vec::new();
    let mut in_string = false;
    let mut escaped = false;

    for (index, ch) in text.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            '}' | ']' => {
                if stack.pop()? != ch {
                    return None;
                }
                if stack.is_empty() {
                    return Some(index + ch.len_utf8());
                }
            }
            _ => {}
        }
    }

    None
}

fn parse_minimax_invoke_payload(
    text: &str,
    tool: &ToolSpec,
) -> Option<(String, serde_json::Value)> {
    let (invoke_tag, invoke_body, _) = extract_tag_block(text, "invoke")?;
    let invoked_tool_name = extract_tag_attribute(invoke_tag, "name")?;
    if invoked_tool_name != tool.name {
        return None;
    }

    let mut arguments = serde_json::Map::new();
    let mut remaining = invoke_body;

    while let Some((parameter_tag, parameter_body, rest)) =
        extract_tag_block(remaining, "parameter")
    {
        let parameter_name = extract_tag_attribute(parameter_tag, "name")?;
        let parameter_value = parse_json_arguments(parameter_body, tool.name.as_str()).ok()?;
        arguments.insert(parameter_name.to_owned(), parameter_value);
        remaining = rest;
    }

    if arguments.is_empty() {
        None
    } else {
        Some((tool.name.clone(), serde_json::Value::Object(arguments)))
    }
}

fn extract_tag_block<'a>(text: &'a str, tag_name: &str) -> Option<(&'a str, &'a str, &'a str)> {
    let open_tag_prefix = format!("<{tag_name}");
    let close_tag = format!("</{tag_name}>");
    let open_start = text.find(&open_tag_prefix)?;
    let open_end = open_start + text[open_start..].find('>')?;
    let body_start = open_end + 1;
    let close_start = body_start + text[body_start..].find(&close_tag)?;
    let remainder_start = close_start + close_tag.len();

    Some((
        &text[open_start..=open_end],
        text[body_start..close_start].trim(),
        &text[remainder_start..],
    ))
}

fn extract_tag_attribute<'a>(tag: &'a str, attribute_name: &str) -> Option<&'a str> {
    let attribute_prefix = format!("{attribute_name}=\"");
    let start = tag.find(&attribute_prefix)? + attribute_prefix.len();
    let end = start + tag[start..].find('"')?;
    Some(&tag[start..end])
}

fn parse_json_arguments(raw_json: &str, tool_name: &str) -> Result<serde_json::Value> {
    let (sanitized_json, sanitized) = sanitize_invalid_json_literals(raw_json);
    if sanitized {
        warn!(
            tool_name,
            "LLM response contained invalid JSON literals; replaced them with null"
        );
    }

    serde_json::from_str::<serde_json::Value>(&sanitized_json)
        .with_context(|| format!("tool call `{tool_name}` did not contain valid JSON arguments"))
}

fn validate_structured_arguments(
    arguments: serde_json::Value,
    tool: &ToolSpec,
) -> Result<serde_json::Value> {
    let arguments = coerce_arguments_to_schema(arguments, &tool.parameters, tool.name.as_str())?;
    validate_arguments_against_schema(&arguments, &tool.parameters, "$")?;

    if matches!(arguments, serde_json::Value::Object(ref map) if map.is_empty())
        && tool_requires_properties(&tool.parameters)
    {
        bail!(
            "LLM provider silently flattened the arguments into an empty object due to invalid JSON generation. You must explicitly provide the required fields."
        );
    }

    Ok(arguments)
}

fn tool_requires_properties(schema: &serde_json::Value) -> bool {
    schema
        .get("required")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|required| !required.is_empty())
}

fn coerce_arguments_to_schema(
    value: serde_json::Value,
    schema: &serde_json::Value,
    tool_name: &str,
) -> Result<serde_json::Value> {
    let value = match value {
        serde_json::Value::String(raw_json)
            if schema_expects_type(schema, "object") || schema_expects_type(schema, "array") =>
        {
            parse_json_arguments(&raw_json, tool_name)?
        }
        other => other,
    };

    match value {
        serde_json::Value::Object(map) => {
            let property_schemas = schema
                .get("properties")
                .and_then(serde_json::Value::as_object);
            let mut coerced = serde_json::Map::with_capacity(map.len());

            for (key, entry) in map {
                let entry = if let Some(property_schema) =
                    property_schemas.and_then(|properties| properties.get(&key))
                {
                    coerce_arguments_to_schema(entry, property_schema, tool_name)?
                } else {
                    entry
                };
                coerced.insert(key, entry);
            }

            Ok(serde_json::Value::Object(coerced))
        }
        serde_json::Value::Array(items) => {
            let item_schema = schema.get("items");
            let mut coerced = Vec::with_capacity(items.len());

            for item in items {
                let item = if let Some(item_schema) = item_schema {
                    coerce_arguments_to_schema(item, item_schema, tool_name)?
                } else {
                    item
                };
                coerced.push(item);
            }

            Ok(serde_json::Value::Array(coerced))
        }
        other => Ok(other),
    }
}

fn validate_arguments_against_schema(
    value: &serde_json::Value,
    schema: &serde_json::Value,
    path: &str,
) -> Result<()> {
    if schema_expects_type(schema, "object") {
        let object = value
            .as_object()
            .with_context(|| format!("structured arguments at `{path}` must be a JSON object"))?;

        if let Some(required) = schema.get("required").and_then(serde_json::Value::as_array) {
            for property in required.iter().filter_map(serde_json::Value::as_str) {
                ensure_schema(
                    object.contains_key(property),
                    format!("structured arguments missing required property `{path}.{property}`"),
                )?;
            }
        }

        if schema
            .get("additionalProperties")
            .and_then(serde_json::Value::as_bool)
            == Some(false)
        {
            let allowed = schema
                .get("properties")
                .and_then(serde_json::Value::as_object)
                .map(|properties| properties.keys().map(String::as_str).collect::<Vec<_>>())
                .unwrap_or_default();
            if let Some(unexpected) = object
                .keys()
                .find(|key| !allowed.iter().any(|allowed_key| allowed_key == key))
            {
                bail!("structured arguments contained unexpected property `{path}.{unexpected}`");
            }
        }

        if let Some(properties) = schema
            .get("properties")
            .and_then(serde_json::Value::as_object)
        {
            for (property, property_schema) in properties {
                if let Some(property_value) = object.get(property) {
                    let property_path = format!("{path}.{property}");
                    validate_arguments_against_schema(
                        property_value,
                        property_schema,
                        &property_path,
                    )?;
                }
            }
        }

        return Ok(());
    }

    if schema_expects_type(schema, "array") {
        let items = value
            .as_array()
            .with_context(|| format!("structured arguments at `{path}` must be a JSON array"))?;

        if let Some(max_items) = schema.get("maxItems").and_then(serde_json::Value::as_u64) {
            ensure_schema(
                items.len() as u64 <= max_items,
                format!(
                    "structured arguments at `{path}` exceeded the maximum item count of {max_items}"
                ),
            )?;
        }

        if let Some(min_items) = schema.get("minItems").and_then(serde_json::Value::as_u64) {
            ensure_schema(
                items.len() as u64 >= min_items,
                format!(
                    "structured arguments at `{path}` did not satisfy the minimum item count of {min_items}"
                ),
            )?;
        }

        if let Some(item_schema) = schema.get("items") {
            for (index, item) in items.iter().enumerate() {
                let item_path = format!("{path}[{index}]");
                validate_arguments_against_schema(item, item_schema, &item_path)?;
            }
        }

        return Ok(());
    }

    Ok(())
}

fn ensure_schema(condition: bool, message: String) -> Result<()> {
    if condition { Ok(()) } else { bail!(message) }
}

fn schema_expects_type(schema: &serde_json::Value, expected: &str) -> bool {
    match schema.get("type") {
        Some(serde_json::Value::String(kind)) => kind == expected,
        Some(serde_json::Value::Array(kinds)) => kinds
            .iter()
            .filter_map(serde_json::Value::as_str)
            .any(|kind| kind == expected),
        _ => false,
    }
}

fn sanitize_invalid_json_literals(raw_json: &str) -> (String, bool) {
    let mut sanitized = String::with_capacity(raw_json.len());
    let mut replaced = false;
    let mut index = 0;
    let mut in_string = false;
    let mut escaped = false;

    while index < raw_json.len() {
        let remaining = &raw_json[index..];
        let ch = remaining
            .chars()
            .next()
            .expect("JSON sanitization should stay on a valid UTF-8 boundary");

        if in_string {
            sanitized.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            index += ch.len_utf8();
            continue;
        }

        if ch == '"' {
            in_string = true;
            sanitized.push(ch);
            index += ch.len_utf8();
            continue;
        }

        if let Some((token_len, replacement)) = detect_invalid_json_literal(remaining) {
            let previous = sanitized.chars().next_back();
            let next = raw_json[index + token_len..].chars().next();
            if is_invalid_literal_token_boundary(previous, true)
                && is_invalid_literal_token_boundary(next, false)
            {
                sanitized.push_str(replacement);
                replaced = true;
                index += token_len;
                continue;
            }
        }

        sanitized.push(ch);
        index += ch.len_utf8();
    }

    (sanitized, replaced)
}

fn detect_invalid_json_literal(value: &str) -> Option<(usize, &'static str)> {
    if value.starts_with("-Infinity") {
        Some(("-Infinity".len(), "null"))
    } else if value.starts_with("Infinity") {
        Some(("Infinity".len(), "null"))
    } else if value.starts_with("NaN") {
        Some(("NaN".len(), "null"))
    } else if value.starts_with("undefined") {
        Some(("undefined".len(), "null"))
    } else {
        None
    }
}

fn is_invalid_literal_token_boundary(ch: Option<char>, is_previous: bool) -> bool {
    match ch {
        None => true,
        Some(value) if value.is_whitespace() => true,
        Some(value) => {
            if is_previous {
                matches!(value, '[' | '{' | ':' | ',')
            } else {
                matches!(value, ']' | '}' | ',' | ':')
            }
        }
    }
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
    use std::sync::{Arc, LazyLock};
    use std::time::Duration;
    use tokio::sync::Mutex;
    use zeroclaw::providers::traits::TokenUsage;
    use zeroclaw::providers::{ChatRequest, ChatResponse, ToolCall};
    use zeroclaw::tools::ToolSpec;

    static ENV_MUTEX: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    async fn lock_env() -> tokio::sync::MutexGuard<'static, ()> {
        ENV_MUTEX.lock().await
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
        responses: std::sync::Mutex<VecDeque<Result<ChatResponse>>>,
    }

    impl ScriptedBackend {
        fn new(responses: Vec<Result<ChatResponse>>) -> Self {
            Self {
                responses: std::sync::Mutex::new(VecDeque::from(responses)),
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

    fn shadow_fixture_tool() -> ToolSpec {
        ToolSpec {
            name: "generate_shadow_fixtures".to_owned(),
            description: "Return shadow execution fixtures".to_owned(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "fixtures": {
                        "type": "array",
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string" },
                                "description": { "type": "string" },
                                "args": {
                                    "type": "array",
                                    "items": {}
                                }
                            },
                            "required": ["id", "description", "args"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["fixtures"],
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

    fn response_with_text_tool_call(text: impl Into<String>) -> ChatResponse {
        ChatResponse {
            text: Some(text.into()),
            tool_calls: Vec::new(),
            usage: None,
            reasoning_content: None,
        }
    }

    fn response_with_raw_tool_call_arguments(
        tool_name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> ChatResponse {
        ChatResponse {
            text: None,
            tool_calls: vec![ToolCall {
                id: "tool_1".to_owned(),
                name: tool_name.into(),
                arguments: arguments.into(),
            }],
            usage: None,
            reasoning_content: None,
        }
    }

    #[tokio::test]
    async fn model_router_uses_default_provider_and_model_when_env_is_missing() {
        let _env_lock = lock_env().await;
        let _provider_env = EnvVarGuard::remove(TaskKind::Blueprinter.provider_env());
        let _model_env = EnvVarGuard::remove(TaskKind::Blueprinter.model_env());
        let route = ModelRouter::from_env(TaskKind::Blueprinter, "google/gemini-3-flash-preview")
            .expect("model route should resolve");

        assert_eq!(route.provider, super::DEFAULT_TASK_PROVIDER);
        assert_eq!(route.model, "google/gemini-3-flash-preview");
        assert_eq!(route.hint, "hint:blueprinter");
    }

    #[tokio::test]
    async fn model_router_uses_task_specific_provider_override() {
        let _env_lock = lock_env().await;
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

    #[tokio::test]
    async fn docker_sandbox_config_uses_defaults_when_env_is_missing() {
        let _env_lock = lock_env().await;
        let _memory_env = EnvVarGuard::remove(super::DOCKER_SANDBOX_MEMORY_ENV);
        let _cpu_env = EnvVarGuard::remove(super::DOCKER_SANDBOX_CPUS_ENV);

        let config = DockerSandboxConfig::from_env().expect("sandbox config should load");

        assert_eq!(config.memory_limit, "256m");
        assert_eq!(config.cpu_limit, "0.5");
    }

    #[tokio::test]
    async fn zeroclaw_client_requires_openrouter_api_key() {
        let _env_lock = lock_env().await;
        let _openrouter_key = EnvVarGuard::remove(super::OPENROUTER_API_KEY_ENV);

        let error = ZeroClawClient::new(build_task_configs())
            .err()
            .expect("client should fail");

        assert!(error.to_string().contains(super::OPENROUTER_API_KEY_ENV));
    }

    #[tokio::test]
    async fn zeroclaw_client_requires_openai_api_key_for_openai_routes() {
        let _env_lock = lock_env().await;
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

    #[tokio::test]
    async fn task_model_config_injects_provider_specific_api_keys() {
        let _env_lock = lock_env().await;
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
    async fn chat_with_schema_falls_back_after_executor_plain_text_response() {
        let _env_lock = lock_env().await;
        let _openai_api_key = EnvVarGuard::set(super::OPENAI_API_KEY_ENV, "test-openai-key");
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
                prompt_tokens: 7,
                completion_tokens: 3,
                total_tokens: 10,
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

    #[tokio::test]
    async fn chat_with_schema_retries_when_provider_flattens_arguments_to_empty_object() {
        let backend = Arc::new(ScriptedBackend::new(vec![
            Ok(response_with_raw_tool_call_arguments(
                "submit_verification_report",
                "{}",
            )),
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
            1,
        );

        let call = client
            .chat_with_schema(
                TaskKind::Executor,
                "Use the tool schema",
                "Verify the code",
                verification_tool(),
            )
            .await
            .expect("structured call should succeed after retrying flattened arguments");

        assert_eq!(call.tool_name, "submit_verification_report");
        assert_eq!(call.arguments["tests_passed"], true);
        assert_eq!(
            call.arguments["sandbox_execution_errors"],
            serde_json::json!([])
        );
    }

    #[tokio::test]
    async fn chat_with_schema_accepts_minimax_json_wrapped_tool_arguments() {
        let backend = Arc::new(ScriptedBackend::new(vec![Ok(
            response_with_text_tool_call(
                r#"<minimax:tool_call>{"tests_passed":true,"sandbox_execution_errors":[]}</minimax:tool_call>"#,
            ),
        )]));
        let client =
            ZeroClawClient::with_backend_for_tests(build_task_configs(), backend, Vec::new(), 0);

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
        assert_eq!(call.arguments["tests_passed"], true);
        assert_eq!(
            call.arguments["sandbox_execution_errors"],
            serde_json::json!([])
        );
    }

    #[tokio::test]
    async fn chat_with_schema_accepts_minimax_xml_parameter_payload() {
        let backend = Arc::new(ScriptedBackend::new(vec![Ok(
            response_with_text_tool_call(
                r#"<minimax:tool_call>
<invoke name="submit_verification_report">
<parameter name="tests_passed">true</parameter>
<parameter name="sandbox_execution_errors">[]</parameter>
</invoke>
</minimax:tool_call>"#,
            ),
        )]));
        let client =
            ZeroClawClient::with_backend_for_tests(build_task_configs(), backend, Vec::new(), 0);

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
        assert_eq!(call.arguments["tests_passed"], true);
        assert_eq!(
            call.arguments["sandbox_execution_errors"],
            serde_json::json!([])
        );
    }

    #[tokio::test]
    async fn chat_with_schema_accepts_markdown_fenced_json_payload() {
        let backend = Arc::new(ScriptedBackend::new(vec![Ok(
            response_with_text_tool_call(
                r#"```json
{"tests_passed":true,"sandbox_execution_errors":[]}
```"#,
            ),
        )]));
        let client =
            ZeroClawClient::with_backend_for_tests(build_task_configs(), backend, Vec::new(), 0);

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
        assert_eq!(call.arguments["tests_passed"], true);
        assert_eq!(
            call.arguments["sandbox_execution_errors"],
            serde_json::json!([])
        );
    }

    #[tokio::test]
    async fn chat_with_schema_accepts_kimi_delimited_tool_arguments() {
        let backend = Arc::new(ScriptedBackend::new(vec![Ok(
            response_with_text_tool_call(
                r#"<|tool_calls_section_begin|>
<|tool_call_begin|> functions.submit_verification_report:0
<|tool_call_argument_begin|>
{"tests_passed":true,"sandbox_execution_errors":[]}
<|tool_call_argument_end|>
<|tool_call_end|>
<|tool_calls_section_end|>"#,
            ),
        )]));
        let client =
            ZeroClawClient::with_backend_for_tests(build_task_configs(), backend, Vec::new(), 0);

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
        assert_eq!(call.arguments["tests_passed"], true);
        assert_eq!(
            call.arguments["sandbox_execution_errors"],
            serde_json::json!([])
        );
    }

    #[tokio::test]
    async fn chat_with_schema_sanitizes_non_finite_literals_in_native_tool_calls() {
        let backend = Arc::new(ScriptedBackend::new(vec![Ok(
            response_with_raw_tool_call_arguments(
                "submit_verification_report",
                r#"{"tests_passed":true,"sandbox_execution_errors":["myNaNValue",NaN,Infinity,-Infinity]}"#,
            ),
        )]));
        let client =
            ZeroClawClient::with_backend_for_tests(build_task_configs(), backend, Vec::new(), 0);

        let call = client
            .chat_with_schema(
                TaskKind::Executor,
                "Use the tool schema",
                "Verify the code",
                verification_tool(),
            )
            .await
            .expect("structured call should succeed");

        assert_eq!(call.arguments["tests_passed"], true);
        assert_eq!(
            call.arguments["sandbox_execution_errors"],
            serde_json::json!(["myNaNValue", null, null, null])
        );
    }

    #[tokio::test]
    async fn chat_with_schema_sanitizes_non_finite_literals_in_minimax_parameters() {
        let backend = Arc::new(ScriptedBackend::new(vec![Ok(
            response_with_text_tool_call(
                r#"<minimax:tool_call>
<invoke name="submit_verification_report">
<parameter name="tests_passed">true</parameter>
<parameter name="sandbox_execution_errors">["Infinity marker", NaN, Infinity, -Infinity]</parameter>
</invoke>
</minimax:tool_call>"#,
            ),
        )]));
        let client =
            ZeroClawClient::with_backend_for_tests(build_task_configs(), backend, Vec::new(), 0);

        let call = client
            .chat_with_schema(
                TaskKind::Executor,
                "Use the tool schema",
                "Verify the code",
                verification_tool(),
            )
            .await
            .expect("structured call should succeed");

        assert_eq!(call.arguments["tests_passed"], true);
        assert_eq!(
            call.arguments["sandbox_execution_errors"],
            serde_json::json!(["Infinity marker", null, null, null])
        );
    }

    #[tokio::test]
    async fn chat_with_schema_coerces_stringified_array_properties() {
        let backend = Arc::new(ScriptedBackend::new(vec![Ok(
            response_with_raw_tool_call_arguments(
                "generate_shadow_fixtures",
                r#"{"fixtures":"[{\"id\":\"happy-path\",\"description\":\"callback fixture\",\"args\":[\"1\",{\"capture\":\"callback\"}]}]"}"#,
            ),
        )]));
        let client =
            ZeroClawClient::with_backend_for_tests(build_task_configs(), backend, Vec::new(), 0);

        let call = client
            .chat_with_schema(
                TaskKind::Verifier,
                "Use the tool schema",
                "Generate fixtures",
                shadow_fixture_tool(),
            )
            .await
            .expect("structured call should succeed");

        assert_eq!(call.tool_name, "generate_shadow_fixtures");
        assert_eq!(call.arguments["fixtures"][0]["id"], "happy-path");
        assert_eq!(
            call.arguments["fixtures"][0]["args"][1],
            serde_json::json!({ "capture": "callback" })
        );
    }

    #[tokio::test]
    async fn chat_with_schema_sanitizes_undefined_inside_stringified_array_properties() {
        let backend = Arc::new(ScriptedBackend::new(vec![Ok(
            response_with_raw_tool_call_arguments(
                "generate_shadow_fixtures",
                r#"{"fixtures":"[{\"id\":\"undefined-arg\",\"description\":\"sanitized fixture\",\"args\":[undefined,{\"__fn__\":true},NaN,Infinity,-Infinity]}]"}"#,
            ),
        )]));
        let client =
            ZeroClawClient::with_backend_for_tests(build_task_configs(), backend, Vec::new(), 0);

        let call = client
            .chat_with_schema(
                TaskKind::Verifier,
                "Use the tool schema",
                "Generate fixtures",
                shadow_fixture_tool(),
            )
            .await
            .expect("structured call should succeed");

        assert_eq!(
            call.arguments["fixtures"][0]["args"],
            serde_json::json!([null, { "__fn__": true }, null, null, null])
        );
    }

    #[tokio::test]
    async fn chat_with_schema_retries_when_top_level_shape_does_not_match_schema() {
        let backend = Arc::new(ScriptedBackend::new(vec![
            Ok(response_with_raw_tool_call_arguments(
                "generate_shadow_fixtures",
                r#"[{"id":"wrong-top-level"}]"#,
            )),
            Ok(response_with_raw_tool_call_arguments(
                "generate_shadow_fixtures",
                r#"{"fixtures":[{"id":"happy-path","description":"callback fixture","args":["1",{"capture":"callback"}]}]}"#,
            )),
        ]));
        let client = ZeroClawClient::with_backend_for_tests(
            build_task_configs(),
            backend,
            vec![Duration::ZERO],
            1,
        );

        let call = client
            .chat_with_schema(
                TaskKind::Verifier,
                "Use the tool schema",
                "Generate fixtures",
                shadow_fixture_tool(),
            )
            .await
            .expect("structured call should succeed after retrying schema mismatch");

        assert_eq!(call.tool_name, "generate_shadow_fixtures");
        assert_eq!(call.arguments["fixtures"][0]["id"], "happy-path");
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
        let _env_lock = lock_env().await;
        let _openai_api_key = EnvVarGuard::remove(super::OPENAI_API_KEY_ENV);
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
