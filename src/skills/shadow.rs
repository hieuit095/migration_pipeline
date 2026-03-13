use crate::config::DockerSandboxConfig;
use crate::utils::path::{docker_bind_mount, normalize_relative_path};
use anyhow::{Context, Result, bail, ensure};
use async_trait::async_trait;
use camino::{Utf8Path, Utf8PathBuf};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::Output;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::fs;
use tracing::warn;

use super::sandbox::{
    SANDBOX_TIMEOUT_SECS, SandboxExecutionEnvironment, SandboxRuntime, build_container_name,
    preferred_output, run_docker_command,
};
use super::{SandboxSkill, Skill};

const SHADOW_MOUNT_POINT: &str = "/shadow";
const CRITICAL_PERFORMANCE_RATIO: u64 = 5;
const CRITICAL_PERFORMANCE_DELTA_MS: u64 = 50;

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ShadowFixture {
    pub id: String,
    pub description: String,
    #[serde(default)]
    pub args: Vec<Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ExecutionTarget {
    pub relative_path: String,
    pub callable: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ShadowTestRequest {
    pub legacy_root: String,
    pub modern_root: String,
    pub legacy_target: ExecutionTarget,
    pub modern_target: ExecutionTarget,
    pub fixtures: Vec<ShadowFixture>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ExecutionRecord {
    pub target: String,
    pub command: String,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub return_value: Option<Value>,
    pub duration_ms: u64,
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct FixtureDiff {
    pub fixture: ShadowFixture,
    pub legacy: ExecutionRecord,
    pub modern: ExecutionRecord,
    pub output_matches: bool,
    pub performance_regression: bool,
    pub differences: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ExecutionDiff {
    pub equivalent: bool,
    pub fixture_diffs: Vec<FixtureDiff>,
    pub sandbox_execution_errors: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ShadowTestSkill {
    sandbox_skill: SandboxSkill,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShadowRuntime {
    Node,
    NodeTypeScript,
    Python,
}

#[derive(Debug)]
struct ShadowInvocation {
    command: String,
    container_name: String,
    docker_args: Vec<String>,
}

#[derive(Debug)]
struct RenderedShadowRunner {
    runner_name: String,
    runner_contents: String,
    inner_command: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RunnerRecord {
    exit_code: i32,
    stdout: String,
    stderr: String,
    return_value: Option<Value>,
    duration_ms: u64,
    error: Option<String>,
}

#[derive(Debug)]
struct ShadowWorkspace {
    root: PathBuf,
}

impl ShadowWorkspace {
    async fn create() -> Result<Self> {
        let unique_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock moved backwards while creating shadow test workspace")?
            .as_nanos();
        let root = std::env::temp_dir().join(format!("migration_pipeline_shadow_{unique_id}"));
        fs::create_dir_all(&root)
            .await
            .with_context(|| format!("failed to create shadow workspace {}", root.display()))?;
        Ok(Self { root })
    }

    async fn write_fixture(&self, fixture: &ShadowFixture) -> Result<PathBuf> {
        let file_name = format!("fixture_{}.json", sanitize_fixture_id(&fixture.id));
        let path = self.root.join(file_name);
        let payload = serde_json::to_string(fixture)
            .with_context(|| format!("failed to serialize shadow fixture `{}`", fixture.id))?;
        fs::write(&path, payload)
            .await
            .with_context(|| format!("failed to write shadow fixture {}", path.display()))?;
        Ok(path)
    }

    async fn cleanup(self) -> Result<()> {
        match tokio::fs::remove_dir_all(&self.root).await {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error).with_context(|| {
                format!("failed to remove shadow workspace {}", self.root.display())
            }),
        }
    }
}

impl ShadowTestSkill {
    pub fn new(config: DockerSandboxConfig) -> Self {
        Self {
            sandbox_skill: SandboxSkill::new(config),
        }
    }

    async fn run_target(
        &self,
        label: &str,
        source_root: &Path,
        target: &ExecutionTarget,
        execution_environment: &SandboxExecutionEnvironment,
        fixture_path: &Path,
    ) -> Result<ExecutionRecord> {
        let invocation = build_invocation(
            &self.sandbox_skill,
            label,
            source_root,
            target,
            execution_environment,
            fixture_path,
        )
        .await?;
        let container_guard = self
            .sandbox_skill
            .arm_container_cleanup(invocation.container_name.clone());
        let output = run_docker_command(
            &invocation.docker_args,
            Duration::from_secs(SANDBOX_TIMEOUT_SECS),
            Some(container_guard),
        )
        .await
        .with_context(|| format!("failed to run {label} shadow container"))?;

        Ok(parse_execution_record(label, invocation.command, output))
    }
}

impl Default for ShadowTestSkill {
    fn default() -> Self {
        Self::new(DockerSandboxConfig::default())
    }
}

#[async_trait]
impl Skill for ShadowTestSkill {
    fn name(&self) -> &str {
        "shadow_test"
    }

    async fn execute(&self, args: Vec<String>) -> Result<String> {
        let request = parse_request(&args)?;
        validate_request(&request)?;
        let execution_result = self.execute_request(&request).await;
        match execution_result {
            Ok(raw_diff) => Ok(raw_diff),
            Err(error) => {
                warn!(error = %error, "Shadow test execution failed before a complete diff could be produced");
                serialize_execution_diff(&fatal_execution_diff(&request, error))
            }
        }
    }
}

impl ShadowTestSkill {
    async fn execute_request(&self, request: &ShadowTestRequest) -> Result<String> {
        self.sandbox_skill.ensure_docker_ready().await?;
        let legacy_root = PathBuf::from(&request.legacy_root);
        let modern_root = PathBuf::from(&request.modern_root);
        let legacy_relative = normalize_relative_path(
            &request.legacy_target.relative_path,
            "shadow execution target",
        )?;
        let modern_relative = normalize_relative_path(
            &request.modern_target.relative_path,
            "shadow execution target",
        )?;
        let legacy_runtime = ShadowRuntime::from_path(&legacy_relative)?;
        let modern_runtime = ShadowRuntime::from_path(&modern_relative)?;
        let legacy_environment = self
            .sandbox_skill
            .prepare_execution_environment(
                "shadow-legacy",
                &legacy_root,
                legacy_runtime.sandbox_runtime(),
                std::slice::from_ref(&legacy_relative),
            )
            .await?;
        let modern_environment = self
            .sandbox_skill
            .prepare_execution_environment(
                "shadow-modern",
                &modern_root,
                modern_runtime.sandbox_runtime(),
                std::slice::from_ref(&modern_relative),
            )
            .await?;

        let workspace = ShadowWorkspace::create().await?;
        let execution_result = async {
            let mut fixture_diffs = Vec::with_capacity(request.fixtures.len());
            let mut sandbox_execution_errors = Vec::new();

            for fixture in &request.fixtures {
                let fixture_path = workspace.write_fixture(fixture).await?;
                let (legacy_result, modern_result) = tokio::join!(
                    self.run_target(
                        "legacy",
                        &legacy_root,
                        &request.legacy_target,
                        &legacy_environment,
                        &fixture_path
                    ),
                    self.run_target(
                        "modern",
                        &modern_root,
                        &request.modern_target,
                        &modern_environment,
                        &fixture_path
                    )
                );

                let legacy_record = match legacy_result {
                    Ok(record) => record,
                    Err(error) => {
                        warn!(
                            fixture_id = fixture.id.as_str(),
                            error = %error,
                            "Shadow execution failed for legacy target"
                        );
                        synthetic_execution_record("legacy", error)
                    }
                };
                let modern_record = match modern_result {
                    Ok(record) => record,
                    Err(error) => {
                        warn!(
                            fixture_id = fixture.id.as_str(),
                            error = %error,
                            "Shadow execution failed for modern target"
                        );
                        synthetic_execution_record("modern", error)
                    }
                };
                let fixture_diff = compare_fixture(fixture.clone(), legacy_record, modern_record);
                sandbox_execution_errors.extend(
                    fixture_diff
                        .differences
                        .iter()
                        .filter(|difference| difference.contains("runner"))
                        .cloned(),
                );
                let should_stop = should_fail_fast(&fixture_diff);
                fixture_diffs.push(fixture_diff);
                if should_stop {
                    break;
                }
            }

            serialize_execution_diff(&ExecutionDiff {
                equivalent: fixture_diffs.iter().all(|fixture_diff| {
                    fixture_diff.output_matches && !fixture_diff.performance_regression
                }) && sandbox_execution_errors.is_empty(),
                fixture_diffs,
                sandbox_execution_errors,
            })
        }
        .await;

        if let Err(cleanup_error) = workspace.cleanup().await {
            warn!(error = %cleanup_error, "Failed to clean up shadow workspace");
        }

        execution_result
    }
}

fn parse_request(args: &[String]) -> Result<ShadowTestRequest> {
    let raw_request = args
        .first()
        .context("ShadowTestSkill expects a single JSON request argument")?;
    serde_json::from_str(raw_request).context("failed to deserialize shadow test request")
}

fn validate_request(request: &ShadowTestRequest) -> Result<()> {
    ensure!(
        !request.fixtures.is_empty(),
        "ShadowTestSkill requires at least one generated fixture"
    );
    validate_target_path(&request.legacy_root, &request.legacy_target.relative_path)?;
    validate_target_path(&request.modern_root, &request.modern_target.relative_path)?;
    ensure!(
        !request.legacy_target.callable.trim().is_empty(),
        "legacy execution target callable cannot be empty"
    );
    ensure!(
        !request.modern_target.callable.trim().is_empty(),
        "modern execution target callable cannot be empty"
    );
    Ok(())
}

fn validate_target_path(root: &str, relative_path: &str) -> Result<()> {
    let root_path = Path::new(root);
    ensure!(
        root_path.exists(),
        "shadow root does not exist: {}",
        root_path.display()
    );
    ensure!(
        root_path.is_dir(),
        "shadow root must be a directory: {}",
        root_path.display()
    );

    let relative = normalize_relative_path(relative_path, "shadow execution target")?;
    let full_path = root_path.join(relative.as_std_path());
    ensure!(
        full_path.exists() && full_path.is_file(),
        "shadow execution target does not exist: {}",
        full_path.display()
    );
    Ok(())
}

async fn build_invocation(
    sandbox_skill: &SandboxSkill,
    label: &str,
    source_root: &Path,
    target: &ExecutionTarget,
    execution_environment: &SandboxExecutionEnvironment,
    fixture_path: &Path,
) -> Result<ShadowInvocation> {
    let normalized_target =
        normalize_relative_path(&target.relative_path, "shadow execution target")?;
    let runtime = ShadowRuntime::from_path(&normalized_target)?;
    let workspace_root = fixture_path
        .parent()
        .context("fixture path must have a parent directory")?;
    let module_path = execution_environment.container_path(normalized_target.as_path());
    let fixture_mount_path = format!(
        "{SHADOW_MOUNT_POINT}/{}",
        fixture_path
            .file_name()
            .and_then(|value| value.to_str())
            .context("fixture file name was not valid unicode")?
    );

    let rendered_runner = runtime.build_command(
        label,
        source_root,
        &normalized_target,
        execution_environment,
        &module_path,
        &target.callable,
        &fixture_mount_path,
    )?;
    let runner_host_path = workspace_root.join(&rendered_runner.runner_name);
    fs::write(&runner_host_path, rendered_runner.runner_contents)
        .await
        .with_context(|| {
            format!(
                "failed to write shadow runner for {label} target {}",
                runner_host_path.display()
            )
        })?;
    let container_name = build_container_name(&format!("shadow-{label}"));

    let mut docker_args = vec![
        "run".to_owned(),
        "--rm".to_owned(),
        "--name".to_owned(),
        container_name.clone(),
        "--network=none".to_owned(),
        format!("--memory={}", sandbox_skill.config().memory_limit),
        format!("--cpus={}", sandbox_skill.config().cpu_limit),
        "--cap-drop=ALL".to_owned(),
        "--security-opt=no-new-privileges:true".to_owned(),
        "--workdir".to_owned(),
        execution_environment.workdir().to_owned(),
        "--mount".to_owned(),
        docker_bind_mount(workspace_root, SHADOW_MOUNT_POINT, true)?,
    ];
    if let Some(source_mount) = execution_environment.source_mount() {
        docker_args.push("--mount".to_owned());
        docker_args.push(source_mount.to_owned());
    }
    docker_args.push(execution_environment.image().to_owned());
    docker_args.extend(rendered_runner.inner_command);

    Ok(ShadowInvocation {
        command: format!("docker {}", docker_args.join(" ")),
        container_name,
        docker_args,
    })
}

fn parse_execution_record(label: &str, command: String, output: Output) -> ExecutionRecord {
    let raw_stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let raw_stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    match serde_json::from_str::<RunnerRecord>(&raw_stdout) {
        Ok(record) => ExecutionRecord {
            target: label.to_owned(),
            command,
            exit_code: record.exit_code,
            stdout: record.stdout,
            stderr: record.stderr,
            return_value: record.return_value,
            duration_ms: record.duration_ms,
            error: record.error,
        },
        Err(_) => ExecutionRecord {
            target: label.to_owned(),
            command,
            exit_code: output.status.code().unwrap_or(-1),
            stdout: raw_stdout.trim().to_owned(),
            stderr: raw_stderr.trim().to_owned(),
            return_value: None,
            duration_ms: 0,
            error: Some(preferred_output(&raw_stderr, &raw_stdout)),
        },
    }
}

fn compare_fixture(
    fixture: ShadowFixture,
    legacy: ExecutionRecord,
    modern: ExecutionRecord,
) -> FixtureDiff {
    let mut differences = Vec::new();

    if legacy.exit_code != modern.exit_code {
        differences.push(format!(
            "fixture `{}` exit code mismatch: legacy={} modern={}",
            fixture.id, legacy.exit_code, modern.exit_code
        ));
    }
    if legacy.return_value != modern.return_value {
        differences.push(format!(
            "fixture `{}` return value mismatch: legacy={:?} modern={:?}",
            fixture.id, legacy.return_value, modern.return_value
        ));
    }
    if normalize_output(&legacy.stdout) != normalize_output(&modern.stdout) {
        differences.push(format!(
            "fixture `{}` stdout mismatch: legacy=`{}` modern=`{}`",
            fixture.id,
            normalize_output(&legacy.stdout),
            normalize_output(&modern.stdout)
        ));
    }
    if normalize_output(&legacy.stderr) != normalize_output(&modern.stderr) {
        differences.push(format!(
            "fixture `{}` stderr mismatch: legacy=`{}` modern=`{}`",
            fixture.id,
            normalize_output(&legacy.stderr),
            normalize_output(&modern.stderr)
        ));
    }
    if let Some(error) = &legacy.error {
        differences.push(format!(
            "fixture `{}` legacy runner error: {error}",
            fixture.id
        ));
    }
    if let Some(error) = &modern.error {
        differences.push(format!(
            "fixture `{}` modern runner error: {error}",
            fixture.id
        ));
    }

    let performance_regression = is_critical_performance_regression(&legacy, &modern);
    if performance_regression {
        differences.push(format!(
            "fixture `{}` critical performance regression: legacy={}ms modern={}ms",
            fixture.id, legacy.duration_ms, modern.duration_ms
        ));
    }

    FixtureDiff {
        fixture,
        legacy,
        modern,
        output_matches: differences.is_empty(),
        performance_regression,
        differences,
    }
}

fn synthetic_execution_record(label: &str, error: anyhow::Error) -> ExecutionRecord {
    ExecutionRecord {
        target: label.to_owned(),
        exit_code: -1,
        stdout: String::new(),
        duration_ms: 0,
        error: Some(format_error_chain(&error)),
        ..Default::default()
    }
}

fn format_error_chain(error: &anyhow::Error) -> String {
    error
        .chain()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(": ")
}

fn fatal_execution_diff(request: &ShadowTestRequest, error: anyhow::Error) -> ExecutionDiff {
    let fixture = request.fixtures.first().cloned().unwrap_or_default();
    let error_message = format_error_chain(&error);
    let legacy = synthetic_execution_record("legacy", anyhow::anyhow!(error_message.clone()));
    let modern = synthetic_execution_record("modern", anyhow::anyhow!(error_message.clone()));
    let differences = vec![format!(
        "shadow execution infrastructure failed before fixture execution: {error_message}"
    )];

    ExecutionDiff {
        equivalent: false,
        fixture_diffs: vec![FixtureDiff {
            fixture,
            legacy,
            modern,
            output_matches: false,
            performance_regression: false,
            differences: differences.clone(),
        }],
        sandbox_execution_errors: differences,
    }
}

fn serialize_execution_diff(diff: &ExecutionDiff) -> Result<String> {
    serde_json::to_string(diff).context("failed to serialize shadow execution diff")
}

fn should_fail_fast(fixture_diff: &FixtureDiff) -> bool {
    !fixture_diff.output_matches || fixture_diff.performance_regression
}

fn is_critical_performance_regression(legacy: &ExecutionRecord, modern: &ExecutionRecord) -> bool {
    if legacy.duration_ms == 0 {
        return modern.duration_ms > CRITICAL_PERFORMANCE_DELTA_MS;
    }

    modern.duration_ms
        > legacy
            .duration_ms
            .saturating_mul(CRITICAL_PERFORMANCE_RATIO)
        && modern.duration_ms.saturating_sub(legacy.duration_ms) > CRITICAL_PERFORMANCE_DELTA_MS
}

fn normalize_output(value: &str) -> String {
    value.trim().replace("\r\n", "\n")
}

fn sanitize_fixture_id(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>();
    sanitized.trim_matches('-').to_owned()
}

fn sanitize_runner_label(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>();
    let trimmed = sanitized.trim_matches('-');
    if trimmed.is_empty() {
        "shadow".to_owned()
    } else {
        trimmed.to_owned()
    }
}

impl ShadowRuntime {
    fn from_path(relative_path: &Utf8PathBuf) -> Result<Self> {
        match relative_path
            .extension()
            .map(|value| value.to_ascii_lowercase())
            .as_deref()
        {
            Some("js") | Some("mjs") | Some("cjs") => Ok(Self::Node),
            Some("ts") | Some("tsx") => Ok(Self::NodeTypeScript),
            Some("py") => Ok(Self::Python),
            Some(other) => bail!(
                "ShadowTestSkill does not support the `.{other}` extension for shadow execution yet"
            ),
            None => bail!("ShadowTestSkill requires a target file with a supported extension"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_command(
        self,
        label: &str,
        _source_root: &Path,
        _relative_path: &Utf8Path,
        _execution_environment: &SandboxExecutionEnvironment,
        module_path: &str,
        callable: &str,
        fixture_path: &str,
    ) -> Result<RenderedShadowRunner> {
        let runner_label = sanitize_runner_label(label);
        let runner = match self {
            Self::Node => RenderedShadowRunner {
                runner_name: format!("{runner_label}_shadow_runner.mjs"),
                runner_contents: NODE_SHADOW_RUNNER.to_owned(),
                inner_command: vec![
                    "node".to_owned(),
                    format!("{SHADOW_MOUNT_POINT}/{runner_label}_shadow_runner.mjs"),
                    module_path.to_owned(),
                    callable.to_owned(),
                    fixture_path.to_owned(),
                ],
            },
            Self::NodeTypeScript => RenderedShadowRunner {
                runner_name: format!("{runner_label}_shadow_runner.mjs"),
                runner_contents: NODE_SHADOW_RUNNER.to_owned(),
                inner_command: vec![
                    "node".to_owned(),
                    "--import=tsx".to_owned(),
                    format!("{SHADOW_MOUNT_POINT}/{runner_label}_shadow_runner.mjs"),
                    module_path.to_owned(),
                    callable.to_owned(),
                    fixture_path.to_owned(),
                ],
            },
            Self::Python => RenderedShadowRunner {
                runner_name: format!("{runner_label}_shadow_runner.py"),
                runner_contents: PYTHON_SHADOW_RUNNER.to_owned(),
                inner_command: vec![
                    "python".to_owned(),
                    format!("{SHADOW_MOUNT_POINT}/{runner_label}_shadow_runner.py"),
                    module_path.to_owned(),
                    callable.to_owned(),
                    fixture_path.to_owned(),
                ],
            },
        };

        Ok(runner)
    }

    fn sandbox_runtime(self) -> SandboxRuntime {
        match self {
            Self::Node => SandboxRuntime::Node,
            Self::NodeTypeScript => SandboxRuntime::NodeTypeScript,
            Self::Python => SandboxRuntime::Python,
        }
    }
}

const NODE_SHADOW_RUNNER: &str = r#"
import fs from 'node:fs/promises';
import Module from 'node:module';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import vm from 'node:vm';

const CALLBACK_WAIT_MS = 250;
const CALLBACK_TIMEOUT = Symbol('callback-timeout');

const stringify = (value) => {
  if (typeof value === 'string') return value;
  try { return JSON.stringify(value); } catch { return String(value); }
};

const toSerializable = (value) => {
  if (value instanceof Error) {
    return {
      name: value.name,
      message: value.message,
      stack: value.stack ?? null,
    };
  }
  if (typeof value === 'bigint') return value.toString();
  if (typeof value === 'function') return `[Function ${value.name || 'anonymous'}]`;
  if (Array.isArray(value)) return value.map((item) => toSerializable(item));
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([key, nested]) => [key, toSerializable(nested)])
    );
  }
  return value ?? null;
};

const isCallbackPlaceholder = (value) =>
  value === null ||
  (value &&
    typeof value === 'object' &&
    (value.capture === 'callback' || value.__fn__ === true));

const serializeCallbackArgs = (callbackArgs) => {
  if (callbackArgs.length === 0) return null;
  if (callbackArgs.length === 1) return toSerializable(callbackArgs[0]);
  return callbackArgs.map((argument) => toSerializable(argument));
};

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const resolveCallable = (moduleNamespace, callableName) => {
  if (callableName === 'default' && typeof moduleNamespace.default === 'function') {
    return moduleNamespace.default;
  }

  const candidates = [
    moduleNamespace[callableName],
    moduleNamespace.default?.[callableName],
  ];

  return candidates.find((candidate) => typeof candidate === 'function') ?? null;
};

const SAFE_IDENTIFIER = /^[A-Za-z_$][A-Za-z0-9_$]*$/;

const resolveCommonJsCallable = async (modulePath, callableName) => {
  if (!SAFE_IDENTIFIER.test(callableName)) {
    return null;
  }

  const source = await fs.readFile(modulePath, 'utf8');
  const wrappedSource = `
(function (exports, require, module, __filename, __dirname) {
${source}
const __shadowCandidates = [
  typeof ${callableName} === 'function' ? ${callableName} : null,
  module.exports?.[${JSON.stringify(callableName)}],
  module.exports?.default?.[${JSON.stringify(callableName)}],
  ${JSON.stringify(callableName)} === 'default' && typeof module.exports === 'function'
    ? module.exports
    : null,
  ${JSON.stringify(callableName)} === 'default' && typeof module.exports?.default === 'function'
    ? module.exports.default
    : null
];
return __shadowCandidates.find((candidate) => typeof candidate === 'function') ?? null;
})
`;
  const compiled = vm.runInThisContext(wrappedSource, { filename: modulePath });
  const module = { exports: {} };
  const require = Module.createRequire(`file://${modulePath}`);
  return compiled(module.exports, require, module, modulePath, path.dirname(modulePath));
};

const loadCallable = async (modulePath, callableName) => {
  const extension = path.extname(modulePath).toLowerCase();
  if (extension === '.js' || extension === '.cjs') {
    try {
      return await resolveCommonJsCallable(modulePath, callableName);
    } catch (commonJsError) {
      const errorText = String(commonJsError?.message ?? commonJsError);
      if (!/Unexpected token 'export'|Cannot use import statement outside a module/.test(errorText)) {
        throw commonJsError;
      }
    }
  }

  const module = await import(`file://${modulePath}`);
  return resolveCallable(module, callableName);
};

const captureConsole = () => {
  const stdout = [];
  const stderr = [];
  const original = { log: console.log, warn: console.warn, error: console.error };
  console.log = (...args) => stdout.push(args.map(stringify).join(' '));
  console.warn = (...args) => stderr.push(args.map(stringify).join(' '));
  console.error = (...args) => stderr.push(args.map(stringify).join(' '));
  return { stdout, stderr, restore: () => { console.log = original.log; console.warn = original.warn; console.error = original.error; } };
};

const [, , modulePath, callableName, fixturePath] = process.argv;
const fixture = JSON.parse(await fs.readFile(fixturePath, 'utf8'));
const captured = captureConsole();
let startedAt = performance.now();
let exitCode = 0;
let returnValue = null;
let error = null;

try {
  const callable = await loadCallable(modulePath, callableName);
  if (typeof callable !== 'function') {
    throw new Error(`callable ${callableName} was not found in ${modulePath}`);
  }

  captured.stdout.length = 0;
  captured.stderr.length = 0;
  startedAt = performance.now();

  const args = Array.isArray(fixture.args) ? [...fixture.args] : [];
  let callbackPromise = null;
  if (args.length > 0 && isCallbackPlaceholder(args.at(-1))) {
    callbackPromise = new Promise((resolve) => {
      args[args.length - 1] = (...callbackArgs) => {
        resolve(serializeCallbackArgs(callbackArgs));
      };
    });
  }

  const directResult = await callable(...args);
  if (callbackPromise) {
    const callbackResult = await Promise.race([
      callbackPromise,
      wait(CALLBACK_WAIT_MS).then(() => CALLBACK_TIMEOUT),
    ]);
    returnValue =
      callbackResult === CALLBACK_TIMEOUT
        ? toSerializable(directResult)
        : callbackResult;
  } else {
    returnValue = toSerializable(directResult);
  }
} catch (failure) {
  exitCode = 1;
  error = String(failure?.stack ?? failure);
  captured.stderr.push(error);
}

await new Promise((resolve, reject) => {
  process.stdout.write(JSON.stringify({
  exit_code: exitCode,
  stdout: captured.stdout.join('\n'),
  stderr: captured.stderr.join('\n'),
  return_value: returnValue,
  duration_ms: Math.max(0, Math.round(performance.now() - startedAt)),
  error,
  }), (writeError) => writeError ? reject(writeError) : resolve());
});
process.exit(exitCode);
"#;

const PYTHON_SHADOW_RUNNER: &str = r#"
import contextlib
import importlib.util
import io
import json
import sys
import time

module_path, callable_name, fixture_path = sys.argv[1:4]
with open(fixture_path, "r", encoding="utf-8") as handle:
    fixture = json.load(handle)

stdout_buffer = io.StringIO()
stderr_buffer = io.StringIO()
started_at = time.perf_counter()
exit_code = 0
return_value = None
error = None

try:
    spec = importlib.util.spec_from_file_location("shadow_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    callable_obj = getattr(module, callable_name if callable_name != "default" else "__call__", None)
    if callable_name == "default" and callable_obj is None:
        callable_obj = getattr(module, "main", None)
    if not callable(callable_obj):
        raise RuntimeError(f"callable {callable_name} was not found in {module_path}")
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        return_value = callable_obj(*(fixture.get("args") or []))
except Exception as failure:
    exit_code = 1
    error = str(failure)
    stderr_buffer.write(error)

payload = {
    "exit_code": exit_code,
    "stdout": stdout_buffer.getvalue(),
    "stderr": stderr_buffer.getvalue(),
    "return_value": return_value,
    "duration_ms": max(0, round((time.perf_counter() - started_at) * 1000)),
    "error": error,
}
sys.stdout.write(json.dumps(payload))
sys.exit(exit_code)
"#;

#[cfg(test)]
mod tests {
    use super::{
        ExecutionRecord, ExecutionTarget, FixtureDiff, ShadowFixture, ShadowRuntime,
        ShadowTestRequest, compare_fixture, is_critical_performance_regression, should_fail_fast,
        synthetic_execution_record,
    };
    use anyhow::anyhow;
    use serde_json::json;

    #[test]
    fn compare_fixture_detects_return_value_mismatch() {
        let fixture = ShadowFixture {
            id: "fx-1".to_owned(),
            description: "basic case".to_owned(),
            args: vec![json!(1)],
        };
        let legacy = ExecutionRecord {
            target: "legacy".to_owned(),
            return_value: Some(json!(1)),
            ..Default::default()
        };
        let modern = ExecutionRecord {
            target: "modern".to_owned(),
            return_value: Some(json!(2)),
            ..Default::default()
        };

        let diff = compare_fixture(fixture, legacy, modern);
        assert!(!diff.output_matches);
        assert!(
            diff.differences
                .iter()
                .any(|difference| difference.contains("return value mismatch"))
        );
    }

    #[test]
    fn performance_regression_is_flagged_for_large_slowdowns() {
        let legacy = ExecutionRecord {
            duration_ms: 20,
            ..Default::default()
        };
        let modern = ExecutionRecord {
            duration_ms: 200,
            ..Default::default()
        };

        assert!(is_critical_performance_regression(&legacy, &modern));
    }

    #[test]
    fn shadow_request_round_trips_through_json() {
        let request = ShadowTestRequest {
            legacy_root: "./legacy_app".to_owned(),
            modern_root: "./modern_app".to_owned(),
            legacy_target: ExecutionTarget {
                relative_path: "src/server.js".to_owned(),
                callable: "bootstrap".to_owned(),
            },
            modern_target: ExecutionTarget {
                relative_path: "src/server.ts".to_owned(),
                callable: "bootstrap".to_owned(),
            },
            fixtures: vec![ShadowFixture {
                id: "fx-1".to_owned(),
                description: "basic".to_owned(),
                args: vec![json!(1), json!(2)],
            }],
        };

        let payload = serde_json::to_string(&request).expect("request should serialize");
        let restored: ShadowTestRequest =
            serde_json::from_str(&payload).expect("request should deserialize");

        assert_eq!(restored.fixtures.len(), 1);
        assert_eq!(restored.modern_target.callable, "bootstrap");
    }

    #[test]
    fn fail_fast_triggers_for_fixture_mismatch() {
        let fixture = ShadowFixture {
            id: "fx-1".to_owned(),
            description: "basic".to_owned(),
            args: Vec::new(),
        };
        let fixture_diff = FixtureDiff {
            fixture,
            output_matches: false,
            performance_regression: false,
            differences: vec!["mismatch".to_owned()],
            ..Default::default()
        };

        assert!(should_fail_fast(&fixture_diff));
    }

    #[test]
    fn fail_fast_does_not_trigger_for_passing_fixture() {
        let fixture = ShadowFixture {
            id: "fx-2".to_owned(),
            description: "basic".to_owned(),
            args: Vec::new(),
        };
        let fixture_diff = FixtureDiff {
            fixture,
            output_matches: true,
            performance_regression: false,
            differences: Vec::new(),
            ..Default::default()
        };

        assert!(!should_fail_fast(&fixture_diff));
    }

    #[test]
    fn synthetic_execution_record_surfaces_runner_failure_in_fixture_diff() {
        let fixture = ShadowFixture {
            id: "fx-runner".to_owned(),
            description: "runner failure".to_owned(),
            args: Vec::new(),
        };
        let legacy = synthetic_execution_record("legacy", anyhow!("docker timeout"));
        let modern = ExecutionRecord {
            target: "modern".to_owned(),
            exit_code: 0,
            ..Default::default()
        };

        let diff = compare_fixture(fixture, legacy, modern);

        assert!(!diff.output_matches);
        assert!(
            diff.differences
                .iter()
                .any(|difference| difference.contains("legacy runner error: docker timeout"))
        );
    }

    #[test]
    fn build_command_uses_label_specific_runner_name() {
        let source_root = std::path::Path::new("/tmp/legacy");
        let relative_path = camino::Utf8PathBuf::from("src/server.js");
        let execution_environment = crate::skills::sandbox::SandboxExecutionEnvironment::mounted(
            "node:alpine".to_owned(),
            "/app".to_owned(),
            "dummy_mount".to_owned(),
        );

        let legacy_runner = ShadowRuntime::Node.build_command(
            "legacy",
            source_root,
            &relative_path,
            &execution_environment,
            "/app/src/server.js",
            "bootstrap",
            "/shadow/fixture.json",
        ).unwrap();
        let modern_runner = ShadowRuntime::Node.build_command(
            "modern",
            source_root,
            &relative_path,
            &execution_environment,
            "/app/src/server.js",
            "bootstrap",
            "/shadow/fixture.json",
        ).unwrap();

        assert_eq!(legacy_runner.runner_name, "legacy_shadow_runner.mjs");
        assert_eq!(modern_runner.runner_name, "modern_shadow_runner.mjs");
        assert!(
            legacy_runner.inner_command
                .iter()
                .any(|argument| argument == "/shadow/legacy_shadow_runner.mjs")
        );
        assert!(
            modern_runner.inner_command
                .iter()
                .any(|argument| argument == "/shadow/modern_shadow_runner.mjs")
        );
    }
}
