use crate::config::DockerSandboxConfig;
use crate::utils::path::{container_path, docker_bind_mount, normalize_relative_path};
use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use camino::{Utf8Path, Utf8PathBuf};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::{Output, Stdio};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::process::Command as TokioCommand;
use tokio::runtime::Builder as TokioRuntimeBuilder;
use tokio::time::timeout;
use tracing::{debug, warn};

use super::{Skill, TerminalCommandOutput};

pub(crate) const APP_MOUNT_POINT: &str = "/app";
pub(crate) const APP_SOURCE_MOUNT_POINT: &str = "/app-source";
pub(crate) const PREPARED_APP_MOUNT_POINT: &str = "/workspace";
const DOCKER_INFO_TIMEOUT_SECS: u64 = 5;
const SANDBOX_WARMUP_TIMEOUT_SECS: u64 = 180;
pub(crate) const SANDBOX_TIMEOUT_SECS: u64 = 20;

#[derive(Debug, Clone)]
pub struct SandboxSkill {
    config: DockerSandboxConfig,
    reaper: Arc<DockerCleanupReaper>,
}

#[derive(Debug)]
struct SandboxInvocation {
    container_name: String,
    docker_args: Vec<String>,
    display_command: String,
    _execution_environment: SandboxExecutionEnvironment,
}

#[derive(Debug)]
pub(crate) struct SandboxExecutionEnvironment {
    image: String,
    workdir: String,
    source_mount: Option<String>,
    _image_guard: Option<ImageGuard>,
}

#[derive(Debug)]
enum DockerCleanupTarget {
    Container(String),
    Image(String),
}

#[derive(Debug)]
pub(crate) struct ContainerGuard {
    container_name: Option<String>,
    reaper: Arc<DockerCleanupReaper>,
}

#[derive(Debug)]
struct ImageGuard {
    image_name: Option<String>,
    reaper: Arc<DockerCleanupReaper>,
}

#[derive(Debug, Clone)]
struct DockerCleanupReaper {
    sender: Option<mpsc::Sender<DockerCleanupTarget>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SandboxRuntime {
    Node,
    Deno,
    Python,
    Go,
}

impl SandboxSkill {
    pub fn new(config: DockerSandboxConfig) -> Self {
        Self {
            config,
            reaper: Arc::new(DockerCleanupReaper::spawn()),
        }
    }

    pub(crate) fn config(&self) -> &DockerSandboxConfig {
        &self.config
    }

    pub(crate) fn arm_container_cleanup(
        &self,
        container_name: impl Into<String>,
    ) -> ContainerGuard {
        ContainerGuard::new(Arc::clone(&self.reaper), container_name.into())
    }

    fn arm_image_cleanup(&self, image_name: impl Into<String>) -> ImageGuard {
        ImageGuard::new(Arc::clone(&self.reaper), image_name.into())
    }

    pub(crate) async fn ensure_docker_ready(&self) -> Result<()> {
        let args = vec![
            "info".to_owned(),
            "--format".to_owned(),
            "{{json .ServerVersion}}".to_owned(),
        ];
        let output = run_docker_command(&args, Duration::from_secs(DOCKER_INFO_TIMEOUT_SECS), None)
            .await
            .context("docker preflight check failed")?;

        if output.status.success() {
            return Ok(());
        }

        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let detail = preferred_output(stderr.as_ref(), stdout.as_ref());
        Err(anyhow!(
            "Docker is installed but unavailable. Ensure the Docker daemon is running: {}",
            detail
        ))
    }

    pub(crate) async fn prepare_execution_environment(
        &self,
        prefix: &str,
        source_root: &Path,
        runtime: SandboxRuntime,
        execution_paths: &[Utf8PathBuf],
    ) -> Result<SandboxExecutionEnvironment> {
        let direct_mount = docker_bind_mount(source_root, APP_MOUNT_POINT, true)?;
        let Some(install_command) =
            runtime.dependency_install_command(source_root, execution_paths)?
        else {
            return Ok(SandboxExecutionEnvironment::mounted(
                runtime.base_image().to_owned(),
                APP_MOUNT_POINT.to_owned(),
                direct_mount,
            ));
        };

        let warmup_container_name = build_container_name(&format!("{prefix}-warmup"));
        let warmup_image_name = build_container_name(&format!("{prefix}-image"));
        let mut warmup_container_guard = self.arm_container_cleanup(warmup_container_name.clone());
        let warmup_args = build_warmup_docker_args(
            source_root,
            &warmup_container_name,
            &self.config,
            runtime.base_image(),
            &install_command,
        )?;
        debug!(
            command = format_command("docker", &warmup_args),
            source_root = %source_root.display(),
            runtime = ?runtime,
            "Launching dependency warm-up container"
        );

        let (warmup_output, returned_guard) = run_docker_command_with_guard(
            &warmup_args,
            Duration::from_secs(SANDBOX_WARMUP_TIMEOUT_SECS),
            Some(warmup_container_guard),
            false,
        )
        .await
        .context("failed to execute sandbox dependency warm-up container")?;
        warmup_container_guard =
            returned_guard.context("warm-up execution unexpectedly dropped its cleanup guard")?;
        ensure!(
            warmup_output.status.success(),
            "sandbox dependency warm-up failed: {}",
            preferred_output(
                String::from_utf8_lossy(&warmup_output.stderr).as_ref(),
                String::from_utf8_lossy(&warmup_output.stdout).as_ref()
            )
        );

        let commit_args = vec![
            "commit".to_owned(),
            warmup_container_name.clone(),
            warmup_image_name.clone(),
        ];
        let commit_output = run_docker_command(
            &commit_args,
            Duration::from_secs(SANDBOX_WARMUP_TIMEOUT_SECS),
            None,
        )
        .await
        .context("failed to commit warmed sandbox image")?;
        ensure!(
            commit_output.status.success(),
            "sandbox image commit failed: {}",
            preferred_output(
                String::from_utf8_lossy(&commit_output.stderr).as_ref(),
                String::from_utf8_lossy(&commit_output.stdout).as_ref()
            )
        );

        drop(warmup_container_guard);

        Ok(SandboxExecutionEnvironment::prepared(
            warmup_image_name.clone(),
            self.arm_image_cleanup(warmup_image_name),
        ))
    }

    async fn build_invocation(
        &self,
        modern_root: &Path,
        relative_paths: &[String],
    ) -> Result<SandboxInvocation> {
        ensure!(
            !relative_paths.is_empty(),
            "SandboxSkill requires at least one generated test file"
        );

        let normalized_paths = validate_sandbox_targets(modern_root, relative_paths)?;
        let runtime = SandboxRuntime::from_paths(&normalized_paths)?;
        let execution_environment = self
            .prepare_execution_environment("sandbox", modern_root, runtime, &normalized_paths)
            .await?;
        let container_paths = normalized_paths
            .iter()
            .map(|path| execution_environment.container_path(path.as_path()))
            .collect::<Vec<_>>();
        let inner_command = runtime.build_command(&container_paths);
        let container_name = build_container_name("sandbox");

        let mut docker_args = vec![
            "run".to_owned(),
            "--rm".to_owned(),
            "--name".to_owned(),
            container_name.clone(),
            "--network=none".to_owned(),
            format!("--memory={}", self.config.memory_limit),
            format!("--cpus={}", self.config.cpu_limit),
            "--cap-drop=ALL".to_owned(),
            "--security-opt=no-new-privileges:true".to_owned(),
            "--workdir".to_owned(),
            execution_environment.workdir.clone(),
        ];
        if let Some(source_mount) = execution_environment.source_mount.as_ref() {
            docker_args.push("--mount".to_owned());
            docker_args.push(source_mount.clone());
        }
        docker_args.push(execution_environment.image.clone());
        docker_args.extend(inner_command);

        Ok(SandboxInvocation {
            container_name,
            display_command: format_command("docker", &docker_args),
            docker_args,
            _execution_environment: execution_environment,
        })
    }
}

impl SandboxExecutionEnvironment {
    fn mounted(image: String, workdir: String, source_mount: String) -> Self {
        Self {
            image,
            workdir,
            source_mount: Some(source_mount),
            _image_guard: None,
        }
    }

    fn prepared(image: String, image_guard: ImageGuard) -> Self {
        Self {
            image,
            workdir: PREPARED_APP_MOUNT_POINT.to_owned(),
            source_mount: None,
            _image_guard: Some(image_guard),
        }
    }

    pub(crate) fn image(&self) -> &str {
        &self.image
    }

    pub(crate) fn workdir(&self) -> &str {
        &self.workdir
    }

    pub(crate) fn source_mount(&self) -> Option<&str> {
        self.source_mount.as_deref()
    }

    pub(crate) fn container_path(&self, relative_path: &Utf8Path) -> String {
        container_path(self.workdir(), relative_path)
    }
}

impl ContainerGuard {
    fn new(reaper: Arc<DockerCleanupReaper>, container_name: String) -> Self {
        Self {
            container_name: Some(container_name),
            reaper,
        }
    }

    fn disarm(&mut self) {
        self.container_name = None;
    }
}

impl Drop for ContainerGuard {
    fn drop(&mut self) {
        if let Some(container_name) = self.container_name.take() {
            self.reaper
                .schedule_cleanup(DockerCleanupTarget::Container(container_name));
        }
    }
}

impl ImageGuard {
    fn new(reaper: Arc<DockerCleanupReaper>, image_name: String) -> Self {
        Self {
            image_name: Some(image_name),
            reaper,
        }
    }
}

impl Drop for ImageGuard {
    fn drop(&mut self) {
        if let Some(image_name) = self.image_name.take() {
            self.reaper
                .schedule_cleanup(DockerCleanupTarget::Image(image_name));
        }
    }
}

impl DockerCleanupReaper {
    fn spawn() -> Self {
        let (sender, receiver) = mpsc::channel::<DockerCleanupTarget>();
        let thread_builder = thread::Builder::new().name("docker-cleanup-reaper".to_owned());

        let sender = match thread_builder.spawn(move || run_cleanup_worker(receiver)) {
            Ok(_handle) => Some(sender),
            Err(error) => {
                warn!(
                    error = %error,
                    "Failed to spawn Docker cleanup reaper thread; Docker resources may require manual cleanup"
                );
                None
            }
        };

        Self { sender }
    }

    fn schedule_cleanup(&self, cleanup_target: DockerCleanupTarget) {
        if self
            .sender
            .as_ref()
            .is_none_or(|sender| sender.send(cleanup_target).is_err())
        {
            warn!("Docker cleanup reaper is unavailable; resource cleanup was skipped");
        }
    }
}

fn run_cleanup_worker(receiver: mpsc::Receiver<DockerCleanupTarget>) {
    let runtime = match TokioRuntimeBuilder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            warn!(
                error = %error,
                "Failed to initialize Docker cleanup runtime; resource cleanup was skipped"
            );
            return;
        }
    };

    for cleanup_target in receiver {
        if let Err(error) = runtime.block_on(run_cleanup_command(&cleanup_target)) {
            warn!(
                resource = cleanup_target.resource_name(),
                error = %error,
                "Docker cleanup command failed"
            );
        }
    }
}

async fn run_cleanup_command(cleanup_target: &DockerCleanupTarget) -> Result<()> {
    let mut command = TokioCommand::new("docker");
    command
        .args(cleanup_target.args())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    let status = command.status().await.with_context(|| {
        format!(
            "failed to spawn docker cleanup for {}",
            cleanup_target.resource_name()
        )
    })?;
    if !status.success() {
        warn!(
            resource = cleanup_target.resource_name(),
            exit_code = status.code().unwrap_or(-1),
            "Docker cleanup returned a non-success status"
        );
    }

    Ok(())
}

impl DockerCleanupTarget {
    fn args(&self) -> Vec<&str> {
        match self {
            Self::Container(container_name) => vec!["rm", "-f", container_name.as_str()],
            Self::Image(image_name) => vec!["rmi", "-f", image_name.as_str()],
        }
    }

    fn resource_name(&self) -> &str {
        match self {
            Self::Container(container_name) => container_name,
            Self::Image(image_name) => image_name,
        }
    }
}

impl Default for SandboxSkill {
    fn default() -> Self {
        Self::new(DockerSandboxConfig::default())
    }
}

impl SandboxRuntime {
    fn from_paths(relative_paths: &[Utf8PathBuf]) -> Result<Self> {
        let mut runtime = None;

        for relative_path in relative_paths {
            let path_runtime = Self::from_path(relative_path)?;
            if let Some(existing) = runtime {
                ensure!(
                    existing == path_runtime,
                    "SandboxSkill cannot run mixed-language test suites in one container: `{}` does not match the existing runtime",
                    relative_path
                );
            } else {
                runtime = Some(path_runtime);
            }
        }

        runtime.context("SandboxSkill could not infer a runtime for the provided test files")
    }

    pub(crate) fn from_path(relative_path: &Utf8PathBuf) -> Result<Self> {
        match relative_path
            .extension()
            .map(|value| value.to_ascii_lowercase())
            .as_deref()
        {
            Some("js") | Some("mjs") | Some("cjs") => Ok(Self::Node),
            Some("ts") | Some("tsx") => Ok(Self::Deno),
            Some("py") => Ok(Self::Python),
            Some("go") => Ok(Self::Go),
            Some(other) => Err(anyhow!(
                "SandboxSkill does not support isolated test execution for the `.{other}` extension yet"
            )),
            None => Err(anyhow!(
                "SandboxSkill requires test files with a supported extension"
            )),
        }
    }

    fn base_image(self) -> &'static str {
        match self {
            Self::Node => "node:alpine",
            Self::Deno => "denoland/deno:alpine",
            Self::Python => "python:alpine",
            Self::Go => "golang:alpine",
        }
    }

    fn build_command(self, container_paths: &[String]) -> Vec<String> {
        match self {
            Self::Node => {
                let mut command = vec!["node".to_owned(), "--test".to_owned()];
                command.extend(container_paths.iter().cloned());
                command
            }
            Self::Deno => {
                let mut command = vec!["deno".to_owned(), "test".to_owned()];
                command.extend(container_paths.iter().cloned());
                command
            }
            Self::Python => {
                let mut command = vec![
                    "python".to_owned(),
                    "-c".to_owned(),
                    PYTHON_UNITTEST_RUNNER.to_owned(),
                ];
                command.extend(container_paths.iter().cloned());
                command
            }
            Self::Go => vec![
                "sh".to_owned(),
                "-lc".to_owned(),
                format!(
                    "cd {} && go test ./...",
                    shell_escape(PREPARED_APP_MOUNT_POINT)
                ),
            ],
        }
    }

    fn dependency_install_command(
        self,
        source_root: &Path,
        execution_paths: &[Utf8PathBuf],
    ) -> Result<Option<String>> {
        match self {
            Self::Node => {
                if source_root.join("package.json").is_file() {
                    Ok(Some(
                        "npm install --ignore-scripts --no-audit --no-fund".to_owned(),
                    ))
                } else {
                    Ok(None)
                }
            }
            Self::Deno => Ok(Some(build_deno_cache_command(execution_paths))),
            Self::Python => {
                if source_root.join("requirements.txt").is_file() {
                    Ok(Some(
                        "pip install --no-cache-dir -r requirements.txt".to_owned(),
                    ))
                } else if source_root.join("pyproject.toml").is_file()
                    || source_root.join("setup.py").is_file()
                {
                    Ok(Some("pip install --no-cache-dir .".to_owned()))
                } else {
                    Ok(None)
                }
            }
            Self::Go => {
                if source_root.join("go.mod").is_file() {
                    Ok(Some("go mod download".to_owned()))
                } else {
                    Ok(None)
                }
            }
        }
    }
}

#[async_trait]
impl Skill for SandboxSkill {
    fn name(&self) -> &str {
        "sandbox"
    }

    async fn execute(&self, args: Vec<String>) -> Result<String> {
        let (modern_root, test_paths) = parse_sandbox_args(&args)?;
        self.ensure_docker_ready().await?;

        let invocation = self.build_invocation(&modern_root, &test_paths).await?;
        let container_guard = self.arm_container_cleanup(invocation.container_name.clone());
        debug!(
            command = invocation.display_command.as_str(),
            test_paths = ?test_paths,
            "Launching sandboxed verification container"
        );

        let output = run_docker_command(
            &invocation.docker_args,
            Duration::from_secs(SANDBOX_TIMEOUT_SECS),
            Some(container_guard),
        )
        .await?;
        let result = TerminalCommandOutput {
            command: invocation.display_command,
            exit_code: output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        };

        serde_json::to_string(&result).context("failed to serialize sandbox command output")
    }
}

const PYTHON_UNITTEST_RUNNER: &str = r#"
import importlib.util
import pathlib
import sys
import unittest

loader = unittest.defaultTestLoader
suite = unittest.TestSuite()
for file_path in sys.argv[1:]:
    module_name = pathlib.Path(file_path).stem.replace('.', '_')
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load test module: {file_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite.addTests(loader.loadTestsFromModule(module))

result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
"#;

fn parse_sandbox_args(args: &[String]) -> Result<(PathBuf, Vec<String>)> {
    let modern_root = args
        .first()
        .context("SandboxSkill expects the modern_app directory as the first argument")?;
    let relative_paths = args.get(1..).filter(|paths| !paths.is_empty()).context(
        "SandboxSkill expects at least one generated test file path after the modern_app directory",
    )?;

    Ok((PathBuf::from(modern_root), relative_paths.to_vec()))
}

fn validate_sandbox_targets(
    modern_root: &Path,
    relative_paths: &[String],
) -> Result<Vec<Utf8PathBuf>> {
    if !modern_root.exists() {
        return Err(anyhow!(
            "sandbox root does not exist: {}",
            modern_root.display()
        ));
    }

    if !modern_root.is_dir() {
        return Err(anyhow!(
            "SandboxSkill expects a directory root, received {}",
            modern_root.display()
        ));
    }

    let mut normalized_paths = Vec::with_capacity(relative_paths.len());
    for relative_path in relative_paths {
        let normalized = normalize_relative_path(relative_path, "sandbox target path")?;
        let full_path = modern_root.join(normalized.as_std_path());
        if !full_path.exists() {
            return Err(anyhow!(
                "sandbox target does not exist: {}",
                full_path.display()
            ));
        }

        if !full_path.is_file() {
            return Err(anyhow!(
                "sandbox target must be a file: {}",
                full_path.display()
            ));
        }

        normalized_paths.push(normalized);
    }

    Ok(normalized_paths)
}

fn build_warmup_docker_args(
    source_root: &Path,
    container_name: &str,
    config: &DockerSandboxConfig,
    base_image: &str,
    install_command: &str,
) -> Result<Vec<String>> {
    let copy_source = shell_escape(&format!("{APP_SOURCE_MOUNT_POINT}/."));
    let workspace = shell_escape(PREPARED_APP_MOUNT_POINT);
    let shell_command = format!(
        "rm -rf {workspace} && mkdir -p {workspace} && cp -R {copy_source} {workspace} && cd {workspace} && {install_command}"
    );

    Ok(vec![
        "run".to_owned(),
        "--name".to_owned(),
        container_name.to_owned(),
        "--network=bridge".to_owned(),
        format!("--memory={}", config.memory_limit),
        format!("--cpus={}", config.cpu_limit),
        "--cap-drop=ALL".to_owned(),
        "--security-opt=no-new-privileges:true".to_owned(),
        "--mount".to_owned(),
        docker_bind_mount(source_root, APP_SOURCE_MOUNT_POINT, true)?,
        base_image.to_owned(),
        "sh".to_owned(),
        "-lc".to_owned(),
        shell_command,
    ])
}

fn build_deno_cache_command(execution_paths: &[Utf8PathBuf]) -> String {
    let cached_paths = execution_paths
        .iter()
        .map(|path| shell_escape(&container_path(PREPARED_APP_MOUNT_POINT, path.as_path())))
        .collect::<Vec<_>>()
        .join(" ");
    format!("deno cache {cached_paths}")
}

fn shell_escape(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

pub(crate) async fn run_docker_command(
    args: &[String],
    timeout_duration: Duration,
    container_guard: Option<ContainerGuard>,
) -> Result<Output> {
    let (output, _) =
        run_docker_command_with_guard(args, timeout_duration, container_guard, true).await?;
    Ok(output)
}

async fn run_docker_command_with_guard(
    args: &[String],
    timeout_duration: Duration,
    mut container_guard: Option<ContainerGuard>,
    disarm_guard_on_success: bool,
) -> Result<(Output, Option<ContainerGuard>)> {
    let mut command = TokioCommand::new("docker");
    command
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let output = timeout(timeout_duration, async {
        command.output().await.map_err(|error| match error.kind() {
            ErrorKind::NotFound => anyhow!(
                "Docker CLI is not installed or not on PATH. Install Docker and ensure the daemon is running."
            ),
            _ => anyhow!("failed to spawn docker command: {error}"),
        })
    })
    .await;

    let output = match output {
        Ok(result) => result?,
        Err(_) => {
            return Err(anyhow!(
                "sandbox execution timed out after {} seconds: {}",
                timeout_duration.as_secs(),
                format_command("docker", args)
            ));
        }
    };

    if disarm_guard_on_success {
        if let Some(container_guard) = &mut container_guard {
            container_guard.disarm();
        }
    }

    Ok((output, container_guard))
}

pub(crate) fn build_container_name(prefix: &str) -> String {
    let unique_id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let sanitized_prefix = prefix
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '.' | '-') {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>();

    format!(
        "migration-pipeline-{}-{}-{}",
        sanitized_prefix.trim_matches('-'),
        std::process::id(),
        unique_id
    )
}

pub(crate) fn preferred_output(stderr: &str, stdout: &str) -> String {
    let stderr = stderr.trim();
    if !stderr.is_empty() {
        return stderr.to_owned();
    }

    let stdout = stdout.trim();
    if !stdout.is_empty() {
        stdout.to_owned()
    } else {
        "no additional error details were returned by Docker".to_owned()
    }
}

fn format_command(program: &str, args: &[String]) -> String {
    if args.is_empty() {
        program.to_owned()
    } else {
        format!("{program} {}", args.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::{APP_SOURCE_MOUNT_POINT, SandboxRuntime, SandboxSkill};
    use crate::config::DockerSandboxConfig;
    use camino::Utf8PathBuf;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn make_temp_modern_root() -> std::path::PathBuf {
        let unique_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be monotonic")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("migration_pipeline_sandbox_{unique_id}"));
        fs::create_dir_all(root.join("tests")).expect("sandbox test root should exist");
        root
    }

    fn sandbox_skill() -> SandboxSkill {
        SandboxSkill::new(DockerSandboxConfig {
            memory_limit: "512m".to_owned(),
            cpu_limit: "1.5".to_owned(),
        })
    }

    #[tokio::test]
    async fn sandbox_invocation_enforces_network_and_read_only_mounts() {
        let root = make_temp_modern_root();
        fs::write(root.join("tests/test_app.py"), "print('ok')").expect("fixture should exist");

        let invocation = sandbox_skill()
            .build_invocation(&root, &[String::from("tests/test_app.py")])
            .await
            .expect("sandbox invocation should build");

        assert!(
            invocation
                .docker_args
                .contains(&"--network=none".to_owned())
        );
        assert!(invocation.docker_args.contains(&"--memory=512m".to_owned()));
        assert!(invocation.docker_args.contains(&"--cpus=1.5".to_owned()));
        assert!(invocation.docker_args.contains(&"--mount".to_owned()));
        assert!(
            invocation
                .docker_args
                .iter()
                .any(|arg| arg.contains("target=/app") && arg.contains("readonly"))
        );

        fs::remove_dir_all(root).expect("sandbox temp root should be removed");
    }

    #[tokio::test]
    async fn sandbox_invocation_selects_python_container_command() {
        let root = make_temp_modern_root();
        fs::write(root.join("tests/test_app.py"), "print('ok')").expect("fixture should exist");

        let invocation = sandbox_skill()
            .build_invocation(&root, &[String::from("tests/test_app.py")])
            .await
            .expect("sandbox invocation should build");

        assert!(invocation.display_command.contains("python:alpine"));
        assert!(
            invocation
                .display_command
                .contains("/app/tests/test_app.py")
        );
        assert!(invocation.display_command.contains("-c"));

        fs::remove_dir_all(root).expect("sandbox temp root should be removed");
    }

    #[tokio::test]
    async fn sandbox_invocation_rejects_parent_directory_traversal() {
        let root = make_temp_modern_root();
        fs::write(root.join("tests/test_app.py"), "print('ok')").expect("fixture should exist");

        let error = sandbox_skill()
            .build_invocation(&root, &[String::from("../secrets.py")])
            .await
            .expect_err("sandbox should reject paths outside modern_app");

        assert!(error.to_string().contains("traverse upwards"));

        fs::remove_dir_all(root).expect("sandbox temp root should be removed");
    }

    #[tokio::test]
    async fn sandbox_invocation_rejects_mixed_language_test_suites() {
        let root = make_temp_modern_root();
        fs::create_dir_all(root.join("src")).expect("src directory should exist");
        fs::write(root.join("tests/test_app.py"), "print('ok')")
            .expect("python test fixture should exist");
        fs::write(root.join("tests/test_app.js"), "test('ok', () => {});")
            .expect("javascript test fixture should exist");

        let error = sandbox_skill()
            .build_invocation(
                &root,
                &[
                    String::from("tests/test_app.py"),
                    String::from("tests/test_app.js"),
                ],
            )
            .await
            .expect_err("sandbox should reject mixed runtimes");

        assert!(error.to_string().contains("mixed-language"));

        fs::remove_dir_all(root).expect("sandbox temp root should be removed");
    }

    #[test]
    fn sandbox_runtime_detects_node_dependency_manifests() {
        let root = make_temp_modern_root();
        fs::write(root.join("package.json"), r#"{"name":"fixture"}"#)
            .expect("package.json should exist");

        let install_command = SandboxRuntime::Node
            .dependency_install_command(&root, &[Utf8PathBuf::from("tests/test_app.js")])
            .expect("dependency detection should succeed");

        assert_eq!(
            install_command.as_deref(),
            Some("npm install --ignore-scripts --no-audit --no-fund")
        );

        fs::remove_dir_all(root).expect("sandbox temp root should be removed");
    }

    #[test]
    fn sandbox_runtime_builds_deno_cache_command_from_test_paths() {
        let install_command = SandboxRuntime::Deno
            .dependency_install_command(
                std::path::Path::new(APP_SOURCE_MOUNT_POINT),
                &[Utf8PathBuf::from("tests/test_app.ts")],
            )
            .expect("dependency detection should succeed");

        assert_eq!(
            install_command.as_deref(),
            Some("deno cache '/workspace/tests/test_app.ts'")
        );
    }
}
