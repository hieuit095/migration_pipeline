use crate::config::DockerSandboxConfig;
use crate::utils::path::{container_path, docker_bind_mount, normalize_relative_path};
use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use camino::Utf8PathBuf;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::{Command as StdCommand, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::process::Command as TokioCommand;
use tokio::time::timeout;
use tracing::{debug, warn};

use super::{Skill, TerminalCommandOutput};

pub(crate) const APP_MOUNT_POINT: &str = "/app";
const DOCKER_INFO_TIMEOUT_SECS: u64 = 5;
pub(crate) const SANDBOX_TIMEOUT_SECS: u64 = 20;

#[derive(Debug, Clone)]
pub struct SandboxSkill {
    config: DockerSandboxConfig,
}

#[derive(Debug)]
struct SandboxInvocation {
    container_name: String,
    docker_args: Vec<String>,
    display_command: String,
}

#[derive(Debug)]
pub(crate) struct ContainerGuard {
    pub container_name: String,
    armed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SandboxRuntime {
    Node,
    Deno,
    Python,
    Go,
}

impl SandboxSkill {
    pub fn new(config: DockerSandboxConfig) -> Self {
        Self { config }
    }

    pub(crate) fn config(&self) -> &DockerSandboxConfig {
        &self.config
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

    fn build_invocation(
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
        let container_paths = normalized_paths
            .iter()
            .map(|path| container_path(APP_MOUNT_POINT, path.as_path()))
            .collect::<Vec<_>>();
        let (image, inner_command) = runtime.build_command(&container_paths);
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
            APP_MOUNT_POINT.to_owned(),
            "--mount".to_owned(),
            docker_bind_mount(modern_root, APP_MOUNT_POINT, true)?,
            image.to_owned(),
        ];
        docker_args.extend(inner_command);

        Ok(SandboxInvocation {
            container_name,
            display_command: format_command("docker", &docker_args),
            docker_args,
        })
    }
}

impl ContainerGuard {
    fn new(container_name: impl Into<String>) -> Self {
        Self {
            container_name: container_name.into(),
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ContainerGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }

        let container_name = self.container_name.clone();
        std::thread::spawn(move || {
            let result = StdCommand::new("docker")
                .args(["rm", "-f", container_name.as_str()])
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .output();

            match result {
                Ok(output) if output.status.success() => {}
                Ok(output) => {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let detail = stderr.trim();
                    if !detail.is_empty()
                        && !detail.to_ascii_lowercase().contains("no such container")
                    {
                        warn!(
                            container_name = container_name.as_str(),
                            detail, "Container guard cleanup returned a non-success status"
                        );
                    }
                }
                Err(error) => {
                    warn!(
                        container_name = container_name.as_str(),
                        error = %error,
                        "Container guard cleanup failed to spawn docker rm -f"
                    );
                }
            }
        });
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

    fn from_path(relative_path: &Utf8PathBuf) -> Result<Self> {
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

    fn build_command(self, container_paths: &[String]) -> (&'static str, Vec<String>) {
        match self {
            Self::Node => {
                let mut command = vec!["node".to_owned(), "--test".to_owned()];
                command.extend(container_paths.iter().cloned());
                ("node:alpine", command)
            }
            Self::Deno => {
                let mut command = vec!["deno".to_owned(), "test".to_owned()];
                command.extend(container_paths.iter().cloned());
                ("denoland/deno:alpine", command)
            }
            Self::Python => {
                let mut command = vec![
                    "python".to_owned(),
                    "-c".to_owned(),
                    PYTHON_UNITTEST_RUNNER.to_owned(),
                ];
                command.extend(container_paths.iter().cloned());
                ("python:alpine", command)
            }
            Self::Go => (
                "golang:alpine",
                vec![
                    "sh".to_owned(),
                    "-lc".to_owned(),
                    "cd /app && go test ./...".to_owned(),
                ],
            ),
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

        let invocation = self.build_invocation(&modern_root, &test_paths)?;
        debug!(
            command = invocation.display_command.as_str(),
            test_paths = ?test_paths,
            "Launching sandboxed verification container"
        );

        let output = run_docker_command(
            &invocation.docker_args,
            Duration::from_secs(SANDBOX_TIMEOUT_SECS),
            Some(invocation.container_name.as_str()),
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

pub(crate) async fn run_docker_command(
    args: &[String],
    timeout_duration: Duration,
    container_name: Option<&str>,
) -> Result<std::process::Output> {
    let mut container_guard = container_name.map(ContainerGuard::new);
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

    if let Some(container_guard) = &mut container_guard {
        container_guard.disarm();
    }

    Ok(output)
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
    use super::SandboxSkill;
    use crate::config::DockerSandboxConfig;
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

    #[test]
    fn sandbox_invocation_enforces_network_and_read_only_mounts() {
        let root = make_temp_modern_root();
        fs::write(root.join("tests/test_app.py"), "print('ok')").expect("fixture should exist");

        let invocation = sandbox_skill()
            .build_invocation(&root, &[String::from("tests/test_app.py")])
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

    #[test]
    fn sandbox_invocation_selects_python_container_command() {
        let root = make_temp_modern_root();
        fs::write(root.join("tests/test_app.py"), "print('ok')").expect("fixture should exist");

        let invocation = sandbox_skill()
            .build_invocation(&root, &[String::from("tests/test_app.py")])
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

    #[test]
    fn sandbox_invocation_rejects_parent_directory_traversal() {
        let root = make_temp_modern_root();
        fs::write(root.join("tests/test_app.py"), "print('ok')").expect("fixture should exist");

        let error = sandbox_skill()
            .build_invocation(&root, &[String::from("../secrets.py")])
            .expect_err("sandbox should reject paths outside modern_app");

        assert!(error.to_string().contains("traverse upwards"));

        fs::remove_dir_all(root).expect("sandbox temp root should be removed");
    }

    #[test]
    fn sandbox_invocation_rejects_mixed_language_test_suites() {
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
            .expect_err("sandbox should reject mixed runtimes");

        assert!(error.to_string().contains("mixed-language"));

        fs::remove_dir_all(root).expect("sandbox temp root should be removed");
    }
}
