use anyhow::{Context, Result, anyhow};
use std::io::ErrorKind;
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command as TokioCommand;
use tokio::runtime::{Builder, Handle};
use tokio::time::timeout;
use tracing::debug;

use super::{Skill, TerminalCommandOutput};

const APP_MOUNT_POINT: &str = "/app";
const DOCKER_INFO_TIMEOUT_SECS: u64 = 5;
const SANDBOX_TIMEOUT_SECS: u64 = 20;
const SANDBOX_MEMORY_LIMIT: &str = "256m";
const SANDBOX_CPU_LIMIT: &str = "0.5";

#[derive(Debug, Default, Clone, Copy)]
pub struct SandboxSkill;

#[derive(Debug)]
struct SandboxInvocation {
    docker_args: Vec<String>,
    display_command: String,
}

impl SandboxSkill {
    async fn execute_async(&self, args: Vec<String>) -> Result<String> {
        let (modern_root, relative_path) = parse_sandbox_args(&args)?;
        self.ensure_docker_ready().await?;

        let invocation = Self::build_invocation(&modern_root, &relative_path)?;
        debug!(
            command = invocation.display_command.as_str(),
            relative_path = relative_path.as_str(),
            "Launching sandboxed verification container"
        );

        let output = run_docker_command(
            &invocation.docker_args,
            Duration::from_secs(SANDBOX_TIMEOUT_SECS),
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

    async fn ensure_docker_ready(&self) -> Result<()> {
        let args = vec![
            "info".to_owned(),
            "--format".to_owned(),
            "{{json .ServerVersion}}".to_owned(),
        ];
        let output = run_docker_command(&args, Duration::from_secs(DOCKER_INFO_TIMEOUT_SECS))
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

    fn build_invocation(modern_root: &Path, relative_path: &str) -> Result<SandboxInvocation> {
        validate_sandbox_target(modern_root, relative_path)?;

        let relative_path = relative_path.replace('\\', "/");
        let host_mount = docker_mount_path(modern_root)?;
        let container_path = format!("{APP_MOUNT_POINT}/{relative_path}");
        let container_parent = Path::new(&container_path)
            .parent()
            .unwrap_or_else(|| Path::new(APP_MOUNT_POINT))
            .to_string_lossy()
            .replace('\\', "/");

        let (image, inner_command) = match Path::new(&relative_path)
            .extension()
            .and_then(|value| value.to_str())
            .map(|value| value.to_ascii_lowercase())
            .as_deref()
        {
            Some("js") | Some("mjs") | Some("cjs") => (
                "node:alpine",
                vec![
                    "node".to_owned(),
                    "--check".to_owned(),
                    container_path.clone(),
                ],
            ),
            Some("ts") | Some("tsx") => (
                "denoland/deno:alpine",
                vec![
                    "deno".to_owned(),
                    "check".to_owned(),
                    container_path.clone(),
                ],
            ),
            Some("py") => (
                "python:alpine",
                vec![
                    "python".to_owned(),
                    "-c".to_owned(),
                    format!(
                        "import py_compile; py_compile.compile({container_path:?}, cfile='/tmp/sandbox.pyc', doraise=True)"
                    ),
                ],
            ),
            Some("php") => (
                "php:alpine",
                vec!["php".to_owned(), "-l".to_owned(), container_path.clone()],
            ),
            Some("rb") => (
                "ruby:alpine",
                vec!["ruby".to_owned(), "-c".to_owned(), container_path.clone()],
            ),
            Some("go") => (
                "golang:alpine",
                vec!["go".to_owned(), "test".to_owned(), container_parent],
            ),
            Some("rs") => (
                "rust:alpine",
                vec![
                    "rustc".to_owned(),
                    "--crate-type".to_owned(),
                    "lib".to_owned(),
                    "--emit".to_owned(),
                    "metadata".to_owned(),
                    "-o".to_owned(),
                    "/tmp/sandbox.rmeta".to_owned(),
                    container_path.clone(),
                ],
            ),
            Some("java") => (
                "eclipse-temurin:21-alpine",
                vec![
                    "javac".to_owned(),
                    "-d".to_owned(),
                    "/tmp".to_owned(),
                    container_path.clone(),
                ],
            ),
            Some(other) => {
                return Err(anyhow!(
                    "SandboxSkill does not support the `.{other}` extension yet"
                ));
            }
            None => {
                return Err(anyhow!(
                    "SandboxSkill requires a file path with a supported extension: {relative_path}"
                ));
            }
        };

        let mut docker_args = vec![
            "run".to_owned(),
            "--rm".to_owned(),
            "--network=none".to_owned(),
            format!("--memory={SANDBOX_MEMORY_LIMIT}"),
            format!("--cpus={SANDBOX_CPU_LIMIT}"),
            "--cap-drop=ALL".to_owned(),
            "--security-opt=no-new-privileges:true".to_owned(),
            "--workdir".to_owned(),
            "/tmp".to_owned(),
            "-v".to_owned(),
            format!("{host_mount}:{APP_MOUNT_POINT}:ro"),
            image.to_owned(),
        ];
        docker_args.extend(inner_command);

        Ok(SandboxInvocation {
            display_command: format_command("docker", &docker_args),
            docker_args,
        })
    }
}

impl Skill for SandboxSkill {
    fn name(&self) -> &str {
        "sandbox"
    }

    fn execute(&self, args: Vec<String>) -> Result<String> {
        if let Ok(handle) = Handle::try_current() {
            tokio::task::block_in_place(|| handle.block_on(self.execute_async(args)))
        } else {
            Builder::new_current_thread()
                .enable_all()
                .build()
                .context("failed to create runtime for SandboxSkill")?
                .block_on(self.execute_async(args))
        }
    }
}

fn parse_sandbox_args(args: &[String]) -> Result<(PathBuf, String)> {
    let modern_root = args
        .first()
        .context("SandboxSkill expects the modern_app directory as the first argument")?;
    let relative_path = args
        .get(1)
        .context("SandboxSkill expects the target file path as the second argument")?;

    Ok((PathBuf::from(modern_root), relative_path.to_owned()))
}

fn validate_sandbox_target(modern_root: &Path, relative_path: &str) -> Result<()> {
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

    let relative = Path::new(relative_path);
    if relative.is_absolute() {
        return Err(anyhow!(
            "sandbox target must be relative to the mounted modern_app directory: {relative_path}"
        ));
    }

    if relative.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        return Err(anyhow!(
            "sandbox target must stay within the mounted modern_app directory: {relative_path}"
        ));
    }

    let full_path = modern_root.join(relative);
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

    Ok(())
}

fn docker_mount_path(modern_root: &Path) -> Result<String> {
    let absolute_path = if modern_root.is_absolute() {
        modern_root.to_path_buf()
    } else {
        std::env::current_dir()
            .context("failed to resolve current working directory for sandbox mount")?
            .join(modern_root)
    };

    Ok(absolute_path.to_string_lossy().replace('\\', "/"))
}

async fn run_docker_command(
    args: &[String],
    timeout_duration: Duration,
) -> Result<std::process::Output> {
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
    .await
    .map_err(|_| {
        anyhow!(
            "sandbox execution timed out after {} seconds: {}",
            timeout_duration.as_secs(),
            format_command("docker", args)
        )
    })??;

    Ok(output)
}

fn preferred_output(stderr: &str, stdout: &str) -> String {
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

    #[test]
    fn sandbox_invocation_enforces_network_and_read_only_mounts() {
        let root = make_temp_modern_root();
        fs::write(root.join("tests/test_app.py"), "print('ok')").expect("fixture should exist");

        let invocation = SandboxSkill::build_invocation(&root, "tests/test_app.py")
            .expect("sandbox invocation should build");

        assert!(
            invocation
                .docker_args
                .contains(&"--network=none".to_owned())
        );
        assert!(invocation.docker_args.contains(&"--memory=256m".to_owned()));
        assert!(invocation.docker_args.contains(&"--cpus=0.5".to_owned()));
        assert!(
            invocation
                .docker_args
                .iter()
                .any(|arg| arg.ends_with(":/app:ro"))
        );

        fs::remove_dir_all(root).expect("sandbox temp root should be removed");
    }

    #[test]
    fn sandbox_invocation_selects_python_container_command() {
        let root = make_temp_modern_root();
        fs::write(root.join("tests/test_app.py"), "print('ok')").expect("fixture should exist");

        let invocation = SandboxSkill::build_invocation(&root, "tests/test_app.py")
            .expect("sandbox invocation should build");

        assert!(invocation.display_command.contains("python:alpine"));
        assert!(invocation.display_command.contains("--network=none"));
        assert!(
            invocation
                .display_command
                .contains("/app/tests/test_app.py")
        );

        fs::remove_dir_all(root).expect("sandbox temp root should be removed");
    }

    #[test]
    fn sandbox_invocation_rejects_parent_directory_traversal() {
        let root = make_temp_modern_root();
        fs::write(root.join("tests/test_app.py"), "print('ok')").expect("fixture should exist");

        let error = SandboxSkill::build_invocation(&root, "../secrets.py")
            .expect_err("sandbox should reject paths outside modern_app");

        assert!(error.to_string().contains("must stay within"));

        fs::remove_dir_all(root).expect("sandbox temp root should be removed");
    }
}
