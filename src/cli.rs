use anyhow::{Context, Result, ensure};
use inquire::{Confirm, InquireError, Password, PasswordDisplayMode, Select, Text};
use std::env;
use std::fmt::{self, Display};
use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};

const ENV_FILE: &str = ".env";
const STATE_DB_FILE: &str = ".migration_state.db";
const STATE_JSON_FILE: &str = ".migration_state.json";
const PIPELINE_LEGACY_ROOT_ENV: &str = "PIPELINE_LEGACY_ROOT";
const PIPELINE_MODERN_ROOT_ENV: &str = "PIPELINE_MODERN_ROOT";
const DEFAULT_NONINTERACTIVE_LEGACY_ROOT: &str = "legacy_app";
const DEFAULT_MODERN_ROOT: &str = "./modern_app";
const DEFAULT_DOCKER_SANDBOX_MEMORY: &str = "256m";
const DEFAULT_DOCKER_SANDBOX_CPUS: &str = "0.5";

const OPENAI_API_KEY_ENV: &str = "OPENAI_API_KEY";
const TOGETHER_API_KEY_ENV: &str = "TOGETHER_API_KEY";
const BLUEPRINTER_MODEL_ENV: &str = "BLUEPRINTER_MODEL";
const EXECUTOR_MODEL_ENV: &str = "EXECUTOR_MODEL";
const VERIFIER_MODEL_ENV: &str = "VERIFIER_MODEL";
const SURGEON_MODEL_ENV: &str = "SURGEON_MODEL";
const DOCKER_SANDBOX_MEMORY_ENV: &str = "DOCKER_SANDBOX_MEMORY";
const DOCKER_SANDBOX_CPUS_ENV: &str = "DOCKER_SANDBOX_CPUS";

#[derive(Debug, Clone)]
pub struct StartMigrationConfig {
    pub legacy_root: PathBuf,
    pub modern_root: PathBuf,
}

#[derive(Debug, Clone, Copy)]
enum MainMenuAction {
    StartMigration,
    SettingsAndConfiguration,
    Exit,
}

#[derive(Debug, Clone, Copy)]
enum SettingsMenuAction {
    ManageApiKeys,
    ConfigureTaskModels,
    ConfigureDockerLimits,
    BackToMainMenu,
}

impl Display for MainMenuAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StartMigration => f.write_str("Start Migration"),
            Self::SettingsAndConfiguration => f.write_str("Settings & Configuration"),
            Self::Exit => f.write_str("Exit"),
        }
    }
}

impl Display for SettingsMenuAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ManageApiKeys => f.write_str("Manage API Keys"),
            Self::ConfigureTaskModels => f.write_str("Configure Task Models"),
            Self::ConfigureDockerLimits => f.write_str("Configure Docker Limits"),
            Self::BackToMainMenu => f.write_str("Back to Main Menu"),
        }
    }
}

pub fn run_cli(startup_root: &Path) -> Result<Option<StartMigrationConfig>> {
    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        return Ok(Some(load_noninteractive_startup_configuration(
            startup_root,
        )?));
    }

    loop {
        let Some(selection) = prompt_select(
            "Main Menu",
            vec![
                MainMenuAction::StartMigration,
                MainMenuAction::SettingsAndConfiguration,
                MainMenuAction::Exit,
            ],
        )?
        else {
            return Ok(None);
        };

        match selection {
            MainMenuAction::StartMigration => {
                if let Some(configuration) = start_migration_flow(startup_root)? {
                    return Ok(Some(configuration));
                }
            }
            MainMenuAction::SettingsAndConfiguration => {
                settings_menu_loop(startup_root)?;
            }
            MainMenuAction::Exit => return Ok(None),
        }
    }
}

fn start_migration_flow(startup_root: &Path) -> Result<Option<StartMigrationConfig>> {
    let legacy_root = loop {
        let Some(input) = prompt_text("Enter the path to the legacy project (source):", None)?
        else {
            return Ok(None);
        };

        match resolve_project_root(startup_root, Path::new(input.trim()), false) {
            Ok(path) => break path,
            Err(error) => {
                println!("{error}");
            }
        }
    };

    let modern_root = loop {
        let Some(input) = prompt_text(
            "Enter the path for the modernized project (destination):",
            Some(DEFAULT_MODERN_ROOT),
        )?
        else {
            return Ok(None);
        };

        match resolve_project_root(startup_root, Path::new(input.trim()), true) {
            Ok(path) => {
                if path == legacy_root {
                    println!(
                        "modernized output directory must differ from the legacy codebase directory"
                    );
                    continue;
                }
                break path;
            }
            Err(error) => {
                println!("{error}");
            }
        }
    };

    let clear_previous_state = prompt_confirm(
        "Do you want to clear previous migration state (.migration_state.db/.json) before starting?",
        true,
    )?
    .unwrap_or(false);
    if clear_previous_state {
        clear_previous_state_files(startup_root)?;
    }

    Ok(Some(StartMigrationConfig {
        legacy_root,
        modern_root,
    }))
}

fn settings_menu_loop(startup_root: &Path) -> Result<()> {
    loop {
        let Some(selection) = prompt_select(
            "Settings & Configuration",
            vec![
                SettingsMenuAction::ManageApiKeys,
                SettingsMenuAction::ConfigureTaskModels,
                SettingsMenuAction::ConfigureDockerLimits,
                SettingsMenuAction::BackToMainMenu,
            ],
        )?
        else {
            return Ok(());
        };

        match selection {
            SettingsMenuAction::ManageApiKeys => manage_api_keys(startup_root)?,
            SettingsMenuAction::ConfigureTaskModels => configure_task_models(startup_root)?,
            SettingsMenuAction::ConfigureDockerLimits => configure_docker_limits(startup_root)?,
            SettingsMenuAction::BackToMainMenu => return Ok(()),
        }
    }
}

fn manage_api_keys(startup_root: &Path) -> Result<()> {
    let Some(openai_api_key) = prompt_password("OPENAI_API_KEY")? else {
        return Ok(());
    };
    persist_env_value(startup_root, OPENAI_API_KEY_ENV, &openai_api_key)?;

    let Some(together_api_key) = prompt_password("TOGETHER_API_KEY")? else {
        return Ok(());
    };
    persist_env_value(startup_root, TOGETHER_API_KEY_ENV, &together_api_key)?;
    Ok(())
}

fn configure_task_models(startup_root: &Path) -> Result<()> {
    for setting in [
        (
            "Blueprinter Model",
            BLUEPRINTER_MODEL_ENV,
            vec![
                "gpt-5.4",
                "google/gemini-3-flash-preview",
                "anthropic/claude-sonnet-4.5",
            ],
        ),
        (
            "Executor Model",
            EXECUTOR_MODEL_ENV,
            vec!["moonshotai/Kimi-K2.5", "minimax/minimax-m2.5", "gpt-5.4"],
        ),
        (
            "Verifier Model",
            VERIFIER_MODEL_ENV,
            vec!["zai-org/GLM-5", "gpt-5.4", "anthropic/claude-sonnet-4.5"],
        ),
        (
            "Surgeon Model",
            SURGEON_MODEL_ENV,
            vec![
                "gpt-5.4",
                "anthropic/claude-sonnet-4.5",
                "google/gemini-3-flash-preview",
            ],
        ),
    ] {
        let Some(value) = prompt_model_value(setting.0, setting.1, &setting.2)? else {
            return Ok(());
        };
        persist_env_value(startup_root, setting.1, &value)?;
    }

    Ok(())
}

fn configure_docker_limits(startup_root: &Path) -> Result<()> {
    let memory_default = env::var(DOCKER_SANDBOX_MEMORY_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_DOCKER_SANDBOX_MEMORY.to_owned());
    let Some(memory_limit) = prompt_text(
        "Enter DOCKER_SANDBOX_MEMORY (e.g., 512m):",
        Some(memory_default.as_str()),
    )?
    else {
        return Ok(());
    };
    persist_env_value(startup_root, DOCKER_SANDBOX_MEMORY_ENV, memory_limit.trim())?;

    let cpu_default = env::var(DOCKER_SANDBOX_CPUS_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_DOCKER_SANDBOX_CPUS.to_owned());
    let Some(cpu_limit) = prompt_text(
        "Enter DOCKER_SANDBOX_CPUS (e.g., 1.0):",
        Some(cpu_default.as_str()),
    )?
    else {
        return Ok(());
    };
    persist_env_value(startup_root, DOCKER_SANDBOX_CPUS_ENV, cpu_limit.trim())?;

    Ok(())
}

fn prompt_password(env_key: &str) -> Result<Option<String>> {
    let prompt_label = format!("Enter {env_key}:");
    let prompt = Password::new(&prompt_label)
        .with_display_mode(PasswordDisplayMode::Masked)
        .without_confirmation();
    let Some(value) = prompt_value(prompt.prompt())? else {
        return Ok(None);
    };

    ensure!(!value.trim().is_empty(), "{env_key} cannot be empty");
    Ok(Some(value))
}

fn prompt_model_value(
    label: &str,
    env_key: &str,
    recommendations: &[&str],
) -> Result<Option<String>> {
    let current_value = env::var(env_key)
        .ok()
        .filter(|value| !value.trim().is_empty());
    let mut options = Vec::new();
    if let Some(current) = current_value.as_ref() {
        options.push(format!("Current: {current}"));
    }
    for recommendation in recommendations {
        if current_value.as_deref() != Some(*recommendation) {
            options.push((*recommendation).to_owned());
        }
    }
    options.push("Custom model...".to_owned());

    let Some(selection) = prompt_select(&format!("Select {label}"), options)? else {
        return Ok(None);
    };

    if selection == "Custom model..." {
        let Some(custom_value) = prompt_text(
            &format!("Enter a custom value for {env_key}:"),
            current_value.as_deref(),
        )?
        else {
            return Ok(None);
        };
        ensure!(!custom_value.trim().is_empty(), "{env_key} cannot be empty");
        Ok(Some(custom_value.trim().to_owned()))
    } else if let Some(current) = selection.strip_prefix("Current: ") {
        Ok(Some(current.to_owned()))
    } else {
        Ok(Some(selection))
    }
}

fn prompt_text(prompt: &str, default: Option<&str>) -> Result<Option<String>> {
    let mut inquiry = Text::new(prompt);
    if let Some(default) = default {
        inquiry = inquiry.with_default(default);
    }

    let Some(value) = prompt_value(inquiry.prompt())? else {
        return Ok(None);
    };

    Ok(Some(value.trim().to_owned()))
}

fn prompt_confirm(prompt: &str, default: bool) -> Result<Option<bool>> {
    prompt_value(Confirm::new(prompt).with_default(default).prompt())
}

fn prompt_select<T>(prompt: &str, options: Vec<T>) -> Result<Option<T>>
where
    T: Display + Clone,
{
    prompt_value(Select::new(prompt, options).prompt())
}

fn prompt_value<T>(result: std::result::Result<T, InquireError>) -> Result<Option<T>> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(InquireError::OperationCanceled | InquireError::OperationInterrupted) => Ok(None),
        Err(error) => Err(error).context("interactive prompt failed"),
    }
}

fn clear_previous_state_files(startup_root: &Path) -> Result<()> {
    for relative_path in [STATE_DB_FILE, STATE_JSON_FILE] {
        let path = startup_root.join(relative_path);
        if path.exists() {
            fs::remove_file(&path)
                .with_context(|| format!("failed to remove {}", path.display()))?;
        }
    }

    Ok(())
}

fn load_noninteractive_startup_configuration(startup_root: &Path) -> Result<StartMigrationConfig> {
    let legacy_root = env::var(PIPELINE_LEGACY_ROOT_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_NONINTERACTIVE_LEGACY_ROOT.to_owned());
    let modern_root = env::var(PIPELINE_MODERN_ROOT_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_MODERN_ROOT.to_owned());

    Ok(StartMigrationConfig {
        legacy_root: resolve_project_root(startup_root, Path::new(legacy_root.trim()), false)?,
        modern_root: resolve_project_root(startup_root, Path::new(modern_root.trim()), true)?,
    })
}

fn persist_env_value(startup_root: &Path, key: &str, value: &str) -> Result<()> {
    let env_path = startup_root.join(ENV_FILE);
    let existing = match fs::read_to_string(&env_path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => {
            return Err(error).with_context(|| format!("failed to read {}", env_path.display()));
        }
    };

    let mut updated_lines = Vec::new();
    let mut replaced = false;
    for line in existing.lines() {
        if env_assignment_key(line).is_some_and(|existing_key| existing_key == key) {
            updated_lines.push(format!("{key}={}", format_env_value(value)));
            replaced = true;
        } else {
            updated_lines.push(line.to_owned());
        }
    }

    if !replaced {
        updated_lines.push(format!("{key}={}", format_env_value(value)));
    }

    let mut rendered = updated_lines.join("\n");
    if !rendered.is_empty() {
        rendered.push('\n');
    }

    fs::write(&env_path, rendered)
        .with_context(|| format!("failed to write {}", env_path.display()))?;
    unsafe {
        env::set_var(key, value);
    }
    Ok(())
}

fn env_assignment_key(line: &str) -> Option<&str> {
    let trimmed = line.trim_start();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return None;
    }

    let assignment = trimmed.strip_prefix("export ").unwrap_or(trimmed);
    let separator = assignment.find('=')?;
    let key = assignment[..separator].trim();
    if key.is_empty() { None } else { Some(key) }
}

fn format_env_value(value: &str) -> String {
    format!(
        "\"{}\"",
        value
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
    )
}

fn resolve_project_root(
    startup_root: &Path,
    configured_path: &Path,
    create_if_missing: bool,
) -> Result<PathBuf> {
    let absolute_path = absolute_project_root(startup_root, configured_path);

    if create_if_missing {
        fs::create_dir_all(&absolute_path).with_context(|| {
            format!(
                "failed to create required project directory {}",
                absolute_path.display()
            )
        })?;
    }

    ensure!(
        absolute_path.exists(),
        "required project directory does not exist: {}",
        absolute_path.display()
    );
    ensure!(
        absolute_path.is_dir(),
        "required project path is not a directory: {}",
        absolute_path.display()
    );

    Ok(absolute_path)
}

fn absolute_project_root(startup_root: &Path, configured_path: &Path) -> PathBuf {
    if configured_path.is_absolute() {
        configured_path.to_path_buf()
    } else {
        startup_root.join(configured_path)
    }
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_MODERN_ROOT, env_assignment_key, format_env_value, persist_env_value};
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::{LazyLock, Mutex};
    use std::time::{SystemTime, UNIX_EPOCH};

    static ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    fn temp_root(prefix: &str) -> PathBuf {
        let unique_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be monotonic")
            .as_nanos();
        let root = env::temp_dir().join(format!("{prefix}_{unique_id}"));
        fs::create_dir_all(&root).expect("temp root should be created");
        root
    }

    #[test]
    fn env_assignment_key_ignores_comments() {
        assert_eq!(env_assignment_key("# OPENAI_API_KEY=foo"), None);
        assert_eq!(
            env_assignment_key(" OPENAI_API_KEY=foo"),
            Some("OPENAI_API_KEY")
        );
        assert_eq!(
            env_assignment_key("export TOGETHER_API_KEY=bar"),
            Some("TOGETHER_API_KEY")
        );
    }

    #[test]
    fn format_env_value_quotes_special_characters() {
        assert_eq!(format_env_value("abc"), "\"abc\"");
        assert_eq!(format_env_value("a\"b"), "\"a\\\"b\"");
        assert_eq!(format_env_value("a\\b"), "\"a\\\\b\"");
    }

    #[test]
    fn persist_env_value_updates_existing_key_without_dropping_others() {
        let _guard = ENV_LOCK.lock().expect("env lock should not be poisoned");
        let root = temp_root("migration_pipeline_cli_env");
        let env_path = root.join(".env");
        fs::write(
            &env_path,
            "OPENAI_API_KEY=\"old\"\n# comment\nTOGETHER_API_KEY=\"keep\"\n",
        )
        .expect(".env should be written");

        persist_env_value(&root, "OPENAI_API_KEY", "new-value").expect("env update should work");

        let contents = fs::read_to_string(&env_path).expect(".env should be readable");
        assert!(contents.contains("OPENAI_API_KEY=\"new-value\""));
        assert!(contents.contains("TOGETHER_API_KEY=\"keep\""));
        assert!(contents.contains("# comment"));

        fs::remove_dir_all(root).expect("temp root should be removed");
    }

    #[test]
    fn default_modern_root_constant_matches_requested_default() {
        assert_eq!(DEFAULT_MODERN_ROOT, "./modern_app");
    }
}
