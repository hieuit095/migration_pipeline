use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::agents::Ticket;

pub fn load_state(path: &str) -> Result<Option<Vec<Ticket>>> {
    let path = Path::new(path);

    if !path.exists() {
        return Ok(None);
    }

    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read migration state from {}", path.display()))?;
    let tickets = serde_json::from_str::<Vec<Ticket>>(&contents)
        .with_context(|| format!("failed to parse migration state from {}", path.display()))?;

    Ok(Some(tickets))
}

pub fn save_state(path: &str, tickets: &[Ticket]) -> Result<()> {
    let path = Path::new(path);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create parent directories for state file {}",
                    path.display()
                )
            })?;
        }
    }

    let temp_path = temporary_state_path(path);
    let payload =
        serde_json::to_string_pretty(tickets).context("failed to serialize migration state")?;

    fs::write(&temp_path, payload).with_context(|| {
        format!(
            "failed to write temporary migration state file {}",
            temp_path.display()
        )
    })?;

    rename_atomically(&temp_path, path)?;
    Ok(())
}

fn temporary_state_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("migration_state");
    path.with_file_name(format!("{file_name}.tmp"))
}

fn rename_atomically(temp_path: &Path, target_path: &Path) -> Result<()> {
    match fs::rename(temp_path, target_path) {
        Ok(()) => Ok(()),
        Err(rename_error) if target_path.exists() => {
            fs::remove_file(target_path).with_context(|| {
                format!(
                    "failed to replace existing migration state file {}",
                    target_path.display()
                )
            })?;
            fs::rename(temp_path, target_path).with_context(|| {
                format!(
                    "failed to rename temporary migration state file {} to {} after replacement attempt: {}",
                    temp_path.display(),
                    target_path.display(),
                    rename_error
                )
            })
        }
        Err(error) => Err(error).with_context(|| {
            format!(
                "failed to rename temporary migration state file {} to {}",
                temp_path.display(),
                target_path.display()
            )
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::{load_state, save_state};
    use crate::agents::{Ticket, TicketStatus};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn state_round_trips_through_disk() {
        let unique_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be monotonic")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("migration_pipeline_state_{unique_id}"));
        fs::create_dir_all(&root).expect("temp directory should be created");
        let state_path = root.join(".migration_state.json");

        let tickets = vec![Ticket {
            id: "STATE-1".to_owned(),
            description: "Persist ticket".to_owned(),
            context_files: vec!["src/server.js".to_owned()],
            status: TicketStatus::InProgress,
            legacy_code_snippet: "http.createServer(...)".to_owned(),
            target_framework: "TypeScript".to_owned(),
            dependencies: vec!["express".to_owned()],
            modern_file_paths: vec!["src/server.ts".to_owned()],
            retries: 1,
            token_usage: crate::agents::TicketTokenUsage {
                llm_calls: 2,
                prompt_tokens: 120,
                completion_tokens: 45,
                total_tokens: 165,
            },
        }];

        save_state(state_path.to_string_lossy().as_ref(), &tickets).expect("state should save");
        let loaded = load_state(state_path.to_string_lossy().as_ref())
            .expect("state should load")
            .expect("state should exist");

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, "STATE-1");
        assert_eq!(loaded[0].retries, 1);
        assert_eq!(loaded[0].token_usage.total_tokens, 165);

        fs::remove_dir_all(root).expect("temp directory should be removed");
    }
}
