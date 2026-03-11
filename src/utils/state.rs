use std::fs;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use tempfile::NamedTempFile;

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

    let payload =
        serde_json::to_string_pretty(tickets).context("failed to serialize migration state")?;
    let state_directory = state_directory(path);
    let mut temp_file = NamedTempFile::new_in(state_directory).with_context(|| {
        format!(
            "failed to create temporary migration state file in {}",
            state_directory.display()
        )
    })?;

    temp_file.write_all(payload.as_bytes()).with_context(|| {
        format!(
            "failed to write temporary migration state file in {}",
            state_directory.display()
        )
    })?;
    temp_file.as_file_mut().sync_all().with_context(|| {
        format!(
            "failed to sync temporary migration state file {}",
            path.display()
        )
    })?;

    temp_file
        .persist(path)
        .map_err(|error| error.error)
        .with_context(|| {
            format!(
                "failed to persist temporary migration state file to {}",
                path.display()
            )
        })?;
    Ok(())
}

fn state_directory(path: &Path) -> &Path {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
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
            test_file_paths: vec!["tests/src/server.test.ts".to_owned()],
            retries: 1,
            token_usage: crate::agents::TicketTokenUsage {
                llm_calls: 2,
                prompt_tokens: 120,
                completion_tokens: 45,
                total_tokens: 165,
            },
            last_execution_diff: None,
            last_ast_diff: None,
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

    #[test]
    fn save_state_replaces_existing_file_atomically() {
        let unique_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be monotonic")
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("migration_pipeline_state_replace_{unique_id}"));
        fs::create_dir_all(&root).expect("temp directory should be created");
        let state_path = root.join(".migration_state.json");

        let initial_tickets = vec![Ticket {
            id: "STATE-OLD".to_owned(),
            description: "Old ticket".to_owned(),
            context_files: vec!["src/old.js".to_owned()],
            status: TicketStatus::Todo,
            legacy_code_snippet: "old();".to_owned(),
            target_framework: "TypeScript".to_owned(),
            dependencies: vec![],
            modern_file_paths: Vec::new(),
            test_file_paths: Vec::new(),
            retries: 0,
            token_usage: crate::agents::TicketTokenUsage::default(),
            last_execution_diff: None,
            last_ast_diff: None,
        }];
        let replacement_tickets = vec![Ticket {
            id: "STATE-NEW".to_owned(),
            description: "New ticket".to_owned(),
            context_files: vec!["src/new.js".to_owned()],
            status: TicketStatus::Verified,
            legacy_code_snippet: "new();".to_owned(),
            target_framework: "TypeScript".to_owned(),
            dependencies: vec!["express".to_owned()],
            modern_file_paths: vec!["src/new.ts".to_owned()],
            test_file_paths: vec!["tests/src/new.test.ts".to_owned()],
            retries: 2,
            token_usage: crate::agents::TicketTokenUsage {
                llm_calls: 1,
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
            last_execution_diff: None,
            last_ast_diff: None,
        }];

        save_state(state_path.to_string_lossy().as_ref(), &initial_tickets)
            .expect("initial state should save");
        save_state(state_path.to_string_lossy().as_ref(), &replacement_tickets)
            .expect("replacement state should save");

        let loaded = load_state(state_path.to_string_lossy().as_ref())
            .expect("state should load")
            .expect("state should exist");

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, "STATE-NEW");
        assert_eq!(loaded[0].status, TicketStatus::Verified);
        assert_eq!(loaded[0].retries, 2);

        fs::remove_dir_all(root).expect("temp directory should be removed");
    }
}
