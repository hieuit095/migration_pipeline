use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rusqlite::{Connection, params};

use crate::agents::{Ticket, TicketStatus};

const CREATE_TICKETS_TABLE_SQL: &str = r#"
    CREATE TABLE IF NOT EXISTS tickets (
        id TEXT PRIMARY KEY NOT NULL,
        status TEXT NOT NULL,
        retries INTEGER NOT NULL,
        payload TEXT NOT NULL
    )
"#;

pub fn load_state(path: &str) -> Result<Option<Vec<Ticket>>> {
    let connection = open_state_connection(path)?;
    let mut statement = connection
        .prepare("SELECT payload FROM tickets ORDER BY id")
        .context("failed to prepare state load query")?;
    let payload_rows = statement
        .query_map([], |row| row.get::<_, String>(0))
        .context("failed to query persisted tickets from state database")?;

    let mut tickets = Vec::new();
    for payload_row in payload_rows {
        let payload = payload_row.context("failed to read ticket payload from state database")?;
        let ticket = serde_json::from_str::<Ticket>(&payload)
            .context("failed to deserialize ticket payload from state database")?;
        tickets.push(ticket);
    }

    if tickets.is_empty() {
        Ok(None)
    } else {
        Ok(Some(tickets))
    }
}

pub fn save_state(path: &str, ticket: &Ticket) -> Result<()> {
    let mut connection = open_state_connection(path)?;
    let payload =
        serde_json::to_string(ticket).context("failed to serialize migration ticket state")?;
    let retries = i64::from(ticket.retries);
    let status = ticket_status_label(&ticket.status);
    let transaction = connection
        .transaction()
        .context("failed to begin ticket state upsert transaction")?;

    transaction
        .execute(
            r#"
                INSERT OR REPLACE INTO tickets (id, status, retries, payload)
                VALUES (?1, ?2, ?3, ?4)
            "#,
            params![ticket.id.as_str(), status, retries, payload],
        )
        .with_context(|| {
            format!(
                "failed to upsert persisted state for ticket `{}` into SQLite",
                ticket.id
            )
        })?;
    transaction.commit().with_context(|| {
        format!(
            "failed to commit persisted state for ticket `{}`",
            ticket.id
        )
    })?;

    Ok(())
}

fn open_state_connection(path: &str) -> Result<Connection> {
    let path = resolve_state_path(Path::new(path))?;
    let state_directory = state_directory(&path)?;
    std::fs::create_dir_all(&state_directory).with_context(|| {
        format!(
            "failed to create parent directories for state database {}",
            path.display()
        )
    })?;

    let connection = Connection::open(&path)
        .with_context(|| format!("failed to open SQLite state database {}", path.display()))?;
    initialize_schema(&connection)?;

    Ok(connection)
}

fn initialize_schema(connection: &Connection) -> Result<()> {
    connection
        .execute_batch(CREATE_TICKETS_TABLE_SQL)
        .context("failed to initialize SQLite migration state schema")
}

fn resolve_state_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()
            .context("failed to resolve current working directory for state database")?
            .join(path))
    }
}

fn state_directory(path: &Path) -> Result<PathBuf> {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .with_context(|| {
            format!(
                "state database path must have a parent directory: {}",
                path.display()
            )
        })
}

fn ticket_status_label(status: &TicketStatus) -> &'static str {
    match status {
        TicketStatus::Todo => "todo",
        TicketStatus::InProgress => "in_progress",
        TicketStatus::Verified => "verified",
        TicketStatus::Failed(_) => "failed",
    }
}

#[cfg(test)]
mod tests {
    use super::{load_state, save_state};
    use crate::agents::{Ticket, TicketStatus, TicketTokenUsage};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn make_temp_state_path(prefix: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let unique_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be monotonic")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("{prefix}_{unique_id}"));
        fs::create_dir_all(&root).expect("temp directory should be created");
        let state_path = root.join(".migration_state.db");
        (root, state_path)
    }

    fn sample_ticket(id: &str) -> Ticket {
        Ticket {
            id: id.to_owned(),
            description: "Persist ticket".to_owned(),
            context_files: vec!["src/server.js".to_owned()],
            status: TicketStatus::InProgress,
            legacy_code_snippet: "http.createServer(...)".to_owned(),
            target_framework: "TypeScript".to_owned(),
            dependencies: vec!["express".to_owned()],
            modern_file_paths: vec!["src/server.ts".to_owned()],
            test_file_paths: vec!["tests/src/server.test.ts".to_owned()],
            retries: 1,
            token_usage: TicketTokenUsage {
                llm_calls: 2,
                prompt_tokens: 120,
                completion_tokens: 45,
                total_tokens: 165,
            },
            last_execution_diff: None,
            last_ast_diff: None,
        }
    }

    #[test]
    fn state_round_trips_through_sqlite() {
        let (root, state_path) = make_temp_state_path("migration_pipeline_state");
        let ticket = sample_ticket("STATE-1");

        save_state(state_path.to_string_lossy().as_ref(), &ticket).expect("state should save");
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
    fn save_state_upserts_existing_ticket_row() {
        let (root, state_path) = make_temp_state_path("migration_pipeline_state_replace");
        let initial_ticket = sample_ticket("STATE-UPSERT");
        let mut replacement_ticket = sample_ticket("STATE-UPSERT");
        replacement_ticket.status = TicketStatus::Verified;
        replacement_ticket.retries = 2;
        replacement_ticket.description = "Updated ticket".to_owned();

        save_state(state_path.to_string_lossy().as_ref(), &initial_ticket)
            .expect("initial state should save");
        save_state(state_path.to_string_lossy().as_ref(), &replacement_ticket)
            .expect("replacement state should save");

        let loaded = load_state(state_path.to_string_lossy().as_ref())
            .expect("state should load")
            .expect("state should exist");

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, "STATE-UPSERT");
        assert_eq!(loaded[0].status, TicketStatus::Verified);
        assert_eq!(loaded[0].retries, 2);
        assert_eq!(loaded[0].description, "Updated ticket");

        fs::remove_dir_all(root).expect("temp directory should be removed");
    }

    #[test]
    fn load_state_creates_empty_database_when_missing() {
        let (root, state_path) = make_temp_state_path("migration_pipeline_state_init");

        let loaded = load_state(state_path.to_string_lossy().as_ref()).expect("state should load");

        assert!(loaded.is_none());
        assert!(state_path.exists());

        fs::remove_dir_all(root).expect("temp directory should be removed");
    }
}
