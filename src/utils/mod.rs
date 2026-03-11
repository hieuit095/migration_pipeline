pub mod state;

use anyhow::Result;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::Layer;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

pub fn setup_telemetry() -> Result<WorkerGuard> {
    std::fs::create_dir_all("logs")?;

    let file_appender = tracing_appender::rolling::daily("logs", "pipeline.log");
    let (file_writer, file_guard) = tracing_appender::non_blocking(file_appender);

    let stdout_layer = fmt::layer()
        .with_ansi(true)
        .with_target(true)
        .with_filter(LevelFilter::INFO);
    let file_layer = fmt::layer()
        .with_ansi(false)
        .with_target(true)
        .with_writer(file_writer)
        .with_filter(LevelFilter::DEBUG);

    tracing_subscriber::registry()
        .with(stdout_layer)
        .with(file_layer)
        .try_init()?;

    Ok(file_guard)
}

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("Agent failed: {0}")]
    AgentFailure(String),
    #[error("Semantic mismatch found")]
    SemanticMismatch,
}

pub fn clean_json_response(raw: &str) -> String {
    let trimmed = raw.trim();
    let without_fence = strip_markdown_fence(trimmed);
    let candidate = without_fence.trim();

    if looks_like_json(candidate) {
        return candidate.to_owned();
    }

    extract_json_block(candidate)
        .map(str::trim)
        .unwrap_or(candidate)
        .to_owned()
}

pub fn clean_code_response(raw: &str) -> String {
    let extracted = extract_fenced_block(raw).unwrap_or(raw).trim();
    let trimmed = strip_markdown_fence(extracted).trim();
    let mut lines = trimmed.lines();

    let mut cleaned = match lines.next() {
        Some(first_line)
            if is_generated_path_header(first_line.trim_start())
                && lines.clone().next().is_some() =>
        {
            lines.collect::<Vec<_>>().join("\n")
        }
        _ => trimmed.to_owned(),
    };

    cleaned = strip_conversational_preamble(&cleaned);
    cleaned = strip_conversational_epilogue(&cleaned);

    cleaned.trim().to_owned()
}

fn strip_markdown_fence(input: &str) -> &str {
    let trimmed = input.trim();

    if let Some(stripped) = trimmed.strip_prefix("```") {
        let stripped = match stripped.find('\n') {
            Some(newline_index) => &stripped[newline_index + 1..],
            None => stripped,
        };

        return stripped.trim().trim_end_matches("```").trim();
    }

    trimmed
}

fn extract_fenced_block(input: &str) -> Option<&str> {
    let start = input.find("```")?;
    let rest = &input[start + 3..];
    let line_break = rest.find('\n')?;
    let content_start = start + 3 + line_break + 1;
    let remainder = &input[content_start..];
    let end = remainder.find("```")?;
    Some(&remainder[..end])
}

fn looks_like_json(value: &str) -> bool {
    let starts_like_json = value.starts_with('{') || value.starts_with('[');
    let ends_like_json = value.ends_with('}') || value.ends_with(']');

    starts_like_json && ends_like_json
}

fn extract_json_block(input: &str) -> Option<&str> {
    let object_block = slice_between(input, '{', '}');
    let array_block = slice_between(input, '[', ']');

    match (object_block, array_block) {
        (Some(object), Some(array)) => {
            if object.len() >= array.len() {
                Some(object)
            } else {
                Some(array)
            }
        }
        (Some(object), None) => Some(object),
        (None, Some(array)) => Some(array),
        (None, None) => None,
    }
}

fn slice_between(input: &str, open: char, close: char) -> Option<&str> {
    let start = input.find(open)?;
    let end = input.rfind(close)?;

    (start < end).then_some(&input[start..=end])
}

fn is_generated_path_header(line: &str) -> bool {
    let normalized = line.to_ascii_lowercase();
    normalized.starts_with("// file:")
        || normalized.starts_with("# file:")
        || normalized.starts_with("// path:")
        || normalized.starts_with("# path:")
}

fn strip_conversational_preamble(input: &str) -> String {
    let lines: Vec<&str> = input.lines().collect();
    let first_code_index = lines
        .iter()
        .position(|line| !is_probable_preamble_line(line.trim()))
        .unwrap_or(0);

    lines[first_code_index..].join("\n")
}

fn strip_conversational_epilogue(input: &str) -> String {
    let lines: Vec<&str> = input.lines().collect();
    let last_code_index = lines
        .iter()
        .rposition(|line| !is_probable_epilogue_line(line.trim()))
        .map(|index| index + 1)
        .unwrap_or(lines.len());

    lines[..last_code_index].join("\n")
}

fn is_probable_preamble_line(line: &str) -> bool {
    if line.is_empty() {
        return true;
    }

    let normalized = line.to_ascii_lowercase();
    is_generated_path_header(line)
        || normalized.starts_with("here is")
        || normalized.starts_with("here's")
        || normalized.starts_with("below is")
        || normalized.starts_with("corrected code")
        || normalized.starts_with("updated code")
        || normalized.starts_with("fixed code")
        || normalized.starts_with("i fixed")
        || normalized.starts_with("i have fixed")
        || normalized.starts_with("the corrected")
        || normalized.starts_with("the updated")
        || normalized.starts_with("sure")
        || normalized.starts_with("certainly")
}

fn is_probable_epilogue_line(line: &str) -> bool {
    if line.is_empty() {
        return true;
    }

    let normalized = line.to_ascii_lowercase();
    normalized.starts_with("let me know")
        || normalized.starts_with("if you want")
        || normalized.starts_with("if you'd like")
        || normalized.starts_with("this should")
}

#[cfg(test)]
mod tests {
    use super::{clean_code_response, clean_json_response};

    #[test]
    fn clean_json_response_strips_markdown_fences() {
        let raw = "```json\n{\"tickets\":[]}\n```";
        assert_eq!(clean_json_response(raw), "{\"tickets\":[]}");
    }

    #[test]
    fn clean_json_response_extracts_json_from_wrapped_text() {
        let raw = "Here is the blueprint:\n{\"tickets\":[{\"id\":\"BP-1\"}]}";
        assert_eq!(
            clean_json_response(raw),
            "{\"tickets\":[{\"id\":\"BP-1\"}]}"
        );
    }

    #[test]
    fn clean_code_response_strips_fences_and_generated_file_headers() {
        let raw = "```ts\n// FILE: src/server.ts\nexport const app = {};\n```";
        assert_eq!(clean_code_response(raw), "export const app = {};");
    }

    #[test]
    fn clean_code_response_strips_conversational_wrapper_text() {
        let raw = "Here is the corrected code:\n```ts\nexport const app = {};\n```\nLet me know if you want tests.";
        assert_eq!(clean_code_response(raw), "export const app = {};");
    }
}
