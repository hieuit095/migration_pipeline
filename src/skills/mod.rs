use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Component, Path, PathBuf};
use walkdir::{DirEntry, WalkDir};

mod sandbox;

pub use sandbox::SandboxSkill;

pub trait Skill: Send + Sync {
    fn name(&self) -> &str;
    fn execute(&self, args: Vec<String>) -> Result<String>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct FileIOSkill;
#[derive(Debug, Default, Clone, Copy)]
pub struct FileWriteSkill;
pub struct ASTParsingSkill;

#[derive(Debug, Serialize, Deserialize)]
pub struct TerminalCommandOutput {
    pub command: String,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
}

impl FileIOSkill {
    fn should_descend(entry: &DirEntry) -> bool {
        if !entry.file_type().is_dir() {
            return true;
        }

        if entry.depth() == 0 {
            return true;
        }

        let directory_name = entry.file_name().to_string_lossy();

        !directory_name.eq_ignore_ascii_case("node_modules")
            && !directory_name.eq_ignore_ascii_case("target")
            && !directory_name.eq_ignore_ascii_case(".git")
    }

    fn read_text_file(path: &Path) -> Result<Option<String>> {
        let bytes =
            fs::read(path).with_context(|| format!("failed to read file {}", path.display()))?;

        if bytes.contains(&0) {
            return Ok(None);
        }

        match String::from_utf8(bytes) {
            Ok(contents) => Ok(Some(contents)),
            Err(_) => Ok(None),
        }
    }

    fn normalize_path(root: &Path, path: &Path) -> String {
        path.strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/")
    }

    fn format_text_file(root: &Path, path: &Path) -> Result<Option<String>> {
        Self::read_text_file(path).map(|contents| {
            contents.map(|contents| {
                let relative_path = Self::normalize_path(root, path);
                format!("// File: {relative_path}\n{contents}")
            })
        })
    }

    fn collect_directory(root_path: &Path) -> Result<Vec<String>> {
        let mut files: Vec<PathBuf> = WalkDir::new(root_path)
            .into_iter()
            .filter_entry(Self::should_descend)
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().is_file())
            .map(|entry| entry.into_path())
            .collect();

        files.sort();

        let mut aggregated_sections = Vec::new();
        for path in files {
            if let Some(section) = Self::format_text_file(root_path, &path)? {
                aggregated_sections.push(section);
            }
        }

        Ok(aggregated_sections)
    }

    fn collect_selected_files(root_path: &Path, relative_paths: &[String]) -> Result<Vec<String>> {
        let mut aggregated_sections = Vec::new();

        for relative_path in relative_paths {
            let relative = Path::new(relative_path);
            if relative.is_absolute() {
                return Err(anyhow!(
                    "context file paths must be relative to the legacy directory: {relative_path}"
                ));
            }

            if relative
                .components()
                .any(|component| matches!(component, Component::ParentDir))
            {
                return Err(anyhow!(
                    "context file paths must stay within the legacy directory: {relative_path}"
                ));
            }

            let full_path = root_path.join(relative);
            if !full_path.exists() {
                return Err(anyhow!(
                    "context file does not exist: {}",
                    full_path.display()
                ));
            }

            if full_path.is_dir() {
                return Err(anyhow!(
                    "context file path points to a directory, expected a file: {relative_path}"
                ));
            }

            let section = Self::format_text_file(root_path, &full_path)?.ok_or_else(|| {
                anyhow!("context file is not a readable text file: {relative_path}")
            })?;
            aggregated_sections.push(section);
        }

        Ok(aggregated_sections)
    }
}

impl Skill for FileIOSkill {
    fn name(&self) -> &str {
        "file_io"
    }

    fn execute(&self, args: Vec<String>) -> Result<String> {
        let root_arg = args
            .first()
            .context("FileIOSkill expects a directory path as the first argument")?;
        let root_path = Path::new(root_arg);

        if !root_path.exists() {
            return Err(anyhow!(
                "legacy directory does not exist: {}",
                root_path.display()
            ));
        }

        if !root_path.is_dir() {
            return Err(anyhow!(
                "FileIOSkill expects a directory, received: {}",
                root_path.display()
            ));
        }

        let aggregated_sections = if args.len() == 1 {
            Self::collect_directory(root_path)?
        } else {
            Self::collect_selected_files(root_path, &args[1..])?
        };

        if aggregated_sections.is_empty() {
            return Err(anyhow!(
                "no readable text files found in legacy directory {}",
                root_path.display()
            ));
        }

        Ok(aggregated_sections.join("\n\n"))
    }
}

impl Skill for FileWriteSkill {
    fn name(&self) -> &str {
        "file_write"
    }

    fn execute(&self, args: Vec<String>) -> Result<String> {
        let target_path = args
            .first()
            .context("FileWriteSkill expects the target path as the first argument")?;
        let file_content = args
            .get(1)
            .context("FileWriteSkill expects the file content as the second argument")?;
        let target_path = Path::new(target_path);

        if let Some(parent) = target_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).with_context(|| {
                    format!(
                        "failed to create parent directories for {}",
                        target_path.display()
                    )
                })?;
            }
        }

        fs::write(target_path, file_content).with_context(|| {
            format!(
                "failed to write generated file to {}",
                target_path.display()
            )
        })?;

        Ok(target_path.to_string_lossy().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::{FileIOSkill, FileWriteSkill, Skill};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn file_io_skill_reads_text_files_and_skips_ignored_directories() {
        let unique_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be monotonic")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("migration_pipeline_file_io_{unique_id}"));

        fs::create_dir_all(root.join("src")).expect("should create src dir");
        fs::create_dir_all(root.join("node_modules")).expect("should create node_modules dir");
        fs::write(root.join("src/app.js"), "console.log('hello');").expect("should write text");
        fs::write(root.join("node_modules/ignored.js"), "module.exports = {};")
            .expect("should write ignored text");
        fs::write(root.join("binary.bin"), [0, 159, 146, 150]).expect("should write binary");

        let output = FileIOSkill
            .execute(vec![root.to_string_lossy().to_string()])
            .expect("skill should succeed");

        assert!(output.contains("// File: src/app.js"));
        assert!(output.contains("console.log('hello');"));
        assert!(!output.contains("ignored.js"));
        assert!(!output.contains("binary.bin"));

        fs::remove_dir_all(root).expect("should clean up temp directory");
    }

    #[test]
    fn file_io_skill_can_read_selected_context_files() {
        let unique_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be monotonic")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("migration_pipeline_context_io_{unique_id}"));

        fs::create_dir_all(root.join("src")).expect("should create src dir");
        fs::write(root.join("src/app.js"), "console.log('app');").expect("should write app");
        fs::write(root.join("src/db.js"), "module.exports = {};").expect("should write db");

        let output = FileIOSkill
            .execute(vec![
                root.to_string_lossy().to_string(),
                "src/db.js".to_owned(),
            ])
            .expect("skill should read selected files");

        assert!(output.contains("// File: src/db.js"));
        assert!(!output.contains("src/app.js"));

        fs::remove_dir_all(root).expect("should clean up temp directory");
    }

    #[test]
    fn file_write_skill_creates_missing_directories_and_writes_file() {
        let unique_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be monotonic")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("migration_pipeline_file_write_{unique_id}"));
        let target_path = root.join("modern_app/src/generated.ts");

        FileWriteSkill
            .execute(vec![
                target_path.to_string_lossy().to_string(),
                "export const value = 1;".to_owned(),
            ])
            .expect("file write skill should succeed");

        let written = fs::read_to_string(&target_path).expect("generated file should exist");
        assert_eq!(written, "export const value = 1;");

        fs::remove_dir_all(root).expect("should clean up temp directory");
    }
}
