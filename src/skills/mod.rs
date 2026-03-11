use crate::utils::path::{normalize_relative_path, relative_path_from_root};
use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use tokio::{fs as tokio_fs, task};
use tree_sitter::{Language, Node, Parser};
use walkdir::{DirEntry, WalkDir};

mod sandbox;
mod shadow;

pub use sandbox::SandboxSkill;
pub use shadow::{
    ExecutionDiff, ExecutionTarget, ShadowFixture, ShadowTestRequest, ShadowTestSkill,
};

#[async_trait]
pub trait Skill: Send + Sync {
    fn name(&self) -> &str;
    async fn execute(&self, args: Vec<String>) -> Result<String>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct FileIOSkill;
#[derive(Debug, Default, Clone, Copy)]
pub struct FileWriteSkill;
#[derive(Debug, Default, Clone, Copy)]
pub struct ASTParsingSkill;

#[derive(Debug, Serialize, Deserialize)]
pub struct TerminalCommandOutput {
    pub command: String,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct DependencyGraph {
    pub files: Vec<FileDependencyNode>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct FileDependencyNode {
    pub path: String,
    pub language: String,
    pub imports: Vec<ImportEdge>,
    pub functions: Vec<FunctionSignature>,
    pub classes: Vec<ClassDefinition>,
    pub variables: Vec<VariableBinding>,
    pub branches: Vec<BranchDescriptor>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ImportEdge {
    pub source: String,
    pub names: Vec<String>,
    pub kind: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct FunctionSignature {
    pub name: String,
    pub parameters: Vec<String>,
    pub is_async: bool,
    pub is_generator: bool,
    pub return_type: Option<String>,
    pub exported: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ClassDefinition {
    pub name: String,
    pub extends: Option<String>,
    pub methods: Vec<MethodSignature>,
    pub exported: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct MethodSignature {
    pub name: String,
    pub parameters: Vec<String>,
    pub is_async: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct VariableBinding {
    pub name: String,
    pub declaration_kind: String,
    pub value_kind: Option<String>,
    pub exported: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct BranchDescriptor {
    pub kind: String,
    pub condition: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct AstDiff {
    pub missing_functions: Vec<String>,
    pub missing_classes: Vec<String>,
    pub missing_imports: Vec<String>,
    pub missing_branches: Vec<String>,
    pub unsupported_files: Vec<String>,
    pub notes: Vec<String>,
}

impl AstDiff {
    pub fn is_empty(&self) -> bool {
        self.missing_functions.is_empty()
            && self.missing_classes.is_empty()
            && self.missing_imports.is_empty()
            && self.missing_branches.is_empty()
            && self.unsupported_files.is_empty()
            && self.notes.is_empty()
    }
}

#[derive(Debug, Clone, Copy)]
enum SourceLanguage {
    JavaScript,
    TypeScript,
    Tsx,
}

impl SourceLanguage {
    fn from_path(path: &Path) -> Option<Self> {
        match path.extension().and_then(|value| value.to_str()) {
            Some("js" | "mjs" | "cjs" | "jsx") => Some(Self::JavaScript),
            Some("ts") => Some(Self::TypeScript),
            Some("tsx") => Some(Self::Tsx),
            _ => None,
        }
    }

    fn language(self) -> Language {
        match self {
            Self::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            Self::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Self::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::JavaScript => "javascript",
            Self::TypeScript => "typescript",
            Self::Tsx => "tsx",
        }
    }
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
        relative_path_from_root(root, path).into_string()
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
            let full_path = validate_relative_file(root_path, relative_path)?;
            let section = Self::format_text_file(root_path, &full_path)?.ok_or_else(|| {
                anyhow!("context file is not a readable text file: {relative_path}")
            })?;
            aggregated_sections.push(section);
        }

        Ok(aggregated_sections)
    }
}

impl ASTParsingSkill {
    fn collect_directory(root_path: &Path) -> Result<Vec<FileDependencyNode>> {
        let mut files: Vec<PathBuf> = WalkDir::new(root_path)
            .into_iter()
            .filter_entry(FileIOSkill::should_descend)
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().is_file())
            .map(|entry| entry.into_path())
            .filter(|path| SourceLanguage::from_path(path).is_some())
            .collect();

        files.sort();
        Self::parse_files(root_path, files)
    }

    fn collect_selected_files(
        root_path: &Path,
        relative_paths: &[String],
    ) -> Result<Vec<FileDependencyNode>> {
        let mut files = Vec::with_capacity(relative_paths.len());

        for relative_path in relative_paths {
            let full_path = validate_relative_file(root_path, relative_path)?;
            if SourceLanguage::from_path(&full_path).is_none() {
                bail!(
                    "ASTParsingSkill supports only JavaScript/TypeScript files, received `{relative_path}`"
                );
            }
            files.push(full_path);
        }

        Self::parse_files(root_path, files)
    }

    fn parse_files(root_path: &Path, files: Vec<PathBuf>) -> Result<Vec<FileDependencyNode>> {
        let mut parsed_files = Vec::new();

        for path in files {
            let Some(language) = SourceLanguage::from_path(&path) else {
                continue;
            };
            let Some(contents) = FileIOSkill::read_text_file(&path)? else {
                continue;
            };
            parsed_files.push(Self::parse_file(root_path, &path, &contents, language)?);
        }

        Ok(parsed_files)
    }

    fn parse_file(
        root_path: &Path,
        path: &Path,
        source: &str,
        language: SourceLanguage,
    ) -> Result<FileDependencyNode> {
        let mut parser = Parser::new();
        parser
            .set_language(&language.language())
            .with_context(|| format!("failed to load {} parser", language.as_str()))?;
        let tree = parser
            .parse(source, None)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let root = tree.root_node();

        let mut summary = FileDependencyNode {
            path: FileIOSkill::normalize_path(root_path, path),
            language: language.as_str().to_owned(),
            imports: Vec::new(),
            functions: Vec::new(),
            classes: Vec::new(),
            variables: Vec::new(),
            branches: Vec::new(),
        };

        let mut cursor = root.walk();
        for child in root.named_children(&mut cursor) {
            Self::process_top_level_node(child, source, false, &mut summary);
        }
        Self::collect_branches(root, source, &mut summary.branches);
        summary.branches.sort();
        summary.branches.dedup();

        Ok(summary)
    }

    fn process_top_level_node(
        node: Node<'_>,
        source: &str,
        exported: bool,
        summary: &mut FileDependencyNode,
    ) {
        match node.kind() {
            "import_statement" => {
                if let Some(import_edge) = Self::extract_import(node, source) {
                    summary.imports.push(import_edge);
                }
            }
            "function_declaration" => {
                if let Some(signature) = Self::extract_function(node, source, exported) {
                    summary.functions.push(signature);
                }
            }
            "class_declaration" => {
                if let Some(class_definition) = Self::extract_class(node, source, exported) {
                    summary.classes.push(class_definition);
                }
            }
            "lexical_declaration" | "variable_declaration" => {
                summary
                    .variables
                    .extend(Self::extract_variables(node, source, exported));
            }
            "export_statement" => {
                let mut cursor = node.walk();
                for child in node.named_children(&mut cursor) {
                    if child.kind() == "export_clause" {
                        continue;
                    }
                    Self::process_top_level_node(child, source, true, summary);
                }
            }
            _ => {}
        }
    }

    fn extract_import(node: Node<'_>, source: &str) -> Option<ImportEdge> {
        let source_node = node.child_by_field_name("source")?;
        let source_text = node_text(source_node, source)?;
        let import_source = trim_quotes(&source_text).to_owned();
        let mut names = Vec::new();
        let mut cursor = node.walk();

        for child in node.named_children(&mut cursor) {
            if child.id() == source_node.id() {
                continue;
            }
            collect_identifier_texts(child, source, &mut names);
        }

        let kind = if names.is_empty() {
            "side_effect"
        } else if node_text(node, source).is_some_and(|text| text.contains('*')) {
            "namespace"
        } else {
            "named"
        };

        Some(ImportEdge {
            source: import_source,
            names,
            kind: kind.to_owned(),
        })
    }

    fn extract_function(node: Node<'_>, source: &str, exported: bool) -> Option<FunctionSignature> {
        let name = node
            .child_by_field_name("name")
            .and_then(|value| node_text(value, source))?;
        let parameters = node
            .child_by_field_name("parameters")
            .map(|value| collect_parameter_names(value, source))
            .unwrap_or_default();
        let return_type = node
            .child_by_field_name("return_type")
            .and_then(|value| node_text(value, source))
            .map(|value| value.trim().to_owned());
        let node_text = node_text(node, source)?;

        Some(FunctionSignature {
            name,
            parameters,
            is_async: node_text.trim_start().starts_with("async "),
            is_generator: node_text.contains("function*"),
            return_type,
            exported,
        })
    }

    fn extract_class(node: Node<'_>, source: &str, exported: bool) -> Option<ClassDefinition> {
        let name = node
            .child_by_field_name("name")
            .and_then(|value| node_text(value, source))?;
        let extends = node
            .child_by_field_name("superclass")
            .and_then(|value| node_text(value, source));
        let methods = node
            .child_by_field_name("body")
            .map(|body| Self::extract_methods(body, source))
            .unwrap_or_default();

        Some(ClassDefinition {
            name,
            extends,
            methods,
            exported,
        })
    }

    fn extract_methods(node: Node<'_>, source: &str) -> Vec<MethodSignature> {
        let mut methods = Vec::new();
        let mut cursor = node.walk();

        for child in node.named_children(&mut cursor) {
            if !matches!(child.kind(), "method_definition" | "method_signature") {
                continue;
            }

            let Some(name) = child
                .child_by_field_name("name")
                .and_then(|value| node_text(value, source))
            else {
                continue;
            };
            let parameters = child
                .child_by_field_name("parameters")
                .map(|value| collect_parameter_names(value, source))
                .unwrap_or_default();
            let is_async = node_text(child, source)
                .is_some_and(|value| value.trim_start().starts_with("async "));

            methods.push(MethodSignature {
                name,
                parameters,
                is_async,
            });
        }

        methods
    }

    fn extract_variables(node: Node<'_>, source: &str, exported: bool) -> Vec<VariableBinding> {
        let declaration_kind = node
            .child(0)
            .and_then(|value| node_text(value, source))
            .unwrap_or_else(|| "unknown".to_owned());
        let mut variables = Vec::new();
        let mut cursor = node.walk();

        for child in node.named_children(&mut cursor) {
            if child.kind() != "variable_declarator" {
                continue;
            }

            let Some(name) = child
                .child_by_field_name("name")
                .and_then(|value| node_text(value, source))
            else {
                continue;
            };
            let value_kind = child
                .child_by_field_name("value")
                .map(|value| value.kind().to_owned());

            variables.push(VariableBinding {
                name,
                declaration_kind: declaration_kind.trim().to_owned(),
                value_kind,
                exported,
            });
        }

        variables
    }

    fn collect_branches(node: Node<'_>, source: &str, branches: &mut Vec<BranchDescriptor>) {
        if let Some(branch) = Self::extract_branch(node, source) {
            branches.push(branch);
        }

        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            Self::collect_branches(child, source, branches);
        }
    }

    fn extract_branch(node: Node<'_>, source: &str) -> Option<BranchDescriptor> {
        match node.kind() {
            "if_statement" => Some(BranchDescriptor {
                kind: "if".to_owned(),
                condition: node
                    .child_by_field_name("condition")
                    .and_then(|value| node_text(value, source))
                    .unwrap_or_else(|| "if".to_owned()),
            }),
            "switch_statement" => Some(BranchDescriptor {
                kind: "switch".to_owned(),
                condition: node
                    .child_by_field_name("value")
                    .and_then(|value| node_text(value, source))
                    .unwrap_or_else(|| "switch".to_owned()),
            }),
            "conditional_expression" => Some(BranchDescriptor {
                kind: "ternary".to_owned(),
                condition: node
                    .child_by_field_name("condition")
                    .and_then(|value| node_text(value, source))
                    .unwrap_or_else(|| "ternary".to_owned()),
            }),
            "try_statement" => Some(BranchDescriptor {
                kind: "try".to_owned(),
                condition: "try/catch".to_owned(),
            }),
            "for_statement" | "while_statement" | "do_statement" => Some(BranchDescriptor {
                kind: node.kind().to_owned(),
                condition: node
                    .child_by_field_name("condition")
                    .and_then(|value| node_text(value, source))
                    .unwrap_or_else(|| node.kind().to_owned()),
            }),
            _ => None,
        }
    }
}

async fn spawn_blocking_result<T, F>(operation: &'static str, work: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    task::spawn_blocking(work)
        .await
        .map_err(|error| anyhow!("blocking task `{operation}` failed to join: {error}"))?
}

#[async_trait]
impl Skill for FileIOSkill {
    fn name(&self) -> &str {
        "file_io"
    }

    async fn execute(&self, args: Vec<String>) -> Result<String> {
        spawn_blocking_result("file_io", move || {
            let root_arg = args
                .first()
                .context("FileIOSkill expects a directory path as the first argument")?;
            let root_path = PathBuf::from(root_arg);

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
                Self::collect_directory(&root_path)?
            } else {
                Self::collect_selected_files(&root_path, &args[1..])?
            };

            if aggregated_sections.is_empty() {
                return Err(anyhow!(
                    "no readable text files found in legacy directory {}",
                    root_path.display()
                ));
            }

            Ok(aggregated_sections.join("\n\n"))
        })
        .await
    }
}

#[async_trait]
impl Skill for ASTParsingSkill {
    fn name(&self) -> &str {
        "ast_parsing"
    }

    async fn execute(&self, args: Vec<String>) -> Result<String> {
        spawn_blocking_result("ast_parsing", move || {
            let root_arg = args
                .first()
                .context("ASTParsingSkill expects a directory path as the first argument")?;
            let root_path = PathBuf::from(root_arg);

            if !root_path.exists() {
                return Err(anyhow!(
                    "legacy directory does not exist: {}",
                    root_path.display()
                ));
            }

            if !root_path.is_dir() {
                return Err(anyhow!(
                    "ASTParsingSkill expects a directory, received: {}",
                    root_path.display()
                ));
            }

            let files = if args.len() == 1 {
                Self::collect_directory(&root_path)?
            } else {
                Self::collect_selected_files(&root_path, &args[1..])?
            };

            if files.is_empty() {
                return Err(anyhow!(
                    "no supported JavaScript/TypeScript source files found in legacy directory {}",
                    root_path.display()
                ));
            }

            serde_json::to_string(&DependencyGraph { files })
                .context("failed to serialize AST dependency graph")
        })
        .await
    }
}

#[async_trait]
impl Skill for FileWriteSkill {
    fn name(&self) -> &str {
        "file_write"
    }

    async fn execute(&self, args: Vec<String>) -> Result<String> {
        let target_path = args
            .first()
            .context("FileWriteSkill expects the target path as the first argument")?;
        let file_content = args
            .get(1)
            .context("FileWriteSkill expects the file content as the second argument")?;
        let target_path = PathBuf::from(target_path);

        if let Some(parent) = target_path.parent() {
            if !parent.as_os_str().is_empty() {
                tokio_fs::create_dir_all(parent).await.with_context(|| {
                    format!(
                        "failed to create parent directories for {}",
                        target_path.display()
                    )
                })?;
            }
        }

        tokio_fs::write(&target_path, file_content)
            .await
            .with_context(|| {
                format!(
                    "failed to write generated file to {}",
                    target_path.display()
                )
            })?;

        Ok(target_path.to_string_lossy().to_string())
    }
}

pub fn parse_dependency_graph_json(input: &str) -> Result<DependencyGraph> {
    serde_json::from_str(input).context("failed to deserialize dependency graph JSON")
}

pub fn diff_dependency_graphs(legacy: &DependencyGraph, modern: &DependencyGraph) -> AstDiff {
    let legacy_functions = collect_function_signatures(legacy);
    let modern_functions = collect_function_signatures(modern);
    let legacy_classes = collect_class_signatures(legacy);
    let modern_classes = collect_class_signatures(modern);
    let legacy_imports = collect_import_signatures(legacy);
    let modern_imports = collect_import_signatures(modern);
    let legacy_branches = collect_branch_signatures(legacy);
    let modern_branches = collect_branch_signatures(modern);

    AstDiff {
        missing_functions: legacy_functions
            .difference(&modern_functions)
            .cloned()
            .collect(),
        missing_classes: legacy_classes
            .difference(&modern_classes)
            .cloned()
            .collect(),
        missing_imports: legacy_imports
            .difference(&modern_imports)
            .cloned()
            .collect(),
        missing_branches: legacy_branches
            .difference(&modern_branches)
            .cloned()
            .collect(),
        unsupported_files: Vec::new(),
        notes: Vec::new(),
    }
}

fn validate_relative_file(root_path: &Path, relative_path: &str) -> Result<PathBuf> {
    let relative = normalize_relative_path(relative_path, "context file path")?;
    let full_path = root_path.join(relative.as_std_path());
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

    Ok(full_path)
}

fn node_text(node: Node<'_>, source: &str) -> Option<String> {
    source.get(node.byte_range()).map(ToOwned::to_owned)
}

fn trim_quotes(value: &str) -> &str {
    value.trim().trim_matches('"').trim_matches('\'')
}

fn collect_parameter_names(node: Node<'_>, source: &str) -> Vec<String> {
    let mut parameters = Vec::new();
    collect_identifier_texts(node, source, &mut parameters);
    parameters
}

fn collect_identifier_texts(node: Node<'_>, source: &str, output: &mut Vec<String>) {
    match node.kind() {
        "identifier"
        | "property_identifier"
        | "type_identifier"
        | "shorthand_property_identifier_pattern" => {
            if let Some(identifier) = node_text(node, source) {
                if !output.contains(&identifier) {
                    output.push(identifier);
                }
            }
        }
        _ => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                collect_identifier_texts(child, source, output);
            }
        }
    }
}

fn collect_function_signatures(graph: &DependencyGraph) -> BTreeSet<String> {
    graph
        .files
        .iter()
        .flat_map(|file| {
            file.functions.iter().map(move |function| {
                format!(
                    "{}::{}({})",
                    file.path,
                    function.name,
                    function.parameters.join(",")
                )
            })
        })
        .collect()
}

fn collect_class_signatures(graph: &DependencyGraph) -> BTreeSet<String> {
    graph
        .files
        .iter()
        .flat_map(|file| {
            file.classes
                .iter()
                .map(move |class_definition| format!("{}::{}", file.path, class_definition.name))
        })
        .collect()
}

fn collect_import_signatures(graph: &DependencyGraph) -> BTreeSet<String> {
    graph
        .files
        .iter()
        .flat_map(|file| {
            file.imports.iter().map(move |import_edge| {
                format!(
                    "{}::{}::{}",
                    file.path, import_edge.kind, import_edge.source
                )
            })
        })
        .collect()
}

fn collect_branch_signatures(graph: &DependencyGraph) -> BTreeSet<String> {
    graph
        .files
        .iter()
        .flat_map(|file| {
            file.branches
                .iter()
                .map(move |branch| format!("{}::{}::{}", file.path, branch.kind, branch.condition))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{ASTParsingSkill, FileIOSkill, FileWriteSkill, Skill};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[tokio::test]
    async fn ast_parsing_skill_builds_dependency_graph_for_typescript() {
        let unique_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be monotonic")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("migration_pipeline_ast_{unique_id}"));

        fs::create_dir_all(root.join("src")).expect("should create src dir");
        fs::write(
            root.join("src/app.ts"),
            r#"
                import { serve } from "./server";
                const version = "1.0.0";
                export class AppService {
                    async run(port: number) {
                        return serve(port);
                    }
                }
                export async function bootstrap(port: number): Promise<void> {
                    return serve(port);
                }
            "#,
        )
        .expect("should write fixture");

        let output = ASTParsingSkill
            .execute(vec![root.to_string_lossy().to_string()])
            .await
            .expect("ASTParsingSkill should succeed");
        let parsed: serde_json::Value =
            serde_json::from_str(&output).expect("dependency graph should be valid JSON");

        assert_eq!(parsed["files"][0]["path"], "src/app.ts");
        assert_eq!(parsed["files"][0]["imports"][0]["source"], "./server");
        assert_eq!(parsed["files"][0]["functions"][0]["name"], "bootstrap");
        assert_eq!(parsed["files"][0]["classes"][0]["name"], "AppService");
        assert_eq!(parsed["files"][0]["variables"][0]["name"], "version");

        fs::remove_dir_all(root).expect("should clean up temp directory");
    }

    #[tokio::test]
    async fn file_io_skill_reads_text_files_and_skips_ignored_directories() {
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
            .await
            .expect("skill should succeed");

        assert!(output.contains("// File: src/app.js"));
        assert!(output.contains("console.log('hello');"));
        assert!(!output.contains("ignored.js"));
        assert!(!output.contains("binary.bin"));

        fs::remove_dir_all(root).expect("should clean up temp directory");
    }

    #[tokio::test]
    async fn file_io_skill_can_read_selected_context_files() {
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
            .await
            .expect("skill should read selected files");

        assert!(output.contains("// File: src/db.js"));
        assert!(!output.contains("src/app.js"));

        fs::remove_dir_all(root).expect("should clean up temp directory");
    }

    #[tokio::test]
    async fn file_write_skill_creates_missing_directories_and_writes_file() {
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
            .await
            .expect("file write skill should succeed");

        let written = fs::read_to_string(&target_path).expect("generated file should exist");
        assert_eq!(written, "export const value = 1;");

        fs::remove_dir_all(root).expect("should clean up temp directory");
    }
}
