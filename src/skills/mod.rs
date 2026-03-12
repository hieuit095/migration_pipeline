use crate::utils::path::{normalize_relative_path, relative_path_from_root};
use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use tempfile::Builder;
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

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImportEdge {
    pub source: String,
    pub names: Vec<String>,
    pub kind: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FunctionSignature {
    pub name: String,
    pub parameters: Vec<String>,
    pub is_async: bool,
    pub is_generator: bool,
    pub return_type: Option<String>,
    pub exported: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClassDefinition {
    pub name: String,
    pub extends: Option<String>,
    pub methods: Vec<MethodSignature>,
    pub exported: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MethodSignature {
    pub name: String,
    pub parameters: Vec<String>,
    pub is_async: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VariableBinding {
    pub name: String,
    pub declaration_kind: String,
    pub value_kind: Option<String>,
    pub exported: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BranchDescriptor {
    pub kind: String,
    pub condition: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq, Eq)]
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
    Python,
    Go,
    Rust,
}

impl SourceLanguage {
    fn from_path(path: &Path) -> Option<Self> {
        match path.extension().and_then(|value| value.to_str()) {
            Some("js" | "mjs" | "cjs" | "jsx") => Some(Self::JavaScript),
            Some("ts") => Some(Self::TypeScript),
            Some("tsx") => Some(Self::Tsx),
            Some("py") => Some(Self::Python),
            Some("go") => Some(Self::Go),
            Some("rs") => Some(Self::Rust),
            _ => None,
        }
    }

    fn language(self) -> Language {
        match self {
            Self::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            Self::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Self::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
            Self::Python => tree_sitter_python::LANGUAGE.into(),
            Self::Go => tree_sitter_go::LANGUAGE.into(),
            Self::Rust => tree_sitter_rust::LANGUAGE.into(),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::JavaScript => "javascript",
            Self::TypeScript => "typescript",
            Self::Tsx => "tsx",
            Self::Python => "python",
            Self::Go => "go",
            Self::Rust => "rust",
        }
    }

    fn supported_labels() -> &'static str {
        "JavaScript, TypeScript, TSX, Python, Go, and Rust"
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

    fn create_context_output() -> Result<(BufWriter<fs::File>, PathBuf)> {
        let temp_file = Builder::new()
            .prefix("migration_pipeline_context_")
            .suffix(".txt")
            .tempfile()
            .context("failed to create temporary context file")?;
        let (file, path) = temp_file
            .keep()
            .map_err(|error| anyhow!(error.error))
            .context("failed to persist temporary context file")?;
        Ok((BufWriter::new(file), path))
    }

    fn append_text_file(
        root: &Path,
        path: &Path,
        writer: &mut BufWriter<fs::File>,
    ) -> Result<bool> {
        let Some(contents) = Self::read_text_file(path)? else {
            return Ok(false);
        };
        let relative_path = Self::normalize_path(root, path);
        writeln!(writer, "// File: {relative_path}")
            .with_context(|| format!("failed to write context header for {}", path.display()))?;
        writer
            .write_all(contents.as_bytes())
            .with_context(|| format!("failed to write context body for {}", path.display()))?;
        writer
            .write_all(b"\n\n")
            .with_context(|| format!("failed to finalize context block for {}", path.display()))?;
        Ok(true)
    }

    fn remove_temp_context_file(path: &Path) {
        let _ = fs::remove_file(path);
    }

    fn collect_directory(root_path: &Path) -> Result<PathBuf> {
        let mut files: Vec<PathBuf> = WalkDir::new(root_path)
            .into_iter()
            .filter_entry(Self::should_descend)
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().is_file())
            .map(|entry| entry.into_path())
            .collect();

        files.sort();

        let (mut writer, output_path) = Self::create_context_output()?;
        let mut wrote_any = false;
        for path in files {
            if Self::append_text_file(root_path, &path, &mut writer)? {
                wrote_any = true;
            }
        }

        writer
            .flush()
            .context("failed to flush aggregated context file")?;
        if !wrote_any {
            Self::remove_temp_context_file(&output_path);
            return Err(anyhow!(
                "no readable text files found in legacy directory {}",
                root_path.display()
            ));
        }

        Ok(output_path)
    }

    fn collect_selected_files(root_path: &Path, relative_paths: &[String]) -> Result<PathBuf> {
        let (mut writer, output_path) = Self::create_context_output()?;
        let mut wrote_any = false;

        for relative_path in relative_paths {
            let full_path = validate_relative_file(root_path, relative_path)?;
            if !Self::append_text_file(root_path, &full_path, &mut writer)? {
                Self::remove_temp_context_file(&output_path);
                return Err(anyhow!(
                    "context file is not a readable text file: {relative_path}"
                ));
            }
            wrote_any = true;
        }

        writer
            .flush()
            .context("failed to flush selected context file")?;
        if !wrote_any {
            Self::remove_temp_context_file(&output_path);
            return Err(anyhow!(
                "no readable context files were selected from {}",
                root_path.display()
            ));
        }

        Ok(output_path)
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
                    "ASTParsingSkill does not support `{relative_path}`. Supported languages are {}",
                    SourceLanguage::supported_labels()
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
            Self::process_top_level_node(child, source, false, language, &mut summary);
        }
        Self::collect_branches(root, source, language, &mut summary.branches);
        summary.branches.sort();
        summary.branches.dedup();
        finalize_summary(&mut summary);

        Ok(summary)
    }

    fn process_top_level_node(
        node: Node<'_>,
        source: &str,
        exported: bool,
        language: SourceLanguage,
        summary: &mut FileDependencyNode,
    ) {
        match language {
            SourceLanguage::JavaScript | SourceLanguage::TypeScript | SourceLanguage::Tsx => {
                Self::process_javascript_top_level_node(node, source, exported, summary)
            }
            SourceLanguage::Python => {
                Self::process_python_top_level_node(node, source, exported, summary)
            }
            SourceLanguage::Go => Self::process_go_top_level_node(node, source, summary),
            SourceLanguage::Rust => Self::process_rust_top_level_node(node, source, summary),
        }
    }

    fn process_javascript_top_level_node(
        node: Node<'_>,
        source: &str,
        exported: bool,
        summary: &mut FileDependencyNode,
    ) {
        match node.kind() {
            "import_statement" => summary.imports.extend(Self::extract_imports(
                node,
                source,
                SourceLanguage::JavaScript,
            )),
            "function_declaration" => {
                if let Some(signature) =
                    Self::extract_function(node, source, SourceLanguage::JavaScript, exported)
                {
                    summary.functions.push(signature);
                }
            }
            "class_declaration" => {
                if let Some(class_definition) =
                    Self::extract_class(node, source, SourceLanguage::JavaScript, exported)
                {
                    Self::upsert_class_definition(summary, class_definition);
                }
            }
            "lexical_declaration" | "variable_declaration" => summary.variables.extend(
                Self::extract_variables(node, source, SourceLanguage::JavaScript, exported),
            ),
            "export_statement" => {
                let mut cursor = node.walk();
                for child in node.named_children(&mut cursor) {
                    if child.kind() == "export_clause" {
                        continue;
                    }
                    Self::process_javascript_top_level_node(child, source, true, summary);
                }
            }
            _ => {}
        }
    }

    fn process_python_top_level_node(
        node: Node<'_>,
        source: &str,
        exported: bool,
        summary: &mut FileDependencyNode,
    ) {
        match node.kind() {
            "import_statement" | "import_from_statement" => summary
                .imports
                .extend(Self::extract_imports(node, source, SourceLanguage::Python)),
            "function_definition" | "async_function_definition" => {
                if let Some(signature) =
                    Self::extract_function(node, source, SourceLanguage::Python, exported)
                {
                    summary.functions.push(signature);
                }
            }
            "class_definition" => {
                if let Some(class_definition) =
                    Self::extract_class(node, source, SourceLanguage::Python, exported)
                {
                    Self::upsert_class_definition(summary, class_definition);
                }
            }
            "assignment" | "augmented_assignment" | "annotated_assignment" => {
                summary.variables.extend(Self::extract_variables(
                    node,
                    source,
                    SourceLanguage::Python,
                    exported,
                ))
            }
            "decorated_definition" | "expression_statement" => {
                let mut cursor = node.walk();
                for child in node.named_children(&mut cursor) {
                    Self::process_python_top_level_node(child, source, exported, summary);
                }
            }
            _ => {}
        }
    }

    fn process_go_top_level_node(node: Node<'_>, source: &str, summary: &mut FileDependencyNode) {
        match node.kind() {
            "import_declaration" => {
                summary
                    .imports
                    .extend(Self::extract_imports(node, source, SourceLanguage::Go))
            }
            "function_declaration" => {
                if let Some(signature) =
                    Self::extract_function(node, source, SourceLanguage::Go, false)
                {
                    summary.functions.push(signature);
                }
            }
            "method_declaration" => {
                if let Some((receiver_name, method)) = extract_go_method(node, source) {
                    Self::upsert_class_definition(
                        summary,
                        ClassDefinition {
                            name: receiver_name,
                            methods: vec![method],
                            ..ClassDefinition::default()
                        },
                    );
                }
            }
            "type_declaration" => {
                for class_definition in extract_go_types(node, source) {
                    Self::upsert_class_definition(summary, class_definition);
                }
            }
            "var_declaration" | "const_declaration" => summary.variables.extend(
                Self::extract_variables(node, source, SourceLanguage::Go, false),
            ),
            _ => {}
        }
    }

    fn process_rust_top_level_node(node: Node<'_>, source: &str, summary: &mut FileDependencyNode) {
        match node.kind() {
            "use_declaration" => {
                summary
                    .imports
                    .extend(Self::extract_imports(node, source, SourceLanguage::Rust))
            }
            "function_item" => {
                if let Some(signature) =
                    Self::extract_function(node, source, SourceLanguage::Rust, false)
                {
                    summary.functions.push(signature);
                }
            }
            "struct_item" | "enum_item" | "trait_item" => {
                if let Some(class_definition) =
                    Self::extract_class(node, source, SourceLanguage::Rust, false)
                {
                    Self::upsert_class_definition(summary, class_definition);
                }
            }
            "impl_item" => {
                if let Some(class_definition) = extract_rust_impl(node, source) {
                    Self::upsert_class_definition(summary, class_definition);
                }
            }
            "const_item" | "static_item" => summary.variables.extend(Self::extract_variables(
                node,
                source,
                SourceLanguage::Rust,
                false,
            )),
            _ => {}
        }
    }

    fn upsert_class_definition(
        summary: &mut FileDependencyNode,
        class_definition: ClassDefinition,
    ) {
        if let Some(existing) = summary
            .classes
            .iter_mut()
            .find(|existing| existing.name == class_definition.name)
        {
            if existing.extends.is_none() {
                existing.extends = class_definition.extends.clone();
            }
            existing.exported |= class_definition.exported;
            for method in class_definition.methods {
                if !existing.methods.contains(&method) {
                    existing.methods.push(method);
                }
            }
            existing.methods.sort();
            existing.methods.dedup();
        } else {
            let mut class_definition = class_definition;
            class_definition.methods.sort();
            class_definition.methods.dedup();
            summary.classes.push(class_definition);
        }
    }

    fn extract_imports(node: Node<'_>, source: &str, language: SourceLanguage) -> Vec<ImportEdge> {
        match language {
            SourceLanguage::JavaScript | SourceLanguage::TypeScript | SourceLanguage::Tsx => {
                Self::extract_javascript_import(node, source)
                    .into_iter()
                    .collect()
            }
            SourceLanguage::Python => Self::extract_python_imports(node, source),
            SourceLanguage::Go => Self::extract_go_imports(node, source),
            SourceLanguage::Rust => Self::extract_rust_imports(node, source),
        }
    }

    fn extract_javascript_import(node: Node<'_>, source: &str) -> Option<ImportEdge> {
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

    fn extract_python_imports(node: Node<'_>, source: &str) -> Vec<ImportEdge> {
        let Some(raw_text) = node_text(node, source) else {
            return Vec::new();
        };
        let trimmed = raw_text.trim();

        if let Some(import_list) = trimmed.strip_prefix("import ") {
            return import_list
                .split(',')
                .filter_map(|entry| {
                    let module_name = entry.split_whitespace().next()?.trim();
                    if module_name.is_empty() {
                        return None;
                    }
                    Some(ImportEdge {
                        source: module_name.to_owned(),
                        names: vec![
                            module_name
                                .rsplit('.')
                                .next()
                                .unwrap_or(module_name)
                                .to_owned(),
                        ],
                        kind: "named".to_owned(),
                    })
                })
                .collect();
        }

        if let Some(from_clause) = trimmed.strip_prefix("from ")
            && let Some((module_name, imported_names)) = from_clause.split_once(" import ")
        {
            let names = imported_names
                .split(',')
                .map(|entry| entry.trim())
                .filter(|entry| !entry.is_empty())
                .map(|entry| entry.split_whitespace().next().unwrap_or(entry).to_owned())
                .collect::<Vec<_>>();
            return vec![ImportEdge {
                source: module_name.trim().to_owned(),
                kind: if names.iter().any(|name| name == "*") {
                    "namespace".to_owned()
                } else {
                    "named".to_owned()
                },
                names,
            }];
        }

        Vec::new()
    }

    fn extract_go_imports(node: Node<'_>, source: &str) -> Vec<ImportEdge> {
        let mut imports = Vec::new();
        let mut cursor = node.walk();

        for child in node.named_children(&mut cursor) {
            if child.kind() != "import_spec" {
                continue;
            }

            let Some(spec_text) = node_text(child, source) else {
                continue;
            };
            if let Some(import_source) = extract_first_quoted_value(&spec_text) {
                imports.push(ImportEdge {
                    source: import_source,
                    names: Vec::new(),
                    kind: "named".to_owned(),
                });
            }
        }

        if imports.is_empty()
            && let Some(raw_text) = node_text(node, source)
            && let Some(import_source) = extract_first_quoted_value(&raw_text)
        {
            imports.push(ImportEdge {
                source: import_source,
                names: Vec::new(),
                kind: "named".to_owned(),
            });
        }

        imports
    }

    fn extract_rust_imports(node: Node<'_>, source: &str) -> Vec<ImportEdge> {
        let Some(raw_text) = node_text(node, source) else {
            return Vec::new();
        };
        let import_source = raw_text
            .trim()
            .trim_start_matches("pub ")
            .trim_start_matches("use ")
            .trim_end_matches(';')
            .trim()
            .to_owned();
        if import_source.is_empty() {
            return Vec::new();
        }

        let names = import_source
            .split("::")
            .last()
            .map(|entry| entry.trim_matches(|character| character == '{' || character == '}'))
            .filter(|entry| !entry.is_empty())
            .map(|entry| {
                entry
                    .split(',')
                    .map(|name| name.trim().to_owned())
                    .collect()
            })
            .unwrap_or_default();

        vec![ImportEdge {
            source: import_source,
            names,
            kind: "named".to_owned(),
        }]
    }

    fn extract_function(
        node: Node<'_>,
        source: &str,
        language: SourceLanguage,
        exported: bool,
    ) -> Option<FunctionSignature> {
        let name = node
            .child_by_field_name("name")
            .and_then(|value| node_text(value, source))?;
        let parameters = node
            .child_by_field_name("parameters")
            .map(|value| collect_parameter_names(value, source))
            .unwrap_or_default();
        let return_type = match language {
            SourceLanguage::Go => node
                .child_by_field_name("result")
                .and_then(|value| node_text(value, source))
                .map(|value| value.trim().to_owned()),
            _ => node
                .child_by_field_name("return_type")
                .and_then(|value| node_text(value, source))
                .map(|value| value.trim().to_owned()),
        };
        let node_text = node_text(node, source)?;
        let exported = match language {
            SourceLanguage::Go => is_exported_name(&name),
            SourceLanguage::Rust => is_rust_exported(&node_text),
            _ => exported,
        };

        Some(FunctionSignature {
            name,
            parameters,
            is_async: node_text.trim_start().starts_with("async "),
            is_generator: matches!(
                language,
                SourceLanguage::JavaScript | SourceLanguage::TypeScript | SourceLanguage::Tsx
            ) && node_text.contains("function*"),
            return_type,
            exported,
        })
    }

    fn extract_class(
        node: Node<'_>,
        source: &str,
        language: SourceLanguage,
        exported: bool,
    ) -> Option<ClassDefinition> {
        let name = node
            .child_by_field_name("name")
            .and_then(|value| node_text(value, source))?;
        let extends = match language {
            SourceLanguage::JavaScript | SourceLanguage::TypeScript | SourceLanguage::Tsx => node
                .child_by_field_name("superclass")
                .and_then(|value| node_text(value, source)),
            SourceLanguage::Python => node
                .child_by_field_name("superclasses")
                .and_then(|value| node_text(value, source))
                .map(|value| value.trim_matches(['(', ')']).trim().to_owned())
                .filter(|value| !value.is_empty()),
            _ => None,
        };
        let methods = match language {
            SourceLanguage::JavaScript | SourceLanguage::TypeScript | SourceLanguage::Tsx => node
                .child_by_field_name("body")
                .map(|body| Self::extract_methods(body, source, language))
                .unwrap_or_default(),
            SourceLanguage::Python => node
                .child_by_field_name("body")
                .map(|body| Self::extract_methods(body, source, language))
                .unwrap_or_default(),
            SourceLanguage::Rust => Self::extract_methods(node, source, language),
            SourceLanguage::Go => Vec::new(),
        };
        let exported = match language {
            SourceLanguage::Go => is_exported_name(&name),
            SourceLanguage::Rust => {
                node_text(node, source).is_some_and(|value| is_rust_exported(&value))
            }
            _ => exported,
        };

        Some(ClassDefinition {
            name,
            extends,
            methods,
            exported,
        })
    }

    fn extract_methods(
        node: Node<'_>,
        source: &str,
        language: SourceLanguage,
    ) -> Vec<MethodSignature> {
        let mut methods = Vec::new();
        let mut cursor = node.walk();

        for child in node.named_children(&mut cursor) {
            match language {
                SourceLanguage::JavaScript | SourceLanguage::TypeScript | SourceLanguage::Tsx => {
                    if !matches!(child.kind(), "method_definition" | "method_signature") {
                        continue;
                    }
                    if let Some(method) = extract_method_signature(child, source) {
                        methods.push(method);
                    }
                }
                SourceLanguage::Python => match child.kind() {
                    "function_definition" | "async_function_definition" => {
                        if let Some(method) = extract_method_signature(child, source) {
                            methods.push(method);
                        }
                    }
                    "decorated_definition" => {
                        methods.extend(Self::extract_methods(child, source, language));
                    }
                    _ => {}
                },
                SourceLanguage::Rust => {
                    if child.kind() != "function_item" {
                        continue;
                    }
                    if let Some(method) = extract_method_signature(child, source) {
                        methods.push(method);
                    }
                }
                SourceLanguage::Go => {}
            }
        }

        methods
    }

    fn extract_variables(
        node: Node<'_>,
        source: &str,
        language: SourceLanguage,
        exported: bool,
    ) -> Vec<VariableBinding> {
        match language {
            SourceLanguage::JavaScript | SourceLanguage::TypeScript | SourceLanguage::Tsx => {
                Self::extract_javascript_variables(node, source, exported)
            }
            SourceLanguage::Python => Self::extract_python_variables(node, source, exported),
            SourceLanguage::Go => Self::extract_go_variables(node, source),
            SourceLanguage::Rust => Self::extract_rust_variables(node, source),
        }
    }

    fn extract_javascript_variables(
        node: Node<'_>,
        source: &str,
        exported: bool,
    ) -> Vec<VariableBinding> {
        let declaration_kind = node
            .child(0)
            .and_then(|value| node_text(value, source))
            .unwrap_or_else(|| node.kind().to_owned())
            .trim()
            .to_owned();
        let mut variables = Vec::new();
        let mut cursor = node.walk();

        for child in node.named_children(&mut cursor) {
            if child.kind() != "variable_declarator" {
                continue;
            }

            let mut names = Vec::new();
            if let Some(name_node) = child.child_by_field_name("name") {
                collect_binding_identifiers(name_node, source, &mut names);
            }
            let value_kind = child
                .child_by_field_name("value")
                .map(|value| value.kind().to_owned());

            for name in names {
                variables.push(VariableBinding {
                    name,
                    declaration_kind: declaration_kind.clone(),
                    value_kind: value_kind.clone(),
                    exported,
                });
            }
        }

        variables
    }

    fn extract_python_variables(
        node: Node<'_>,
        source: &str,
        exported: bool,
    ) -> Vec<VariableBinding> {
        let mut names = Vec::new();
        if let Some(left) = node
            .child_by_field_name("left")
            .or_else(|| node.child_by_field_name("name"))
        {
            collect_binding_identifiers(left, source, &mut names);
        } else if let Some(left) = node.named_child(0) {
            collect_binding_identifiers(left, source, &mut names);
        }

        let value_kind = node
            .child_by_field_name("right")
            .or_else(|| node.child_by_field_name("value"))
            .or_else(|| node.named_child(node.named_child_count().saturating_sub(1)))
            .map(|value| value.kind().to_owned());

        names
            .into_iter()
            .map(|name| VariableBinding {
                name,
                declaration_kind: "assignment".to_owned(),
                value_kind: value_kind.clone(),
                exported,
            })
            .collect()
    }

    fn extract_go_variables(node: Node<'_>, source: &str) -> Vec<VariableBinding> {
        let declaration_kind = if node.kind() == "const_declaration" {
            "const"
        } else {
            "var"
        };
        let mut variables = Vec::new();
        let mut cursor = node.walk();

        for child in node.named_children(&mut cursor) {
            if !matches!(child.kind(), "var_spec" | "const_spec") {
                continue;
            }

            let mut names = Vec::new();
            if let Some(name_node) = child.child_by_field_name("name") {
                collect_binding_identifiers(name_node, source, &mut names);
            } else {
                collect_binding_identifiers(child, source, &mut names);
            }

            let value_kind = child
                .child_by_field_name("value")
                .or_else(|| {
                    let mut child_cursor = child.walk();
                    child
                        .named_children(&mut child_cursor)
                        .find(|candidate| candidate.kind().ends_with("expression"))
                })
                .map(|value| value.kind().to_owned());

            for name in names {
                let exported = is_exported_name(&name);
                variables.push(VariableBinding {
                    name,
                    declaration_kind: declaration_kind.to_owned(),
                    value_kind: value_kind.clone(),
                    exported,
                });
            }
        }

        variables
    }

    fn extract_rust_variables(node: Node<'_>, source: &str) -> Vec<VariableBinding> {
        let mut names = Vec::new();
        if let Some(name_node) = node.child_by_field_name("name") {
            collect_binding_identifiers(name_node, source, &mut names);
        }

        let declaration_kind = if node.kind() == "static_item" {
            "static"
        } else {
            "const"
        };
        let value_kind = node
            .child_by_field_name("value")
            .map(|value| value.kind().to_owned());
        let exported = node_text(node, source).is_some_and(|value| is_rust_exported(&value));

        names
            .into_iter()
            .map(|name| VariableBinding {
                name,
                declaration_kind: declaration_kind.to_owned(),
                value_kind: value_kind.clone(),
                exported,
            })
            .collect()
    }

    fn collect_branches(
        node: Node<'_>,
        source: &str,
        language: SourceLanguage,
        branches: &mut Vec<BranchDescriptor>,
    ) {
        if let Some(branch) = Self::extract_branch(node, source, language) {
            branches.push(branch);
        }

        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            Self::collect_branches(child, source, language, branches);
        }
    }

    fn extract_branch(
        node: Node<'_>,
        source: &str,
        language: SourceLanguage,
    ) -> Option<BranchDescriptor> {
        match language {
            SourceLanguage::JavaScript | SourceLanguage::TypeScript | SourceLanguage::Tsx => {
                match node.kind() {
                    "if_statement" => Some(branch(
                        "if",
                        field_or_fallback(node, source, "condition", &[])
                            .unwrap_or_else(|| "if".to_owned()),
                    )),
                    "switch_statement" => Some(branch(
                        "switch",
                        field_or_fallback(node, source, "value", &[])
                            .unwrap_or_else(|| "switch".to_owned()),
                    )),
                    "conditional_expression" => Some(branch(
                        "ternary",
                        field_or_fallback(node, source, "condition", &[])
                            .unwrap_or_else(|| "ternary".to_owned()),
                    )),
                    "try_statement" => Some(branch("try", "try/catch".to_owned())),
                    "for_statement" | "while_statement" | "do_statement" => Some(branch(
                        node.kind(),
                        field_or_fallback(node, source, "condition", &[])
                            .unwrap_or_else(|| node.kind().to_owned()),
                    )),
                    _ => None,
                }
            }
            SourceLanguage::Python => match node.kind() {
                "if_statement" => Some(branch(
                    "if",
                    field_or_fallback(
                        node,
                        source,
                        "condition",
                        &["comparison_operator", "boolean_operator"],
                    )
                    .unwrap_or_else(|| "if".to_owned()),
                )),
                "for_statement" => Some(branch(
                    "for_statement",
                    field_or_fallback(node, source, "right", &["in"])
                        .unwrap_or_else(|| "for".to_owned()),
                )),
                "while_statement" => Some(branch(
                    "while_statement",
                    field_or_fallback(node, source, "condition", &[])
                        .unwrap_or_else(|| "while".to_owned()),
                )),
                "try_statement" => Some(branch("try", "try/except".to_owned())),
                "conditional_expression" => Some(branch(
                    "ternary",
                    field_or_fallback(node, source, "condition", &[])
                        .unwrap_or_else(|| "ternary".to_owned()),
                )),
                "match_statement" => Some(branch(
                    "match",
                    field_or_fallback(node, source, "subject", &[])
                        .unwrap_or_else(|| "match".to_owned()),
                )),
                _ => None,
            },
            SourceLanguage::Go => match node.kind() {
                "if_statement" => Some(branch(
                    "if",
                    field_or_fallback(node, source, "condition", &["expression"])
                        .unwrap_or_else(|| "if".to_owned()),
                )),
                "expression_switch_statement" | "type_switch_statement" => Some(branch(
                    "switch",
                    field_or_fallback(node, source, "value", &["expression"])
                        .unwrap_or_else(|| "switch".to_owned()),
                )),
                "select_statement" => Some(branch("select", "select".to_owned())),
                "for_statement" => Some(branch(
                    "for_statement",
                    field_or_fallback(node, source, "condition", &["expression"])
                        .unwrap_or_else(|| "for".to_owned()),
                )),
                _ => None,
            },
            SourceLanguage::Rust => match node.kind() {
                "if_expression" => Some(branch(
                    "if",
                    field_or_fallback(node, source, "condition", &["condition"])
                        .unwrap_or_else(|| "if".to_owned()),
                )),
                "match_expression" => Some(branch(
                    "match",
                    field_or_fallback(node, source, "value", &["expression"])
                        .unwrap_or_else(|| "match".to_owned()),
                )),
                "for_expression" => Some(branch(
                    "for_statement",
                    field_or_fallback(node, source, "value", &["iterator", "expression"])
                        .unwrap_or_else(|| "for".to_owned()),
                )),
                "while_expression" => Some(branch(
                    "while_statement",
                    field_or_fallback(node, source, "condition", &["expression"])
                        .unwrap_or_else(|| "while".to_owned()),
                )),
                "loop_expression" => Some(branch("loop", "loop".to_owned())),
                "try_expression" => Some(branch("try", "try".to_owned())),
                _ => None,
            },
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

            let context_path = if args.len() == 1 {
                Self::collect_directory(&root_path)?
            } else {
                Self::collect_selected_files(&root_path, &args[1..])?
            };

            Ok(context_path.to_string_lossy().to_string())
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
                    "no supported source files ({}) found in legacy directory {}",
                    SourceLanguage::supported_labels(),
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
    AstDiff {
        missing_functions: diff_function_signatures(legacy, modern),
        missing_classes: diff_class_signatures(legacy, modern),
        missing_imports: diff_import_signatures(legacy, modern),
        missing_branches: diff_branch_signatures(legacy, modern),
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
        | "field_identifier"
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

fn collect_binding_identifiers(node: Node<'_>, source: &str, output: &mut Vec<String>) {
    match node.kind() {
        "identifier"
        | "property_identifier"
        | "type_identifier"
        | "field_identifier"
        | "shorthand_property_identifier_pattern" => {
            if let Some(identifier) = node_text(node, source) {
                let identifier = identifier.trim().to_owned();
                if !identifier.is_empty() && !output.contains(&identifier) {
                    output.push(identifier);
                }
            }
        }
        "pair_pattern"
        | "object_pattern"
        | "array_pattern"
        | "assignment_pattern"
        | "rest_pattern"
        | "tuple_pattern"
        | "list_pattern"
        | "dictionary_splat_pattern"
        | "parameter_declaration"
        | "typed_parameter"
        | "variadic_parameter_declaration"
        | "variadic_parameter"
        | "parameter"
        | "receiver"
        | "mutable_specifier"
        | "reference_pattern"
        | "tuple_struct_pattern"
        | "slice_pattern" => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                collect_binding_identifiers(child, source, output);
            }
        }
        _ => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                collect_binding_identifiers(child, source, output);
            }
        }
    }
}

fn extract_method_signature(node: Node<'_>, source: &str) -> Option<MethodSignature> {
    let name = node
        .child_by_field_name("name")
        .and_then(|value| node_text(value, source))
        .or_else(|| {
            field_or_fallback(node, source, "name", &["identifier", "property_identifier"])
        })?;
    let parameters = node
        .child_by_field_name("parameters")
        .map(|value| collect_parameter_names(value, source))
        .unwrap_or_default();
    let body_text = node_text(node, source)?;

    Some(MethodSignature {
        name,
        parameters,
        is_async: body_text.trim_start().starts_with("async "),
    })
}

fn field_or_fallback(
    node: Node<'_>,
    source: &str,
    field_name: &str,
    fallback_kinds: &[&str],
) -> Option<String> {
    if let Some(value) = node
        .child_by_field_name(field_name)
        .and_then(|value| node_text(value, source))
    {
        let value = value.trim().to_owned();
        if !value.is_empty() {
            return Some(value);
        }
    }

    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if fallback_kinds.contains(&child.kind()) {
            if let Some(value) = node_text(child, source) {
                let value = value.trim().to_owned();
                if !value.is_empty() {
                    return Some(value);
                }
            }
        }
    }

    None
}

fn extract_first_quoted_value(input: &str) -> Option<String> {
    for delimiter in ['"', '`'] {
        let Some(start) = input.find(delimiter) else {
            continue;
        };
        let Some(remainder) = input.get(start + delimiter.len_utf8()..) else {
            continue;
        };
        let Some(end) = remainder.find(delimiter) else {
            continue;
        };
        return remainder.get(..end).map(ToOwned::to_owned);
    }

    None
}

fn normalize_type_name(input: &str) -> Option<String> {
    let mut value = input.trim().trim_matches(['(', ')']).trim().to_owned();
    while let Some(stripped) = value.strip_prefix('&') {
        value = stripped.trim_start().to_owned();
    }
    while let Some(stripped) = value.strip_prefix('*') {
        value = stripped.trim_start().to_owned();
    }
    while let Some(stripped) = value.strip_prefix("mut ") {
        value = stripped.trim_start().to_owned();
    }
    while let Some(stripped) = value.strip_prefix("[]") {
        value = stripped.trim_start().to_owned();
    }

    let value = value
        .split_whitespace()
        .last()
        .unwrap_or(&value)
        .split('<')
        .next()
        .unwrap_or(&value)
        .rsplit("::")
        .next()
        .unwrap_or(&value)
        .rsplit('.')
        .next()
        .unwrap_or(&value)
        .trim()
        .to_owned();

    (!value.is_empty()).then_some(value)
}

fn extract_receiver_type_name(node: Node<'_>, source: &str) -> Option<String> {
    field_or_fallback(
        node,
        source,
        "type",
        &[
            "type_identifier",
            "qualified_type",
            "generic_type",
            "pointer_type",
            "scoped_type_identifier",
            "identifier",
        ],
    )
    .or_else(|| node_text(node, source))
    .and_then(|value| normalize_type_name(&value))
}

fn extract_go_method(node: Node<'_>, source: &str) -> Option<(String, MethodSignature)> {
    let receiver_name = node
        .child_by_field_name("receiver")
        .and_then(|receiver| extract_receiver_type_name(receiver, source))?;
    let method = extract_method_signature(node, source)?;
    Some((receiver_name, method))
}

fn extract_go_types(node: Node<'_>, source: &str) -> Vec<ClassDefinition> {
    let mut classes = Vec::new();
    let mut cursor = node.walk();

    for child in node.named_children(&mut cursor) {
        if child.kind() != "type_spec" {
            continue;
        }

        let Some(name) = child
            .child_by_field_name("name")
            .and_then(|value| node_text(value, source))
        else {
            continue;
        };
        let type_kind = child
            .child_by_field_name("type")
            .map(|value| value.kind().to_owned())
            .unwrap_or_default();
        if !matches!(type_kind.as_str(), "struct_type" | "interface_type") {
            continue;
        }

        classes.push(ClassDefinition {
            name: name.clone(),
            extends: None,
            methods: Vec::new(),
            exported: is_exported_name(&name),
        });
    }

    classes
}

fn extract_rust_impl(node: Node<'_>, source: &str) -> Option<ClassDefinition> {
    let name = node
        .child_by_field_name("type")
        .and_then(|value| extract_receiver_type_name(value, source))
        .or_else(|| {
            field_or_fallback(
                node,
                source,
                "type",
                &["type_identifier", "scoped_type_identifier", "generic_type"],
            )
            .and_then(|value| normalize_type_name(&value))
        })?;

    Some(ClassDefinition {
        name,
        extends: None,
        methods: ASTParsingSkill::extract_methods(node, source, SourceLanguage::Rust),
        exported: false,
    })
}

fn branch(kind: impl Into<String>, condition: impl Into<String>) -> BranchDescriptor {
    BranchDescriptor {
        kind: kind.into(),
        condition: condition.into(),
    }
}

fn is_exported_name(name: &str) -> bool {
    name.chars()
        .next()
        .is_some_and(|character| character.is_uppercase())
}

fn is_rust_exported(item_text: &str) -> bool {
    item_text.trim_start().starts_with("pub ")
}

fn finalize_summary(summary: &mut FileDependencyNode) {
    summary.imports.sort();
    summary.imports.dedup();
    summary.functions.sort();
    summary.functions.dedup();
    for class_definition in &mut summary.classes {
        class_definition.methods.sort();
        class_definition.methods.dedup();
    }
    summary.classes.sort();
    summary.classes.dedup();
    summary.variables.sort();
    summary.variables.dedup();
    summary.branches.sort();
    summary.branches.dedup();
}

fn stable_hash(value: impl Hash) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn diff_function_signatures(legacy: &DependencyGraph, modern: &DependencyGraph) -> Vec<String> {
    let modern_hashes = collect_function_signature_hashes(modern);
    let mut seen_missing = HashSet::new();
    let mut missing_functions = Vec::new();

    for file in &legacy.files {
        for function in &file.functions {
            let signature_hash = stable_hash((&file.path, &function.name, &function.parameters));
            if !modern_hashes.contains(&signature_hash) && seen_missing.insert(signature_hash) {
                missing_functions.push(format!(
                    "{}::{}({})",
                    file.path,
                    function.name,
                    function.parameters.join(",")
                ));
            }
        }
    }

    missing_functions
}

fn collect_function_signature_hashes(graph: &DependencyGraph) -> HashSet<u64> {
    graph
        .files
        .iter()
        .flat_map(|file| {
            file.functions.iter().map(move |function| {
                stable_hash((&file.path, &function.name, &function.parameters))
            })
        })
        .collect()
}

fn diff_class_signatures(legacy: &DependencyGraph, modern: &DependencyGraph) -> Vec<String> {
    let modern_hashes = collect_class_signature_hashes(modern);
    let mut seen_missing = HashSet::new();
    let mut missing_classes = Vec::new();

    for file in &legacy.files {
        for class_definition in &file.classes {
            let signature_hash = stable_hash((&file.path, &class_definition.name));
            if !modern_hashes.contains(&signature_hash) && seen_missing.insert(signature_hash) {
                missing_classes.push(format!("{}::{}", file.path, class_definition.name));
            }
        }
    }

    missing_classes
}

fn collect_class_signature_hashes(graph: &DependencyGraph) -> HashSet<u64> {
    graph
        .files
        .iter()
        .flat_map(|file| {
            file.classes
                .iter()
                .map(move |class_definition| stable_hash((&file.path, &class_definition.name)))
        })
        .collect()
}

fn diff_import_signatures(legacy: &DependencyGraph, modern: &DependencyGraph) -> Vec<String> {
    let modern_hashes = collect_import_signature_hashes(modern);
    let mut seen_missing = HashSet::new();
    let mut missing_imports = Vec::new();

    for file in &legacy.files {
        for import_edge in &file.imports {
            let signature_hash = stable_hash((&file.path, &import_edge.kind, &import_edge.source));
            if !modern_hashes.contains(&signature_hash) && seen_missing.insert(signature_hash) {
                missing_imports.push(format!(
                    "{}::{}::{}",
                    file.path, import_edge.kind, import_edge.source
                ));
            }
        }
    }

    missing_imports
}

fn collect_import_signature_hashes(graph: &DependencyGraph) -> HashSet<u64> {
    graph
        .files
        .iter()
        .flat_map(|file| {
            file.imports.iter().map(move |import_edge| {
                stable_hash((&file.path, &import_edge.kind, &import_edge.source))
            })
        })
        .collect()
}

fn diff_branch_signatures(legacy: &DependencyGraph, modern: &DependencyGraph) -> Vec<String> {
    let modern_hashes = collect_branch_signature_hashes(modern);
    let mut seen_missing = HashSet::new();
    let mut missing_branches = Vec::new();

    for file in &legacy.files {
        for branch in &file.branches {
            let signature_hash = stable_hash((&file.path, &branch.kind, &branch.condition));
            if !modern_hashes.contains(&signature_hash) && seen_missing.insert(signature_hash) {
                missing_branches.push(format!(
                    "{}::{}::{}",
                    file.path, branch.kind, branch.condition
                ));
            }
        }
    }

    missing_branches
}

fn collect_branch_signature_hashes(graph: &DependencyGraph) -> HashSet<u64> {
    graph
        .files
        .iter()
        .flat_map(|file| {
            file.branches
                .iter()
                .map(move |branch| stable_hash((&file.path, &branch.kind, &branch.condition)))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        ASTParsingSkill, AstDiff, BranchDescriptor, ClassDefinition, DependencyGraph,
        FileDependencyNode, FileIOSkill, FileWriteSkill, FunctionSignature, ImportEdge, Skill,
        diff_dependency_graphs, parse_dependency_graph_json,
    };
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_root(prefix: &str) -> std::path::PathBuf {
        let unique_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be monotonic")
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}_{unique_id}"))
    }

    fn read_context_file(path: &str) -> String {
        let path = std::path::Path::new(path);
        let contents = fs::read_to_string(path).expect("context temp file should be readable");
        fs::remove_file(path).expect("context temp file should be removable");
        contents
    }

    #[test]
    fn diff_dependency_graphs_preserves_missing_signature_reporting() {
        let legacy = DependencyGraph {
            files: vec![FileDependencyNode {
                path: "src/app.ts".to_owned(),
                functions: vec![FunctionSignature {
                    name: "bootstrap".to_owned(),
                    parameters: vec!["port".to_owned()],
                    ..FunctionSignature::default()
                }],
                classes: vec![ClassDefinition {
                    name: "AppService".to_owned(),
                    ..ClassDefinition::default()
                }],
                imports: vec![ImportEdge {
                    source: "./server".to_owned(),
                    kind: "named".to_owned(),
                    ..ImportEdge::default()
                }],
                branches: vec![BranchDescriptor {
                    kind: "if".to_owned(),
                    condition: "ready".to_owned(),
                }],
                ..FileDependencyNode::default()
            }],
        };
        let modern = DependencyGraph::default();

        let diff = diff_dependency_graphs(&legacy, &modern);

        assert_eq!(
            diff,
            AstDiff {
                missing_functions: vec!["src/app.ts::bootstrap(port)".to_owned()],
                missing_classes: vec!["src/app.ts::AppService".to_owned()],
                missing_imports: vec!["src/app.ts::named::./server".to_owned()],
                missing_branches: vec!["src/app.ts::if::ready".to_owned()],
                ..AstDiff::default()
            }
        );
    }

    #[tokio::test]
    async fn ast_parsing_skill_builds_dependency_graph_for_typescript() {
        let root = temp_root("migration_pipeline_ast");

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
    async fn ast_parsing_skill_builds_dependency_graph_for_python() {
        let root = temp_root("migration_pipeline_ast_python");

        fs::create_dir_all(root.join("src")).expect("should create src dir");
        fs::write(
            root.join("src/app.py"),
            r#"
import requests
from services.server import serve

VERSION = "1.0.0"

class AppService(BaseService):
    async def run(self, port):
        return serve(port)

async def bootstrap(port):
    if port > 0:
        return serve(port)
    return None
"#,
        )
        .expect("should write fixture");

        let output = ASTParsingSkill
            .execute(vec![root.to_string_lossy().to_string()])
            .await
            .expect("ASTParsingSkill should succeed");
        let parsed = parse_dependency_graph_json(&output).expect("dependency graph should parse");
        let file = parsed.files.first().expect("python file should be present");

        assert_eq!(file.path, "src/app.py");
        assert_eq!(file.language, "python");
        assert!(
            file.imports
                .iter()
                .any(|import_edge| import_edge.source == "requests")
        );
        assert!(
            file.functions
                .iter()
                .any(|function| function.name == "bootstrap")
        );
        assert!(
            file.classes
                .iter()
                .any(|class_definition| class_definition.name == "AppService")
        );
        assert!(
            file.variables
                .iter()
                .any(|variable| variable.name == "VERSION")
        );
        assert!(file.branches.iter().any(|branch| branch.kind == "if"));

        fs::remove_dir_all(root).expect("should clean up temp directory");
    }

    #[tokio::test]
    async fn ast_parsing_skill_builds_dependency_graph_for_go() {
        let root = temp_root("migration_pipeline_ast_go");

        fs::create_dir_all(root.join("src")).expect("should create src dir");
        fs::write(
            root.join("src/app.go"),
            r#"
package main

import "fmt"

var version = "1.0.0"

type AppService struct{}

func Bootstrap(port int) string {
    if port > 0 {
        return fmt.Sprintf("%d", port)
    }
    return ""
}
"#,
        )
        .expect("should write fixture");

        let output = ASTParsingSkill
            .execute(vec![root.to_string_lossy().to_string()])
            .await
            .expect("ASTParsingSkill should succeed");
        let parsed = parse_dependency_graph_json(&output).expect("dependency graph should parse");
        let file = parsed.files.first().expect("go file should be present");

        assert_eq!(file.path, "src/app.go");
        assert_eq!(file.language, "go");
        assert!(
            file.imports
                .iter()
                .any(|import_edge| import_edge.source == "fmt")
        );
        assert!(
            file.functions
                .iter()
                .any(|function| function.name == "Bootstrap")
        );
        assert!(
            file.classes
                .iter()
                .any(|class_definition| class_definition.name == "AppService")
        );
        assert!(
            file.variables
                .iter()
                .any(|variable| variable.name == "version")
        );
        assert!(file.branches.iter().any(|branch| branch.kind == "if"));

        fs::remove_dir_all(root).expect("should clean up temp directory");
    }

    #[tokio::test]
    async fn ast_parsing_skill_builds_dependency_graph_for_rust() {
        let root = temp_root("migration_pipeline_ast_rust");

        fs::create_dir_all(root.join("src")).expect("should create src dir");
        fs::write(
            root.join("src/app.rs"),
            r#"
use crate::server::serve;

pub const VERSION: &str = "1.0.0";

pub struct AppService;

pub fn bootstrap(port: u16) -> String {
    if port > 0 {
        return serve(port);
    }
    String::new()
}
"#,
        )
        .expect("should write fixture");

        let output = ASTParsingSkill
            .execute(vec![root.to_string_lossy().to_string()])
            .await
            .expect("ASTParsingSkill should succeed");
        let parsed = parse_dependency_graph_json(&output).expect("dependency graph should parse");
        let file = parsed.files.first().expect("rust file should be present");

        assert_eq!(file.path, "src/app.rs");
        assert_eq!(file.language, "rust");
        assert!(
            file.imports
                .iter()
                .any(|import_edge| import_edge.source == "crate::server::serve")
        );
        assert!(
            file.functions
                .iter()
                .any(|function| function.name == "bootstrap")
        );
        assert!(
            file.classes
                .iter()
                .any(|class_definition| class_definition.name == "AppService")
        );
        assert!(
            file.variables
                .iter()
                .any(|variable| variable.name == "VERSION")
        );
        assert!(file.branches.iter().any(|branch| branch.kind == "if"));

        fs::remove_dir_all(root).expect("should clean up temp directory");
    }

    #[tokio::test]
    async fn file_io_skill_reads_text_files_and_skips_ignored_directories() {
        let root = temp_root("migration_pipeline_file_io");

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
        let output = read_context_file(&output);

        assert!(output.contains("// File: src/app.js"));
        assert!(output.contains("console.log('hello');"));
        assert!(!output.contains("ignored.js"));
        assert!(!output.contains("binary.bin"));

        fs::remove_dir_all(root).expect("should clean up temp directory");
    }

    #[tokio::test]
    async fn file_io_skill_can_read_selected_context_files() {
        let root = temp_root("migration_pipeline_context_io");

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
        let output = read_context_file(&output);

        assert!(output.contains("// File: src/db.js"));
        assert!(!output.contains("src/app.js"));

        fs::remove_dir_all(root).expect("should clean up temp directory");
    }

    #[tokio::test]
    async fn file_write_skill_creates_missing_directories_and_writes_file() {
        let root = temp_root("migration_pipeline_file_write");
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
