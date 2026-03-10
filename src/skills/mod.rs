use anyhow::Result;

pub trait Skill: Send + Sync {
    fn name(&self) -> &str;
    fn execute(&self, args: Vec<String>) -> Result<String>;
}

pub struct FileIOSkill;
pub struct ASTParsingSkill;
pub struct SandboxSkill;

// Implementations for skills would go here
