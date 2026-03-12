use anyhow::{Context, Result};
use minijinja::Environment;
use serde::Serialize;

pub(crate) fn render<T>(template_name: &str, context: T) -> Result<String>
where
    T: Serialize,
{
    let environment = prompt_environment()?;
    let template = environment
        .get_template(template_name)
        .with_context(|| format!("failed to load prompt template `{template_name}`"))?;

    template
        .render(context)
        .with_context(|| format!("failed to render prompt template `{template_name}`"))
}

fn prompt_environment() -> Result<Environment<'static>> {
    let mut environment = Environment::new();
    environment.set_trim_blocks(true);
    environment.set_lstrip_blocks(true);
    environment
        .add_template(
            "blueprinter/system.j2",
            include_str!("templates/blueprinter_system.j2"),
        )
        .context("failed to register blueprinter system prompt template")?;
    environment
        .add_template(
            "blueprinter/user.j2",
            include_str!("templates/blueprinter_user.j2"),
        )
        .context("failed to register blueprinter user prompt template")?;

    Ok(environment)
}
