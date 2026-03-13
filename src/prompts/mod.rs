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

#[cfg(test)]
mod tests {
    use super::render;
    use serde::Serialize;

    #[derive(Serialize)]
    struct BlueprinterUserContext {
        legacy_dir_path: String,
        dependency_graph: String,
    }

    #[test]
    fn render_blueprinter_system_template() {
        let result = render::<()>("blueprinter/system.j2", ());
        assert!(result.is_ok());
        let rendered = result.unwrap();
        assert!(rendered.contains("Staff Software Engineer"));
        assert!(rendered.contains("legacy migrations"));
    }

    #[test]
    fn render_blueprinter_user_template() {
        let context = BlueprinterUserContext {
            legacy_dir_path: "legacy_app".to_string(),
            dependency_graph: String::from(r#"{"files": []}"#),
        };
        let result = render("blueprinter/user.j2", context);
        assert!(result.is_ok());
        let rendered = result.unwrap();
        assert!(rendered.contains("legacy_app"));
        let expected = r#"{"files": []}"#;
        assert!(rendered.contains(expected));
    }

    #[test]
    fn render_returns_error_for_nonexistent_template() {
        let result = render::<()>("nonexistent/template.j2", ());
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("failed to load prompt template"));
    }
}
