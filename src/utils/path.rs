use anyhow::{Result, bail, ensure};
use camino::{Utf8Path, Utf8PathBuf};
use std::path::{Component, Path};

pub fn normalize_relative_path(path: &str, description: &str) -> Result<Utf8PathBuf> {
    let trimmed = path.trim();
    ensure!(!trimmed.is_empty(), "{description} cannot be empty");
    ensure!(
        !Utf8Path::new(trimmed).is_absolute()
            && !trimmed.starts_with(['/', '\\'])
            && !looks_like_windows_prefix(trimmed),
        "{description} must be relative, received `{trimmed}`"
    );

    let mut segments = Vec::new();
    for segment in trimmed.split(['/', '\\']) {
        match segment {
            "" | "." => {}
            ".." => bail!("{description} cannot traverse upwards: `{trimmed}`"),
            value => segments.push(value.to_owned()),
        }
    }

    ensure!(
        !segments.is_empty(),
        "{description} normalized to an empty value"
    );
    Ok(posix_path_from_segments(segments))
}

pub fn relative_path_from_root(root: &Path, path: &Path) -> Utf8PathBuf {
    let relative = path.strip_prefix(root).unwrap_or(path);
    let segments = relative
        .components()
        .filter_map(|component| match component {
            Component::Normal(value) => Some(value.to_string_lossy().into_owned()),
            Component::CurDir => None,
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => None,
        })
        .collect::<Vec<_>>();

    posix_path_from_segments(segments)
}

pub fn container_path(mount_point: &str, relative_path: &Utf8Path) -> String {
    format!("{mount_point}/{}", relative_path.as_str())
}

pub fn docker_bind_mount(source: &Path, target: &str, readonly: bool) -> Result<String> {
    ensure!(
        source.is_absolute(),
        "docker bind mount source must be absolute, received {}",
        source.display()
    );
    let mut mount = format!(
        "type=bind,source={},target={target}",
        source.to_string_lossy()
    );
    if readonly {
        mount.push_str(",readonly");
    }
    Ok(mount)
}

fn posix_path_from_segments(segments: Vec<String>) -> Utf8PathBuf {
    Utf8PathBuf::from(segments.join("/"))
}

fn looks_like_windows_prefix(path: &str) -> bool {
    let bytes = path.as_bytes();
    bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}

#[cfg(test)]
mod tests {
    use super::{
        container_path, docker_bind_mount, normalize_relative_path, relative_path_from_root,
    };
    use camino::Utf8Path;
    use std::path::Path;

    #[test]
    fn normalize_relative_path_rejects_absolute_windows_paths() {
        let error = normalize_relative_path("C:\\temp\\app.ts", "generated file path")
            .expect_err("absolute windows path should fail");

        assert!(error.to_string().contains("must be relative"));
    }

    #[test]
    fn normalize_relative_path_preserves_posix_output() {
        let path = normalize_relative_path("src\\server.ts", "generated file path")
            .expect("path should normalize");

        assert_eq!(path.as_str(), "src/server.ts");
    }

    #[test]
    fn relative_path_from_root_uses_posix_separators() {
        let root = Path::new("modern_app");
        let path = Path::new("modern_app/tests/server.test.ts");

        assert_eq!(
            relative_path_from_root(root, path).as_str(),
            "tests/server.test.ts"
        );
    }

    #[test]
    fn container_path_joins_mount_and_relative_path() {
        assert_eq!(
            container_path("/app", Utf8Path::new("tests/server.test.ts")),
            "/app/tests/server.test.ts"
        );
    }

    #[test]
    fn docker_bind_mount_rejects_relative_source_paths() {
        let error = docker_bind_mount(Path::new("modern_app"), "/app", true)
            .expect_err("relative source path should fail");

        assert!(error.to_string().contains("must be absolute"));
    }
}
