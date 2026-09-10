//! The agent's tool surface.

mod edit_file;
mod list_files;
mod read_file;
mod run_command;
mod search;
mod write_file;

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use anyhow::{Result, bail};
use async_trait::async_trait;
use serde_json::Value;

use crate::agent::message::ToolSpec;
use crate::backup::BackupStore;
use crate::config::Config;

/// What a tool does to the machine, which is what approval policy keys off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effect {
    /// Only observes: safe to auto-run under the `writes` policy.
    Read,
    /// Changes files or runs commands.
    Mutate,
}

pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
}

impl ToolOutput {
    pub fn ok(content: impl Into<String>) -> Self {
        ToolOutput {
            content: content.into(),
            is_error: false,
        }
    }

    pub fn error(content: impl Into<String>) -> Self {
        ToolOutput {
            content: content.into(),
            is_error: true,
        }
    }
}

pub struct ToolContext {
    pub workdir: PathBuf,
    pub config: Arc<Config>,
    pub backups: Arc<BackupStore>,
}

impl ToolContext {
    /// Resolves a model-supplied path against the working directory and, unless
    /// configured otherwise, refuses anything that escapes it.
    ///
    /// The path is normalized lexically rather than with `canonicalize`, which
    /// would fail for files the agent is about to create.
    pub fn resolve(&self, raw: &str) -> Result<PathBuf> {
        let requested = Path::new(raw);
        let joined = if requested.is_absolute() {
            requested.to_path_buf()
        } else {
            self.workdir.join(requested)
        };
        let normalized = normalize(&joined);

        if !self.config.security.allow_outside_workdir {
            let root = normalize(&self.workdir);
            if !normalized.starts_with(&root) {
                bail!(
                    "path {} is outside the working directory {}. Set security.allow_outside_workdir = true to permit this.",
                    normalized.display(),
                    root.display()
                );
            }
        }
        Ok(normalized)
    }

    /// Path as shown to the model: relative to the working directory when it
    /// lives underneath it, which keeps transcripts readable.
    pub fn display(&self, path: &Path) -> String {
        path.strip_prefix(&self.workdir)
            .unwrap_or(path)
            .display()
            .to_string()
    }
}

/// Lexical `..`/`.` resolution, with no filesystem access.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn input_schema(&self) -> Value;
    fn effect(&self) -> Effect;
    /// One line describing this specific call, shown in the approval prompt and
    /// in the transcript.
    fn summarize(&self, input: &Value) -> String;
    async fn run(&self, ctx: &ToolContext, input: &Value) -> Result<ToolOutput>;
}

pub struct ToolRegistry {
    tools: Vec<Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn with_defaults() -> Self {
        ToolRegistry {
            tools: vec![
                Arc::new(read_file::ReadFile),
                Arc::new(write_file::WriteFile),
                Arc::new(edit_file::EditFile),
                Arc::new(list_files::ListFiles),
                Arc::new(search::Search),
                Arc::new(run_command::RunCommand),
            ],
        }
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.iter().find(|t| t.name() == name).cloned()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Arc<dyn Tool>> {
        self.tools.iter()
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools
            .iter()
            .map(|t| ToolSpec {
                name: t.name().to_string(),
                description: t.description().to_string(),
                input_schema: t.input_schema(),
            })
            .collect()
    }
}

/// Trims oversized tool output, keeping the head and the tail. The tail usually
/// holds the error a command died with, so dropping it would defeat the point.
pub fn truncate_output(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let head_len = max_bytes * 2 / 3;
    let tail_len = max_bytes - head_len;
    let head = floor_char_boundary(text, head_len);
    let tail_start = ceil_char_boundary(text, text.len() - tail_len);
    format!(
        "{}\n\n[... {} bytes truncated ...]\n\n{}",
        &text[..head],
        text.len() - head - (text.len() - tail_start),
        &text[tail_start..]
    )
}

fn floor_char_boundary(text: &str, mut index: usize) -> usize {
    index = index.min(text.len());
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn ceil_char_boundary(text: &str, mut index: usize) -> usize {
    index = index.min(text.len());
    while index < text.len() && !text.is_char_boundary(index) {
        index += 1;
    }
    index
}

/// Reads a required string argument.
pub fn arg_str<'a>(input: &'a Value, key: &str) -> Result<&'a str> {
    match input.get(key).and_then(|v| v.as_str()) {
        Some(v) => Ok(v),
        None => bail!("missing required string argument `{key}`"),
    }
}

pub fn arg_u64(input: &Value, key: &str, default: u64) -> u64 {
    input.get(key).and_then(|v| v.as_u64()).unwrap_or(default)
}

pub fn arg_bool(input: &Value, key: &str, default: bool) -> bool {
    input.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(allow_outside: bool) -> ToolContext {
        let mut config = Config::default();
        config.security.allow_outside_workdir = allow_outside;
        ToolContext {
            workdir: PathBuf::from("/srv/app"),
            config: Arc::new(config),
            backups: Arc::new(BackupStore::new(Path::new("/tmp/none"), "s", false)),
        }
    }

    #[test]
    fn relative_paths_resolve_under_the_workdir() {
        let ctx = ctx(false);
        assert_eq!(
            ctx.resolve("infra/main.tf").unwrap(),
            PathBuf::from("/srv/app/infra/main.tf")
        );
    }

    #[test]
    fn traversal_out_of_the_workdir_is_refused() {
        let ctx = ctx(false);
        let err = ctx.resolve("../../etc/shadow").unwrap_err().to_string();
        assert!(err.contains("outside the working directory"));
        assert!(ctx.resolve("/etc/shadow").is_err());
    }

    #[test]
    fn traversal_inside_the_workdir_is_fine() {
        let ctx = ctx(false);
        assert_eq!(
            ctx.resolve("infra/../main.tf").unwrap(),
            PathBuf::from("/srv/app/main.tf")
        );
    }

    #[test]
    fn opting_out_permits_absolute_paths() {
        assert!(ctx(true).resolve("/etc/hosts").is_ok());
    }

    #[test]
    fn truncation_keeps_head_and_tail() {
        let text = format!("{}{}", "a".repeat(500), "TAIL_MARKER");
        let out = truncate_output(&text, 100);
        assert!(out.starts_with("aaa"));
        assert!(out.contains("truncated"));
        assert!(out.ends_with("TAIL_MARKER"));
    }

    #[test]
    fn short_output_is_untouched() {
        assert_eq!(truncate_output("hello", 100), "hello");
    }

    #[test]
    fn truncation_respects_char_boundaries() {
        let text = "é".repeat(400);
        let out = truncate_output(&text, 100);
        assert!(out.contains("truncated"));
    }

    #[test]
    fn every_default_tool_has_an_object_schema() {
        for tool in ToolRegistry::with_defaults().iter() {
            let schema = tool.input_schema();
            assert_eq!(schema["type"], "object", "{}", tool.name());
            assert!(schema.get("properties").is_some(), "{}", tool.name());
            assert!(!tool.description().is_empty(), "{}", tool.name());
        }
    }
}
