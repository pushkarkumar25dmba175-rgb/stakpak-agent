//! `write_file` — create or overwrite a file, backing up what was there.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};

use crate::tools::{Effect, Tool, ToolContext, ToolOutput, arg_str};

pub struct WriteFile;

#[async_trait]
impl Tool for WriteFile {
    fn name(&self) -> &'static str {
        "write_file"
    }

    fn description(&self) -> &'static str {
        "Write a file, creating parent directories as needed and overwriting any existing \
         contents. The previous contents are backed up first and can be restored with \
         `stakpak restore`. Prefer edit_file for changing part of an existing file."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to write, relative to the working directory."},
                "content": {"type": "string", "description": "Full contents of the file."}
            },
            "required": ["path", "content"]
        })
    }

    fn effect(&self) -> Effect {
        Effect::Mutate
    }

    fn summarize(&self, input: &Value) -> String {
        let path = input.get("path").and_then(|p| p.as_str()).unwrap_or("?");
        let lines = input
            .get("content")
            .and_then(|c| c.as_str())
            .map(|c| c.lines().count())
            .unwrap_or(0);
        format!("write {path} ({lines} lines)")
    }

    async fn run(&self, ctx: &ToolContext, input: &Value) -> Result<ToolOutput> {
        let path = ctx.resolve(arg_str(input, "path")?)?;
        let content = arg_str(input, "content")?;

        if path.is_dir() {
            return Ok(ToolOutput::error(format!(
                "{} is a directory",
                ctx.display(&path)
            )));
        }

        let existed = path.exists();
        let backup = ctx.backups.snapshot(&path, self.name())?;

        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&path, content).await?;

        let verb = if existed { "Overwrote" } else { "Created" };
        let mut out = format!(
            "{verb} {} ({} lines, {} bytes)",
            ctx.display(&path),
            content.lines().count(),
            content.len()
        );
        if let Some(entry) = backup {
            out.push_str(&format!("\nBackup id: {}", entry.id));
        }
        Ok(ToolOutput::ok(out))
    }
}
