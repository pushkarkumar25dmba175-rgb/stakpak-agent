//! `edit_file` — exact string replacement in an existing file.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};

use crate::tools::{Effect, Tool, ToolContext, ToolOutput, arg_bool, arg_str};

pub struct EditFile;

#[async_trait]
impl Tool for EditFile {
    fn name(&self) -> &'static str {
        "edit_file"
    }

    fn description(&self) -> &'static str {
        "Replace an exact string in a file. `old_string` must appear exactly once unless \
         `replace_all` is true, so include enough surrounding context to make it unique. \
         The file is backed up before the change."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File to edit, relative to the working directory."},
                "old_string": {"type": "string", "description": "Exact text to replace, including indentation."},
                "new_string": {"type": "string", "description": "Replacement text."},
                "replace_all": {"type": "boolean", "description": "Replace every occurrence instead of requiring exactly one. Defaults to false."}
            },
            "required": ["path", "old_string", "new_string"]
        })
    }

    fn effect(&self) -> Effect {
        Effect::Mutate
    }

    fn summarize(&self, input: &Value) -> String {
        format!(
            "edit {}",
            input.get("path").and_then(|p| p.as_str()).unwrap_or("?")
        )
    }

    async fn run(&self, ctx: &ToolContext, input: &Value) -> Result<ToolOutput> {
        let path = ctx.resolve(arg_str(input, "path")?)?;
        let old = arg_str(input, "old_string")?;
        let new = arg_str(input, "new_string")?;
        let replace_all = arg_bool(input, "replace_all", false);

        if !path.exists() {
            return Ok(ToolOutput::error(format!(
                "{} does not exist; use write_file to create it",
                ctx.display(&path)
            )));
        }
        if old == new {
            return Ok(ToolOutput::error(
                "old_string and new_string are identical; nothing to do".to_string(),
            ));
        }

        let original = tokio::fs::read_to_string(&path).await?;
        let occurrences = original.matches(old).count();
        match occurrences {
            0 => {
                return Ok(ToolOutput::error(format!(
                    "old_string was not found in {}. Read the file again — whitespace and \
                     indentation must match exactly.",
                    ctx.display(&path)
                )));
            }
            n if n > 1 && !replace_all => {
                return Ok(ToolOutput::error(format!(
                    "old_string appears {n} times in {}. Add surrounding context to make it \
                     unique, or pass replace_all: true.",
                    ctx.display(&path)
                )));
            }
            _ => {}
        }

        let updated = if replace_all {
            original.replace(old, new)
        } else {
            original.replacen(old, new, 1)
        };

        let backup = ctx.backups.snapshot(&path, self.name())?;
        tokio::fs::write(&path, &updated).await?;

        let mut out = format!(
            "Edited {} ({} replacement{})\n\n{}",
            ctx.display(&path),
            occurrences.min(if replace_all { occurrences } else { 1 }),
            if replace_all && occurrences > 1 {
                "s"
            } else {
                ""
            },
            crate::ui::unified_diff(&original, &updated, &ctx.display(&path))
        );
        if let Some(entry) = backup {
            out.push_str(&format!("\nBackup id: {}", entry.id));
        }
        Ok(ToolOutput::ok(out))
    }
}
