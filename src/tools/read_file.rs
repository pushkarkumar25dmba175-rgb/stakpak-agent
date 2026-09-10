//! `read_file` — read a file with line numbers.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};

use crate::tools::{Effect, Tool, ToolContext, ToolOutput, arg_str, arg_u64, truncate_output};

pub struct ReadFile;

const DEFAULT_LIMIT: u64 = 2000;

#[async_trait]
impl Tool for ReadFile {
    fn name(&self) -> &'static str {
        "read_file"
    }

    fn description(&self) -> &'static str {
        "Read a text file from the working directory. Output is prefixed with line numbers so \
         you can refer to specific lines. Use `offset` and `limit` to page through large files."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file, relative to the working directory."},
                "offset": {"type": "integer", "description": "1-based line to start from. Defaults to 1."},
                "limit": {"type": "integer", "description": "Maximum lines to return. Defaults to 2000."}
            },
            "required": ["path"]
        })
    }

    fn effect(&self) -> Effect {
        Effect::Read
    }

    fn summarize(&self, input: &Value) -> String {
        format!(
            "read {}",
            input.get("path").and_then(|p| p.as_str()).unwrap_or("?")
        )
    }

    async fn run(&self, ctx: &ToolContext, input: &Value) -> Result<ToolOutput> {
        let path = ctx.resolve(arg_str(input, "path")?)?;
        if !path.exists() {
            return Ok(ToolOutput::error(format!(
                "{} does not exist",
                ctx.display(&path)
            )));
        }
        if path.is_dir() {
            return Ok(ToolOutput::error(format!(
                "{} is a directory; use list_files instead",
                ctx.display(&path)
            )));
        }

        let bytes = tokio::fs::read(&path).await?;
        let text = match String::from_utf8(bytes) {
            Ok(text) => text,
            Err(err) => {
                return Ok(ToolOutput::error(format!(
                    "{} is not valid UTF-8 ({} bytes); it looks like a binary file",
                    ctx.display(&path),
                    err.as_bytes().len()
                )));
            }
        };

        let offset = arg_u64(input, "offset", 1).max(1) as usize;
        let limit = arg_u64(input, "limit", DEFAULT_LIMIT).max(1) as usize;
        let total = text.lines().count();

        let numbered: String = text
            .lines()
            .enumerate()
            .skip(offset - 1)
            .take(limit)
            .map(|(i, line)| format!("{:>6}\t{}\n", i + 1, line))
            .collect();

        if numbered.is_empty() {
            return Ok(ToolOutput::ok(format!(
                "{} has {total} lines; nothing to show from line {offset}",
                ctx.display(&path)
            )));
        }

        let shown_to = (offset - 1 + limit).min(total);
        let mut out = numbered;
        if shown_to < total {
            out.push_str(&format!(
                "\n[showing lines {offset}-{shown_to} of {total}; call again with offset={} for more]\n",
                shown_to + 1
            ));
        }
        Ok(ToolOutput::ok(truncate_output(
            &out,
            ctx.config.tools.max_output_bytes,
        )))
    }
}
