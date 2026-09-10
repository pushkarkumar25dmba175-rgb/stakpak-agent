//! `list_files` — walk the tree, honouring .gitignore.

use anyhow::Result;
use async_trait::async_trait;
use ignore::WalkBuilder;
use serde_json::{Value, json};

use crate::tools::{Effect, Tool, ToolContext, ToolOutput, arg_u64, truncate_output};

pub struct ListFiles;

const DEFAULT_DEPTH: u64 = 3;
const MAX_ENTRIES: usize = 1000;

#[async_trait]
impl Tool for ListFiles {
    fn name(&self) -> &'static str {
        "list_files"
    }

    fn description(&self) -> &'static str {
        "List files and directories, skipping anything matched by .gitignore and hidden \
         entries. Use this to orient yourself before reading files."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to list. Defaults to the working directory."},
                "depth": {"type": "integer", "description": "How many levels to descend. Defaults to 3."}
            }
        })
    }

    fn effect(&self) -> Effect {
        Effect::Read
    }

    fn summarize(&self, input: &Value) -> String {
        format!(
            "list {}",
            input.get("path").and_then(|p| p.as_str()).unwrap_or(".")
        )
    }

    async fn run(&self, ctx: &ToolContext, input: &Value) -> Result<ToolOutput> {
        let raw = input.get("path").and_then(|p| p.as_str()).unwrap_or(".");
        let root = ctx.resolve(raw)?;
        if !root.exists() {
            return Ok(ToolOutput::error(format!(
                "{} does not exist",
                ctx.display(&root)
            )));
        }

        let depth = arg_u64(input, "depth", DEFAULT_DEPTH).max(1) as usize;
        let mut entries = Vec::new();
        let mut truncated = false;

        for result in WalkBuilder::new(&root)
            .max_depth(Some(depth))
            .hidden(true)
            .git_ignore(true)
            .git_global(false)
            // A directory with a .gitignore but no .git is still worth
            // respecting; the agent often works on a copied or vendored tree.
            .require_git(false)
            .build()
        {
            let entry = match result {
                Ok(e) => e,
                // An unreadable subdirectory should not abort the whole listing.
                Err(err) => {
                    entries.push(format!("  (skipped: {err})"));
                    continue;
                }
            };
            if entry.path() == root {
                continue;
            }
            if entries.len() >= MAX_ENTRIES {
                truncated = true;
                break;
            }
            let is_dir = entry.file_type().is_some_and(|t| t.is_dir());
            let rendered = ctx.display(entry.path());
            entries.push(if is_dir {
                format!("{rendered}/")
            } else {
                rendered
            });
        }

        entries.sort();
        if entries.is_empty() {
            return Ok(ToolOutput::ok(format!("{} is empty", ctx.display(&root))));
        }

        let mut out = entries.join("\n");
        if truncated {
            out.push_str(&format!(
                "\n\n[stopped after {MAX_ENTRIES} entries; narrow the path or reduce depth]"
            ));
        }
        Ok(ToolOutput::ok(truncate_output(
            &out,
            ctx.config.tools.max_output_bytes,
        )))
    }
}
