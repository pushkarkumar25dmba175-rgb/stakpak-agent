//! `search` — regex search across the tree, honouring .gitignore.

use anyhow::Result;
use async_trait::async_trait;
use ignore::WalkBuilder;
use regex::RegexBuilder;
use serde_json::{Value, json};

use crate::tools::{
    Effect, Tool, ToolContext, ToolOutput, arg_bool, arg_str, arg_u64, truncate_output,
};

pub struct Search;

const DEFAULT_MAX_RESULTS: u64 = 200;
/// Files larger than this are almost always build output or data dumps.
const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;

#[async_trait]
impl Tool for Search {
    fn name(&self) -> &'static str {
        "search"
    }

    fn description(&self) -> &'static str {
        "Search file contents with a regular expression, skipping .gitignore'd, hidden, and \
         binary files. Returns `path:line: text` for each match. Use `glob` to restrict the \
         search to matching filenames, e.g. \"*.tf\"."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Rust regex to search for."},
                "path": {"type": "string", "description": "Directory to search. Defaults to the working directory."},
                "glob": {"type": "string", "description": "Only search files whose name matches this glob, e.g. \"*.yaml\"."},
                "case_sensitive": {"type": "boolean", "description": "Defaults to false."},
                "max_results": {"type": "integer", "description": "Defaults to 200."}
            },
            "required": ["pattern"]
        })
    }

    fn effect(&self) -> Effect {
        Effect::Read
    }

    fn summarize(&self, input: &Value) -> String {
        format!(
            "search /{}/",
            input.get("pattern").and_then(|p| p.as_str()).unwrap_or("?")
        )
    }

    async fn run(&self, ctx: &ToolContext, input: &Value) -> Result<ToolOutput> {
        let pattern = arg_str(input, "pattern")?;
        let case_sensitive = arg_bool(input, "case_sensitive", false);
        let regex = match RegexBuilder::new(pattern)
            .case_insensitive(!case_sensitive)
            .build()
        {
            Ok(r) => r,
            Err(err) => return Ok(ToolOutput::error(format!("invalid regex: {err}"))),
        };

        let root = ctx.resolve(input.get("path").and_then(|p| p.as_str()).unwrap_or("."))?;
        if !root.exists() {
            return Ok(ToolOutput::error(format!(
                "{} does not exist",
                ctx.display(&root)
            )));
        }

        let mut walker = WalkBuilder::new(&root);
        walker
            .standard_filters(true)
            .hidden(true)
            .git_ignore(true)
            .git_global(false)
            .require_git(false);

        if let Some(glob) = input.get("glob").and_then(|g| g.as_str()) {
            let mut builder = ignore::overrides::OverrideBuilder::new(&root);
            if let Err(err) = builder.add(glob) {
                return Ok(ToolOutput::error(format!("invalid glob {glob:?}: {err}")));
            }
            match builder.build() {
                Ok(overrides) => {
                    walker.overrides(overrides);
                }
                Err(err) => return Ok(ToolOutput::error(format!("invalid glob {glob:?}: {err}"))),
            }
        }

        let max_results = arg_u64(input, "max_results", DEFAULT_MAX_RESULTS).max(1) as usize;
        let mut hits = Vec::new();
        let mut files_matched = 0usize;
        let mut truncated = false;

        'walk: for result in walker.build() {
            let Ok(entry) = result else { continue };
            if !entry.file_type().is_some_and(|t| t.is_file()) {
                continue;
            }
            if entry.metadata().map(|m| m.len()).unwrap_or(0) > MAX_FILE_BYTES {
                continue;
            }
            // Binary files are skipped rather than reported as garbage.
            let Ok(text) = std::fs::read_to_string(entry.path()) else {
                continue;
            };

            let mut matched_here = false;
            for (i, line) in text.lines().enumerate() {
                if !regex.is_match(line) {
                    continue;
                }
                matched_here = true;
                if hits.len() >= max_results {
                    truncated = true;
                    break 'walk;
                }
                hits.push(format!(
                    "{}:{}: {}",
                    ctx.display(entry.path()),
                    i + 1,
                    line.trim_end().chars().take(300).collect::<String>()
                ));
            }
            if matched_here {
                files_matched += 1;
            }
        }

        if hits.is_empty() {
            return Ok(ToolOutput::ok(format!(
                "No matches for /{pattern}/ under {}",
                ctx.display(&root)
            )));
        }

        let mut out = format!(
            "{} match{} across {files_matched} file{}:\n\n{}",
            hits.len(),
            if hits.len() == 1 { "" } else { "es" },
            if files_matched == 1 { "" } else { "s" },
            hits.join("\n")
        );
        if truncated {
            out.push_str(&format!(
                "\n\n[stopped at max_results={max_results}; narrow the pattern for the rest]"
            ));
        }
        Ok(ToolOutput::ok(truncate_output(
            &out,
            ctx.config.tools.max_output_bytes,
        )))
    }
}
