//! `run_command` — run a shell command in the working directory.
//!
//! By the time this runs, secret placeholders in the command have already been
//! swapped back for real values by the agent loop, so a command can use a
//! credential the model never saw.

use std::process::Stdio;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::process::Command;

use crate::config::SecurityConfig;
use crate::tools::{Effect, Tool, ToolContext, ToolOutput, arg_str, arg_u64, truncate_output};

pub struct RunCommand;

#[async_trait]
impl Tool for RunCommand {
    fn name(&self) -> &'static str {
        "run_command"
    }

    fn description(&self) -> &'static str {
        "Run a shell command in the working directory and return its stdout, stderr, and exit \
         code. Use this for git, terraform, kubectl, docker, package managers, and tests. \
         Commands run non-interactively, so pass flags like -y or --no-input rather than \
         waiting for a prompt."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command line to run, interpreted by the shell."},
                "workdir": {"type": "string", "description": "Directory to run in, relative to the working directory. Defaults to the working directory."},
                "timeout_secs": {"type": "integer", "description": "Kill the command after this many seconds."}
            },
            "required": ["command"]
        })
    }

    fn effect(&self) -> Effect {
        Effect::Mutate
    }

    fn summarize(&self, input: &Value) -> String {
        let command = input
            .get("command")
            .and_then(|c| c.as_str())
            .unwrap_or("?")
            .replace('\n', " ");
        format!("run: {}", command.chars().take(120).collect::<String>())
    }

    async fn run(&self, ctx: &ToolContext, input: &Value) -> Result<ToolOutput> {
        let command = arg_str(input, "command")?;

        if let Some(pattern) = denied_by(&ctx.config.security, command) {
            return Ok(ToolOutput::error(format!(
                "Refused: the command matches the denied pattern {pattern:?} from \
                 security.denied_command_patterns."
            )));
        }

        let cwd = match input.get("workdir").and_then(|w| w.as_str()) {
            Some(dir) => ctx.resolve(dir)?,
            None => ctx.workdir.clone(),
        };
        if !cwd.is_dir() {
            return Ok(ToolOutput::error(format!(
                "{} is not a directory",
                ctx.display(&cwd)
            )));
        }

        let timeout = Duration::from_secs(
            arg_u64(input, "timeout_secs", ctx.config.tools.command_timeout_secs).clamp(1, 3600),
        );

        let child = Command::new("sh")
            .arg("-c")
            .arg(command)
            .current_dir(&cwd)
            // Nothing is going to answer a prompt, so close stdin outright.
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()?;

        let output = match tokio::time::timeout(timeout, child.wait_with_output()).await {
            Ok(result) => result?,
            Err(_) => {
                return Ok(ToolOutput::error(format!(
                    "Command timed out after {}s and was killed. Re-run it in the background, \
                     or raise timeout_secs.",
                    timeout.as_secs()
                )));
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let code = output.status.code();

        let mut rendered = String::new();
        if !stdout.trim().is_empty() {
            rendered.push_str(stdout.trim_end());
            rendered.push('\n');
        }
        if !stderr.trim().is_empty() {
            if !rendered.is_empty() {
                rendered.push('\n');
            }
            rendered.push_str("stderr:\n");
            rendered.push_str(stderr.trim_end());
            rendered.push('\n');
        }
        if rendered.is_empty() {
            rendered.push_str("(no output)\n");
        }

        let succeeded = code == Some(0);
        let status_line = match code {
            Some(0) => "exit 0".to_string(),
            Some(c) => format!("exit {c}"),
            None => "killed by signal".to_string(),
        };

        let body = truncate_output(&rendered, ctx.config.tools.max_output_bytes);
        let content = format!("{body}\n[{status_line}]");
        Ok(if succeeded {
            ToolOutput::ok(content)
        } else {
            ToolOutput::error(content)
        })
    }
}

/// Returns the deny pattern a command matched, if any.
fn denied_by<'a>(security: &'a SecurityConfig, command: &str) -> Option<&'a str> {
    let normalized = command.split_whitespace().collect::<Vec<_>>().join(" ");
    security
        .denied_command_patterns
        .iter()
        .find(|p| normalized.contains(p.as_str()) || command.contains(p.as_str()))
        .map(|p| p.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backup::BackupStore;
    use crate::config::Config;
    use std::path::Path;
    use std::sync::Arc;

    fn ctx() -> ToolContext {
        ToolContext {
            workdir: std::env::temp_dir(),
            config: Arc::new(Config::default()),
            backups: Arc::new(BackupStore::new(Path::new("/tmp/none"), "s", false)),
        }
    }

    #[test]
    fn destructive_commands_are_denied() {
        let security = crate::config::SecurityConfig::default();
        assert!(denied_by(&security, "sudo rm -rf / --no-preserve-root").is_some());
        // Extra whitespace should not slip past the check.
        assert!(denied_by(&security, "rm   -rf   /").is_some());
        assert!(denied_by(&security, "rm -rf ./build").is_none());
    }

    #[tokio::test]
    async fn captures_stdout_and_exit_code() {
        let out = RunCommand
            .run(&ctx(), &json!({"command": "echo hello"}))
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("hello"));
        assert!(out.content.contains("[exit 0]"));
    }

    #[tokio::test]
    async fn a_failing_command_is_reported_as_an_error() {
        let out = RunCommand
            .run(&ctx(), &json!({"command": "echo oops >&2; exit 3"}))
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("stderr:"));
        assert!(out.content.contains("[exit 3]"));
    }

    #[tokio::test]
    async fn slow_commands_are_killed() {
        let out = RunCommand
            .run(&ctx(), &json!({"command": "sleep 5", "timeout_secs": 1}))
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("timed out"));
    }
}
