//! The agent loop: prompt the model, run the tools it asks for, feed the
//! results back, repeat until it stops asking.

pub mod message;
pub mod provider;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};

use crate::config::{ApprovalMode, Config};
use crate::redact::SecretStore;
use crate::session::Session;
use crate::tools::{Effect, ToolContext, ToolRegistry, truncate_output};
use crate::ui::{self, Approval};

use message::{Completion, CompletionRequest, ContentBlock, Message, StopReason, Usage};
use provider::LlmProvider;

pub struct Agent {
    provider: Box<dyn LlmProvider>,
    tools: ToolRegistry,
    tool_ctx: ToolContext,
    config: Arc<Config>,
    secrets: SecretStore,
    state_dir: PathBuf,
    pub session: Session,
    /// Set by `--yes` or by answering "a" at an approval prompt.
    auto_approve: bool,
    /// Running token totals for the session, reported after each turn.
    usage: Usage,
}

impl Agent {
    pub fn new(
        config: Config,
        workdir: &Path,
        state_dir: &Path,
        session: Session,
        auto_approve: bool,
    ) -> Result<Self> {
        let config = Arc::new(config);
        let provider = provider::build(&config.provider)?;

        let mut secrets = if config.security.redact_secrets {
            SecretStore::new(true, &config.security.extra_secret_patterns)?
        } else {
            SecretStore::disabled()
        };
        if config.security.redact_env {
            secrets.register_env();
        }

        let backups = Arc::new(crate::backup::BackupStore::new(
            state_dir,
            &session.id,
            config.tools.backup_files,
        ));

        Ok(Agent {
            tool_ctx: ToolContext {
                workdir: workdir.to_path_buf(),
                config: Arc::clone(&config),
                backups,
            },
            tools: ToolRegistry::with_defaults(),
            provider,
            config,
            secrets,
            state_dir: state_dir.to_path_buf(),
            session,
            auto_approve,
            usage: Usage::default(),
        })
    }

    pub fn model(&self) -> &str {
        self.provider.model()
    }

    pub fn provider_name(&self) -> &'static str {
        self.provider.name()
    }

    pub fn secrets(&self) -> &SecretStore {
        &self.secrets
    }

    /// Runs one user request to completion, including every tool round it takes.
    pub async fn turn(&mut self, input: &str) -> Result<String> {
        self.session.messages.push(Message::user(input));
        self.session.checkpoint(input);

        let mut final_text = String::new();

        for iteration in 1..=self.config.agent.max_iterations {
            let completion = self.complete().await?;
            self.usage.input_tokens += completion.usage.input_tokens;
            self.usage.output_tokens += completion.usage.output_tokens;
            let text = completion.message.text();
            ui::assistant(&text);
            if !text.trim().is_empty() {
                final_text = text;
            }

            let tool_uses: Vec<(String, String, serde_json::Value)> = completion
                .message
                .tool_uses()
                .iter()
                .map(|u| (u.id.to_string(), u.name.to_string(), u.input.clone()))
                .collect();

            self.session.messages.push(completion.message);

            if tool_uses.is_empty() {
                if completion.stop_reason == StopReason::MaxTokens {
                    ui::warn(
                        "the model hit its output limit; ask it to continue if the answer looks cut off",
                    );
                }
                self.save()?;
                self.report_usage();
                return Ok(final_text);
            }

            let mut results = Vec::with_capacity(tool_uses.len());
            for (id, name, input) in tool_uses {
                results.push(self.execute(&id, &name, &input).await);
            }
            self.session.messages.push(Message::tool_results(results));
            self.save()?;

            if iteration == self.config.agent.max_iterations {
                ui::warn(&format!(
                    "stopped after {} tool rounds (agent.max_iterations). Send another message to continue.",
                    self.config.agent.max_iterations
                ));
            }
        }

        Ok(final_text)
    }

    /// Prints the running token total, when the provider reported one.
    fn report_usage(&self) {
        if self.usage.input_tokens == 0 && self.usage.output_tokens == 0 {
            return;
        }
        ui::info(&format!(
            "{} tokens in, {} out this session",
            self.usage.input_tokens, self.usage.output_tokens
        ));
    }

    async fn complete(&self) -> Result<Completion> {
        let request = CompletionRequest {
            system: self.system_prompt(),
            messages: self.session.messages.clone(),
            tools: self.tools.specs(),
            max_tokens: self.config.provider.max_tokens,
            temperature: self.config.provider.temperature,
        };
        self.provider
            .complete(&request)
            .await
            .context("requesting a completion")
    }

    /// Runs a single tool call and returns the block to send back to the model.
    async fn execute(&mut self, id: &str, name: &str, input: &serde_json::Value) -> ContentBlock {
        let Some(tool) = self.tools.get(name) else {
            let known: Vec<&str> = self.tools.iter().map(|t| t.name()).collect();
            return ContentBlock::ToolResult {
                tool_use_id: id.to_string(),
                content: format!(
                    "unknown tool `{name}`. Available tools: {}",
                    known.join(", ")
                ),
                is_error: true,
            };
        };

        let summary = tool.summarize(input);
        ui::tool_call(&summary);

        if self.needs_approval(tool.effect())
            && let Some(denial) = self.request_approval(&summary, input)
        {
            ui::info(&denial);
            return ContentBlock::ToolResult {
                tool_use_id: id.to_string(),
                content: denial,
                is_error: true,
            };
        }

        // Placeholders go back to real credentials only here, on the way into
        // the tool — never into the transcript.
        let restored = self.secrets.restore_value(input);

        let output = match tool.run(&self.tool_ctx, &restored).await {
            Ok(output) => output,
            Err(err) => crate::tools::ToolOutput::error(format!("{name} failed: {err:#}")),
        };

        let redacted = self.secrets.redact(&output.content);
        let content = truncate_output(&redacted, self.config.tools.max_output_bytes);
        ui::tool_result(&content, output.is_error);

        ContentBlock::ToolResult {
            tool_use_id: id.to_string(),
            content,
            is_error: output.is_error,
        }
    }

    fn needs_approval(&self, effect: Effect) -> bool {
        if self.auto_approve {
            return false;
        }
        match self.config.security.approval {
            ApprovalMode::Never => false,
            ApprovalMode::Writes => effect == Effect::Mutate,
            ApprovalMode::Always => true,
        }
    }

    /// Returns `None` when approved, or the message to hand back to the model
    /// when the user declined.
    fn request_approval(&mut self, summary: &str, input: &serde_json::Value) -> Option<String> {
        let detail = serde_json::to_string_pretty(input).ok();
        match ui::ask_approval(summary, detail.as_deref()) {
            Approval::Once => None,
            Approval::Always => {
                self.auto_approve = true;
                None
            }
            Approval::Denied(reason) => Some(format!(
                "This action was not run: {reason}. Do not retry it; ask what to do instead."
            )),
        }
    }

    pub fn save(&mut self) -> Result<()> {
        self.session.save(&self.state_dir)
    }

    fn system_prompt(&self) -> String {
        let mut prompt = format!(
            "You are Stakpak, an autonomous DevOps agent working in a terminal on a real \
             machine. You help with infrastructure, deployments, debugging, and \
             infrastructure-as-code (Terraform, Kubernetes, Docker, CI pipelines, shell).\n\
             \n\
             Working directory: {}\n\
             Platform: {}\n\
             \n\
             How to work:\n\
             - Investigate before you act. Read the files and run the read-only commands that \
               tell you what is actually deployed, rather than assuming.\n\
             - Prefer the smallest change that solves the problem, and follow the conventions \
               already present in the repository.\n\
             - Run one tool at a time when a step depends on the previous result; batch \
               independent lookups together.\n\
             - After changing infrastructure code, validate it (terraform validate, kubectl \
               --dry-run, a linter, the project's tests) before reporting success.\n\
             - Never report something as done that you did not verify. If a command failed, \
               say so and show the error.\n\
             - Destructive actions (deleting resources, force-pushing, dropping data) need the \
               user's explicit go-ahead first. Ask.\n\
             \n\
             Answer in plain prose for the terminal. Keep it short: state what you found, what \
             you changed, and what is left. No headings or bullet lists unless the content \
             really is a list.",
            self.tool_ctx.workdir.display(),
            std::env::consts::OS,
        );

        if self.secrets.is_enabled() {
            prompt.push_str(
                "\n\nSecrets: credentials in tool output are replaced with placeholders of the form \
                 {{SECRET_<kind>_<n>}}. The real values are substituted back in just \
                 before a tool runs, so pass the placeholder through verbatim whenever a \
                 command needs the credential. Never guess at or invent a real value, and \
                 never write a placeholder into a file that gets committed.",
            );
        }

        if !self.config.agent.system_prompt_append.trim().is_empty() {
            prompt.push_str("\n\n");
            prompt.push_str(self.config.agent.system_prompt_append.trim());
        }

        prompt
    }
}
