//! OpenAI-compatible chat-completions endpoints (OpenAI itself, vLLM,
//! Ollama's compat layer, OpenRouter, and friends).
//!
//! The internal message model is Anthropic-shaped, so this module's real work
//! is translating tool calls in both directions.

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde_json::{Value, json};

use crate::agent::message::{
    Completion, CompletionRequest, ContentBlock, Message, Role, StopReason, Usage,
};
use crate::agent::provider::{LlmProvider, http_client};
use crate::config::ProviderConfig;

pub struct OpenAiProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl OpenAiProvider {
    pub fn new(config: &ProviderConfig) -> Result<Self> {
        Ok(OpenAiProvider {
            client: http_client(config)?,
            api_key: config.api_key()?,
            base_url: config.base_url.trim_end_matches('/').to_string(),
            model: config.model.clone(),
        })
    }
}

#[async_trait]
impl LlmProvider for OpenAiProvider {
    fn name(&self) -> &'static str {
        "openai"
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn complete(&self, request: &CompletionRequest) -> Result<Completion> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let mut body = json!({
            "model": self.model,
            "max_tokens": request.max_tokens.max(1),
            "temperature": request.temperature,
            "messages": to_openai_messages(request),
        });
        if !request.tools.is_empty() {
            body["tools"] = json!(
                request
                    .tools
                    .iter()
                    .map(|t| json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.input_schema,
                        }
                    }))
                    .collect::<Vec<_>>()
            );
        }

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("calling {url}"))?;

        let status = response.status();
        let text = response.text().await.context("reading response body")?;
        if !status.is_success() {
            let detail: String = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|v| {
                    v.get("error")
                        .and_then(|e| e.get("message"))
                        .and_then(|m| m.as_str())
                        .map(str::to_string)
                })
                .unwrap_or_else(|| text.chars().take(400).collect());
            bail!("OpenAI-compatible API error {status}: {detail}");
        }

        let parsed: Value = serde_json::from_str(&text).context("parsing response as JSON")?;
        parse_completion(&parsed)
    }
}

fn to_openai_messages(request: &CompletionRequest) -> Vec<Value> {
    let mut out = Vec::new();
    if !request.system.is_empty() {
        out.push(json!({"role": "system", "content": request.system}));
    }

    for message in &request.messages {
        match message.role {
            Role::User => {
                // A user turn is either plain text or a batch of tool results,
                // and tool results have to become their own `tool` messages.
                let mut text = String::new();
                for block in &message.content {
                    match block {
                        ContentBlock::Text { text: t } => {
                            if !text.is_empty() {
                                text.push('\n');
                            }
                            text.push_str(t);
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => out.push(json!({
                            "role": "tool",
                            "tool_call_id": tool_use_id,
                            "content": content,
                        })),
                        ContentBlock::ToolUse { .. } => {}
                    }
                }
                if !text.is_empty() {
                    out.push(json!({"role": "user", "content": text}));
                }
            }
            Role::Assistant => {
                let text = message.text();
                let tool_calls: Vec<Value> = message
                    .tool_uses()
                    .iter()
                    .map(|u| {
                        json!({
                            "id": u.id,
                            "type": "function",
                            "function": {
                                "name": u.name,
                                // OpenAI wants arguments as a JSON string.
                                "arguments": u.input.to_string(),
                            }
                        })
                    })
                    .collect();

                let mut msg = json!({"role": "assistant"});
                msg["content"] = if text.is_empty() {
                    Value::Null
                } else {
                    json!(text)
                };
                if !tool_calls.is_empty() {
                    msg["tool_calls"] = json!(tool_calls);
                }
                out.push(msg);
            }
        }
    }
    out
}

fn parse_completion(parsed: &Value) -> Result<Completion> {
    let choice = parsed
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|c| c.first())
        .context("response had no choices")?;
    let message = choice.get("message").context("choice had no message")?;

    let mut content = Vec::new();
    if let Some(text) = message.get("content").and_then(|c| c.as_str())
        && !text.is_empty()
    {
        content.push(ContentBlock::Text {
            text: text.to_string(),
        });
    }

    let mut saw_tool_call = false;
    if let Some(calls) = message.get("tool_calls").and_then(|c| c.as_array()) {
        for call in calls {
            saw_tool_call = true;
            let function = call.get("function").unwrap_or(&Value::Null);
            let raw_args = function
                .get("arguments")
                .and_then(|a| a.as_str())
                .unwrap_or("{}");
            // Some compatible servers send an object rather than a string, and
            // some send arguments that do not parse at all.
            let input = serde_json::from_str::<Value>(raw_args).unwrap_or_else(|_| {
                function
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(|| json!({}))
            });
            content.push(ContentBlock::ToolUse {
                id: call
                    .get("id")
                    .and_then(|i| i.as_str())
                    .unwrap_or_default()
                    .to_string(),
                name: function
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or_default()
                    .to_string(),
                input,
            });
        }
    }

    let stop_reason = match choice.get("finish_reason").and_then(|f| f.as_str()) {
        Some("tool_calls") | Some("function_call") => StopReason::ToolUse,
        Some("length") => StopReason::MaxTokens,
        Some("stop") => StopReason::EndTurn,
        Some(other) => StopReason::Other(other.to_string()),
        // Not every compatible server sets finish_reason.
        None if saw_tool_call => StopReason::ToolUse,
        None => StopReason::EndTurn,
    };

    let usage = parsed
        .get("usage")
        .map(|u| Usage {
            input_tokens: u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
            output_tokens: u
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
        })
        .unwrap_or_default();

    Ok(Completion {
        message: Message::assistant(content),
        stop_reason,
        usage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::message::ToolSpec;

    #[test]
    fn tool_results_become_tool_role_messages() {
        let request = CompletionRequest {
            system: "be careful".into(),
            messages: vec![
                Message::user("check the cluster"),
                Message::assistant(vec![ContentBlock::ToolUse {
                    id: "tu_1".into(),
                    name: "run_command".into(),
                    input: json!({"command": "kubectl get pods"}),
                }]),
                Message::tool_results(vec![ContentBlock::ToolResult {
                    tool_use_id: "tu_1".into(),
                    content: "no pods".into(),
                    is_error: false,
                }]),
            ],
            tools: vec![ToolSpec {
                name: "run_command".into(),
                description: "run".into(),
                input_schema: json!({"type": "object"}),
            }],
            max_tokens: 100,
            temperature: 0.0,
        };

        let messages = to_openai_messages(&request);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[2]["role"], "assistant");
        // Arguments are stringified for the OpenAI schema.
        assert!(messages[2]["tool_calls"][0]["function"]["arguments"].is_string());
        assert_eq!(messages[3]["role"], "tool");
        assert_eq!(messages[3]["tool_call_id"], "tu_1");
    }

    #[test]
    fn parses_tool_calls_back_out() {
        let raw = json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{\"path\":\"main.tf\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7}
        });
        let completion = parse_completion(&raw).unwrap();
        assert_eq!(completion.stop_reason, StopReason::ToolUse);
        let uses = completion.message.tool_uses();
        assert_eq!(uses[0].name, "read_file");
        assert_eq!(uses[0].input["path"], "main.tf");
        assert_eq!(completion.usage.input_tokens, 5);
    }

    #[test]
    fn tool_calls_without_finish_reason_still_count_as_tool_use() {
        let raw = json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "c1",
                        "function": {"name": "list_files", "arguments": "{}"}
                    }]
                }
            }]
        });
        assert_eq!(
            parse_completion(&raw).unwrap().stop_reason,
            StopReason::ToolUse
        );
    }
}
