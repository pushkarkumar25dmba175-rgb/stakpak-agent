//! Anthropic Messages API.

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde_json::{Value, json};

use crate::agent::message::{
    Completion, CompletionRequest, ContentBlock, Message, StopReason, Usage,
};
use crate::agent::provider::{LlmProvider, http_client};
use crate::config::ProviderConfig;

const API_VERSION: &str = "2023-06-01";

pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl AnthropicProvider {
    pub fn new(config: &ProviderConfig) -> Result<Self> {
        Ok(AnthropicProvider {
            client: http_client(config)?,
            api_key: config.api_key()?,
            base_url: config.base_url.trim_end_matches('/').to_string(),
            model: config.model.clone(),
        })
    }

    fn body(&self, request: &CompletionRequest) -> Value {
        let mut body = json!({
            "model": self.model,
            "max_tokens": request.max_tokens.max(1),
            "temperature": request.temperature.clamp(0.0, 1.0),
            "messages": request.messages,
        });
        if !request.system.is_empty() {
            body["system"] = json!(request.system);
        }
        if !request.tools.is_empty() {
            body["tools"] = json!(request.tools);
        }
        body
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    fn name(&self) -> &'static str {
        "anthropic"
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn complete(&self, request: &CompletionRequest) -> Result<Completion> {
        let url = format!("{}/v1/messages", self.base_url);
        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .json(&self.body(request))
            .send()
            .await
            .with_context(|| format!("calling {url}"))?;

        let status = response.status();
        let text = response.text().await.context("reading response body")?;
        if !status.is_success() {
            bail!("{}", describe_error(status, &text));
        }

        let parsed: Value =
            serde_json::from_str(&text).context("parsing Anthropic response as JSON")?;
        parse_completion(&parsed)
    }
}

fn parse_completion(parsed: &Value) -> Result<Completion> {
    let blocks = parsed
        .get("content")
        .and_then(|c| c.as_array())
        .context("response had no content array")?;

    let mut content = Vec::new();
    for block in blocks {
        match block.get("type").and_then(|t| t.as_str()) {
            Some("text") => content.push(ContentBlock::Text {
                text: block["text"].as_str().unwrap_or_default().to_string(),
            }),
            Some("tool_use") => content.push(ContentBlock::ToolUse {
                id: block["id"].as_str().unwrap_or_default().to_string(),
                name: block["name"].as_str().unwrap_or_default().to_string(),
                input: block.get("input").cloned().unwrap_or_else(|| json!({})),
            }),
            // Thinking and other block types carry nothing we need to replay.
            _ => {}
        }
    }

    let stop_reason = match parsed.get("stop_reason").and_then(|s| s.as_str()) {
        Some("end_turn") | Some("stop_sequence") => StopReason::EndTurn,
        Some("tool_use") => StopReason::ToolUse,
        Some("max_tokens") => StopReason::MaxTokens,
        Some(other) => StopReason::Other(other.to_string()),
        None => StopReason::EndTurn,
    };

    let usage = parsed
        .get("usage")
        .map(|u| Usage {
            input_tokens: u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
            output_tokens: u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
        })
        .unwrap_or_default();

    Ok(Completion {
        message: Message::assistant(content),
        stop_reason,
        usage,
    })
}

/// Turns an API error body into something worth printing, falling back to the
/// raw body when it is not the shape we expect.
fn describe_error(status: reqwest::StatusCode, body: &str) -> String {
    let detail = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| {
            v.get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| body.chars().take(400).collect());

    let hint = match status.as_u16() {
        401 => " (check the API key in your configured api_key_env)",
        404 => " (check provider.model and provider.base_url)",
        429 => " (rate limited — retry shortly)",
        _ => "",
    };
    format!("Anthropic API error {status}: {detail}{hint}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_tool_use_response() {
        let raw = json!({
            "content": [
                {"type": "text", "text": "Let me look."},
                {"type": "tool_use", "id": "tu_1", "name": "run_command",
                 "input": {"command": "kubectl get pods"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 12, "output_tokens": 34}
        });
        let completion = parse_completion(&raw).unwrap();
        assert_eq!(completion.stop_reason, StopReason::ToolUse);
        assert_eq!(completion.usage.output_tokens, 34);
        assert_eq!(completion.message.text(), "Let me look.");
        assert_eq!(completion.message.tool_uses()[0].name, "run_command");
    }

    #[test]
    fn unknown_block_types_are_ignored() {
        let raw = json!({
            "content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "hi"}],
            "stop_reason": "end_turn"
        });
        let completion = parse_completion(&raw).unwrap();
        assert_eq!(completion.message.content.len(), 1);
        assert_eq!(completion.stop_reason, StopReason::EndTurn);
    }

    #[test]
    fn error_bodies_are_summarized() {
        let msg = describe_error(
            reqwest::StatusCode::UNAUTHORIZED,
            r#"{"error":{"message":"invalid x-api-key"}}"#,
        );
        assert!(msg.contains("invalid x-api-key"));
        assert!(msg.contains("api_key_env"));
    }
}
