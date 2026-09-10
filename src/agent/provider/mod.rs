//! LLM providers.

mod anthropic;
mod openai;

use anyhow::Result;
use async_trait::async_trait;

use crate::agent::message::{Completion, CompletionRequest};
use crate::config::{ProviderConfig, ProviderKind};

#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn name(&self) -> &'static str;
    fn model(&self) -> &str;
    async fn complete(&self, request: &CompletionRequest) -> Result<Completion>;
}

pub fn build(config: &ProviderConfig) -> Result<Box<dyn LlmProvider>> {
    match config.kind {
        ProviderKind::Anthropic => Ok(Box::new(anthropic::AnthropicProvider::new(config)?)),
        ProviderKind::Openai => Ok(Box::new(openai::OpenAiProvider::new(config)?)),
    }
}

/// Shared HTTP client construction, so both providers get the same timeout and
/// proxy handling.
fn http_client(config: &ProviderConfig) -> Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(config.timeout_secs))
        .user_agent(concat!("stakpak-agent/", env!("CARGO_PKG_VERSION")));

    // A model served from this machine (Ollama, vLLM, llama.cpp) should not be
    // dialled through an outbound HTTP proxy.
    if is_loopback(&config.base_url) {
        builder = builder.no_proxy();
    }
    Ok(builder.build()?)
}

fn is_loopback(base_url: &str) -> bool {
    let without_scheme = base_url
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(base_url);
    let host = without_scheme
        .split(['/', ':'])
        .next()
        .unwrap_or_default()
        .trim_start_matches('[')
        .trim_end_matches(']');
    matches!(host, "localhost" | "127.0.0.1" | "::1" | "0.0.0.0")
}

#[cfg(test)]
mod tests {
    use super::is_loopback;

    #[test]
    fn local_model_servers_bypass_the_proxy() {
        assert!(is_loopback("http://127.0.0.1:11434"));
        assert!(is_loopback("http://localhost:8000/v1"));
        assert!(!is_loopback("https://api.anthropic.com"));
        assert!(!is_loopback("https://localhost.example.com"));
    }
}
