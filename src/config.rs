//! Layered configuration: built-in defaults, then the user config file, then
//! the project config file, then a handful of environment overrides.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub const PROJECT_CONFIG: &str = "stakpak.toml";
pub const STATE_DIR: &str = ".stakpak";

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub provider: ProviderConfig,
    pub agent: AgentConfig,
    pub security: SecurityConfig,
    pub tools: ToolsConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Anthropic,
    /// Any OpenAI-compatible chat-completions endpoint.
    Openai,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProviderConfig {
    pub kind: ProviderKind,
    pub model: String,
    /// Name of the environment variable holding the API key. The key itself is
    /// deliberately never stored in the config file.
    pub api_key_env: String,
    pub base_url: String,
    pub max_tokens: u32,
    pub temperature: f32,
    /// Request timeout for a single completion call.
    pub timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AgentConfig {
    /// Upper bound on tool-call rounds in a single turn, so a confused model
    /// cannot loop forever.
    pub max_iterations: u32,
    /// Extra instructions appended to the built-in system prompt.
    pub system_prompt_append: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalMode {
    /// Run everything without asking.
    Never,
    /// Ask before anything that writes to disk or executes a command.
    Writes,
    /// Ask before every tool call, reads included.
    Always,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SecurityConfig {
    /// Replace detected secrets with placeholders before the model sees them.
    pub redact_secrets: bool,
    /// Also register the values of secret-looking environment variables.
    pub redact_env: bool,
    pub approval: ApprovalMode,
    /// Additional regexes treated as secrets. Capture group 1 is redacted when
    /// present, otherwise the whole match.
    pub extra_secret_patterns: Vec<String>,
    /// Substrings that block a command outright, checked after placeholders are
    /// restored. A blocked command is never executed, with or without approval.
    pub denied_command_patterns: Vec<String>,
    /// Whether tools may touch paths outside the working directory.
    pub allow_outside_workdir: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ToolsConfig {
    pub command_timeout_secs: u64,
    /// Tool output beyond this is truncated in the middle before being sent to
    /// the model, keeping both the start and the (usually more useful) tail.
    pub max_output_bytes: usize,
    /// Back up every file before it is modified or overwritten.
    pub backup_files: bool,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        ProviderConfig {
            kind: ProviderKind::Anthropic,
            model: "claude-sonnet-5".to_string(),
            api_key_env: "ANTHROPIC_API_KEY".to_string(),
            base_url: "https://api.anthropic.com".to_string(),
            max_tokens: 8192,
            temperature: 0.0,
            timeout_secs: 300,
        }
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        AgentConfig {
            max_iterations: 40,
            system_prompt_append: String::new(),
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        SecurityConfig {
            redact_secrets: true,
            redact_env: true,
            approval: ApprovalMode::Writes,
            extra_secret_patterns: Vec::new(),
            denied_command_patterns: vec![
                "rm -rf /".to_string(),
                "mkfs".to_string(),
                "dd if=/dev/zero".to_string(),
                ":(){:|:&};:".to_string(),
                "shutdown".to_string(),
                "reboot".to_string(),
            ],
            allow_outside_workdir: false,
        }
    }
}

impl Default for ToolsConfig {
    fn default() -> Self {
        ToolsConfig {
            command_timeout_secs: 120,
            max_output_bytes: 30_000,
            backup_files: true,
        }
    }
}

impl ProviderConfig {
    /// Reads the API key from the configured environment variable.
    pub fn api_key(&self) -> Result<String> {
        match std::env::var(&self.api_key_env) {
            Ok(key) if !key.trim().is_empty() => Ok(key),
            _ => anyhow::bail!(
                "{} is not set. Export it, or point provider.api_key_env at the variable \
                 holding your key.",
                self.api_key_env
            ),
        }
    }
}

impl Config {
    /// Loads defaults, then the user config, then the project config, then env
    /// overrides. Later layers win key by key.
    pub fn load(workdir: &Path, explicit: Option<&Path>) -> Result<Self> {
        let mut merged = toml::Value::Table(toml::Table::new());

        let layers: Vec<PathBuf> = match explicit {
            Some(p) => vec![p.to_path_buf()],
            None => {
                let mut v = Vec::new();
                if let Some(p) = user_config_path() {
                    v.push(p);
                }
                v.push(workdir.join(PROJECT_CONFIG));
                v
            }
        };

        for path in &layers {
            if !path.exists() {
                if explicit.is_some() {
                    anyhow::bail!("config file not found: {}", path.display());
                }
                continue;
            }
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?;
            let value: toml::Value =
                toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
            merge(&mut merged, value);
        }

        let mut cfg: Config = merged.try_into().context("applying configuration values")?;
        cfg.apply_env_overrides();
        Ok(cfg)
    }

    fn apply_env_overrides(&mut self) {
        if let Ok(v) = std::env::var("STAKPAK_MODEL") {
            self.provider.model = v;
        }
        if let Ok(v) = std::env::var("STAKPAK_BASE_URL") {
            self.provider.base_url = v;
        }
        if let Ok(v) = std::env::var("STAKPAK_API_KEY_ENV") {
            self.provider.api_key_env = v;
        }
        if let Ok(v) = std::env::var("STAKPAK_PROVIDER") {
            match v.as_str() {
                "anthropic" => self.provider.kind = ProviderKind::Anthropic,
                "openai" => self.provider.kind = ProviderKind::Openai,
                other => eprintln!("ignoring unknown STAKPAK_PROVIDER={other}"),
            }
        }
        if let Ok(v) = std::env::var("STAKPAK_APPROVAL") {
            match v.as_str() {
                "never" => self.security.approval = ApprovalMode::Never,
                "writes" => self.security.approval = ApprovalMode::Writes,
                "always" => self.security.approval = ApprovalMode::Always,
                other => eprintln!("ignoring unknown STAKPAK_APPROVAL={other}"),
            }
        }
    }

    pub fn to_toml(&self) -> Result<String> {
        toml::to_string_pretty(self).context("serializing configuration")
    }
}

pub fn user_config_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("stakpak-agent").join("config.toml"))
}

/// Deep-merges `next` into `base`; tables recurse, everything else replaces.
fn merge(base: &mut toml::Value, next: toml::Value) {
    match (base, next) {
        (toml::Value::Table(base), toml::Value::Table(next)) => {
            for (k, v) in next {
                match base.get_mut(&k) {
                    Some(existing) => merge(existing, v),
                    None => {
                        base.insert(k, v);
                    }
                }
            }
        }
        (base, next) => *base = next,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_round_trip_through_toml() {
        let cfg = Config::default();
        let text = cfg.to_toml().unwrap();
        let parsed: Config = toml::from_str(&text).unwrap();
        assert_eq!(parsed.provider.model, cfg.provider.model);
        assert_eq!(parsed.security.approval, cfg.security.approval);
    }

    #[test]
    fn later_layers_override_key_by_key() {
        let mut base: toml::Value =
            toml::from_str("[provider]\nmodel = \"a\"\nmax_tokens = 10\n").unwrap();
        let next: toml::Value = toml::from_str("[provider]\nmodel = \"b\"\n").unwrap();
        merge(&mut base, next);
        assert_eq!(base["provider"]["model"].as_str(), Some("b"));
        // Untouched keys survive the merge.
        assert_eq!(base["provider"]["max_tokens"].as_integer(), Some(10));
    }

    #[test]
    fn partial_config_fills_in_defaults() {
        let cfg: Config = toml::from_str("[agent]\nmax_iterations = 5\n").unwrap();
        assert_eq!(cfg.agent.max_iterations, 5);
        assert_eq!(
            cfg.provider.max_tokens,
            ProviderConfig::default().max_tokens
        );
    }
}
