//! Secret substitution.
//!
//! Tool output is scanned for credentials before it reaches the model; each
//! distinct secret is swapped for a stable placeholder like
//! `{{SECRET_github_token_1}}`. When the model later feeds that placeholder
//! back to us in a tool argument, `restore` puts the real value back just
//! before execution. The model gets to orchestrate with credentials it never
//! actually sees.

use std::collections::HashMap;

use anyhow::{Context, Result};
use regex::Regex;
use serde_json::Value;

/// A named pattern.
///
/// The secret is the `secret` capture group if the regex defines one, else
/// group 1, else the whole match — so a pattern can keep surrounding context
/// (`password = `) visible in the redacted text and hide only the value. A
/// `name` capture group overrides the pattern's kind, which lets an assignment
/// like `DB_PASSWORD=...` produce `{{SECRET_db_password_1}}` rather than a
/// generic label.
struct SecretPattern {
    kind: &'static str,
    regex: Regex,
}

pub struct SecretStore {
    enabled: bool,
    patterns: Vec<SecretPattern>,
    /// Real secret -> placeholder.
    by_secret: HashMap<String, String>,
    /// Placeholder -> real secret.
    by_placeholder: HashMap<String, String>,
    counter: usize,
}

/// Values shorter than this are too likely to be false positives to be worth
/// hiding, and hiding them makes output harder to read for no gain.
const MIN_SECRET_LEN: usize = 8;

impl SecretStore {
    pub fn new(enabled: bool, extra_patterns: &[String]) -> Result<Self> {
        let mut patterns = builtin_patterns()?;
        for (i, p) in extra_patterns.iter().enumerate() {
            let regex = Regex::new(p)
                .with_context(|| format!("compiling security.extra_secret_patterns[{i}]"))?;
            // Leaked so the kind can stay a `&'static str` like the built-ins;
            // the count is bounded by the config file.
            let kind: &'static str = Box::leak(format!("custom_{}", i + 1).into_boxed_str());
            patterns.push(SecretPattern { kind, regex });
        }
        Ok(SecretStore {
            enabled,
            patterns,
            by_secret: HashMap::new(),
            by_placeholder: HashMap::new(),
            counter: 0,
        })
    }

    pub fn disabled() -> Self {
        SecretStore {
            enabled: false,
            patterns: Vec::new(),
            by_secret: HashMap::new(),
            by_placeholder: HashMap::new(),
            counter: 0,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn len(&self) -> usize {
        self.by_secret.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_secret.is_empty()
    }

    /// Registers the values of environment variables whose names look like they
    /// hold credentials, so those values are hidden anywhere they show up —
    /// including in output from a command that echoed its own environment.
    pub fn register_env(&mut self) {
        if !self.enabled {
            return;
        }
        let name_re = Regex::new(
            r"(?i)(secret|token|password|passwd|credential|api[_-]?key|access[_-]?key|private[_-]?key|auth)",
        )
        .expect("static regex");
        let path_like = Regex::new(r"(?i)_(path|file|dir|home|sock|socket|url|type|method)$")
            .expect("static regex");

        let vars: Vec<(String, String)> = std::env::vars().collect();
        for (name, value) in vars {
            if !name_re.is_match(&name) || path_like.is_match(&name) {
                continue;
            }
            if value.len() < MIN_SECRET_LEN || value.starts_with('/') {
                continue;
            }
            let kind = name.to_lowercase();
            self.register(&value, &kind);
        }
    }

    /// Interns a secret and returns its placeholder, reusing the placeholder if
    /// the same value has been seen before.
    pub fn register(&mut self, secret: &str, kind: &str) -> String {
        if let Some(existing) = self.by_secret.get(secret) {
            return existing.clone();
        }
        self.counter += 1;
        let placeholder = format!("{{{{SECRET_{}_{}}}}}", sanitize_kind(kind), self.counter);
        self.by_secret
            .insert(secret.to_string(), placeholder.clone());
        self.by_placeholder
            .insert(placeholder.clone(), secret.to_string());
        placeholder
    }

    /// Replaces every known and newly detected secret in `text` with placeholders.
    pub fn redact(&mut self, text: &str) -> String {
        if !self.enabled {
            return text.to_string();
        }

        // Already-known secrets first: a value registered from the environment
        // should get its existing placeholder rather than a second one.
        let mut out = text.to_string();
        if !self.by_secret.is_empty() {
            let mut known: Vec<(String, String)> = self
                .by_secret
                .iter()
                .map(|(s, p)| (s.clone(), p.clone()))
                .collect();
            // Longest first, so a secret that contains another is matched whole.
            known.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));
            for (secret, placeholder) in known {
                if out.contains(&secret) {
                    out = out.replace(&secret, &placeholder);
                }
            }
        }

        // Then pattern detection, collecting matches before mutating the string.
        for i in 0..self.patterns.len() {
            let matches: Vec<(String, String)> = {
                let pattern = &self.patterns[i];
                pattern
                    .regex
                    .captures_iter(&out)
                    .filter_map(|caps| {
                        let value = caps
                            .name("secret")
                            .or_else(|| caps.get(1))
                            .or_else(|| caps.get(0))?;
                        let kind = caps
                            .name("name")
                            .map(|m| m.as_str().to_string())
                            .unwrap_or_else(|| pattern.kind.to_string());
                        Some((kind, value.as_str().to_string()))
                    })
                    .filter(|(_, value)| value.len() >= MIN_SECRET_LEN)
                    .collect()
            };
            for (kind, found) in matches {
                // A placeholder from an earlier pass must not be re-redacted.
                if found.contains("{{SECRET_") {
                    continue;
                }
                let placeholder = self.register(&found, &kind);
                out = out.replace(&found, &placeholder);
            }
        }
        out
    }

    /// Puts real secrets back wherever a placeholder appears.
    pub fn restore(&self, text: &str) -> String {
        if self.by_placeholder.is_empty() {
            return text.to_string();
        }
        let mut out = text.to_string();
        for (placeholder, secret) in &self.by_placeholder {
            if out.contains(placeholder) {
                out = out.replace(placeholder, secret);
            }
        }
        out
    }

    /// Restores placeholders in every string inside a JSON value, which is how
    /// tool arguments arrive from the model.
    pub fn restore_value(&self, value: &Value) -> Value {
        match value {
            Value::String(s) => Value::String(self.restore(s)),
            Value::Array(items) => {
                Value::Array(items.iter().map(|v| self.restore_value(v)).collect())
            }
            Value::Object(map) => Value::Object(
                map.iter()
                    .map(|(k, v)| (k.clone(), self.restore_value(v)))
                    .collect(),
            ),
            other => other.clone(),
        }
    }

    /// Placeholder/kind pairs, for `stakpak secrets`.
    pub fn placeholders(&self) -> Vec<String> {
        let mut v: Vec<String> = self.by_placeholder.keys().cloned().collect();
        v.sort();
        v
    }
}

fn sanitize_kind(kind: &str) -> String {
    let cleaned: String = kind
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    cleaned.trim_matches('_').to_lowercase()
}

fn builtin_patterns() -> Result<Vec<SecretPattern>> {
    let specs: &[(&'static str, &str)] = &[
        (
            "private_key",
            r"(?s)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        ),
        (
            "aws_access_key_id",
            r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA)[0-9A-Z]{16}\b",
        ),
        ("github_token", r"\bgh[pousr]_[A-Za-z0-9]{16,}\b"),
        ("anthropic_api_key", r"\bsk-ant-[A-Za-z0-9_\-]{20,}"),
        ("openai_api_key", r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}"),
        ("google_api_key", r"\bAIza[0-9A-Za-z_\-]{35}\b"),
        ("slack_token", r"\bxox[baprs]-[A-Za-z0-9\-]{10,}"),
        ("stripe_key", r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}\b"),
        (
            "jwt",
            r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}",
        ),
        ("bearer_token", r"(?i)\bbearer\s+([A-Za-z0-9._~+/=\-]{20,})"),
        // `scheme://user:password@host`
        (
            "connection_string",
            r"(?i)\b[a-z][a-z0-9+.\-]*://[^\s:@/]+:([^\s@/]{4,})@",
        ),
        // `PASSWORD=...`, `DB_PASSWORD=...`, `api_key: "..."`, `--token ...`
        (
            "assigned_secret",
            r#"(?i)(?<name>[a-z0-9_\-]*(?:password|passwd|secret|token|api[_-]?key|access[_-]?key|secret[_-]?key|client[_-]?secret|auth[_-]?token))\b["']?\s*[:=]\s*["']?(?<secret>[^\s"'`,;{}()\[\]]{8,})"#,
        ),
    ];

    specs
        .iter()
        .map(|(kind, src)| {
            Ok(SecretPattern {
                kind,
                regex: Regex::new(src)
                    .with_context(|| format!("compiling built-in pattern {kind}"))?,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> SecretStore {
        SecretStore::new(true, &[]).unwrap()
    }

    #[test]
    fn aws_key_is_replaced_and_restored() {
        let mut s = store();
        let text = "aws_access_key_id = AKIAIOSFODNN7EXAMPLE";
        let redacted = s.redact(text);
        assert!(!redacted.contains("AKIAIOSFODNN7EXAMPLE"));
        assert!(redacted.contains("{{SECRET_"));
        assert_eq!(s.restore(&redacted), text);
    }

    #[test]
    fn the_same_secret_reuses_one_placeholder() {
        let mut s = store();
        let redacted = s.redact("ghp_aaaaaaaaaaaaaaaaaaaa and again ghp_aaaaaaaaaaaaaaaaaaaa");
        assert_eq!(s.len(), 1);
        let first = redacted.find("{{SECRET_").unwrap();
        let last = redacted.rfind("{{SECRET_").unwrap();
        assert_ne!(first, last, "both occurrences should be replaced");
    }

    #[test]
    fn assignment_keeps_the_key_name_visible() {
        let mut s = store();
        let redacted = s.redact("DB_PASSWORD=hunter2000supersecret");
        assert!(redacted.starts_with("DB_PASSWORD="));
        assert!(!redacted.contains("hunter2000supersecret"));
    }

    #[test]
    fn the_placeholder_is_named_after_the_key() {
        let mut s = store();
        let redacted = s.redact("DB_PASSWORD=hunter2000supersecret");
        assert!(
            redacted.contains("{{SECRET_db_password_"),
            "expected a key-derived name, got {redacted}"
        );
    }

    #[test]
    fn short_values_are_left_alone() {
        let mut s = store();
        let text = "password=abc";
        assert_eq!(s.redact(text), text);
        assert!(s.is_empty());
    }

    #[test]
    fn private_key_block_is_captured_whole() {
        let mut s = store();
        let text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEabcdef\n-----END RSA PRIVATE KEY-----";
        let redacted = s.redact(text);
        assert!(!redacted.contains("MIIEabcdef"));
        assert_eq!(s.restore(&redacted), text);
    }

    #[test]
    fn connection_string_password_only() {
        let mut s = store();
        let redacted = s.redact("postgres://admin:s3cr3tpassword@db.internal:5432/app");
        assert!(redacted.contains("postgres://admin:"));
        assert!(redacted.contains("@db.internal:5432/app"));
        assert!(!redacted.contains("s3cr3tpassword"));
    }

    #[test]
    fn restore_walks_nested_json() {
        let mut s = store();
        let placeholder = s.register("ghp_realtokenvalue", "github_token");
        let input = serde_json::json!({
            "command": format!("curl -H 'token: {placeholder}'"),
            "env": [placeholder.clone()],
        });
        let restored = s.restore_value(&input);
        assert!(
            restored["command"]
                .as_str()
                .unwrap()
                .contains("ghp_realtokenvalue")
        );
        assert_eq!(restored["env"][0], "ghp_realtokenvalue");
    }

    #[test]
    fn disabled_store_is_a_passthrough() {
        let mut s = SecretStore::disabled();
        let text = "AKIAIOSFODNN7EXAMPLE";
        assert_eq!(s.redact(text), text);
    }

    #[test]
    fn redaction_is_idempotent() {
        let mut s = store();
        let once = s.redact("token=ghp_aaaaaaaaaaaaaaaaaaaa");
        let twice = s.redact(&once);
        assert_eq!(once, twice);
    }
}
