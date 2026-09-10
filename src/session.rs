//! Session persistence and checkpoints.
//!
//! The whole conversation is written to `.stakpak/sessions/<id>.json` after
//! every turn, and each turn also records a checkpoint (a message index) so a
//! session can be resumed or rewound to an earlier point.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::agent::message::Message;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub index: usize,
    pub timestamp: DateTime<Utc>,
    /// The user request that opened this turn, trimmed for display.
    pub label: String,
    /// Number of messages in the transcript at this point.
    pub message_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub workdir: PathBuf,
    pub model: String,
    pub messages: Vec<Message>,
    pub checkpoints: Vec<Checkpoint>,
    #[serde(skip)]
    path: Option<PathBuf>,
}

impl Session {
    pub fn new(workdir: &Path, model: &str) -> Self {
        let now = Utc::now();
        Session {
            id: uuid::Uuid::new_v4().simple().to_string()[..12].to_string(),
            created_at: now,
            updated_at: now,
            workdir: workdir.to_path_buf(),
            model: model.to_string(),
            messages: Vec::new(),
            checkpoints: Vec::new(),
            path: None,
        }
    }

    /// Records a checkpoint at the current end of the transcript.
    pub fn checkpoint(&mut self, label: &str) {
        self.checkpoints.push(Checkpoint {
            index: self.checkpoints.len() + 1,
            timestamp: Utc::now(),
            label: truncate(label, 80),
            message_count: self.messages.len(),
        });
    }

    /// Drops every message recorded after the given checkpoint.
    pub fn rewind_to(&mut self, index: usize) -> Result<&Checkpoint> {
        let position = self
            .checkpoints
            .iter()
            .position(|c| c.index == index)
            .with_context(|| format!("no checkpoint {index} in session {}", self.id))?;
        let count = self.checkpoints[position].message_count;
        self.messages.truncate(count);
        self.checkpoints.truncate(position + 1);
        Ok(&self.checkpoints[position])
    }

    pub fn save(&mut self, state_dir: &Path) -> Result<()> {
        let dir = state_dir.join("sessions");
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        let path = dir.join(format!("{}.json", self.id));
        self.updated_at = Utc::now();
        let text = serde_json::to_string_pretty(self).context("serializing session")?;
        std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))?;
        self.path = Some(path);
        Ok(())
    }

    pub fn load(state_dir: &Path, id: &str) -> Result<Self> {
        let path = state_dir.join("sessions").join(format!("{id}.json"));
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("no saved session {id} at {}", path.display()))?;
        let mut session: Session =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        session.path = Some(path);
        Ok(session)
    }

    /// Every saved session, most recently updated first.
    pub fn list(state_dir: &Path) -> Result<Vec<Session>> {
        let dir = state_dir.join("sessions");
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let mut sessions = Vec::new();
        for entry in
            std::fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))?
        {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            match std::fs::read_to_string(&path)
                .ok()
                .and_then(|t| serde_json::from_str::<Session>(&t).ok())
            {
                Some(mut s) => {
                    s.path = Some(path);
                    sessions.push(s);
                }
                None => eprintln!("skipping unreadable session file {}", path.display()),
            }
        }
        sessions.sort_by_key(|s| std::cmp::Reverse(s.updated_at));
        Ok(sessions)
    }

    pub fn latest(state_dir: &Path) -> Result<Option<Session>> {
        Ok(Self::list(state_dir)?.into_iter().next())
    }

    /// First user message, used as a one-line description in `stakpak sessions`.
    pub fn summary(&self) -> String {
        self.messages
            .first()
            .map(|m| truncate(&m.text(), 60))
            .unwrap_or_else(|| "(empty)".to_string())
    }
}

fn truncate(text: &str, max: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max {
        return flat;
    }
    let kept: String = flat.chars().take(max.saturating_sub(1)).collect();
    format!("{kept}…")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::message::{ContentBlock, Message};

    fn state_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "stakpak-session-{tag}-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn saves_and_loads_a_transcript() {
        let dir = state_dir("roundtrip");
        let mut s = Session::new(Path::new("/srv/app"), "claude-sonnet-5");
        s.messages.push(Message::user("deploy the staging cluster"));
        s.checkpoint("deploy the staging cluster");
        s.save(&dir).unwrap();

        let loaded = Session::load(&dir, &s.id).unwrap();
        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(loaded.checkpoints.len(), 1);
        assert_eq!(loaded.summary(), "deploy the staging cluster");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rewinding_drops_later_messages() {
        let mut s = Session::new(Path::new("."), "m");
        s.messages.push(Message::user("first"));
        s.checkpoint("first");
        s.messages.push(Message::assistant(vec![ContentBlock::Text {
            text: "done".into(),
        }]));
        s.messages.push(Message::user("second"));
        s.checkpoint("second");

        s.rewind_to(1).unwrap();
        assert_eq!(s.messages.len(), 1);
        assert_eq!(s.checkpoints.len(), 1);
    }

    #[test]
    fn rewinding_to_a_missing_checkpoint_errors() {
        let mut s = Session::new(Path::new("."), "m");
        assert!(s.rewind_to(7).is_err());
    }

    #[test]
    fn latest_returns_the_newest_session() {
        let dir = state_dir("latest");
        let mut a = Session::new(Path::new("."), "m");
        a.messages.push(Message::user("older"));
        a.save(&dir).unwrap();
        let mut b = Session::new(Path::new("."), "m");
        b.messages.push(Message::user("newer"));
        b.save(&dir).unwrap();

        assert_eq!(Session::latest(&dir).unwrap().unwrap().id, b.id);
        std::fs::remove_dir_all(&dir).ok();
    }
}
