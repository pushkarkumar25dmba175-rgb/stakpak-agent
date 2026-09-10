//! Reversible file operations.
//!
//! Every write goes through here first: the previous contents are copied into
//! `.stakpak/backups/` and recorded in an append-only index, so any edit the
//! agent makes can be undone later with `stakpak restore`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupEntry {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub session_id: String,
    /// The file that was about to be changed.
    pub original_path: PathBuf,
    /// Copy of the previous contents, or `None` when the file did not exist.
    pub backup_path: Option<PathBuf>,
    pub tool: String,
}

impl BackupEntry {
    pub fn was_created(&self) -> bool {
        self.backup_path.is_none()
    }
}

pub struct BackupStore {
    dir: PathBuf,
    index: PathBuf,
    session_id: String,
    enabled: bool,
}

impl BackupStore {
    pub fn new(state_dir: &Path, session_id: &str, enabled: bool) -> Self {
        let dir = state_dir.join("backups");
        let index = dir.join("index.jsonl");
        BackupStore {
            dir,
            index,
            session_id: session_id.to_string(),
            enabled,
        }
    }

    /// Records the current state of `path` before it is modified. Returns the
    /// entry so callers can mention the backup id in tool output.
    pub fn snapshot(&self, path: &Path, tool: &str) -> Result<Option<BackupEntry>> {
        if !self.enabled {
            return Ok(None);
        }
        std::fs::create_dir_all(&self.dir)
            .with_context(|| format!("creating {}", self.dir.display()))?;

        let id = short_id();
        let backup_path = if path.exists() {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "file".to_string());
            let dest = self.dir.join(format!("{id}-{name}"));
            std::fs::copy(path, &dest).with_context(|| format!("backing up {}", path.display()))?;
            Some(dest)
        } else {
            None
        };

        let entry = BackupEntry {
            id,
            timestamp: Utc::now(),
            session_id: self.session_id.clone(),
            original_path: path.to_path_buf(),
            backup_path,
            tool: tool.to_string(),
        };
        self.append(&entry)?;
        Ok(Some(entry))
    }

    fn append(&self, entry: &BackupEntry) -> Result<()> {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.index)
            .with_context(|| format!("opening {}", self.index.display()))?;
        writeln!(file, "{}", serde_json::to_string(entry)?)
            .with_context(|| format!("writing {}", self.index.display()))?;
        Ok(())
    }

    /// All recorded backups, newest last.
    pub fn list(&self) -> Result<Vec<BackupEntry>> {
        if !self.index.exists() {
            return Ok(Vec::new());
        }
        let text = std::fs::read_to_string(&self.index)
            .with_context(|| format!("reading {}", self.index.display()))?;
        let mut entries = Vec::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            match serde_json::from_str::<BackupEntry>(line) {
                Ok(e) => entries.push(e),
                // A truncated final line should not make the whole index unusable.
                Err(err) => eprintln!("skipping unreadable backup record: {err}"),
            }
        }
        Ok(entries)
    }

    /// Puts a file back the way it was. A backup with no stored contents means
    /// the agent created the file, so restoring removes it again.
    pub fn restore(&self, id: &str) -> Result<BackupEntry> {
        let entry = self
            .list()?
            .into_iter()
            .find(|e| e.id == id)
            .with_context(|| format!("no backup with id {id}"))?;

        match &entry.backup_path {
            Some(backup) => {
                if let Some(parent) = entry.original_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::copy(backup, &entry.original_path)
                    .with_context(|| format!("restoring {}", entry.original_path.display()))?;
            }
            None => {
                if entry.original_path.exists() {
                    std::fs::remove_file(&entry.original_path)
                        .with_context(|| format!("removing {}", entry.original_path.display()))?;
                }
            }
        }
        Ok(entry)
    }
}

fn short_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()[..12].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("stakpak-test-{tag}-{}", short_id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn restores_previous_contents() {
        let root = temp_dir("backup");
        let file = root.join("app.conf");
        std::fs::write(&file, "original").unwrap();

        let store = BackupStore::new(&root.join(".stakpak"), "s1", true);
        let entry = store.snapshot(&file, "write_file").unwrap().unwrap();
        std::fs::write(&file, "modified").unwrap();

        store.restore(&entry.id).unwrap();
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "original");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn restoring_a_created_file_deletes_it() {
        let root = temp_dir("created");
        let file = root.join("new.tf");

        let store = BackupStore::new(&root.join(".stakpak"), "s1", true);
        let entry = store.snapshot(&file, "write_file").unwrap().unwrap();
        assert!(entry.was_created());
        std::fs::write(&file, "resource {}").unwrap();

        store.restore(&entry.id).unwrap();
        assert!(!file.exists());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn disabled_store_records_nothing() {
        let root = temp_dir("disabled");
        let file = root.join("a.txt");
        std::fs::write(&file, "x").unwrap();
        let store = BackupStore::new(&root.join(".stakpak"), "s1", false);
        assert!(store.snapshot(&file, "write_file").unwrap().is_none());
        assert!(store.list().unwrap().is_empty());
        std::fs::remove_dir_all(&root).ok();
    }
}
