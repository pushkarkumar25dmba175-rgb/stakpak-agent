//! End-to-end exercises of the tool surface against a real temporary tree.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{Value, json};
use stakpak_agent::backup::BackupStore;
use stakpak_agent::config::Config;
use stakpak_agent::tools::{ToolContext, ToolOutput, ToolRegistry};

struct Fixture {
    root: PathBuf,
    ctx: ToolContext,
    registry: ToolRegistry,
}

impl Fixture {
    fn new(tag: &str) -> Self {
        let root = std::env::temp_dir().join(format!("stakpak-it-{tag}-{}", uuid_like()));
        std::fs::create_dir_all(&root).unwrap();

        let config = Arc::new(Config::default());
        let backups = Arc::new(BackupStore::new(&root.join(".stakpak"), "test", true));
        Fixture {
            ctx: ToolContext {
                workdir: root.clone(),
                config,
                backups,
            },
            registry: ToolRegistry::with_defaults(),
            root,
        }
    }

    async fn call(&self, tool: &str, input: Value) -> ToolOutput {
        self.registry
            .get(tool)
            .unwrap_or_else(|| panic!("no tool named {tool}"))
            .run(&self.ctx, &input)
            .await
            .unwrap()
    }

    fn write(&self, relative: &str, contents: &str) -> PathBuf {
        let path = self.root.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&path, contents).unwrap();
        path
    }

    fn read(&self, relative: &str) -> String {
        std::fs::read_to_string(self.root.join(relative)).unwrap()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.root).ok();
    }
}

fn uuid_like() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{nanos:x}")
}

#[tokio::test]
async fn read_file_numbers_lines_and_pages() {
    let fx = Fixture::new("read");
    fx.write("main.tf", "one\ntwo\nthree\nfour\n");

    let out = fx.call("read_file", json!({"path": "main.tf"})).await;
    assert!(!out.is_error);
    assert!(out.content.contains("     1\tone"));
    assert!(out.content.contains("     4\tfour"));

    let paged = fx
        .call(
            "read_file",
            json!({"path": "main.tf", "offset": 3, "limit": 1}),
        )
        .await;
    assert!(paged.content.contains("three"));
    assert!(!paged.content.contains("four"));
    assert!(paged.content.contains("offset=4"));
}

#[tokio::test]
async fn read_file_reports_missing_and_binary_files() {
    let fx = Fixture::new("read-err");
    let missing = fx.call("read_file", json!({"path": "nope.txt"})).await;
    assert!(missing.is_error);
    assert!(missing.content.contains("does not exist"));

    std::fs::write(fx.root.join("blob.bin"), [0xff, 0xfe, 0x00, 0x01]).unwrap();
    let binary = fx.call("read_file", json!({"path": "blob.bin"})).await;
    assert!(binary.is_error);
    assert!(binary.content.contains("binary"));
}

#[tokio::test]
async fn write_file_creates_parents_and_records_a_backup() {
    let fx = Fixture::new("write");

    let created = fx
        .call(
            "write_file",
            json!({"path": "infra/vpc.tf", "content": "resource \"aws_vpc\" \"main\" {}\n"}),
        )
        .await;
    assert!(!created.is_error);
    assert!(created.content.starts_with("Created"));
    assert!(fx.read("infra/vpc.tf").contains("aws_vpc"));

    let overwritten = fx
        .call(
            "write_file",
            json!({"path": "infra/vpc.tf", "content": "# replaced\n"}),
        )
        .await;
    assert!(overwritten.content.starts_with("Overwrote"));
    assert_eq!(fx.read("infra/vpc.tf"), "# replaced\n");

    // The overwrite is reversible.
    let store = BackupStore::new(&fx.root.join(".stakpak"), "test", true);
    let entries = store.list().unwrap();
    assert_eq!(entries.len(), 2);
    store.restore(&entries[1].id).unwrap();
    assert!(fx.read("infra/vpc.tf").contains("aws_vpc"));
}

#[tokio::test]
async fn edit_file_requires_a_unique_match() {
    let fx = Fixture::new("edit");
    fx.write("values.yaml", "replicas: 1\nimage: app:1.0\nreplicas: 1\n");

    let ambiguous = fx
        .call(
            "edit_file",
            json!({"path": "values.yaml", "old_string": "replicas: 1", "new_string": "replicas: 3"}),
        )
        .await;
    assert!(ambiguous.is_error);
    assert!(ambiguous.content.contains("appears 2 times"));

    let all = fx
        .call(
            "edit_file",
            json!({
                "path": "values.yaml",
                "old_string": "replicas: 1",
                "new_string": "replicas: 3",
                "replace_all": true
            }),
        )
        .await;
    assert!(!all.is_error, "{}", all.content);
    assert_eq!(fx.read("values.yaml").matches("replicas: 3").count(), 2);
    // The output carries a diff so the model can see what it did.
    assert!(all.content.contains("+replicas: 3"));
}

#[tokio::test]
async fn edit_file_rejects_a_missing_match() {
    let fx = Fixture::new("edit-miss");
    fx.write("app.py", "print('hi')\n");
    let out = fx
        .call(
            "edit_file",
            json!({"path": "app.py", "old_string": "nowhere", "new_string": "x"}),
        )
        .await;
    assert!(out.is_error);
    assert!(out.content.contains("not found"));
    // The file is untouched.
    assert_eq!(fx.read("app.py"), "print('hi')\n");
}

#[tokio::test]
async fn search_finds_matches_and_honours_globs() {
    let fx = Fixture::new("search");
    fx.write("infra/main.tf", "resource \"aws_s3_bucket\" \"logs\" {}\n");
    fx.write("README.md", "the aws_s3_bucket resource\n");

    let all = fx.call("search", json!({"pattern": "aws_s3_bucket"})).await;
    assert!(all.content.contains("infra/main.tf:1"));
    assert!(all.content.contains("README.md:1"));

    let scoped = fx
        .call(
            "search",
            json!({"pattern": "aws_s3_bucket", "glob": "*.tf"}),
        )
        .await;
    assert!(scoped.content.contains("infra/main.tf"));
    assert!(!scoped.content.contains("README.md"));

    let none = fx
        .call("search", json!({"pattern": "not_present_here"}))
        .await;
    assert!(none.content.contains("No matches"));
}

#[tokio::test]
async fn search_reports_a_bad_regex_instead_of_failing() {
    let fx = Fixture::new("search-bad");
    let out = fx.call("search", json!({"pattern": "([unclosed"})).await;
    assert!(out.is_error);
    assert!(out.content.contains("invalid regex"));
}

#[tokio::test]
async fn list_files_skips_gitignored_paths() {
    let fx = Fixture::new("list");
    fx.write(".gitignore", "target/\n");
    fx.write("src/main.rs", "fn main() {}\n");
    fx.write("target/debug/app", "binary\n");

    let out = fx.call("list_files", json!({})).await;
    assert!(out.content.contains("src/main.rs"));
    assert!(
        !out.content.contains("target/debug"),
        "gitignored path leaked: {}",
        out.content
    );
}

#[tokio::test]
async fn tools_cannot_escape_the_working_directory() {
    let fx = Fixture::new("sandbox");
    let registry = ToolRegistry::with_defaults();
    let escaping = registry
        .get("write_file")
        .unwrap()
        .run(
            &fx.ctx,
            &json!({"path": "../escaped.txt", "content": "nope"}),
        )
        .await;
    assert!(escaping.is_err(), "path traversal should be refused");
    assert!(!Path::new("/tmp/escaped.txt").exists());
}

#[tokio::test]
async fn run_command_executes_in_the_working_directory() {
    let fx = Fixture::new("cmd");
    fx.write("marker.txt", "found me\n");
    let out = fx
        .call("run_command", json!({"command": "cat marker.txt"}))
        .await;
    assert!(!out.is_error, "{}", out.content);
    assert!(out.content.contains("found me"));
}
