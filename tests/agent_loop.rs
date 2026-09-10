//! End-to-end exercises of the agent loop against a stub provider.
//!
//! A minimal HTTP server stands in for an OpenAI-compatible endpoint, so the
//! full path — request shaping, tool dispatch, secret substitution, transcript
//! persistence — runs for real without touching a hosted model.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};
use stakpak_agent::agent::Agent;
use stakpak_agent::config::{ApprovalMode, Config, ProviderKind};
use stakpak_agent::session::Session;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Every request body the stub received, in order.
type Requests = Arc<Mutex<Vec<Value>>>;

/// Starts a stub chat-completions endpoint. `respond` is handed each request
/// body and returns the JSON to reply with.
async fn stub_provider<F>(respond: F) -> (SocketAddr, Requests)
where
    F: Fn(&Value, usize) -> Value + Send + Sync + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let requests: Requests = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&requests);
    let respond = Arc::new(respond);

    tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let respond = Arc::clone(&respond);
            let seen = Arc::clone(&seen);
            tokio::spawn(async move {
                let Some(body) = read_request(&mut stream).await else {
                    return;
                };
                let parsed: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
                let index = {
                    let mut guard = seen.lock().unwrap();
                    guard.push(parsed.clone());
                    guard.len() - 1
                };
                let payload = respond(&parsed, index).to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    payload.len(),
                    payload
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.shutdown().await;
            });
        }
    });

    (addr, requests)
}

/// Reads one HTTP request and returns its body.
async fn read_request(stream: &mut tokio::net::TcpStream) -> Option<String> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        let read = stream.read(&mut chunk).await.ok()?;
        if read == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..read]);

        let Some(split) = buf.windows(4).position(|w| w == b"\r\n\r\n") else {
            continue;
        };
        let headers = String::from_utf8_lossy(&buf[..split]).to_lowercase();
        let length: usize = headers
            .lines()
            .find_map(|l| l.strip_prefix("content-length:"))
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0);
        let body_start = split + 4;
        if buf.len() - body_start >= length {
            return Some(
                String::from_utf8_lossy(&buf[body_start..body_start + length]).to_string(),
            );
        }
    }
}

fn tool_call_response(name: &str, arguments: Value) -> Value {
    json!({
        "choices": [{
            "message": {
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments.to_string()}
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}
    })
}

fn text_response(text: &str) -> Value {
    json!({
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 6}
    })
}

fn workspace(tag: &str) -> PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("stakpak-loop-{tag}-{nanos:x}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn config_for(addr: SocketAddr, redact_env: bool) -> Config {
    let mut config = Config::default();
    config.provider.kind = ProviderKind::Openai;
    config.provider.base_url = format!("http://{addr}");
    config.provider.model = "stub-model".into();
    // The stub ignores the key; PATH is read only so the constructor finds
    // *some* value without mutating the test process's environment.
    config.provider.api_key_env = "PATH".into();
    config.security.approval = ApprovalMode::Never;
    config.security.redact_env = redact_env;
    config
}

#[tokio::test]
async fn runs_a_tool_then_reports_back() {
    let root = workspace("tool");
    let (addr, requests) = stub_provider(|_request, index| match index {
        0 => tool_call_response(
            "write_file",
            json!({"path": "deploy.sh", "content": "#!/bin/sh\nkubectl apply -f .\n"}),
        ),
        _ => text_response("Wrote deploy.sh."),
    })
    .await;

    let state = root.join(".stakpak");
    let session = Session::new(&root, "stub-model");
    let mut agent = Agent::new(config_for(addr, false), &root, &state, session, true).unwrap();

    let reply = agent.turn("add a deploy script").await.unwrap();
    assert_eq!(reply, "Wrote deploy.sh.");

    // The tool really ran.
    let written = std::fs::read_to_string(root.join("deploy.sh")).unwrap();
    assert!(written.contains("kubectl apply"));

    // The second request carried the tool result back to the model.
    let sent = requests.lock().unwrap();
    assert_eq!(sent.len(), 2);
    let followup = &sent[1]["messages"];
    let tool_message = followup
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "tool")
        .expect("tool result should be sent back");
    assert!(
        tool_message["content"]
            .as_str()
            .unwrap()
            .contains("Created")
    );

    // And the transcript was persisted.
    let saved = Session::load(&state, &agent.session.id).unwrap();
    assert_eq!(saved.messages.len(), 4);
    assert_eq!(saved.checkpoints.len(), 1);

    std::fs::remove_dir_all(&root).ok();
}

#[tokio::test]
async fn secrets_in_tool_output_never_reach_the_model() {
    let root = workspace("redact");
    std::fs::write(
        root.join("credentials"),
        "aws_access_key_id = AKIAIOSFODNN7EXAMPLE\n",
    )
    .unwrap();

    let (addr, requests) = stub_provider(|_request, index| match index {
        0 => tool_call_response("read_file", json!({"path": "credentials"})),
        _ => text_response("Found the key."),
    })
    .await;

    let state = root.join(".stakpak");
    let session = Session::new(&root, "stub-model");
    let mut agent = Agent::new(config_for(addr, false), &root, &state, session, true).unwrap();
    agent
        .turn("what credentials are configured?")
        .await
        .unwrap();

    let sent = requests.lock().unwrap();
    let followup = serde_json::to_string(&sent[1]).unwrap();
    assert!(
        !followup.contains("AKIAIOSFODNN7EXAMPLE"),
        "the raw credential was sent to the model"
    );
    assert!(followup.contains("{{SECRET_"));

    // The transcript on disk is redacted too.
    let saved = serde_json::to_string(&Session::load(&state, &agent.session.id).unwrap()).unwrap();
    assert!(!saved.contains("AKIAIOSFODNN7EXAMPLE"));

    std::fs::remove_dir_all(&root).ok();
}

#[tokio::test]
async fn a_placeholder_is_restored_before_the_command_runs() {
    let root = workspace("restore");
    std::fs::write(root.join("credentials"), "token=ghp_abcdefghijklmnopqrst\n").unwrap();

    // The second turn feeds the placeholder from the first tool result straight
    // back into a shell command, the way a model would when it needs the
    // credential it was never shown.
    let (addr, _requests) = stub_provider(|request, index| match index {
        0 => tool_call_response("read_file", json!({"path": "credentials"})),
        1 => {
            let tool_result = request["messages"]
                .as_array()
                .unwrap()
                .iter()
                .find(|m| m["role"] == "tool")
                .expect("the tool result should be in the transcript")["content"]
                .as_str()
                .unwrap()
                .to_string();
            let start = tool_result
                .find("{{SECRET_")
                .expect("a placeholder should be present");
            let end = tool_result[start..].find("}}").unwrap() + start + 2;
            let placeholder = &tool_result[start..end];
            tool_call_response(
                "run_command",
                json!({"command": format!("printf '%s' '{placeholder}' > used.txt")}),
            )
        }
        _ => text_response("Used the token."),
    })
    .await;

    let state = root.join(".stakpak");
    let session = Session::new(&root, "stub-model");
    let mut agent = Agent::new(config_for(addr, false), &root, &state, session, true).unwrap();
    agent.turn("use the token").await.unwrap();

    let used = std::fs::read_to_string(root.join("used.txt")).unwrap();
    assert_eq!(
        used, "ghp_abcdefghijklmnopqrst",
        "the real credential should reach the shell"
    );

    std::fs::remove_dir_all(&root).ok();
}

#[tokio::test]
async fn a_denied_command_comes_back_as_a_tool_error() {
    let root = workspace("denied");
    let (addr, requests) = stub_provider(|_request, index| match index {
        0 => tool_call_response(
            "run_command",
            json!({"command": "rm -rf / --no-preserve-root"}),
        ),
        _ => text_response("I will not do that."),
    })
    .await;

    let state = root.join(".stakpak");
    let session = Session::new(&root, "stub-model");
    let mut agent = Agent::new(config_for(addr, false), &root, &state, session, true).unwrap();
    agent.turn("clean up the disk").await.unwrap();

    let sent = requests.lock().unwrap();
    let followup = serde_json::to_string(&sent[1]).unwrap();
    assert!(followup.contains("denied pattern"));

    std::fs::remove_dir_all(&root).ok();
}

#[tokio::test]
async fn the_tool_round_limit_is_enforced() {
    let root = workspace("limit");
    // A model that only ever asks for another tool call must not loop forever.
    let (addr, requests) =
        stub_provider(|_request, _index| tool_call_response("list_files", json!({}))).await;

    let state = root.join(".stakpak");
    let mut config = config_for(addr, false);
    config.agent.max_iterations = 3;
    let session = Session::new(&root, "stub-model");
    let mut agent = Agent::new(config, &root, &state, session, true).unwrap();
    agent.turn("look around forever").await.unwrap();

    assert_eq!(requests.lock().unwrap().len(), 3);
    std::fs::remove_dir_all(&root).ok();
}

#[tokio::test]
async fn an_unknown_tool_is_reported_rather_than_fatal() {
    let root = workspace("unknown");
    let (addr, requests) = stub_provider(|_request, index| match index {
        0 => tool_call_response("terraform_apply", json!({})),
        _ => text_response("Sorry, I do not have that tool."),
    })
    .await;

    let state = root.join(".stakpak");
    let session = Session::new(&root, "stub-model");
    let mut agent = Agent::new(config_for(addr, false), &root, &state, session, true).unwrap();
    let reply = agent.turn("apply the terraform").await.unwrap();
    assert_eq!(reply, "Sorry, I do not have that tool.");

    let sent = requests.lock().unwrap();
    let followup = serde_json::to_string(&sent[1]).unwrap();
    assert!(followup.contains("unknown tool"));
    assert!(
        followup.contains("run_command"),
        "should list what is available"
    );

    std::fs::remove_dir_all(&root).ok();
}

#[tokio::test]
async fn the_system_prompt_and_tools_are_sent() {
    let root = workspace("prompt");
    let (addr, requests) = stub_provider(|_request, _index| text_response("Nothing to do.")).await;

    let state = root.join(".stakpak");
    let session = Session::new(&root, "stub-model");
    let mut agent = Agent::new(config_for(addr, false), &root, &state, session, true).unwrap();
    agent.turn("hello").await.unwrap();

    let sent = requests.lock().unwrap();
    let request = &sent[0];
    assert_eq!(request["messages"][0]["role"], "system");
    let system = request["messages"][0]["content"].as_str().unwrap();
    assert!(system.contains("DevOps"));
    assert!(system.contains(&root.display().to_string()));
    assert!(
        system.contains("{{SECRET_"),
        "secret handling should be explained"
    );

    let tools = request["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 6);
    assert!(tools.iter().any(|t| t["function"]["name"] == "run_command"));

    std::fs::remove_dir_all(&root).ok();
}
