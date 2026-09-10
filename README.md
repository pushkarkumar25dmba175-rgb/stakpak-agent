# stakpak-agent

An autonomous DevOps agent for your terminal. You describe what you want done to
your infrastructure; the agent reads the repository, runs commands, edits files,
and reports what actually happened — with credentials hidden from the model,
every write reversible, and the whole session resumable.

Written in Rust, ships as a single binary, and works against Anthropic or any
OpenAI-compatible endpoint (including a model running on your own machine).

```
$ stakpak "the staging deployment is crashlooping, find out why"

I checked the deployment and the pod is failing its readiness probe.

  ⏺ run: kubectl get pods -n staging
  ⎿ web-7d4f8c9b6-x2k9p   0/1   CrashLoopBackOff   7   14m
  ⏺ run: kubectl logs web-7d4f8c9b6-x2k9p -n staging --previous
  ⎿ Error: DATABASE_URL is not set
  ⏺ read k8s/staging/deployment.yaml

The deployment references a `db-credentials` secret that no longer exists in the
staging namespace — it was renamed to `postgres-credentials` in commit 4a91c02
but the staging manifest still points at the old name.
```

## Why this exists

An agent with shell access to your infrastructure is useful and alarming in
equal measure. This one is built around three properties that make it possible
to actually leave it running:

**The model never sees your credentials.** Tool output is scanned before it goes
anywhere near the model, and anything that looks like a credential — AWS keys,
GitHub tokens, private key blocks, JWTs, connection-string passwords,
`PASSWORD=` assignments — is replaced with a placeholder like
`{{SECRET_github_token_1}}`. When the model passes that placeholder back in a
command, the real value is substituted in at the last moment, inside the tool.
So the agent can `curl` an authenticated endpoint using a token it was never
shown.

**Every write is reversible.** Before any file is created or modified, the
previous contents are copied into `.stakpak/backups/` and recorded in an
append-only index. `stakpak backups` lists them; `stakpak restore <id>` puts a
file back — including deleting a file the agent created.

**Nothing runs without your say-so, by default.** Reads happen freely; anything
that writes a file or runs a command asks first, showing you the exact
arguments. `--yes` turns that off when you want it to run unattended, and
`security.denied_command_patterns` refuses a command outright regardless of
approval mode.

## Install

```sh
git clone https://github.com/pushkarkumar25dmba175-rgb/stakpak-agent
cd stakpak-agent
cargo install --path .
```

Requires Rust 1.88 or newer. Then point it at a model:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
stakpak config init      # writes a commented stakpak.toml
```

## Usage

Interactive session in the current directory:

```sh
stakpak
```

One-shot, then exit:

```sh
stakpak "add a health check to the web service and validate the manifest"
```

Useful flags:

| Flag | Effect |
| --- | --- |
| `-C, --workdir <DIR>` | Work in a different directory |
| `-y, --yes` | Approve every tool call (unattended runs) |
| `--approval never\|writes\|always` | Override the approval policy for one run |
| `--model <MODEL>` | Override the configured model |
| `--continue` | Resume the most recent session here |
| `--resume <ID>` | Resume a specific session |

Subcommands:

| Command | Effect |
| --- | --- |
| `stakpak sessions` | List saved sessions in this directory |
| `stakpak show [ID]` | Print a session transcript |
| `stakpak backups` | List file backups the agent has taken |
| `stakpak restore <ID>` | Put a file back the way it was |
| `stakpak tools` | List the tools the agent has |
| `stakpak config show\|init\|path` | Inspect or create configuration |

Inside an interactive session, `/help` lists slash commands: `/tools`,
`/secrets`, `/checkpoints`, `/rewind <n>`, `/session`, `/clear`, `/exit`.

## Tools

| Tool | Effect | What it does |
| --- | --- | --- |
| `read_file` | read | Read a file with line numbers, with paging for large files |
| `list_files` | read | Walk the tree, skipping `.gitignore`d and hidden paths |
| `search` | read | Regex search across files, optionally filtered by glob |
| `write_file` | mutates | Create or overwrite a file, backing up what was there |
| `edit_file` | mutates | Exact string replacement, with a diff in the result |
| `run_command` | mutates | Run a shell command with a timeout and captured output |

All paths are resolved against the working directory and refused if they escape
it, unless `security.allow_outside_workdir` says otherwise.

## Configuration

Two optional files are merged, key by key, with built-in defaults underneath:

1. `~/.config/stakpak-agent/config.toml`
2. `./stakpak.toml` (wins)

`stakpak config init` writes a commented starter file. The interesting knobs:

```toml
[provider]
kind = "anthropic"              # or "openai" for any OpenAI-compatible endpoint
model = "claude-sonnet-5"
api_key_env = "ANTHROPIC_API_KEY"   # the key is read from the environment, never stored
base_url = "https://api.anthropic.com"

[agent]
max_iterations = 40             # tool rounds allowed in one turn
system_prompt_append = ""       # your deploy conventions, house rules, etc.

[security]
redact_secrets = true
redact_env = true               # also hide secret-looking environment variables
approval = "writes"             # "never" | "writes" | "always"
denied_command_patterns = ["rm -rf /", "mkfs", "shutdown"]
allow_outside_workdir = false

[tools]
command_timeout_secs = 120
max_output_bytes = 30000        # long output is truncated in the middle
backup_files = true
```

`STAKPAK_MODEL`, `STAKPAK_PROVIDER`, `STAKPAK_BASE_URL`, `STAKPAK_API_KEY_ENV`,
and `STAKPAK_APPROVAL` override the corresponding settings.

### Running against a local model

```toml
[provider]
kind = "openai"
model = "qwen2.5-coder:32b"
base_url = "http://localhost:11434"
api_key_env = "OLLAMA_API_KEY"   # any non-empty value
```

Loopback endpoints bypass any configured HTTP proxy.

## Sessions

Every turn is written to `.stakpak/sessions/<id>.json` as it happens, and each
turn records a checkpoint. So a session survives a crash, `--continue` picks up
where you left off, and `/rewind 3` drops everything after the third turn when
the agent has gone down a dead end.

## How it works

```
src/
  agent/           the model/tool loop, system prompt, approval gating
    provider/      Anthropic Messages API and OpenAI chat-completions
  tools/           the tool surface, plus path sandboxing and output limits
  redact.rs        secret detection, placeholder substitution, restoration
  backup.rs        pre-write snapshots and restoration
  session.rs       transcript persistence and checkpoints
  ui.rs            terminal rendering, diffs, approval prompts
```

One turn is: send the transcript plus tool schemas to the model; if it asks for
tools, restore any placeholders in the arguments, gate on approval, run them,
redact the results, and send them back; repeat until it stops asking or
`agent.max_iterations` is hit.

## Development

```sh
cargo test          # unit tests plus end-to-end tests against a stub provider
cargo clippy --all-targets
cargo fmt
```

The integration tests in `tests/agent_loop.rs` run the whole loop — request
shaping, tool dispatch, secret substitution, transcript persistence — against a
stub HTTP server standing in for the model, so no API key is needed to test.

## Roadmap

- Streaming responses, so long answers appear as they are generated
- MCP support, both as a client for external servers and as a server exposing
  these tools to editors
- A full TUI with a scrollable transcript and a checkpoint browser
- An unattended mode with a schedule, for recurring infrastructure checks

## Prior art

Inspired by [Stakpak](https://github.com/stakpak/agent), an open-source DevOps
agent with a similar security posture. This is an independent implementation,
not a fork.

## License

Apache-2.0. See [LICENSE](LICENSE).
