# PersonalOS Agent

A local AI coworker that lives on your computer, learns how you work, and does
what you ask — while you keep control of everything that matters.

It reads and organises your files, summarises documents, writes reports, runs
approved commands, and notices the workflows you repeat. What it does *not* do
is act on its own authority: every consequential action is classified, shown to
you, approved by you, logged, verified and — wherever it is possible — undoable.

```
$ agent ask "organise the PDFs in my research inbox and write me an index"

Plan: Organise research documents
  1. List what is in ~/Research/inbox                     filesystem.list
  2. Extract metadata from the 12 PDFs                    documents.metadata
  3. Rename them to your YYYY-MM-DD convention            filesystem.rename
  4. Write research-index.md                              filesystem.write

┌─ Approval needed ────────────────────────────────────────────────┐
│ Action      Rename 2024-report.pdf → 2026-09-15-acme-breach.pdf  │
│ Tool        filesystem.rename                                     │
│ Risk        level 2 (moderate local change)                       │
│ Affects     ~/Research/inbox/2024-report.pdf                      │
│ Reversible  yes — the inverse rename is recorded                  │
└───────────────────────────────────────────────────────────────────┘
Go ahead? [y/N]:
```

---

## Why it is built this way

An agent with a shell and your home directory is useful and alarming in equal
measure. Three properties make it something you can actually leave running.

**Nothing consequential happens without you.** Every action is classified into
one of five risk levels. Reads run freely; a level-3 action (deleting,
installing, anything privileged) and a level-4 one (anything that leaves this
machine) require you to *type a word*, every single time. No standing rule, no
`--yes` flag and no amount of repetition can waive that — it is enforced in the
policy engine and covered by tests.

**Everything is reversible and everything is logged.** Before a file is
overwritten its contents are copied aside. Before it is moved, the inverse move
is journalled. "Delete" means "move to `~/.personalos/agent_trash/`" unless you
insist otherwise. `agent rollback TASK_ID` replays the journal backwards. Every
action is appended to `logs/actions-YYYY-MM-DD.jsonl` with its risk level,
approval status and verification result.

**Learning is explicit, inspectable and reversible.** The agent notices when you
repeat a workflow — but noticing is not automating. It *asks*, and only if you
accept does it write a skill, as a YAML file you can read, edit and delete.
Preferences carry provenance: something you stated has confidence 1.0 and never
decays, something inferred decays with age and stops being used once it gets
weak. `agent memory` shows you all of it; `agent memory forget` removes any of it.

---

## Install

Python 3.12 or newer.

```sh
git clone https://github.com/pushkarkumar25dmba175-rgb/stakpak-agent
cd stakpak-agent/personal-agent

python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"        # or just `pip install -e .` for the core

agent init --workspace ~/PersonalOS
agent doctor
```

`agent doctor` tells you what is configured, what is missing and what is
optional. You can run everything below without an API key — the agent falls
back to a keyword planner and says so rather than pretending.

To use a model, set one environment variable:

```sh
export ANTHROPIC_API_KEY=...            # or OPENAI_API_KEY
```

or run entirely locally, with nothing leaving your machine:

```sh
export PERSONALOS_LLM_PROVIDER=ollama
export PERSONALOS_LLM_MODEL=llama3.1
```

Without installing, `python main.py ask "..."` works from a checkout.

---

## First five minutes

```sh
agent ask "what is in my PersonalOS folder?"     # read-only: runs immediately
agent plan "tidy up my downloads"                # see the plan, execute nothing
agent skills                                      # the skills you have
agent run daily_summary --input folder=~/PersonalOS
agent history --today                             # everything it did
agent explain TASK_ID                             # everything about one task
agent rollback TASK_ID --dry-run                  # what undoing would do
agent feedback "always save reports as Markdown"  # teach it something
```

---

## The security model

Every operation declares a base risk level, and every *specific call* can be
escalated above it — writing a new file is level 1, overwriting an existing one
is level 2, and a shell command that touches `~/.ssh` is level 3 no matter how
innocuous it looked.

| Level | Meaning | Examples | Default |
|---|---|---|---|
| **0** | Read-only | list, read, search, summarise | runs automatically |
| **1** | Additive local change | create a file, copy something | automatic when you asked for it |
| **2** | Changes existing state | rename, move, overwrite, edit | asks, unless a standing rule covers it |
| **3** | Destructive or privileged | delete, install, `sudo`, stop a process | **always asks — type `DELETE`** |
| **4** | Leaves this machine | send, publish, deploy, pay, call an API | **always asks, immediately before it happens — type `SEND`** |

Approval happens per step, *at the moment the step runs* — not once for the
whole plan up front. A plan whose fourth step turns out riskier than expected
stops and asks.

**What cannot be loosened.** These are schema- and policy-level invariants, not
conventions:

- `security.auto_approve_max_level` is capped at 2 by the config schema.
- Levels 3 and 4 skip rule evaluation entirely and go straight to a human.
- An allow rule is clamped to level 2 when it is written *and* when it is read.
- `AutoApproveHandler`, used by the scheduler, clamps its own ceiling to level 2.
- `learning.auto_create_skills` cannot be set to true.
- The dashboard refuses to bind to anything but loopback.

**Where it can reach.** `workspace.allowed_roots` is an allow-list: a path
outside it is refused before any tool runs. `workspace.denied_paths` is carved
back out of it — `~/.ssh`, `~/.aws`, `~/.gnupg` and friends are unreachable by
default even if they sit inside an allowed root. Symlinks are not followed.

**Shell commands are screened, not just confirmed.** Commands are parsed and
checked for destructive patterns, privilege escalation, credential access,
persistence mechanisms and suspicious chains. Anything matching
`security.shell_denied_patterns` is blocked outright — not offered for
approval, because "are you sure?" is the wrong question for `rm -rf /`.
Child processes get an environment with every credential-looking variable
stripped out, so a script the agent runs cannot read your API keys.

**Secrets are referenced, never stored.** The config names an *environment
variable* (`api_key_env: ANTHROPIC_API_KEY`), and the HTTP tool takes
`auth_secret_ref: env:GITHUB_TOKEN` rather than a token — it refuses a raw
`Authorization` header. Everything on its way to a log, the database or the
model passes through pattern-based redaction first.

**External content is data, never instructions.** Text from files, web pages and
API responses is wrapped in an `<untrusted_content>` fence, scanned for
injection patterns, and flagged in the prompt. A document saying "ignore your
previous instructions" is a document that says that. The layer that actually
stops an injected instruction from doing damage, though, is the permission
model: it asks you before anything consequential happens, no matter who
suggested it.

---

## How it learns

```
OBSERVE → UNDERSTAND → RETRIEVE MEMORY → PLAN → CHECK PERMISSIONS
        → EXECUTE → VERIFY → LOG → LEARN
```

Five memory layers, all in one local SQLite file, all inspectable:

| Layer | Holds | Lives |
|---|---|---|
| **Working** | the current request, files in focus, step results | in memory, discarded at the end |
| **Episodic** | past tasks, what was done, whether it worked, your corrections | `episodes` |
| **Semantic** | durable facts: what a folder is for, what a project means | `semantic_memory` |
| **Preference** | how you like things done, with source and confidence | `preferences` |
| **Workflow** | shapes of task you keep repeating | `workflows` |

**Retrieval is budgeted.** The whole database is never sent to the model. Each
request is classified, memory is searched and ranked, and only what fits in
`llm.memory_budget_tokens` is injected — explicit preferences first, because
they are instructions rather than background.

**Workflow discovery.** Each task is fingerprinted by its chain of
`tool.operation` calls, so "rename → summarise → move" is the same shape whether
it ran over three files or thirty. Similar fingerprints cluster; once a cluster
crosses `learning.min_occurrences_for_suggestion`, the agent says so:

```
┌─ I noticed something ────────────────────────────────────────────┐
│ You have run this sequence 3 times. Shall I turn it into a       │
│ reusable skill?                                                  │
│                                                                  │
│ Steps: filesystem.rename → documents.extract_text →              │
│        filesystem.move (3 times)                                 │
│ Accepting writes a skill definition you can read and edit. It    │
│ does not run anything, and it does not change what needs         │
│ approval.                                                        │
└──────────────────────────────────────────────────────────────────┘
```

`agent suggestions accept 1` writes the YAML and prints it. Nothing is created
otherwise. Suggestions are rate-limited by a cooldown and a per-session cap, so
the agent does not become the thing that interrupts you all day.

**Corrections outrank inferences, always.** `agent feedback "never rename files
in this folder"` becomes a deny rule. `agent feedback "don't ask before moving
files in this project"` becomes an allow rule — clamped to level 2, so it
covers routine moves and nothing else. An inferred preference can never
overwrite one you stated; the store returns the stated value unchanged.

---

## Skills

A skill is a YAML file describing a plan with named inputs — declarative on
purpose. A skill you can read is a skill you can audit, and one the agent cannot
use to smuggle in arbitrary code. **A skill is a saved plan, not a granted
permission**: running one goes through the same policy engine as anything else.

```yaml
name: archive_reports
description: Collect reports from a folder into a dated archive directory.
version: 1
inputs:
  folder: { type: string, required: true }
steps:
  - id: survey
    description: See what is in the folder
    tool: filesystem
    operation: list
    arguments: { path: "{{ inputs.folder }}", pattern: "*.md" }
    verification: { kind: count_at_least, field: count, minimum: 0 }
  - id: manifest
    description: Write a manifest of what was found
    tool: filesystem
    operation: write
    arguments:
      path: "{{ inputs.folder }}/manifest.md"
      content: "Reports found: {{ steps.survey.data.count }}"
    depends_on: [survey]
    verification: { kind: path_nonempty }
```

```sh
agent skills                          # list, with success rates
agent skills inspect archive_reports  # print the definition
agent skills version NAME FILE        # supersede, keeping the old version
agent skills disable NAME             # stop it running, without deleting it
agent skills delete NAME              # delete, keeping a copy
```

Full example: [`config/skill.example.yaml`](config/skill.example.yaml).

---

## Verification

A tool returning success means the call did not raise. It does not mean the file
is where it should be. Every step is checked afterwards — `path_exists`,
`path_nonempty`, `output_contains`, `return_code_zero`, `count_at_least` — and a
step with no applicable check is recorded as *unverified* rather than
optimistically passed. `agent explain TASK_ID` shows the verification result for
every action.

---

## Commands

```
agent init                  set up ~/.personalos
agent doctor                check the installation and security posture
agent ask "…"               do something
agent plan "…"              show the plan, execute nothing
agent run SKILL             run a saved skill
agent status                what the agent is doing, and how it is configured
agent tools                 every operation and what it risks

agent history [--today] [--failures] [--task ID]
agent explain TASK_ID       everything about one task
agent rollback TASK_ID [--dry-run]

agent memory show|search|set|forget|workflows
agent feedback "…"          teach it something

agent skills list|inspect|import|version|enable|disable|delete
agent suggestions list|accept|dismiss|scan

agent schedule add|list|run|remove|enable|disable|serve
agent permissions show|allow|deny|revoke
agent project list|create|switch|show
agent observe status|events|poll|pause

agent morning               what is scheduled and what is open
agent evening               what happened, and what you keep repeating
agent pause / agent resume
agent dashboard             the local web view
agent config                the effective configuration
```

---

## Scheduling, observation and projects

**Scheduling.** `agent schedule add tidy --when "every day at 18:00" --ask
"organise my downloads"`. Scheduled jobs run with nobody watching, so they get a
handler that can approve at most level 2 — and a job whose plan needs level 3
stops and records why, where you will see it in `agent history`.

**Observation is off by default** and narrow when on: it records *that* a file
appeared, with its path and size, never its contents. It watches only the
folders you list. It never starts a task on its own — a watcher that triggers
work is a watcher that can be triggered by anyone who can drop a file in your
Downloads folder. `agent observe pause` is immediate.

**Projects** scope the agent to one body of work: their own directories, their
own memory namespace, their own rules. That makes project scoping a security
feature — "don't ask before moving files here" can be granted for one project
without applying anywhere else.

```sh
agent project create threat-research -d ~/Research/threat-intel
agent project switch threat-research
```

See [`config/project.example.json`](config/project.example.json).

---

## Dashboard

```sh
pip install -e ".[dashboard]"
agent dashboard          # http://127.0.0.1:8765
```

Read-only: tasks, memory, skills, schedules, permissions, audit log, settings.
It deliberately does not expose approvals — a browser tab is a poor place to
make a consequential decision, and an HTTP endpoint that grants permissions is
an endpoint worth attacking. Approvals stay on the terminal.

---

## Starting with your computer

[`packaging/`](packaging/) has templates for systemd (`--user`), launchd and
Windows Task Scheduler. They start the **scheduler service only**, as your user,
never elevated. Starting the service does not start doing things.

---

## Layout

```
src/personalos/
  agent/        orchestrator, planner, executor, reasoning, verification, state
  memory/       working, episodic, semantic, preference, workflow + retrieval
  tools/        filesystem, shell, python, documents, http, notifications,
                processes, clipboard, browser — behind one Tool contract
  skills/       declarative YAML skills, registry, manager, built-ins
  learning/     pattern detection, workflow discovery, feedback, skill generation
  security/     risk levels, policy engine, sandbox, secrets, injection defence
  audit/        JSONL action log, rollback journal
  scheduler/    jobs, schedule parsing, local scheduler
  observers/    opt-in folder watching
  projects/     project definitions and scoping
  database/     SQLAlchemy models, engine
  llm/          provider abstraction: Anthropic, OpenAI, Ollama, offline
  interface/    CLI, approval prompts, doctor, daily briefs
  dashboard/    optional FastAPI + static UI
```

Layers depend only downwards. The `Tool` contract is the extension point: give
a class a name, a description, an input schema per operation, a risk level and
an `_run`, register it, and the planner, policy engine, audit log and rollback
journal pick it up with no other changes.

---

## Development

```sh
pip install -e ".[all,dev]"
pytest                     # the full suite, no network or API key needed
ruff check src tests
mypy src
```

Tests run against a throwaway home under `tmp_path` and an offline provider, so
they never touch your real state. The suite covers the security invariants
directly — that level 3 asks even with a matching allow rule, that an inferred
preference cannot overwrite a stated one, that a repeated workflow produces a
suggestion and not an automation, that a failed step's output never reaches a
later step.

### Roadmap

Phases 1–5 are implemented: CLI, provider abstraction, tools, memory, planner,
audit, permissions, risk classification, rollback, shell and Python execution,
workflow memory, pattern detection, feedback learning, skill generation,
scheduler, folder monitoring, notifications, dashboard, projects and retrieval.

Phase 6 — browser and desktop automation — is deliberately only an abstraction
(`tools/browser.py`). The contract requires an isolated profile, because a
driver that inherits your logged-in browser session can act as you on every site
you are signed in to. Vector search is the same: `memory/vector_index.py` defines
the seam, and lexical retrieval is what ships.

---

## Licence

Apache-2.0. See [LICENSE](../LICENSE).
