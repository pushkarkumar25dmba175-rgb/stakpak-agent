//! Command-line shell around the [`stakpak_agent`] library.

mod cli;

use std::path::Path;

use anyhow::{Context, Result};
use clap::Parser;
use colored::Colorize;

use cli::{Cli, Command, ConfigAction};
use stakpak_agent::agent::{self, Agent};
use stakpak_agent::backup::BackupStore;
use stakpak_agent::config::{self, Config, STATE_DIR};
use stakpak_agent::session::Session;
use stakpak_agent::tools::{self, ToolRegistry};
use stakpak_agent::ui;

#[tokio::main]
async fn main() {
    if let Err(err) = run().await {
        ui::error(&format!("{err:#}"));
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let cli = Cli::parse();

    let workdir = match &cli.workdir {
        Some(dir) => dir
            .canonicalize()
            .with_context(|| format!("no such directory: {}", dir.display()))?,
        None => std::env::current_dir().context("determining the current directory")?,
    };
    let state_dir = workdir.join(STATE_DIR);

    // Subcommands that only inspect state do not need a configured provider or
    // an API key, so they are handled before the config is validated.
    match &cli.command {
        Some(Command::Config { action }) => return run_config(&cli, &workdir, action),
        Some(Command::Sessions) => return list_sessions(&state_dir),
        Some(Command::Show { id }) => return show_session(&state_dir, id.as_deref()),
        Some(Command::Backups) => return list_backups(&state_dir),
        Some(Command::Restore { id }) => return restore_backup(&state_dir, id),
        Some(Command::Tools) => return list_tools(),
        None => {}
    }

    let config = load_config(&cli, &workdir)?;
    let session = resolve_session(&cli, &workdir, &state_dir, &config)?;
    let resumed = !session.messages.is_empty();

    let mut agent = Agent::new(config, &workdir, &state_dir, session, cli.yes)?;

    let prompt = cli.prompt.join(" ");
    if !prompt.trim().is_empty() {
        agent.turn(prompt.trim()).await?;
        return Ok(());
    }

    interactive(&mut agent, &workdir, &state_dir, resumed).await
}

fn load_config(cli: &Cli, workdir: &Path) -> Result<Config> {
    let mut config = Config::load(workdir, cli.config.as_deref())?;
    if let Some(model) = &cli.model {
        config.provider.model = model.clone();
    }
    if let Some(approval) = cli.approval {
        config.security.approval = approval.into();
    }
    Ok(config)
}

fn resolve_session(
    cli: &Cli,
    workdir: &Path,
    state_dir: &Path,
    config: &Config,
) -> Result<Session> {
    if let Some(id) = &cli.resume {
        let session = Session::load(state_dir, id)?;
        ui::info(&format!(
            "resumed session {} ({} messages)",
            session.id,
            session.messages.len()
        ));
        return Ok(session);
    }
    if cli.continue_ {
        return match Session::latest(state_dir)? {
            Some(session) => {
                ui::info(&format!(
                    "continuing session {} ({} messages)",
                    session.id,
                    session.messages.len()
                ));
                Ok(session)
            }
            None => {
                ui::warn("no saved session here yet; starting a new one");
                Ok(Session::new(workdir, &config.provider.model))
            }
        };
    }
    Ok(Session::new(workdir, &config.provider.model))
}

async fn interactive(
    agent: &mut Agent,
    workdir: &Path,
    state_dir: &Path,
    resumed: bool,
) -> Result<()> {
    ui::banner(
        &format!("{} ({})", agent.model(), agent.provider_name()),
        &workdir.display().to_string(),
        &agent.session.id,
    );
    if agent.secrets().is_enabled() && !agent.secrets().is_empty() {
        ui::info(&format!(
            "{} secret(s) from the environment will be hidden from the model",
            agent.secrets().len()
        ));
    }
    if resumed {
        ui::info("previous messages are still in context");
    }

    let mut editor = rustyline::DefaultEditor::new()?;
    let history = state_dir.join("history");
    let _ = editor.load_history(&history);

    loop {
        let line = match editor.readline(&format!("{} ", "›".cyan().bold())) {
            Ok(line) => line,
            Err(rustyline::error::ReadlineError::Interrupted) => {
                println!("(ctrl-c — use /exit to quit)");
                continue;
            }
            Err(rustyline::error::ReadlineError::Eof) => break,
            Err(err) => return Err(err.into()),
        };

        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        let _ = editor.add_history_entry(input);

        if input.starts_with('/') {
            match handle_slash(agent, input) {
                Ok(true) => break,
                Ok(false) => continue,
                Err(err) => {
                    ui::error(&format!("{err:#}"));
                    continue;
                }
            }
        }

        println!();
        if let Err(err) = agent.turn(input).await {
            // A failed turn should not end the session; the user may want to
            // fix an API key or a config file and try again.
            ui::error(&format!("{err:#}"));
        }
    }

    if let Some(parent) = history.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let _ = editor.save_history(&history);
    agent.save()?;
    ui::info(&format!(
        "session {} saved — resume with `stakpak --resume {}`",
        agent.session.id, agent.session.id
    ));
    Ok(())
}

/// Handles a `/command`. Returns true when the session should end.
fn handle_slash(agent: &mut Agent, input: &str) -> Result<bool> {
    let mut parts = input.splitn(2, char::is_whitespace);
    let command = parts.next().unwrap_or_default();
    let rest = parts.next().unwrap_or_default().trim();

    match command {
        "/exit" | "/quit" => return Ok(true),
        "/help" => {
            println!("  /help                this message");
            println!("  /tools               list available tools");
            println!("  /secrets             list redacted secret placeholders");
            println!("  /checkpoints         list checkpoints in this session");
            println!("  /rewind <n>          drop everything after checkpoint n");
            println!("  /session             show session id and message count");
            println!("  /clear               start a fresh transcript");
            println!("  /exit                save and quit");
            println!();
        }
        "/tools" => list_tools()?,
        "/secrets" => {
            let placeholders = agent.secrets().placeholders();
            if placeholders.is_empty() {
                println!("  no secrets registered yet");
            } else {
                for placeholder in placeholders {
                    println!("  {placeholder}");
                }
            }
            println!();
        }
        "/checkpoints" => {
            if agent.session.checkpoints.is_empty() {
                println!("  no checkpoints yet");
            } else {
                for checkpoint in &agent.session.checkpoints {
                    println!(
                        "  {:>3}  {}  {}",
                        checkpoint.index,
                        checkpoint.timestamp.format("%H:%M:%S"),
                        checkpoint.label
                    );
                }
            }
            println!();
        }
        "/rewind" => {
            let index: usize = rest
                .parse()
                .with_context(|| format!("usage: /rewind <checkpoint>, got {rest:?}"))?;
            let label = agent.session.rewind_to(index)?.label.clone();
            agent.save()?;
            ui::info(&format!("rewound to checkpoint {index}: {label}"));
        }
        "/session" => {
            ui::info(&format!(
                "session {} — {} messages, {} checkpoints",
                agent.session.id,
                agent.session.messages.len(),
                agent.session.checkpoints.len()
            ));
        }
        "/clear" => {
            agent.session.messages.clear();
            agent.session.checkpoints.clear();
            agent.save()?;
            ui::info("transcript cleared");
        }
        other => ui::warn(&format!("unknown command {other} — try /help")),
    }
    Ok(false)
}

fn run_config(cli: &Cli, workdir: &Path, action: &ConfigAction) -> Result<()> {
    match action {
        ConfigAction::Show => {
            let config = load_config(cli, workdir)?;
            print!("{}", config.to_toml()?);
        }
        ConfigAction::Path => {
            match config::user_config_path() {
                Some(path) => println!("user:    {}", path.display()),
                None => println!("user:    (no config directory on this platform)"),
            }
            println!(
                "project: {}",
                workdir.join(config::PROJECT_CONFIG).display()
            );
            println!();
            println!("Both are optional; the project file wins key by key.");
        }
        ConfigAction::Init { force } => {
            let path = workdir.join(config::PROJECT_CONFIG);
            if path.exists() && !force {
                anyhow::bail!(
                    "{} already exists; pass --force to overwrite",
                    path.display()
                );
            }
            std::fs::write(&path, starter_config())
                .with_context(|| format!("writing {}", path.display()))?;
            ui::info(&format!("wrote {}", path.display()));
        }
    }
    Ok(())
}

fn starter_config() -> String {
    let defaults = Config::default();
    format!(
        "# stakpak-agent configuration.\n\
         # Every key is optional; anything omitted falls back to the built-in default.\n\
         # A user-wide file is merged first, then this one.\n\n\
         [provider]\n\
         # \"anthropic\" or \"openai\" (any OpenAI-compatible endpoint).\n\
         kind = \"anthropic\"\n\
         model = \"{model}\"\n\
         # The key is read from this environment variable, never stored here.\n\
         api_key_env = \"{api_key_env}\"\n\
         base_url = \"{base_url}\"\n\
         max_tokens = {max_tokens}\n\
         temperature = {temperature:.1}\n\n\
         [agent]\n\
         # Tool rounds allowed in a single turn.\n\
         max_iterations = {max_iterations}\n\
         # Extra instructions appended to the system prompt, e.g. your deploy conventions.\n\
         system_prompt_append = \"\"\n\n\
         [security]\n\
         # Replace credentials with placeholders before the model sees them.\n\
         redact_secrets = true\n\
         redact_env = true\n\
         # \"never\", \"writes\", or \"always\".\n\
         approval = \"writes\"\n\
         # Commands containing any of these are refused outright.\n\
         denied_command_patterns = [{denied}]\n\
         allow_outside_workdir = false\n\n\
         [tools]\n\
         command_timeout_secs = {timeout}\n\
         max_output_bytes = {max_output}\n\
         backup_files = true\n",
        model = defaults.provider.model,
        api_key_env = defaults.provider.api_key_env,
        base_url = defaults.provider.base_url,
        max_tokens = defaults.provider.max_tokens,
        temperature = defaults.provider.temperature,
        max_iterations = defaults.agent.max_iterations,
        denied = defaults
            .security
            .denied_command_patterns
            .iter()
            .map(|p| format!("{p:?}"))
            .collect::<Vec<_>>()
            .join(", "),
        timeout = defaults.tools.command_timeout_secs,
        max_output = defaults.tools.max_output_bytes,
    )
}

fn list_sessions(state_dir: &Path) -> Result<()> {
    let sessions = Session::list(state_dir)?;
    if sessions.is_empty() {
        println!("No saved sessions here yet.");
        return Ok(());
    }
    for session in sessions {
        println!(
            "{}  {}  {:>3} msgs  {}",
            session.id.cyan(),
            session.updated_at.format("%Y-%m-%d %H:%M"),
            session.messages.len(),
            session.summary()
        );
    }
    Ok(())
}

fn show_session(state_dir: &Path, id: Option<&str>) -> Result<()> {
    let session = match id {
        Some(id) => Session::load(state_dir, id)?,
        None => Session::latest(state_dir)?.context("no saved sessions here yet")?,
    };
    println!(
        "{} {} — {} messages\n",
        "session".dimmed(),
        session.id.cyan(),
        session.messages.len()
    );
    for message in &session.messages {
        let (label, text) = match message.role {
            agent::message::Role::User => ("user".green(), message.text()),
            agent::message::Role::Assistant => ("agent".cyan(), message.text()),
        };
        for tool_use in message.tool_uses() {
            println!("{} {}", "tool".yellow(), tool_use.name);
        }
        if !text.trim().is_empty() {
            println!("{label} {text}\n");
        }
    }
    Ok(())
}

fn list_backups(state_dir: &Path) -> Result<()> {
    let store = BackupStore::new(state_dir, "", true);
    let entries = store.list()?;
    if entries.is_empty() {
        println!("No backups recorded here yet.");
        return Ok(());
    }
    for entry in entries {
        println!(
            "{}  {}  {:<10}  {}{}",
            entry.id.cyan(),
            entry.timestamp.format("%Y-%m-%d %H:%M:%S"),
            entry.tool,
            entry.original_path.display(),
            if entry.was_created() {
                "  (created by the agent)".dimmed().to_string()
            } else {
                String::new()
            }
        );
    }
    println!();
    println!("Restore one with `stakpak restore <id>`.");
    Ok(())
}

fn restore_backup(state_dir: &Path, id: &str) -> Result<()> {
    let store = BackupStore::new(state_dir, "", true);
    let entry = store.restore(id)?;
    if entry.was_created() {
        ui::info(&format!(
            "removed {} (it did not exist before the agent created it)",
            entry.original_path.display()
        ));
    } else {
        ui::info(&format!("restored {}", entry.original_path.display()));
    }
    Ok(())
}

fn list_tools() -> Result<()> {
    for tool in ToolRegistry::with_defaults().iter() {
        let effect = match tool.effect() {
            tools::Effect::Read => "read".green(),
            tools::Effect::Mutate => "mutates".yellow(),
        };
        println!("{:<14} {}", tool.name().cyan(), effect);
        for line in wrap(tool.description(), 76) {
            println!("               {}", line.dimmed());
        }
        println!();
    }
    Ok(())
}

/// Minimal greedy word wrap for help text.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if !current.is_empty() && current.len() + 1 + word.len() > width {
            lines.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starter_config_parses_as_config() {
        let parsed: Config = toml::from_str(&starter_config()).unwrap();
        assert_eq!(parsed.provider.model, Config::default().provider.model);
        assert!(parsed.security.redact_secrets);
    }

    #[test]
    fn wrapping_respects_the_width() {
        let lines = wrap("one two three four five six seven eight", 12);
        assert!(lines.iter().all(|l| l.len() <= 12));
        assert_eq!(lines.join(" "), "one two three four five six seven eight");
    }

    #[test]
    fn state_dir_hangs_off_the_workdir() {
        let dir = std::path::PathBuf::from("/srv/app").join(STATE_DIR);
        assert!(dir.ends_with(".stakpak"));
    }
}
