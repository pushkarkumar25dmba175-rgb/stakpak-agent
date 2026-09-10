//! Command-line surface.

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

use stakpak_agent::config::ApprovalMode;

#[derive(Parser, Debug)]
#[command(
    name = "stakpak",
    version,
    about = "An autonomous DevOps agent for your terminal",
    long_about = "Stakpak runs an LLM agent against your infrastructure with tool access, \
                  secret redaction, file backups, and resumable sessions.\n\n\
                  Run it with no arguments for an interactive session, or pass a prompt to \
                  run a single request and exit."
)]
pub struct Cli {
    /// Prompt to run non-interactively. Omit for an interactive session.
    #[arg(trailing_var_arg = true)]
    pub prompt: Vec<String>,

    /// Directory the agent works in. Defaults to the current directory.
    #[arg(short = 'C', long, global = true, value_name = "DIR")]
    pub workdir: Option<PathBuf>,

    /// Use this config file instead of the usual layered lookup.
    #[arg(short, long, global = true, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Override the configured model.
    #[arg(long, global = true, value_name = "MODEL")]
    pub model: Option<String>,

    /// Approve every tool call without asking.
    #[arg(short = 'y', long, global = true)]
    pub yes: bool,

    /// Override the approval policy for this run.
    #[arg(long, global = true, value_name = "MODE")]
    pub approval: Option<ApprovalArg>,

    /// Resume a saved session by id.
    #[arg(long, value_name = "ID", conflicts_with = "continue_")]
    pub resume: Option<String>,

    /// Resume the most recent session in this directory.
    #[arg(long = "continue")]
    pub continue_: bool,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum ApprovalArg {
    /// Run everything without asking.
    Never,
    /// Ask before writes and commands.
    Writes,
    /// Ask before every tool call.
    Always,
}

impl From<ApprovalArg> for ApprovalMode {
    fn from(value: ApprovalArg) -> Self {
        match value {
            ApprovalArg::Never => ApprovalMode::Never,
            ApprovalArg::Writes => ApprovalMode::Writes,
            ApprovalArg::Always => ApprovalMode::Always,
        }
    }
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// List saved sessions in this working directory.
    Sessions,
    /// Show the transcript of a session.
    Show {
        /// Session id. Defaults to the most recent session.
        id: Option<String>,
    },
    /// List file backups the agent has taken.
    Backups,
    /// Restore a file from a backup id (see `stakpak backups`).
    Restore {
        /// Backup id.
        id: String,
    },
    /// Inspect or create configuration.
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// List the tools available to the agent.
    Tools,
}

#[derive(Subcommand, Debug)]
pub enum ConfigAction {
    /// Print the effective configuration after all layers are merged.
    Show,
    /// Write a commented starter config to ./stakpak.toml.
    Init {
        /// Overwrite an existing file.
        #[arg(long)]
        force: bool,
    },
    /// Print the config file paths that are consulted.
    Path,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn a_bare_prompt_is_captured() {
        let cli = Cli::parse_from(["stakpak", "why", "is", "the", "pod", "crashlooping"]);
        assert!(cli.command.is_none());
        assert_eq!(cli.prompt.join(" "), "why is the pod crashlooping");
    }

    #[test]
    fn subcommands_still_parse() {
        let cli = Cli::parse_from(["stakpak", "sessions"]);
        assert!(matches!(cli.command, Some(Command::Sessions)));
    }

    #[test]
    fn resume_and_continue_are_mutually_exclusive() {
        assert!(Cli::try_parse_from(["stakpak", "--resume", "abc", "--continue"]).is_err());
    }
}
