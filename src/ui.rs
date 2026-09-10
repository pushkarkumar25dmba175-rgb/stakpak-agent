//! Terminal rendering: agent output, tool activity, diffs, and prompts.

use std::io::{self, IsTerminal, Write};

use colored::Colorize;
use similar::{ChangeTag, TextDiff};

/// What the user decided about a pending tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Approval {
    Once,
    /// Approve this and everything else for the rest of the session.
    Always,
    Denied(String),
}

pub fn banner(model: &str, workdir: &str, session: &str) {
    println!(
        "{} {}",
        "stakpak".bold().cyan(),
        format!("v{}", env!("CARGO_PKG_VERSION")).dimmed()
    );
    println!("  {} {}", "model  ".dimmed(), model);
    println!("  {} {}", "workdir".dimmed(), workdir);
    println!("  {} {}", "session".dimmed(), session);
    println!(
        "  {} {}",
        "help   ".dimmed(),
        "/help for commands, /exit to quit".dimmed()
    );
    println!();
}

pub fn assistant(text: &str) {
    if text.trim().is_empty() {
        return;
    }
    println!("{}", text);
    println!();
}

pub fn tool_call(summary: &str) {
    println!("  {} {}", "⏺".cyan(), summary.bold());
}

pub fn tool_result(content: &str, is_error: bool) {
    let prefix = if is_error {
        "  ⎿ ".red().to_string()
    } else {
        "  ⎿ ".dimmed().to_string()
    };
    // Long results are already in the transcript; the terminal only needs a
    // glimpse to show progress.
    let lines: Vec<&str> = content.lines().collect();
    let shown = lines.iter().take(6);
    for (i, line) in shown.enumerate() {
        let body = line.chars().take(160).collect::<String>();
        if i == 0 {
            println!("{prefix}{}", body.dimmed());
        } else {
            println!("    {}", body.dimmed());
        }
    }
    if lines.len() > 6 {
        println!(
            "    {}",
            format!("… {} more lines", lines.len() - 6).dimmed()
        );
    }
}

pub fn info(message: &str) {
    println!("{} {}", "•".dimmed(), message.dimmed());
}

pub fn warn(message: &str) {
    eprintln!("{} {}", "!".yellow(), message.yellow());
}

pub fn error(message: &str) {
    eprintln!("{} {}", "✗".red(), message.red());
}

/// Asks whether to run a tool call. Falls back to denying when there is no
/// terminal to ask on, so a non-interactive run cannot hang forever.
pub fn ask_approval(summary: &str, detail: Option<&str>) -> Approval {
    if !io::stdin().is_terminal() {
        return Approval::Denied(
            "no terminal available to approve this action; re-run with --yes to auto-approve"
                .to_string(),
        );
    }

    println!();
    println!("  {} {}", "?".yellow().bold(), summary.bold());
    if let Some(detail) = detail {
        for line in detail.lines().take(20) {
            println!("    {}", line.dimmed());
        }
    }
    print!(
        "  {} ",
        "[y] run  [a] run everything  [n] skip  [reason] skip and explain:".dimmed()
    );
    let _ = io::stdout().flush();

    let mut answer = String::new();
    if io::stdin().read_line(&mut answer).is_err() {
        return Approval::Denied("could not read a response".to_string());
    }
    println!();

    match answer.trim() {
        "y" | "Y" | "" => Approval::Once,
        "a" | "A" => Approval::Always,
        "n" | "N" => Approval::Denied("the user skipped this action".to_string()),
        other => Approval::Denied(format!("the user skipped this action: {other}")),
    }
}

/// A compact unified diff, used in edit output and approval prompts.
pub fn unified_diff(before: &str, after: &str, label: &str) -> String {
    let diff = TextDiff::from_lines(before, after);
    let mut out = String::new();
    for group in diff.grouped_ops(3) {
        for op in group {
            for change in diff.iter_changes(&op) {
                let sign = match change.tag() {
                    ChangeTag::Delete => '-',
                    ChangeTag::Insert => '+',
                    ChangeTag::Equal => ' ',
                };
                out.push(sign);
                out.push_str(change.value().trim_end_matches('\n'));
                out.push('\n');
            }
        }
        out.push_str("...\n");
    }
    if out.is_empty() {
        return format!("{label}: no changes\n");
    }
    // The trailing separator after the final hunk adds nothing.
    if out.ends_with("...\n") {
        out.truncate(out.len() - 4);
    }
    format!("--- {label}\n+++ {label}\n{out}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_marks_added_and_removed_lines() {
        let diff = unified_diff("a\nb\nc\n", "a\nB\nc\n", "f.txt");
        assert!(diff.contains("-b"));
        assert!(diff.contains("+B"));
        assert!(diff.contains(" a"));
    }

    #[test]
    fn identical_input_reports_no_changes() {
        assert!(unified_diff("same\n", "same\n", "f.txt").contains("no changes"));
    }
}
