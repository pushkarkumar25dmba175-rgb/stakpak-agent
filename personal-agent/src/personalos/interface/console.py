"""Rendering helpers shared by every CLI command."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from personalos.agent.plan import Plan
from personalos.agent.state import AgentState
from personalos.errors import PersonalOSError
from personalos.security.risk import RiskLevel

console = Console()

RISK_STYLE: dict[RiskLevel, str] = {
    RiskLevel.READ_ONLY: "green",
    RiskLevel.LOW: "cyan",
    RiskLevel.MODERATE: "yellow",
    RiskLevel.HIGH: "red",
    RiskLevel.EXTERNAL: "bold red",
}

STATE_STYLE: dict[AgentState, str] = {
    AgentState.IDLE: "green",
    AgentState.OFFLINE: "dim",
    AgentState.PAUSED: "yellow",
    AgentState.ERROR: "red",
    AgentState.WAITING_FOR_APPROVAL: "yellow",
}


def risk_text(level: RiskLevel) -> Text:
    """A coloured ``level N (label)`` chip."""
    return Text(f"level {int(level)} ({level.label})", style=RISK_STYLE.get(level, "white"))


def state_text(state: AgentState) -> Text:
    return Text(str(state), style=STATE_STYLE.get(state, "cyan"))


def error(exc: Exception) -> None:
    """Print an error the way the agent is supposed to: what, and what to do."""
    if isinstance(exc, PersonalOSError):
        body = Text(exc.message, style="red")
        if exc.remediation:
            body.append("\n\n")
            body.append(exc.remediation, style="yellow")
    else:
        body = Text(f"{type(exc).__name__}: {exc}", style="red")
    console.print(Panel(body, title="Something went wrong", border_style="red"))


def info(message: str) -> None:
    console.print(message)


def success(message: str) -> None:
    console.print(f"[green]✓[/green] {message}")


def warn(message: str) -> None:
    console.print(f"[yellow]•[/yellow] {message}")


def render_plan(plan: Plan, *, title: str = "Plan") -> None:
    """Show a plan as a numbered table, including how each step is verified."""
    if plan.is_empty():
        console.print(
            Panel(
                plan.notes or "Nothing to do for this request.",
                title=title,
                border_style="cyan",
            )
        )
        return

    table = Table(title=f"{title}: {plan.goal}", show_lines=False, expand=False)
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Step")
    table.add_column("Tool", style="cyan")
    table.add_column("Checks", style="dim")

    for index, step in enumerate(plan.ordered(), start=1):
        verification = step.verification.kind if step.verification else "auto"
        table.add_row(str(index), step.description, f"{step.tool}.{step.operation}", verification)

    console.print(table)
    if plan.notes:
        console.print(f"[dim]{plan.notes}[/dim]")


def render_table(title: str, columns: list[str], rows: list[list[Any]]) -> None:
    """Generic table rendering, used by listing commands."""
    table = Table(title=title, expand=False)
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(*[Text(str(cell)) if not isinstance(cell, Text) else cell for cell in row])
    console.print(table)


def render_answer(answer: str, *, success_state: bool) -> None:
    console.print(
        Panel(
            answer.strip() or "(no output)",
            title="Result" if success_state else "Result (incomplete)",
            border_style="green" if success_state else "yellow",
        )
    )
