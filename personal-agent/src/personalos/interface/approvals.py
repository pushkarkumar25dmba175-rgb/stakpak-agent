"""The interactive approval prompt.

This is where the human is actually in the loop, so it has to show enough to
decide: what will happen, to which files, what is worrying about it, and
whether it can be undone. A prompt that just says "run shell command? [y/n]" is
a prompt people learn to answer "y" to without reading.

Level 3 and 4 actions also require typing a word rather than pressing a key —
enough friction that a destructive action cannot be approved by a stray
keystroke.
"""

from __future__ import annotations

from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from personalos.database.models import ApprovalStatus
from personalos.interface.console import console, risk_text
from personalos.security.permissions import ApprovalDecision, ApprovalRequest
from personalos.security.risk import RiskLevel


class InteractiveApprovalHandler:
    """Asks the user, on the terminal, whether an action may proceed.

    Args:
        assume_yes: Auto-approve up to level 2 without prompting — the ``--yes``
            flag. It cannot cover levels 3 and 4; those always prompt.
        quiet: Suppress the explanatory panel for level 0–1 actions.
    """

    def __init__(self, *, assume_yes: bool = False, quiet: bool = False) -> None:
        self.assume_yes = assume_yes
        self.quiet = quiet
        self.remembered: list[str] = []

    def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        if self.assume_yes and request.risk_level <= RiskLevel.MODERATE:
            return ApprovalDecision.by_user("Approved by --yes (level 2 and below).")

        self._render(request)

        if request.risk_level.always_requires_confirmation:
            return self._confirm_by_typing(request)

        approved = Confirm.ask("[bold]Go ahead?[/bold]", default=False)
        if not approved:
            return ApprovalDecision.denied("You declined this action.")

        remember = False
        if request.risk_level == RiskLevel.MODERATE:
            remember = Confirm.ask(
                "[dim]Stop asking for this tool and folder in future?[/dim]", default=False
            )
        return ApprovalDecision(
            approved=True,
            status=ApprovalStatus.USER_APPROVED,
            reason="You approved this action.",
            remember_as_rule=remember,
        )

    def _confirm_by_typing(self, request: ApprovalRequest) -> ApprovalDecision:
        """Require a typed word for high-risk and external actions."""
        word = "DELETE" if request.risk_level == RiskLevel.HIGH else "SEND"
        console.print(
            f"[bold red]This is a level {int(request.risk_level)} action "
            f"({request.risk_level.label}).[/bold red] "
            f"Type [bold]{word}[/bold] to confirm, or anything else to cancel."
        )
        typed = Prompt.ask("Confirm", default="")
        if typed.strip() != word:
            return ApprovalDecision.denied("Not confirmed — nothing was done.")
        return ApprovalDecision.by_user(f"You confirmed by typing {word}.")

    def _render(self, request: ApprovalRequest) -> None:
        body = Table.grid(padding=(0, 2))
        body.add_column(style="dim", justify="right")
        body.add_column()

        body.add_row("Action", request.description)
        body.add_row("Tool", f"{request.tool}.{request.operation}")
        body.add_row("Risk", risk_text(request.risk_level))

        if request.affected_resources:
            resources = "\n".join(request.affected_resources[:8])
            if len(request.affected_resources) > 8:
                resources += f"\n… and {len(request.affected_resources) - 8} more"
            body.add_row("Affects", resources)

        if request.concerns:
            body.add_row(
                "Concerns",
                Text("\n".join(f"• {concern}" for concern in request.concerns), style="yellow"),
            )

        body.add_row(
            "Reversible",
            Text("yes — " + (request.rollback_hint or "recorded in the rollback journal"), style="green")
            if request.reversible
            else Text(request.rollback_hint or "no, this cannot be undone", style="red"),
        )

        console.print(
            Panel(
                body,
                title="Approval needed",
                border_style="red" if request.risk_level >= RiskLevel.HIGH else "yellow",
            )
        )
