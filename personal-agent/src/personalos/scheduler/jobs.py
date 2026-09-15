"""Job definitions and the plain-English schedule parser.

``agent schedule "organize downloads" --when "every day at 18:00"`` has to turn
into something a scheduler understands. The parser handles the phrasings people
actually use and refuses anything it does not understand rather than guessing —
a misparsed schedule that runs at the wrong time is worse than an error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from personalos.errors import ConfigurationError

DAY_ALIASES: dict[str, str] = {
    "monday": "mon", "mon": "mon",
    "tuesday": "tue", "tue": "tue", "tues": "tue",
    "wednesday": "wed", "wed": "wed",
    "thursday": "thu", "thu": "thu", "thurs": "thu",
    "friday": "fri", "fri": "fri",
    "saturday": "sat", "sat": "sat",
    "sunday": "sun", "sun": "sun",
}
WEEKDAYS = "mon-fri"
WEEKEND = "sat,sun"

_TIME = re.compile(r"(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?")
_EVERY_N = re.compile(r"every\s+(\d+)\s*(minute|minutes|min|hour|hours|hr|day|days)")


@dataclass
class ScheduleSpec:
    """A parsed schedule, in a form APScheduler can consume directly."""

    kind: Literal["cron", "interval"]
    fields: dict[str, Any] = field(default_factory=dict)
    human: str = ""

    def describe(self) -> str:
        return self.human or f"{self.kind} {self.fields}"


def _parse_time(text: str) -> tuple[int, int] | None:
    match = _TIME.search(text)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = match.group(3)
    if meridiem == "pm" and hour < 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def parse_schedule(text: str) -> ScheduleSpec:
    """Parse a plain-English schedule.

    Understood forms::

        every day at 18:00
        daily at 6pm
        every weekday at 17:30
        every monday at 09:00
        every 30 minutes
        every 2 hours
        hourly

    Raises:
        ConfigurationError: when the phrasing is not recognised.
    """
    lowered = text.strip().lower()
    if not lowered:
        raise ConfigurationError("A schedule cannot be empty.")

    if lowered in {"hourly", "every hour"}:
        return ScheduleSpec("cron", {"minute": 0}, "every hour, on the hour")

    interval = _EVERY_N.search(lowered)
    if interval:
        amount, unit = int(interval.group(1)), interval.group(2)
        if amount < 1:
            raise ConfigurationError("An interval must be at least 1.")
        if unit.startswith(("minute", "min")):
            return ScheduleSpec("interval", {"minutes": amount}, f"every {amount} minutes")
        if unit.startswith(("hour", "hr")):
            return ScheduleSpec("interval", {"hours": amount}, f"every {amount} hours")
        return ScheduleSpec("interval", {"days": amount}, f"every {amount} days")

    when = _parse_time(lowered)
    time_text = f"{when[0]:02d}:{when[1]:02d}" if when else "00:00"
    hour, minute = when or (0, 0)

    if "weekday" in lowered:
        return ScheduleSpec(
            "cron", {"day_of_week": WEEKDAYS, "hour": hour, "minute": minute},
            f"every weekday at {time_text}",
        )
    if "weekend" in lowered:
        return ScheduleSpec(
            "cron", {"day_of_week": WEEKEND, "hour": hour, "minute": minute},
            f"every weekend day at {time_text}",
        )
    for name, short in DAY_ALIASES.items():
        if re.search(rf"\b{name}s?\b", lowered):
            return ScheduleSpec(
                "cron", {"day_of_week": short, "hour": hour, "minute": minute},
                f"every {name} at {time_text}",
            )
    if "daily" in lowered or "every day" in lowered or lowered.startswith("at "):
        return ScheduleSpec(
            "cron", {"hour": hour, "minute": minute}, f"every day at {time_text}"
        )

    raise ConfigurationError(
        f"I could not understand the schedule {text!r}.",
        remediation=(
            "Try one of: 'every day at 18:00', 'every weekday at 17:30', "
            "'every monday at 9am', 'every 30 minutes', 'hourly'."
        ),
    )


@dataclass
class JobDefinition:
    """What a scheduled job does when it fires."""

    job_id: str
    name: str
    kind: Literal["ask", "skill"]
    payload: dict[str, Any]
    schedule: ScheduleSpec
    max_risk_level: int = 1
    project: str | None = None
    enabled: bool = True

    def describe(self) -> str:
        what = (
            f"ask: {self.payload.get('request', '')}"
            if self.kind == "ask"
            else f"skill: {self.payload.get('skill', '')}"
        )
        return f"{self.name} — {what} — {self.schedule.describe()}"
