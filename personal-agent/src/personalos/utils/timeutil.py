"""Timezone-aware time helpers.

Every timestamp stored by PersonalOS is UTC and timezone-aware. Naive
datetimes are a recurring source of subtle scheduling bugs, so they are
normalised at the boundary rather than tolerated.
"""

from __future__ import annotations

from datetime import UTC, datetime


def utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def isoformat(value: datetime | None = None) -> str:
    """Render ``value`` (default: now) as an ISO-8601 UTC string."""
    return (value or utcnow()).astimezone(UTC).isoformat()


def parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 string, forcing the result to be UTC-aware."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def ensure_aware(value: datetime) -> datetime:
    """Attach UTC to a naive datetime, otherwise convert to UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
