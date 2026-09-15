"""Text helpers used by memory ranking, pattern detection and prompts."""

from __future__ import annotations

import difflib
import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")
_NON_SLUG = re.compile(r"[^a-z0-9]+")


def normalise_whitespace(text: str) -> str:
    """Collapse runs of whitespace so comparisons are not formatting-sensitive."""
    return _WHITESPACE.sub(" ", text).strip()


def slugify(text: str, *, separator: str = "_") -> str:
    """Turn arbitrary text into a filesystem- and identifier-safe slug."""
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = decomposed.encode("ascii", "ignore").decode("ascii").lower()
    slug = _NON_SLUG.sub(separator, ascii_text).strip(separator)
    return slug or "unnamed"


def truncate(text: str, limit: int, *, suffix: str = "…") -> str:
    """Shorten ``text`` to ``limit`` characters, marking that it was cut."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))] + suffix


def token_estimate(text: str) -> int:
    """Cheap token estimate (~4 characters per token) for context budgeting.

    This deliberately avoids a tokenizer dependency: budgets are enforced with
    a safety margin, so an approximation within ~20% is good enough and keeps
    the agent usable with any provider.
    """
    return max(1, (len(text) + 3) // 4)


def sequence_similarity(left: str, right: str) -> float:
    """Return a 0..1 similarity ratio between two strings."""
    if not left and not right:
        return 1.0
    return difflib.SequenceMatcher(
        None, normalise_whitespace(left).lower(), normalise_whitespace(right).lower()
    ).ratio()


def keyword_set(text: str, *, minimum_length: int = 3) -> set[str]:
    """Extract lowercase keywords, dropping very short and very common words."""
    stop = {
        "the", "and", "for", "with", "from", "into", "that", "this", "then",
        "are", "was", "were", "you", "your", "all", "any", "can", "will",
        "please", "make", "have", "has", "not", "但", "them", "they",
    }
    # Split on anything non-alphanumeric so that `research_folder` and
    # `~/Documents/threat-research` both yield the word "research". Keeping
    # paths whole would mean a query never matches a stored path.
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) >= minimum_length and w not in stop}
