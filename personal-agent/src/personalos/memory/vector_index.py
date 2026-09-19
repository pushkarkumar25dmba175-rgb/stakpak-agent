"""Vector search abstraction.

PersonalOS ships with lexical retrieval, which needs no model, no downloads and
no background service — and for a few thousand personal episodes it works well.
The interface below is what a local vector database plugs into when the corpus
outgrows that.

To add one, implement :class:`VectorIndex` and register it in
:func:`build_vector_index`. The rest of the memory layer talks only to this
interface, so nothing else has to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from personalos.errors import ConfigurationError


@dataclass
class VectorHit:
    """One result from a similarity search."""

    identifier: str
    score: float
    payload: dict[str, Any]


@runtime_checkable
class VectorIndex(Protocol):
    """The contract a vector backend has to satisfy."""

    name: str

    def upsert(self, identifier: str, text: str, payload: dict[str, Any]) -> None:
        """Add or replace one document."""
        ...

    def search(self, query: str, *, limit: int = 5) -> list[VectorHit]:
        """Return the nearest documents to ``query``."""
        ...

    def delete(self, identifier: str) -> bool:
        """Remove a document. Returns whether it was there."""
        ...


class NullVectorIndex:
    """The default: no vector search, and honest about it.

    Returning an empty result rather than raising means retrieval degrades to
    lexical-only instead of failing, which is the behaviour you want when a
    backend is unconfigured.
    """

    name = "none"

    def upsert(self, identifier: str, text: str, payload: dict[str, Any]) -> None:
        return None

    def search(self, query: str, *, limit: int = 5) -> list[VectorHit]:
        return []

    def delete(self, identifier: str) -> bool:
        return False


def build_vector_index(backend: str, **options: Any) -> VectorIndex:
    """Construct the configured vector backend.

    Raises:
        ConfigurationError: for a backend that is named but not implemented,
            so a typo in ``memory.vector_backend`` is reported rather than
            silently ignored.
    """
    normalised = (backend or "none").strip().lower()
    if normalised in {"none", "", "lexical"}:
        return NullVectorIndex()
    if normalised in {"chroma", "qdrant", "faiss"}:
        raise ConfigurationError(
            f"The {normalised!r} vector backend is not wired up in this build.",
            remediation=(
                "Implement the VectorIndex protocol in personalos/memory/vector_index.py "
                "and register it in build_vector_index(), or set memory.vector_backend: none."
            ),
        )
    raise ConfigurationError(
        f"Unknown vector backend {backend!r}.",
        remediation="Valid values today: none. Planned: chroma, qdrant, faiss.",
    )
