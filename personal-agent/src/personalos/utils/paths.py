"""Path handling with an explicit allow-list.

The agent is only ever permitted to touch paths inside directories the user has
listed as workspace roots. ``PathResolver`` is the single choke point that turns
a user-supplied string into an absolute path *and* proves it is inside an
allowed root. Tools must never call ``Path(...).resolve()`` directly.
"""

from __future__ import annotations

import os
from pathlib import Path

from personalos.errors import SandboxViolationError


def expand_path(value: str | os.PathLike[str]) -> Path:
    """Expand ``~`` and environment variables, then make the path absolute."""
    text = os.path.expandvars(str(value))
    return Path(text).expanduser().absolute()


def is_within(path: Path, root: Path) -> bool:
    """Return True when ``path`` is ``root`` or lives underneath it."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def safe_relative(path: Path, root: Path) -> str:
    """Render ``path`` relative to ``root`` when possible, else absolute."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def unique_destination(destination: Path) -> Path:
    """Return a non-existing path near ``destination`` by appending a counter.

    Used by reversible operations: the agent never silently overwrites a file
    that is already there, it parks the new one beside it.
    """
    if not destination.exists():
        return destination
    stem, suffix, parent = destination.stem, destination.suffix, destination.parent
    for index in range(1, 1000):
        candidate = parent / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise SandboxViolationError(
        f"Could not find a free filename near {destination}",
        remediation="Clean up the numbered duplicates in that directory.",
    )


class PathResolver:
    """Resolves paths and enforces the workspace allow-list.

    ``allowed_roots`` are the directories the user opted into. ``denied_paths``
    are carved back out of them (e.g. ``~/.ssh`` inside an allowed home
    directory) and always win over an allow.
    """

    def __init__(
        self,
        allowed_roots: list[Path] | list[str],
        denied_paths: list[Path] | list[str] | None = None,
        *,
        follow_symlinks: bool = False,
    ) -> None:
        self.allowed_roots = [expand_path(root).resolve() for root in allowed_roots]
        self.denied_paths = [expand_path(p).resolve() for p in (denied_paths or [])]
        self.follow_symlinks = follow_symlinks

    def _resolve(self, value: str | os.PathLike[str]) -> Path:
        candidate = expand_path(value)
        # ``strict=False`` lets us validate a path that does not exist yet, which
        # is exactly the case for "write this new file here".
        resolved = candidate.resolve(strict=False)
        if not self.follow_symlinks and candidate.is_symlink():
            raise SandboxViolationError(
                f"Refusing to follow the symlink {candidate}",
                remediation="Pass the real path, or enable security.follow_symlinks.",
            )
        return resolved

    def is_allowed(self, value: str | os.PathLike[str]) -> bool:
        """Return whether ``value`` may be touched, without raising."""
        try:
            self.resolve(value)
        except SandboxViolationError:
            return False
        return True

    def resolve(self, value: str | os.PathLike[str]) -> Path:
        """Resolve ``value`` and assert it is inside an allowed root.

        Raises:
            SandboxViolationError: if the path escapes every allowed root or
                falls inside a denied path.
        """
        resolved = self._resolve(value)

        for denied in self.denied_paths:
            if resolved == denied or is_within(resolved, denied):
                raise SandboxViolationError(
                    f"{resolved} is on the deny list and is never accessible to the agent.",
                    remediation="Remove it from security.denied_paths if this is intentional.",
                )

        if not self.allowed_roots:
            raise SandboxViolationError(
                "No workspace roots are configured, so no path is accessible.",
                remediation="Add directories under workspace.allowed_roots in config.yaml.",
            )

        for root in self.allowed_roots:
            if resolved == root or is_within(resolved, root):
                return resolved

        roots = ", ".join(str(r) for r in self.allowed_roots)
        raise SandboxViolationError(
            f"{resolved} is outside every allowed workspace root.",
            remediation=f"Allowed roots are: {roots}. Add one with `agent permissions allow <path>`.",
        )

    def describe(self) -> dict[str, list[str]]:
        """Return a serialisable view of the allow-list, for `agent permissions`."""
        return {
            "allowed_roots": [str(p) for p in self.allowed_roots],
            "denied_paths": [str(p) for p in self.denied_paths],
        }
