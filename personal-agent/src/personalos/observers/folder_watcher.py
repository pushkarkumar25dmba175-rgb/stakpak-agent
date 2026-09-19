"""Watching folders you explicitly opted into.

This is the only module that observes anything without being asked, and it is
off by default. What it does when enabled is deliberately narrow:

* it records *that* a file appeared, with its path and size — never its contents;
* it only watches directories listed in ``observation.watched_folders``;
* it honours ``observation.paused`` on every event, so pausing is immediate;
* it never starts a task on its own. Events accumulate, and the user decides
  what to do with them (``agent observe events``, or a scheduled job).

The last point is the important one. A watcher that triggers work is a watcher
that can be made to trigger work by anyone who can drop a file in your
Downloads folder.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select

from personalos.database.database import Database
from personalos.database.models import ObservationEvent
from personalos.settings import Settings
from personalos.utils.paths import expand_path


@dataclass
class WatchEvent:
    """Something that happened in a watched folder."""

    event_type: str
    path: Path
    observer: str = "folder"
    detail: dict[str, Any] | None = None


class FolderWatcher:
    """Records filesystem events in opted-in directories.

    Uses ``watchdog`` when it is installed. Without it, :meth:`poll` does a
    one-shot scan instead, which is enough for scheduled "what is new?" checks
    and keeps the dependency optional.
    """

    def __init__(self, settings: Settings, database: Database) -> None:
        self.settings = settings
        self.database = database
        self._observer: Any | None = None
        self._seen: dict[str, float] = {}

    # ---- gating ------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return (
            self.settings.observation.watch_folders_enabled
            and not self.settings.observation.paused
            and bool(self.settings.observation.watched_folders)
        )

    def folders(self) -> list[Path]:
        return [expand_path(folder) for folder in self.settings.observation.watched_folders]

    def _ignored(self, path: Path) -> bool:
        return any(
            fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(str(path), pattern)
            for pattern in self.settings.observation.ignore_patterns
        )

    # ---- recording ---------------------------------------------------------
    def record(self, event: WatchEvent) -> ObservationEvent | None:
        """Store an event, unless observation is paused or the path is ignored."""
        if self.settings.observation.paused or self._ignored(event.path):
            return None
        detail = dict(event.detail or {})
        try:
            if event.path.is_file():
                detail.setdefault("size_bytes", event.path.stat().st_size)
                detail.setdefault("suffix", event.path.suffix.lower())
        except OSError:
            pass
        with self.database.session() as session:
            row = ObservationEvent(
                observer=event.observer,
                event_type=event.event_type,
                path=str(event.path),
                detail=detail,
            )
            session.add(row)
            session.flush()
            session.refresh(row)
            session.expunge(row)
            return row

    def events(self, *, limit: int = 50, unhandled_only: bool = False) -> list[ObservationEvent]:
        with self.database.session() as session:
            statement = (
                select(ObservationEvent).order_by(ObservationEvent.created_at.desc()).limit(limit)
            )
            if unhandled_only:
                statement = statement.where(ObservationEvent.handled.is_(False))
            rows = list(session.scalars(statement))
            for row in rows:
                session.expunge(row)
            return rows

    def mark_handled(self, event_ids: list[int]) -> int:
        with self.database.session() as session:
            count = 0
            for event_id in event_ids:
                row = session.get(ObservationEvent, event_id)
                if row is not None:
                    row.handled = True
                    count += 1
            return count

    # ---- polling fallback --------------------------------------------------
    def poll(self) -> list[ObservationEvent]:
        """Scan the watched folders once and record anything new.

        Used when ``watchdog`` is unavailable, and by scheduled jobs that want
        a point-in-time answer to "what showed up since last time?".
        """
        if not self.enabled:
            return []
        recorded: list[ObservationEvent] = []
        for folder in self.folders():
            if not folder.is_dir():
                continue
            for path in folder.iterdir():
                if self._ignored(path):
                    continue
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue
                key = str(path)
                if self._seen.get(key) == mtime:
                    continue
                event_type = "created" if key not in self._seen else "modified"
                self._seen[key] = mtime
                row = self.record(WatchEvent(event_type, path))
                if row is not None:
                    recorded.append(row)
        return recorded

    # ---- watchdog ----------------------------------------------------------
    def start(self) -> bool:
        """Begin watching. Returns False when disabled or watchdog is missing."""
        if not self.enabled:
            return False
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            return False
        if self._observer is not None:
            return True

        watcher = self

        class _Handler(FileSystemEventHandler):
            def on_created(self, event: Any) -> None:
                if not event.is_directory:
                    watcher.record(WatchEvent("created", Path(event.src_path)))

            def on_modified(self, event: Any) -> None:
                if not event.is_directory:
                    watcher.record(WatchEvent("modified", Path(event.src_path)))

            def on_moved(self, event: Any) -> None:
                if not event.is_directory:
                    watcher.record(
                        WatchEvent(
                            "moved",
                            Path(event.dest_path),
                            detail={"from": str(event.src_path)},
                        )
                    )

        self._observer = Observer()
        for folder in self.folders():
            if folder.is_dir():
                self._observer.schedule(_Handler(), str(folder), recursive=False)
        self._observer.start()
        return True

    def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
