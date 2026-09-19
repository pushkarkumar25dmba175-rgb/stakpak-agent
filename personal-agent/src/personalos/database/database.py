"""Engine and session management for the local SQLite database."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from personalos.database.models import Base


def _sqlite_url(path: Path | str) -> str:
    if str(path) == ":memory:":
        return "sqlite+pysqlite:///:memory:"
    return f"sqlite+pysqlite:///{Path(path).absolute()}"


def _apply_pragmas(dbapi_connection, _record) -> None:  # type: ignore[no-untyped-def]
    """WAL plus enforced foreign keys: safe concurrent reads, real cascades."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


class Database:
    """Owns the engine and hands out sessions.

    A single instance is shared by every subsystem; it is cheap to construct
    and safe to use from several threads because ``sessionmaker`` produces a
    fresh session per unit of work.
    """

    def __init__(self, path: Path | str, *, echo: bool = False) -> None:
        self.path = path
        self.engine: Engine = create_engine(
            _sqlite_url(path),
            echo=echo,
            future=True,
            # ``check_same_thread=False`` is required because APScheduler and the
            # folder watcher run jobs on worker threads.
            connect_args={"check_same_thread": False},
        )
        event.listen(self.engine, "connect", _apply_pragmas)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def create_all(self) -> None:
        """Create any missing tables. Safe to call on every start-up."""
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Yield a session, committing on success and rolling back on error."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        self.engine.dispose()


def build_database(path: Path | str, *, echo: bool = False) -> Database:
    """Construct a :class:`Database` with its schema already created."""
    database = Database(path, echo=echo)
    database.create_all()
    return database
