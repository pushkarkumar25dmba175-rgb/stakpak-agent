"""The optional local dashboard."""

__all__ = ["build_app", "serve"]


def __getattr__(name: str):  # pragma: no cover - thin lazy re-export
    # Imported lazily so the package works without FastAPI installed.
    if name in __all__:
        from personalos.dashboard import app as module

        return getattr(module, name)
    raise AttributeError(name)
