"""PersonalOS Agent — a local-first, auditable personal AI coworker.

The package is organised in layers that only depend downwards:

    interface / dashboard      user-facing entry points
    agent                      orchestrator, planner, executor, verification
    skills / learning          reusable workflows and how they are discovered
    tools                      the things the agent can actually do
    security / audit           risk classification, approval, logging, rollback
    memory / projects          what the agent knows and remembers
    database / llm / settings  infrastructure
"""

from personalos.version import __version__

__all__ = ["__version__"]
