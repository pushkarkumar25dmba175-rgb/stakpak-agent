"""The agent's lifecycle states, and a small observable holding the current one."""

from __future__ import annotations

import threading
from collections.abc import Callable
from enum import StrEnum

from personalos.utils.timeutil import isoformat


class AgentState(StrEnum):
    """What the agent is doing right now. Shown by `agent status` and the dashboard."""

    OFFLINE = "OFFLINE"
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    LEARNING = "LEARNING"
    ERROR = "ERROR"
    PAUSED = "PAUSED"

    @property
    def is_busy(self) -> bool:
        return self in {
            AgentState.PLANNING,
            AgentState.EXECUTING,
            AgentState.VERIFYING,
            AgentState.LEARNING,
        }

    @property
    def description(self) -> str:
        return {
            AgentState.OFFLINE: "Not running.",
            AgentState.IDLE: "Ready and waiting for a task.",
            AgentState.PLANNING: "Working out the steps for a request.",
            AgentState.WAITING_FOR_APPROVAL: "Stopped, waiting for you to approve an action.",
            AgentState.EXECUTING: "Carrying out an approved plan.",
            AgentState.VERIFYING: "Checking that the steps actually did what they claimed.",
            AgentState.LEARNING: "Recording what happened and updating memory.",
            AgentState.ERROR: "Stopped after a failure it could not recover from.",
            AgentState.PAUSED: "Paused by you. No task will start until you resume.",
        }[self]


StateListener = Callable[["AgentState", "AgentState"], None]


class StateMachine:
    """Holds the current state and notifies listeners on every transition.

    Thread-safe because the scheduler and folder watcher change state from
    worker threads while the CLI reads it.
    """

    def __init__(self, initial: AgentState = AgentState.IDLE) -> None:
        self._state = initial
        self._lock = threading.RLock()
        self._listeners: list[StateListener] = []
        self.history: list[tuple[str, AgentState]] = [(isoformat(), initial)]

    @property
    def state(self) -> AgentState:
        with self._lock:
            return self._state

    def subscribe(self, listener: StateListener) -> None:
        self._listeners.append(listener)

    def transition(self, new_state: AgentState) -> AgentState:
        """Move to ``new_state``, returning the state that was left behind."""
        with self._lock:
            previous, self._state = self._state, new_state
            self.history.append((isoformat(), new_state))
            self.history = self.history[-200:]
        for listener in list(self._listeners):
            try:
                listener(previous, new_state)
            except Exception:  # noqa: BLE001 - a bad listener must not break the agent
                continue
        return previous

    def paused(self) -> bool:
        return self.state is AgentState.PAUSED

    def pause(self) -> None:
        self.transition(AgentState.PAUSED)

    def resume(self) -> None:
        if self.state is AgentState.PAUSED:
            self.transition(AgentState.IDLE)
