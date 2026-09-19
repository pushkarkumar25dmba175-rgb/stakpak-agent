"""Process isolation for the shell and Python tools.

This is *containment*, not a security boundary against a determined attacker:
a subprocess on the same machine as the user can generally do what the user
can. What the sandbox does guarantee is that ordinary mistakes stay bounded —
a runaway script is killed, a scripted `print` loop cannot fill the disk with
log output, and a child process does not inherit the user's API keys.

Real isolation (containers, seccomp, a VM) plugs in behind the same interface;
:class:`SubprocessSandbox` is the default local implementation.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from personalos.security.secrets import redact, safe_environment
from personalos.utils.textutil import truncate

DEFAULT_OUTPUT_LIMIT = 100_000


@dataclass
class ProcessResult:
    """The outcome of a sandboxed process."""

    command: str
    return_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    truncated: bool = False
    working_directory: str = ""

    @property
    def success(self) -> bool:
        return self.return_code == 0 and not self.timed_out

    def summary(self, limit: int = 2000) -> str:
        """Human-readable, already redacted, safe to log or show a model."""
        body = self.stdout.strip() or self.stderr.strip() or "(no output)"
        head = f"exit {self.return_code}" + (" (timed out)" if self.timed_out else "")
        return f"{head}\n{truncate(body, limit)}"


class Sandbox(Protocol):
    """Runs a command somewhere contained, and reports what happened."""

    async def run(
        self,
        command: str | list[str],
        *,
        cwd: Path,
        timeout: float,
        env: dict[str, str] | None = None,
        stdin: str | None = None,
    ) -> ProcessResult:
        ...


class SubprocessSandbox:
    """Runs commands as child processes with limits applied.

    Args:
        output_limit: Bytes of stdout/stderr kept. Beyond this the result is
            marked ``truncated`` — output is evidence, not a data channel.
        inherit_environment: When False (the default) the child gets an
            environment with every credential-looking variable removed.
        use_process_group: Start the child in its own process group so a
            timeout kills the whole tree, not just the shell that spawned it.
    """

    def __init__(
        self,
        *,
        output_limit: int = DEFAULT_OUTPUT_LIMIT,
        inherit_environment: bool = False,
        use_process_group: bool = True,
    ) -> None:
        self.output_limit = output_limit
        self.inherit_environment = inherit_environment
        self.use_process_group = use_process_group and hasattr(os, "setsid")

    def _environment(self, extra: dict[str, str] | None) -> dict[str, str]:
        base = dict(os.environ) if self.inherit_environment else safe_environment()
        # Keep child processes from spawning their own interactive prompts.
        base.setdefault("PYTHONUNBUFFERED", "1")
        base["DEBIAN_FRONTEND"] = "noninteractive"
        base["GIT_TERMINAL_PROMPT"] = "0"
        if extra:
            base.update(extra)
        return base

    def _clip(self, raw: bytes) -> tuple[str, bool]:
        text = raw.decode("utf-8", errors="replace")
        if len(text) <= self.output_limit:
            return redact(text), False
        return redact(text[: self.output_limit]) + "\n… output truncated …", True

    async def run(
        self,
        command: str | list[str],
        *,
        cwd: Path,
        timeout: float,
        env: dict[str, str] | None = None,
        stdin: str | None = None,
    ) -> ProcessResult:
        """Execute ``command`` in ``cwd``, enforcing ``timeout`` and output limits."""
        cwd.mkdir(parents=True, exist_ok=True)
        environment = self._environment(env)
        loop = asyncio.get_running_loop()
        started = loop.time()
        preexec = os.setsid if self.use_process_group else None
        display = command if isinstance(command, str) else " ".join(command)

        if isinstance(command, str):
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(cwd),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                preexec_fn=preexec,
            )
        else:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                preexec_fn=preexec,
            )

        timed_out = False
        try:
            raw_out, raw_err = await asyncio.wait_for(
                process.communicate(stdin.encode() if stdin is not None else None),
                timeout=timeout,
            )
        except TimeoutError:
            timed_out = True
            self._terminate(process)
            try:
                raw_out, raw_err = await asyncio.wait_for(process.communicate(), timeout=5)
            except (TimeoutError, ProcessLookupError):  # pragma: no cover - race
                raw_out, raw_err = b"", b""

        stdout, clipped_out = self._clip(raw_out or b"")
        stderr, clipped_err = self._clip(raw_err or b"")
        return ProcessResult(
            command=display,
            return_code=process.returncode if process.returncode is not None else -1,
            stdout=stdout,
            stderr=stderr if not timed_out else (stderr + f"\nKilled after {timeout}s.").strip(),
            duration_seconds=loop.time() - started,
            timed_out=timed_out,
            truncated=clipped_out or clipped_err,
            working_directory=str(cwd),
        )

    def _terminate(self, process: asyncio.subprocess.Process) -> None:
        """Kill the child and, where supported, everything it started."""
        try:
            if self.use_process_group and process.pid:
                os.killpg(os.getpgid(process.pid), 9)
            else:
                process.kill()
        except (ProcessLookupError, PermissionError):  # pragma: no cover - race
            pass
