"""Filesystem tool: read, search, organise and edit files inside the workspace.

Every path argument goes through ``context.paths``, so an operation can only
ever touch a directory the user opted into. Two behaviours are worth calling
out because they are policy, not convenience:

* **Writes never silently clobber.** Overwriting an existing file is level 2,
  and the previous contents are copied into the backup store first, so the
  change can be undone.
* **Deleting means moving to the trash.** With
  ``security.prefer_trash_over_delete`` (the default) the file goes to
  ``~/.personalos/agent_trash/`` and stays recoverable. Permanent deletion is
  still level 3 and still asks.
"""

from __future__ import annotations

import fnmatch
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from personalos.security.risk import RiskLevel
from personalos.tools.base import (
    OperationSpec,
    PreparedCall,
    Tool,
    ToolContext,
    ToolResult,
)
from personalos.utils.paths import unique_destination
from personalos.utils.textutil import truncate
from personalos.utils.timeutil import utcnow


# ---- input schemas ---------------------------------------------------------
class ListInput(BaseModel):
    path: str = Field(description="Directory to list.")
    pattern: str | None = Field(default=None, description="Optional glob, e.g. '*.pdf'.")
    recursive: bool = Field(default=False)
    limit: int = Field(default=200, ge=1, le=5000)
    include_hidden: bool = Field(default=False)


class ReadInput(BaseModel):
    path: str
    max_bytes: int | None = Field(default=None, ge=1)
    encoding: str = "utf-8"


class SearchInput(BaseModel):
    path: str = Field(description="Directory to search under.")
    query: str = Field(description="Text to look for inside files.")
    pattern: str = Field(default="*", description="Filename glob to limit the search.")
    max_results: int = Field(default=50, ge=1, le=1000)
    case_sensitive: bool = False


class StatInput(BaseModel):
    path: str


class WriteInput(BaseModel):
    path: str
    content: str
    encoding: str = "utf-8"
    create_parents: bool = True


class AppendInput(BaseModel):
    path: str
    content: str
    encoding: str = "utf-8"


class MkdirInput(BaseModel):
    path: str
    parents: bool = True


class CopyInput(BaseModel):
    source: str
    destination: str
    overwrite: bool = False


class MoveInput(BaseModel):
    source: str
    destination: str
    overwrite: bool = False


class RenameInput(BaseModel):
    path: str
    new_name: str = Field(description="New file name only, not a full path.")


class DeleteInput(BaseModel):
    path: str
    permanent: bool = Field(
        default=False,
        description="When false (the default) the file is moved to the agent trash.",
    )


def _describe(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "name": path.name,
        "is_dir": path.is_dir(),
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
    }


class FilesystemTool(Tool):
    """Local file and folder operations."""

    name: ClassVar[str] = "filesystem"
    description: ClassVar[str] = "Read, search, create, move, rename and organise local files."

    operations: ClassVar[dict[str, OperationSpec]] = {
        "list": OperationSpec(
            name="list",
            description="List the entries in a directory, optionally filtered by a glob.",
            schema=ListInput,
            risk=RiskLevel.READ_ONLY,
            permissions=["filesystem.read"],
        ),
        "read": OperationSpec(
            name="read",
            description="Read a text file's contents.",
            schema=ReadInput,
            risk=RiskLevel.READ_ONLY,
            permissions=["filesystem.read"],
        ),
        "search": OperationSpec(
            name="search",
            description="Search file contents under a directory for a string.",
            schema=SearchInput,
            risk=RiskLevel.READ_ONLY,
            permissions=["filesystem.read"],
        ),
        "stat": OperationSpec(
            name="stat",
            description="Report size, type and modification time for a path.",
            schema=StatInput,
            risk=RiskLevel.READ_ONLY,
            permissions=["filesystem.read"],
        ),
        "write": OperationSpec(
            name="write",
            description="Write a text file, backing up any existing contents first.",
            schema=WriteInput,
            risk=RiskLevel.LOW,
            permissions=["filesystem.write"],
        ),
        "append": OperationSpec(
            name="append",
            description="Append text to a file, creating it when absent.",
            schema=AppendInput,
            risk=RiskLevel.LOW,
            permissions=["filesystem.write"],
        ),
        "mkdir": OperationSpec(
            name="mkdir",
            description="Create a directory.",
            schema=MkdirInput,
            risk=RiskLevel.LOW,
            permissions=["filesystem.write"],
        ),
        "copy": OperationSpec(
            name="copy",
            description="Copy a file to a new location.",
            schema=CopyInput,
            risk=RiskLevel.LOW,
            permissions=["filesystem.write"],
        ),
        "move": OperationSpec(
            name="move",
            description="Move a file or directory to a new location.",
            schema=MoveInput,
            risk=RiskLevel.MODERATE,
            permissions=["filesystem.write"],
        ),
        "rename": OperationSpec(
            name="rename",
            description="Rename a file in place.",
            schema=RenameInput,
            risk=RiskLevel.MODERATE,
            permissions=["filesystem.write"],
        ),
        "delete": OperationSpec(
            name="delete",
            description="Move a file to the agent trash, or delete it permanently.",
            schema=DeleteInput,
            risk=RiskLevel.HIGH,
            permissions=["filesystem.delete"],
            destructive=True,
        ),
    }

    # ---- risk classification per call -------------------------------------
    def prepare_operation(
        self, spec: OperationSpec, call: PreparedCall, context: ToolContext
    ) -> PreparedCall:
        args = call.arguments
        resolver = context.paths

        def safe(value: str) -> str:
            try:
                return str(resolver.resolve(value))
            except Exception as exc:  # noqa: BLE001 - reported as a block
                call.blocked_reason = str(exc)
                return value

        if isinstance(args, (ListInput, ReadInput, SearchInput, StatInput, MkdirInput)):
            call.affected_resources = [safe(args.path)]

        elif isinstance(args, (WriteInput, AppendInput)):
            target = Path(safe(args.path))
            call.affected_resources = [str(target)]
            if target.exists():
                # Overwriting existing content is a level-2 change, not a level-1 one.
                call.risk_level = RiskLevel.MODERATE
                call.description = f"Overwrite the existing file {target.name}"
                call.concerns.append(f"{target} already exists and will be replaced")
                call.rollback_hint = "The previous contents are backed up and restorable."
            else:
                call.description = f"Create a new file at {target}"
            if len(args.content.encode(getattr(args, "encoding", "utf-8"), errors="replace")) > (
                context.settings.workspace.max_write_bytes
            ):
                call.blocked_reason = (
                    f"The content is larger than the configured write limit of "
                    f"{context.settings.workspace.max_write_bytes} bytes."
                )

        elif isinstance(args, (CopyInput, MoveInput)):
            source, destination = Path(safe(args.source)), Path(safe(args.destination))
            call.affected_resources = [str(source), str(destination)]
            if destination.exists() and args.overwrite:
                call.risk_level = RiskLevel.MODERATE
                call.concerns.append(f"{destination} already exists and would be overwritten")
            call.description = (
                f"{'Copy' if isinstance(args, CopyInput) else 'Move'} {source.name} → {destination}"
            )
            call.rollback_hint = (
                "Undoable: the inverse move is recorded."
                if isinstance(args, MoveInput)
                else "Undoable: the copy can be removed."
            )

        elif isinstance(args, RenameInput):
            source = Path(safe(args.path))
            if os.sep in args.new_name or (os.altsep and os.altsep in args.new_name):
                call.blocked_reason = (
                    "rename takes a file name, not a path. Use move to relocate a file."
                )
            call.affected_resources = [str(source), str(source.parent / args.new_name)]
            call.description = f"Rename {source.name} → {args.new_name}"
            call.rollback_hint = "Undoable: the inverse rename is recorded."

        elif isinstance(args, DeleteInput):
            target = Path(safe(args.path))
            call.affected_resources = [str(target)]
            if args.permanent or not context.settings.security.prefer_trash_over_delete:
                call.reversible = False
                call.risk_level = RiskLevel.HIGH
                call.concerns.append("this removes the file permanently and cannot be undone")
                call.description = f"Permanently delete {target}"
                call.rollback_hint = None
            else:
                call.description = f"Move {target.name} to the agent trash"
                call.rollback_hint = (
                    f"Recoverable from {context.settings.trash_dir} until you empty it."
                )
                call.concerns.append("prefer this over permanent deletion; the file stays recoverable")

        return call

    # ---- execution ---------------------------------------------------------
    async def _run(self, call: PreparedCall, context: ToolContext) -> ToolResult:
        handler = getattr(self, f"_op_{call.operation}")
        return handler(call.arguments, context)

    # -- read-only ----------------------------------------------------------
    def _op_list(self, args: ListInput, context: ToolContext) -> ToolResult:
        root = context.paths.resolve(args.path)
        if not root.is_dir():
            return ToolResult.failed(f"{root} is not a directory.")
        walker = root.rglob("*") if args.recursive else root.iterdir()
        entries: list[dict[str, Any]] = []
        for entry in walker:
            if not args.include_hidden and entry.name.startswith("."):
                continue
            if args.pattern and not fnmatch.fnmatch(entry.name, args.pattern):
                continue
            try:
                entries.append(_describe(entry))
            except OSError:
                continue
            if len(entries) >= args.limit:
                break
        entries.sort(key=lambda item: (not item["is_dir"], item["name"].lower()))
        lines = [
            f"{'d' if item['is_dir'] else '-'} {item['name']}  {item['size_bytes']}B  {item['modified'][:16]}"
            for item in entries
        ]
        return ToolResult.ok(
            f"{len(entries)} entries in {root}:\n" + "\n".join(lines[:200]),
            data={"path": str(root), "count": len(entries), "entries": entries},
        )

    def _op_read(self, args: ReadInput, context: ToolContext) -> ToolResult:
        path = context.paths.resolve(args.path)
        if not path.is_file():
            return ToolResult.failed(f"{path} is not a file.")
        limit = args.max_bytes or context.settings.workspace.max_read_bytes
        size = path.stat().st_size
        raw = path.read_bytes()[:limit]
        text = raw.decode(args.encoding, errors="replace")
        truncated = size > limit
        return ToolResult.ok(
            text,
            data={
                "path": str(path),
                "size_bytes": size,
                "truncated": truncated,
                "content": text,
            },
        )

    def _op_search(self, args: SearchInput, context: ToolContext) -> ToolResult:
        root = context.paths.resolve(args.path)
        needle = args.query if args.case_sensitive else args.query.lower()
        matches: list[dict[str, Any]] = []
        for candidate in root.rglob(args.pattern):
            if not candidate.is_file():
                continue
            try:
                if candidate.stat().st_size > context.settings.workspace.max_read_bytes:
                    continue
                content = candidate.read_text("utf-8", errors="ignore")
            except OSError:
                continue
            haystack = content if args.case_sensitive else content.lower()
            if needle not in haystack:
                continue
            for number, line in enumerate(content.splitlines(), start=1):
                probe = line if args.case_sensitive else line.lower()
                if needle in probe:
                    matches.append(
                        {"path": str(candidate), "line": number, "text": truncate(line.strip(), 200)}
                    )
                    break
            if len(matches) >= args.max_results:
                break
        lines = [f"{m['path']}:{m['line']}: {m['text']}" for m in matches]
        return ToolResult.ok(
            f"{len(matches)} matches for {args.query!r}:\n" + "\n".join(lines),
            data={"matches": matches, "count": len(matches)},
        )

    def _op_stat(self, args: StatInput, context: ToolContext) -> ToolResult:
        path = context.paths.resolve(args.path)
        if not path.exists():
            return ToolResult.failed(f"{path} does not exist.")
        info = _describe(path)
        return ToolResult.ok(
            f"{info['path']}: {'directory' if info['is_dir'] else 'file'}, "
            f"{info['size_bytes']} bytes, modified {info['modified']}",
            data=info,
        )

    # -- writes -------------------------------------------------------------
    def _op_write(self, args: WriteInput, context: ToolContext) -> ToolResult:
        path = context.paths.resolve(args.path)
        if context.dry_run:
            return ToolResult.ok(f"[dry run] Would write {len(args.content)} characters to {path}.")
        if args.create_parents:
            path.parent.mkdir(parents=True, exist_ok=True)

        rollback_ids: list[int] = []
        if path.exists():
            backup = context.journal.backup_file(path, context.task_id)
            rollback_ids.append(
                context.journal.record_modify(
                    context.task_id, path, backup, step_id=context.step_id
                )
            )
        else:
            rollback_ids.append(
                context.journal.record_create(context.task_id, path, step_id=context.step_id)
            )

        path.write_text(args.content, encoding=args.encoding)
        return ToolResult.ok(
            f"Wrote {len(args.content)} characters to {path}.",
            data={"path": str(path), "bytes": path.stat().st_size},
            rollback_ids=rollback_ids,
            artifacts=[str(path)],
        )

    def _op_append(self, args: AppendInput, context: ToolContext) -> ToolResult:
        path = context.paths.resolve(args.path)
        if context.dry_run:
            return ToolResult.ok(f"[dry run] Would append {len(args.content)} characters to {path}.")
        path.parent.mkdir(parents=True, exist_ok=True)
        rollback_ids: list[int] = []
        if path.exists():
            backup = context.journal.backup_file(path, context.task_id)
            rollback_ids.append(
                context.journal.record_modify(context.task_id, path, backup, step_id=context.step_id)
            )
        else:
            rollback_ids.append(
                context.journal.record_create(context.task_id, path, step_id=context.step_id)
            )
        with path.open("a", encoding=args.encoding) as handle:
            handle.write(args.content)
        return ToolResult.ok(
            f"Appended {len(args.content)} characters to {path}.",
            data={"path": str(path)},
            rollback_ids=rollback_ids,
            artifacts=[str(path)],
        )

    def _op_mkdir(self, args: MkdirInput, context: ToolContext) -> ToolResult:
        path = context.paths.resolve(args.path)
        if context.dry_run:
            return ToolResult.ok(f"[dry run] Would create the directory {path}.")
        existed = path.exists()
        path.mkdir(parents=args.parents, exist_ok=True)
        return ToolResult.ok(
            f"{'Directory already existed' if existed else 'Created directory'}: {path}.",
            data={"path": str(path), "created": not existed},
            artifacts=[str(path)],
        )

    def _op_copy(self, args: CopyInput, context: ToolContext) -> ToolResult:
        source = context.paths.resolve(args.source)
        destination = context.paths.resolve(args.destination)
        if not source.exists():
            return ToolResult.failed(f"{source} does not exist.")
        if destination.is_dir():
            destination = destination / source.name
        if destination.exists() and not args.overwrite:
            destination = unique_destination(destination)
        if context.dry_run:
            return ToolResult.ok(f"[dry run] Would copy {source} to {destination}.")

        destination.parent.mkdir(parents=True, exist_ok=True)
        rollback_ids: list[int] = []
        if destination.exists():
            backup = context.journal.backup_file(destination, context.task_id)
            rollback_ids.append(
                context.journal.record_modify(
                    context.task_id, destination, backup, step_id=context.step_id
                )
            )
        else:
            rollback_ids.append(
                context.journal.record_create(context.task_id, destination, step_id=context.step_id)
            )

        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=args.overwrite)
        else:
            shutil.copy2(source, destination)
        return ToolResult.ok(
            f"Copied {source} → {destination}.",
            data={"source": str(source), "destination": str(destination)},
            rollback_ids=rollback_ids,
            artifacts=[str(destination)],
        )

    def _op_move(self, args: MoveInput, context: ToolContext) -> ToolResult:
        source = context.paths.resolve(args.source)
        destination = context.paths.resolve(args.destination)
        if not source.exists():
            return ToolResult.failed(f"{source} does not exist.")
        if destination.is_dir():
            destination = destination / source.name
        if destination.exists() and not args.overwrite:
            destination = unique_destination(destination)
        if context.dry_run:
            return ToolResult.ok(f"[dry run] Would move {source} to {destination}.")

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        rollback_id = context.journal.record_move(
            context.task_id, source, destination, step_id=context.step_id
        )
        return ToolResult.ok(
            f"Moved {source} → {destination}.",
            data={"source": str(source), "destination": str(destination)},
            rollback_ids=[rollback_id],
            artifacts=[str(destination)],
        )

    def _op_rename(self, args: RenameInput, context: ToolContext) -> ToolResult:
        source = context.paths.resolve(args.path)
        if not source.exists():
            return ToolResult.failed(f"{source} does not exist.")
        destination = context.paths.resolve(source.parent / args.new_name)
        if destination.exists():
            destination = unique_destination(destination)
        if context.dry_run:
            return ToolResult.ok(f"[dry run] Would rename {source.name} to {destination.name}.")

        source.rename(destination)
        rollback_id = context.journal.record_move(
            context.task_id, source, destination, step_id=context.step_id
        )
        return ToolResult.ok(
            f"Renamed {source.name} → {destination.name}.",
            data={"source": str(source), "destination": str(destination)},
            rollback_ids=[rollback_id],
            artifacts=[str(destination)],
        )

    def _op_delete(self, args: DeleteInput, context: ToolContext) -> ToolResult:
        target = context.paths.resolve(args.path)
        if not target.exists():
            return ToolResult.failed(f"{target} does not exist.")

        permanent = args.permanent or not context.settings.security.prefer_trash_over_delete
        if context.dry_run:
            verb = "permanently delete" if permanent else "move to the trash"
            return ToolResult.ok(f"[dry run] Would {verb}: {target}.")

        if permanent:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            context.journal.record_irreversible(
                context.task_id,
                f"permanently deleted {target}",
                step_id=context.step_id,
            )
            return ToolResult.ok(
                f"Permanently deleted {target}. This cannot be undone.",
                data={"path": str(target), "permanent": True},
            )

        trash_root = context.settings.trash_dir / utcnow().strftime("%Y-%m-%d")
        trash_root.mkdir(parents=True, exist_ok=True)
        destination = unique_destination(trash_root / target.name)
        shutil.move(str(target), str(destination))
        rollback_id = context.journal.record_move(
            context.task_id, target, destination, step_id=context.step_id
        )
        return ToolResult.ok(
            f"Moved {target} to the agent trash at {destination}. "
            "It stays recoverable until you empty the trash.",
            data={"path": str(target), "trash_path": str(destination), "permanent": False},
            rollback_ids=[rollback_id],
        )
