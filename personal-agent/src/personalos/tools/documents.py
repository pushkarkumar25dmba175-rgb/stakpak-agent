"""Document tool: gets text and metadata out of files so they can be summarised.

Extraction is where untrusted content enters the agent, so everything this tool
returns is wrapped by :func:`wrap_untrusted` and scanned for text that is
trying to look like instructions. Downstream, the reasoning layer inserts it
into the prompt inside an ``<untrusted_content>`` fence.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from personalos.security.injection import wrap_untrusted
from personalos.security.risk import RiskLevel
from personalos.tools.base import OperationSpec, PreparedCall, Tool, ToolContext, ToolResult
from personalos.utils.textutil import truncate

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".log", ".yaml", ".yml", ".ini", ".cfg", ".toml"}


class ExtractInput(BaseModel):
    path: str
    max_characters: int = Field(default=20000, ge=100, le=500000)


class MetadataInput(BaseModel):
    path: str


def _extract_pdf(path: Path, limit: int) -> tuple[str, dict[str, Any]]:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Reading PDFs needs the optional `pypdf` package. "
            "Install it with `pip install 'personalos-agent[docs]'`."
        ) from exc
    reader = PdfReader(str(path))
    chunks: list[str] = []
    total = 0
    for page in reader.pages:
        text = page.extract_text() or ""
        chunks.append(text)
        total += len(text)
        if total >= limit:
            break
    info = reader.metadata or {}
    return "\n\n".join(chunks)[:limit], {
        "pages": len(reader.pages),
        "title": str(info.get("/Title", "") or ""),
        "author": str(info.get("/Author", "") or ""),
    }


def _extract_csv(path: Path, limit: int) -> tuple[str, dict[str, Any]]:
    rows: list[list[str]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for index, row in enumerate(csv.reader(handle)):
            rows.append(row)
            if index > 200:
                break
    rendered = "\n".join(", ".join(row) for row in rows)[:limit]
    return rendered, {"rows_sampled": len(rows), "columns": len(rows[0]) if rows else 0}


class DocumentTool(Tool):
    """Reads documents into plain text, tagged as untrusted."""

    name: ClassVar[str] = "documents"
    description: ClassVar[str] = "Extract text and metadata from documents (txt, md, pdf, csv, json)."

    operations: ClassVar[dict[str, OperationSpec]] = {
        "extract_text": OperationSpec(
            name="extract_text",
            description="Extract plain text from a document for summarising.",
            schema=ExtractInput,
            risk=RiskLevel.READ_ONLY,
            permissions=["filesystem.read"],
        ),
        "metadata": OperationSpec(
            name="metadata",
            description="Report a document's type, size and embedded metadata.",
            schema=MetadataInput,
            risk=RiskLevel.READ_ONLY,
            permissions=["filesystem.read"],
        ),
    }

    def prepare_operation(
        self, spec: OperationSpec, call: PreparedCall, context: ToolContext
    ) -> PreparedCall:
        path = getattr(call.arguments, "path", None)
        if path:
            try:
                call.affected_resources = [str(context.paths.resolve(path))]
            except Exception as exc:  # noqa: BLE001
                call.blocked_reason = str(exc)
        return call

    async def _run(self, call: PreparedCall, context: ToolContext) -> ToolResult:
        if call.operation == "extract_text":
            return self._extract(call.arguments, context)  # type: ignore[arg-type]
        return self._metadata(call.arguments, context)  # type: ignore[arg-type]

    def _extract(self, args: ExtractInput, context: ToolContext) -> ToolResult:
        path = context.paths.resolve(args.path)
        if not path.is_file():
            return ToolResult.failed(f"{path} is not a file.")

        suffix = path.suffix.lower()
        extra: dict[str, Any] = {}
        try:
            if suffix == ".pdf":
                text, extra = _extract_pdf(path, args.max_characters)
            elif suffix == ".csv":
                text, extra = _extract_csv(path, args.max_characters)
            elif suffix == ".json":
                payload = json.loads(path.read_text("utf-8", errors="replace"))
                text = json.dumps(payload, indent=2)[: args.max_characters]
            elif suffix in TEXT_SUFFIXES or suffix == "":
                text = path.read_text("utf-8", errors="replace")[: args.max_characters]
            else:
                return ToolResult.failed(
                    f"I do not know how to read {suffix or 'files without an extension'}. "
                    "Supported: .txt, .md, .pdf, .csv, .json and other plain-text formats."
                )
        except RuntimeError as exc:
            return ToolResult.failed(str(exc))

        untrusted = wrap_untrusted(text, source=f"file:{path}")
        return ToolResult.ok(
            truncate(text, 1500),
            data={
                "path": str(path),
                "characters": len(text),
                "text": text,
                "untrusted_block": untrusted.for_prompt(),
                "injection_flags": untrusted.flags,
                **extra,
            },
        )

    def _metadata(self, args: MetadataInput, context: ToolContext) -> ToolResult:
        path = context.paths.resolve(args.path)
        if not path.exists():
            return ToolResult.failed(f"{path} does not exist.")
        stat = path.stat()
        info: dict[str, Any] = {
            "path": str(path),
            "name": path.name,
            "suffix": path.suffix.lower(),
            "size_bytes": stat.st_size,
            "modified_epoch": stat.st_mtime,
        }
        if path.suffix.lower() == ".pdf":
            try:
                _, extra = _extract_pdf(path, 1)
                info.update(extra)
            except RuntimeError:
                info["pdf_metadata"] = "unavailable (pypdf not installed)"
        return ToolResult.ok(
            f"{info['name']}: {info['size_bytes']} bytes, type {info['suffix'] or 'unknown'}",
            data=info,
        )
