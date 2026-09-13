"""Perplexity connector (spec §5.3): the subject's research-question voice.

Reads the Perplexity account export (`conversations-*.json`). Only the
subject's `query` is emitted -- the `answer` is a model's writing, dropped for
the same reason the claude.ai connector drops the assistant half.

Register note: this source is short-form by nature. Across the real
subject-01 export the median query runs ~84 characters, so it reads more like
iMessage than like mail -- how the subject *asks*, not how he explains. That is
a genuine register and worth keeping, but it is not interchangeable with prose
sources and should not be weighted as if it were.

Paste shape differs from claude_code, and deliberately does NOT get its own
heuristic: the threshold and formula are copied from that connector verbatim,
because two definitions of "pasted" would make the corpus's composition depend
on which tool the subject happened to have open. What differs is only what
trips it -- fenced code blocks are rare here (8 of 3,145 queries), while raw
unfenced markdown documents are the dominant paste, and those are caught by the
absolute `prose_chars` threshold rather than by fence-stripping. The measured
consequence of keeping the shared formula is recorded in the task report: 22
entries carrying 376K characters are flagged, and a 61-entry band between 1K
and 8K characters is a genuine mixture of long typed questions and short
pastes that this filter does not separate.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from persona_twin.connectors.base import SubjectContext, scan_snapshot_exports
from persona_twin.schema import RawEnvelope

_FENCE = re.compile(r"```.*?```", re.S)

# Identical to the claude_code / claude_ai threshold, by design -- see module
# docstring. Changing it here without changing it there splits the corpus.
_PASTE_CHAR_THRESHOLD = 8000


def _load(path: Path) -> list[dict]:
    data = json.loads(path.read_bytes())
    if not isinstance(data, dict) or "conversations" not in data:
        raise ValueError(
            f"perplexity connector: {path} has no 'conversations' key")
    convs = data["conversations"]
    if not isinstance(convs, list):
        raise ValueError(
            f"perplexity connector: {path} 'conversations' is not a list")
    return convs


def _ts(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


class PerplexityConnector:
    name = "perplexity"

    def __init__(self, export_paths: list[Path]) -> None:
        self.paths = [Path(p) for p in export_paths]

    def fetch(self, ctx: SubjectContext,
              cursor: str | None) -> Iterator[tuple[RawEnvelope, str]]:
        """Whole-file snapshot cursor; see base.scan_snapshot_exports.

        Safe here because `source_id` is the server-assigned entry uuid,
        which is stable across re-exports, so a re-scan writes zero new rows.
        """
        yield from scan_snapshot_exports(
            self.name, self.paths, cursor,
            lambda path: self._envelopes(ctx, path))

    def _envelopes(self, ctx: SubjectContext, path: Path) -> Iterator[RawEnvelope]:
        key = str(path)
        for conv in _load(path):
            ctx_uuid = str(conv.get("context_uuid") or "")
            for idx, entry in enumerate(conv.get("entries") or []):
                text = str(entry.get("query") or "").strip()
                if not text:
                    continue
                prose = _FENCE.sub("", text).strip()
                raw_chars, prose_chars = len(text), len(prose)
                # `entry_uuid` is server-assigned and unique across all 3,145
                # entries of the real export; the hash is only a fallback for
                # an export shaped unexpectedly enough to omit it.
                source_id = str(entry.get("entry_uuid") or "").strip() or hashlib.sha256(
                    f"{key}:{ctx_uuid}:{idx}:{text}".encode("utf-8")
                ).hexdigest()[:24]
                yield RawEnvelope(
                    subject_id=ctx.config.subject_id, source=self.name,
                    source_id=source_id, ts=_ts(entry.get("created_at")),
                    payload={
                        "text": text,
                        "context_uuid": ctx_uuid,
                        "raw_chars": raw_chars,
                        "prose_chars": prose_chars,
                        "looks_pasted": (
                            prose_chars > _PASTE_CHAR_THRESHOLD
                            or (raw_chars - prose_chars) > raw_chars / 2),
                        # A BLOCKED or FAILED query is still something the
                        # subject typed, so it is kept; the status is recorded
                        # rather than used as a filter.
                        "query_status": str(entry.get("query_status") or ""),
                        "engine_mode": str(entry.get("engine_mode") or ""),
                    },
                    ingested_at=datetime.now(timezone.utc))
