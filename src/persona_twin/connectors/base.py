from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Protocol, runtime_checkable
from persona_twin.config import SubjectConfig
from persona_twin.cursors import CursorStore
from persona_twin.ledger import LearningLedger
from persona_twin.paths import SubjectPaths
from persona_twin.schema import RawEnvelope
from persona_twin.vault import VaultWriter

@dataclass(frozen=True)
class SubjectContext:
    config: SubjectConfig
    paths: SubjectPaths

@runtime_checkable
class Connector(Protocol):
    name: str
    def fetch(self, ctx: SubjectContext,
              cursor: str | None) -> Iterator[tuple[RawEnvelope, str]]: ...

def scan_snapshot_exports(
        name: str, paths: list[Path], cursor: str | None,
        load: Callable[[Path], Iterator[RawEnvelope]],
) -> Iterator[tuple[RawEnvelope, str]]:
    """Whole-file cursor for static snapshot exports (claude.ai, perplexity).

    Cursor is a JSON map {export_path: byte_size_when_fully_read}. An export
    is a snapshot, not an append-only log, so the useful unit of progress is
    the whole file: the marker is emitted only with the LAST envelope of a
    file, which means a crash midway leaves that export unrecorded and it is
    re-scanned from the top next run. That re-scan is free of consequence
    only because these sources carry stable server-assigned ids, so a repeat
    pass writes zero new rows -- do NOT reuse this for a source whose
    source_id is positional or content-derived (see the mbox connector, where
    a regenerated export shifts indices and the cursor is genuinely fragile).

    Recording the byte size rather than a bare "done" is what lets a
    *regenerated* export at the same path be picked up: a new snapshot is a
    different length, so it no longer matches and gets re-read.

    Shared rather than copied per connector: the last-envelope-only marker is
    subtle enough that two copies would drift, and this codebase has already
    paid for that once (the mbox Message-ID guard scoped per-file when the
    case it existed for was cross-file).
    """
    seen: dict[str, int] = json.loads(cursor) if cursor else {}
    for path in sorted(paths, key=str):
        key = str(path)
        if not path.exists():
            # A missing export must not look like "zero new messages" -- that
            # is indistinguishable from a fully-caught-up cursor and would
            # silently drop an entire export.
            raise FileNotFoundError(f"{name} connector: export not found: {path}")
        size = path.stat().st_size
        if seen.get(key) == size:
            continue
        envelopes = list(load(path))
        for idx, env in enumerate(envelopes):
            if idx == len(envelopes) - 1:
                seen[key] = size
            yield env, json.dumps(seen)
        if not envelopes:
            # Nothing to emit, but the file is still fully read; record it so
            # an empty export isn't re-parsed on every run.
            seen[key] = size


@dataclass(frozen=True)
class IngestResult:
    source: str
    new: int
    duplicates: int
    cursor: str | None

def run_connector(conn: Connector, ctx: SubjectContext,
                  ledger: LearningLedger) -> IngestResult:
    """Backfill and incremental share this one path; only the cursor differs (§9.8).

    Every vault row must be traceable to the ingest that produced it, so the
    ledger entry is recorded even when conn.fetch() raises partway through:
    rows written before the failure are marked as coming from a partial
    ingest rather than left with no provenance at all. The cursor is
    deliberately NOT advanced when fetch() raises, even though some
    envelopes were written before the failure — resuming from `start` on
    retry is safe because VaultWriter.write() is idempotent.
    """
    cursors = CursorStore(ctx.paths.cursors)
    writer = VaultWriter(ctx.paths)
    start = cursors.get(conn.name)
    new = dup = 0
    last = start
    try:
        for env, next_cursor in conn.fetch(ctx, start):
            if writer.write(env):
                new += 1
            else:
                dup += 1
            last = next_cursor
    except Exception as exc:
        # Note: cursor is intentionally left at `start`, not advanced to `last`.
        ledger.append("ingest", ctx.config.subject_id,
                      {"source": conn.name, "new": new, "duplicates": dup,
                       "cursor_from": start, "cursor_to": last,
                       "partial": True, "error": str(exc)})
        raise
    if last is not None and last != start:
        cursors.set(conn.name, last)
    ledger.append("ingest", ctx.config.subject_id,
                  {"source": conn.name, "new": new, "duplicates": dup,
                   "cursor_from": start, "cursor_to": last})
    return IngestResult(conn.name, new, dup, last)
