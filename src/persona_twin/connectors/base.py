from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Protocol, runtime_checkable
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

@dataclass(frozen=True)
class IngestResult:
    source: str
    new: int
    duplicates: int
    cursor: str | None

def run_connector(conn: Connector, ctx: SubjectContext,
                  ledger: LearningLedger) -> IngestResult:
    """Backfill and incremental share this one path; only the cursor differs (§9.8)."""
    cursors = CursorStore(ctx.paths.cursors)
    writer = VaultWriter(ctx.paths)
    start = cursors.get(conn.name)
    new = dup = 0
    last = start
    for env, next_cursor in conn.fetch(ctx, start):
        if writer.write(env):
            new += 1
        else:
            dup += 1
        last = next_cursor
    if last is not None and last != start:
        cursors.set(conn.name, last)
    ledger.append("ingest", ctx.config.subject_id,
                  {"source": conn.name, "new": new, "duplicates": dup,
                   "cursor_from": start, "cursor_to": last})
    return IngestResult(conn.name, new, dup, last)
