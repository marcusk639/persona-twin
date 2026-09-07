from __future__ import annotations
import sqlite3
from pathlib import Path
from persona_twin.paths import SubjectPaths
from persona_twin.schema import RawEnvelope

_DDL = """
create table if not exists envelopes (
    subject_id text not null,
    source     text not null,
    source_id  text not null,
    ts         text not null,
    ingested_at text not null,
    payload    text not null,
    primary key (subject_id, source, source_id)
);
"""

class VaultWriter:
    """Immutable raw store with idempotent upsert (spec §9.8)."""

    def __init__(self, paths: SubjectPaths) -> None:
        paths.ensure()
        self.db = Path(paths.vault) / "envelopes.db"
        with self._con() as con:
            con.executescript(_DDL)

    def _con(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db)

    def write(self, env: RawEnvelope) -> bool:
        with self._con() as con:
            cur = con.execute(
                "insert or ignore into envelopes"
                " (subject_id, source, source_id, ts, ingested_at, payload)"
                " values (?, ?, ?, ?, ?, ?)",
                (env.subject_id, env.source, env.source_id, env.ts.isoformat(),
                 env.ingested_at.isoformat(), env.model_dump_json()))
            return cur.rowcount == 1

    def count(self, source: str) -> int:
        with self._con() as con:
            (n,) = con.execute(
                "select count(*) from envelopes where source = ?", (source,)).fetchone()
            return n

    def iter_source(self, source: str):
        with self._con() as con:
            # Lexicographic ordering of these ISO-8601 strings is equivalent to
            # chronological ordering because every timestamp is UTC-normalized
            # by RawEnvelope's schema validator (_require_aware in schema.py)
            # before it is ever stored, so every row carries the same fixed
            # "+00:00" offset suffix. This equivalence would break if a
            # non-UTC offset were ever stored.
            for (payload,) in con.execute(
                "select payload from envelopes where source = ? order by ts", (source,)):
                yield RawEnvelope.model_validate_json(payload)
