from __future__ import annotations
import sqlite3
from pathlib import Path
from persona_twin.paths import SubjectPaths
from persona_twin.schema import Turn

_DDL = """
create table if not exists turns (
    corpus_version text not null,
    subject_id text not null,
    source     text not null,
    source_id  text not null,
    thread_id  text not null,
    ts         text not null,
    author_id  text not null,
    is_subject integer not null,
    assisted   integer not null,
    text       text not null,
    primary key (corpus_version, source, source_id)
);
create index if not exists idx_turns_version on turns(corpus_version);
"""

class UnknownCorpusVersion(Exception):
    """A corpus version that was never built was asked for by name.

    `read` answers an unknown version with an empty list, which is the right
    answer for a query but the wrong basis for a report: a mistyped version
    renders as a corpus of zero turns, zero train, zero held out, zero
    quarantined -- a plausible-looking report for a corpus that does not
    exist. Callers that present numbers to a human raise this instead.
    """

    def __init__(self, version: str, known: list[str]) -> None:
        self.version = version
        self.known = known
        super().__init__(
            f"corpus version {version!r} does not exist; built versions are "
            f"{known if known else 'none'}")


class CorpusStore:
    """Versioned, immutable clean corpus — the single source of T1 truth (spec §6.1)."""

    def __init__(self, paths: SubjectPaths) -> None:
        paths.ensure()
        self.db = Path(paths.clean) / "corpus.db"
        with sqlite3.connect(self.db) as con:
            con.executescript(_DDL)

    def versions(self) -> list[str]:
        with sqlite3.connect(self.db) as con:
            return [r[0] for r in con.execute(
                "select distinct corpus_version from turns order by corpus_version")]

    def write(self, version: str, turns: list[Turn]) -> int:
        if version in self.versions():
            raise ValueError(
                f"corpus version {version!r} already exists; builds are immutable")
        rows = [(version, t.subject_id, t.source, t.source_id, t.thread_id,
                 t.ts.isoformat(), t.author_id, int(t.is_subject), int(t.assisted), t.text)
                for t in turns]
        with sqlite3.connect(self.db) as con:
            con.executemany("insert into turns values (?,?,?,?,?,?,?,?,?,?)", rows)
        return len(rows)

    def read(self, version: str) -> list[Turn]:
        from datetime import datetime
        with sqlite3.connect(self.db) as con:
            rows = con.execute(
                "select subject_id, source, source_id, thread_id, ts, author_id,"
                " is_subject, assisted, text from turns where corpus_version = ?"
                " order by ts, source_id", (version,)).fetchall()
        return [Turn(subject_id=r[0], source=r[1], source_id=r[2], thread_id=r[3],
                     ts=datetime.fromisoformat(r[4]), author_id=r[5],
                     is_subject=bool(r[6]), assisted=bool(r[7]), text=r[8],
                     corpus_version=version) for r in rows]
