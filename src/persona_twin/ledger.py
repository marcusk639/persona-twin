from __future__ import annotations
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from pydantic import BaseModel, ConfigDict, field_validator

from persona_twin.schema import _require_aware


class LedgerEntry(BaseModel):
    model_config = ConfigDict(frozen=True)
    entry_id: str
    ts: datetime
    kind: str
    subject_id: str
    payload: dict[str, Any]
    parents: list[str] = []

    _aware = field_validator("ts")(_require_aware)

class LearningLedger:
    """Append-only provenance record: source -> corpus -> persona -> adapter."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, kind: str, subject_id: str, payload: dict[str, Any],
               parents: list[str] | None = None) -> LedgerEntry:
        entry = LedgerEntry(
            entry_id=uuid.uuid4().hex, ts=datetime.now(timezone.utc),
            kind=kind, subject_id=subject_id, payload=payload, parents=parents or [])
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")
        return entry

    def read_all(self) -> list[LedgerEntry]:
        if not self.path.exists():
            return []
        # split("\n"), never splitlines(): the latter also breaks on U+2028,
        # U+2029 and U+0085, which JSON does not escape. A payload carrying one
        # -- run_connector records str(exc), and corpus text reaches it -- would
        # be sheared into unparseable halves, taking the whole append-only chain
        # down with it. read_text already normalizes \r\n to \n.
        return [LedgerEntry.model_validate_json(line)
                for line in self.path.read_text(encoding="utf-8").split("\n") if line.strip()]

    def provenance(self, entry_id: str) -> list[LedgerEntry]:
        by_id = {e.entry_id: e for e in self.read_all()}
        if entry_id not in by_id:
            raise KeyError(entry_id)
        out, queue, seen = [], [entry_id], set()
        while queue:
            cur = queue.pop(0)
            if cur in seen:
                continue
            seen.add(cur)
            e = by_id[cur]
            out.append(e)
            queue.extend(e.parents)
        return out
