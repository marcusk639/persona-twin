from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path

_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

@dataclass(frozen=True)
class SubjectPaths:
    subject_id: str
    root: Path

    def __post_init__(self) -> None:
        if not _SLUG.match(self.subject_id):
            raise ValueError(f"subject_id must be a lowercase slug: {self.subject_id!r}")

    @property
    def _base(self) -> Path:
        return Path(self.root) / "data" / "subjects" / self.subject_id

    @property
    def vault(self) -> Path: return self._base / "vault"
    @property
    def clean(self) -> Path: return self._base / "clean"
    @property
    def exportable(self) -> Path: return self._base / "exportable"
    @property
    def golden(self) -> Path: return self._base / "golden"
    @property
    def probes(self) -> Path: return self._base / "probes"
    @property
    def cursors(self) -> Path: return self._base / "vault" / "cursors.json"
    @property
    def ledger(self) -> Path: return self._base / "ledger.jsonl"

    def ensure(self) -> None:
        for d in (self.vault, self.clean, self.exportable, self.golden):
            d.mkdir(parents=True, exist_ok=True)
