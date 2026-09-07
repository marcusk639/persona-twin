from __future__ import annotations
import json
from pathlib import Path

class CursorStore:
    """Durable per-source ingest watermarks (spec §9.8)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, source: str) -> str | None:
        return self._load().get(source)

    def set(self, source: str, cursor: str) -> None:
        data = self._load()
        data[source] = cursor
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.path)
