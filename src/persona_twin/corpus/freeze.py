from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from pathlib import Path
from persona_twin.corpus.store import CorpusStore
from persona_twin.paths import SubjectPaths

class GoldenCorpusTampered(Exception):
    """CC2: the frozen golden corpus no longer matches its recorded checksum."""

    def __init__(self, recorded: str, computed: str) -> None:
        self.recorded = recorded
        self.computed = computed
        super().__init__(
            "golden corpus checksum mismatch: recorded="
            f"{recorded} computed={computed}; the frozen snapshot may have "
            "been altered after freezing"
        )

@dataclass(frozen=True)
class GoldenSnapshot:
    version: str
    path: Path
    sha256: str
    turn_count: int

def _meta_path(paths: SubjectPaths) -> Path:
    return Path(paths.golden) / "golden.json"

def freeze_golden(paths: SubjectPaths, version: str) -> GoldenSnapshot:
    """CC2: immutable pre-deployment reference for the subject's voice."""
    meta = _meta_path(paths)
    if meta.exists():
        raise ValueError("golden corpus already frozen; CC2 forbids overwriting it")
    turns = CorpusStore(paths).read(version)
    body = "\n".join(t.model_dump_json() for t in turns)
    out = Path(paths.golden) / f"golden-{version}.jsonl"
    out.write_text(body, encoding="utf-8")
    out.chmod(0o444)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    meta.write_text(json.dumps(
        {"version": version, "path": str(out), "sha256": digest,
         "turn_count": len(turns)}, indent=2), encoding="utf-8")
    meta.chmod(0o444)
    return GoldenSnapshot(version, out, digest, len(turns))

def load_golden(paths: SubjectPaths) -> GoldenSnapshot | None:
    """Load the frozen golden snapshot, verifying its checksum on every read.

    CC2's guarantee is only as good as this check: the recorded sha256 in
    golden.json is evidence the snapshot hasn't been edited since it was
    frozen, but evidence nobody verifies is decorative. We recompute the
    digest from the raw bytes on disk (not from parsed/re-serialized turns,
    so any byte-level change is caught) and raise on any mismatch rather
    than returning a possibly-tampered snapshot.
    """
    meta = _meta_path(paths)
    if not meta.exists():
        return None
    d = json.loads(meta.read_text(encoding="utf-8"))
    snapshot_path = Path(d["path"])
    body = snapshot_path.read_bytes()
    computed = hashlib.sha256(body).hexdigest()
    recorded = d["sha256"]
    if computed != recorded:
        raise GoldenCorpusTampered(recorded, computed)
    return GoldenSnapshot(d["version"], snapshot_path, recorded, d["turn_count"])
