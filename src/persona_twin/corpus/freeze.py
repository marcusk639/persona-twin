from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from pathlib import Path
from persona_twin.corpus.store import CorpusStore
from persona_twin.ledger import LearningLedger
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

class GoldenNotASuperset(Exception):
    """CC2: a re-cut may only ADD to the baseline, never drop or rewrite it."""

    def __init__(self, dropped: int, rewritten: int) -> None:
        self.dropped = dropped
        self.rewritten = rewritten
        super().__init__(
            f"refusing to supersede the golden corpus: {dropped} baseline turn(s) "
            f"absent from the new version and {rewritten} rewritten. A re-cut must "
            "be a strict superset -- otherwise drift measured against it reflects "
            "the baseline changing, not the subject."
        )


class GoldenPostDeployment(Exception):
    """CC2: the baseline is the voice BEFORE the system existed."""

    def __init__(self, assisted: int) -> None:
        self.assisted = assisted
        super().__init__(
            f"refusing to supersede the golden corpus: {assisted} turn(s) are "
            "assisted=True. CC2 is a pre-deployment reference; admitting "
            "system-assisted text is precisely the contamination it exists to "
            "exclude, and it is unrecoverable once frozen."
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


def supersede_golden(paths: SubjectPaths, version: str, ledger: LearningLedger,
                     *, reason: str) -> GoldenSnapshot:
    """Re-point the CC2 baseline at a newer PRE-deployment build.

    CC2 calls the golden corpus immutable, and `freeze_golden` enforces that
    literally. But the constraint's purpose is a reference for "the subject's
    voice as it was before the system existed" -- so a build that adds sources
    from that same pre-deployment era makes the baseline more complete, not
    more contaminated. Refusing forever would freeze an accident of which
    connectors happened to exist on the day of the first freeze.

    What is NOT relaxed is what CC2 actually protects, enforced here as two
    hard refusals:

      * The new version must be a strict superset -- no baseline turn dropped,
        no text rewritten. Otherwise drift measured against the baseline
        reflects the baseline moving rather than the subject changing, which
        silently invalidates every comparison ever made against it.
      * No turn may be assisted=True. Once the system's own output enters the
        reference for the subject's unassisted voice, the contamination cannot
        be undone by any later build.

    The previous snapshot is never destroyed: its .jsonl stays on disk and its
    metadata is archived alongside as golden-superseded-<version>.json, so the
    original baseline remains loadable and auditable. The supersede is recorded
    in the append-only ledger, which is what keeps the source -> corpus ->
    persona chain intact across the re-cut.
    """
    meta = _meta_path(paths)
    if not meta.exists():
        raise ValueError(
            "no golden corpus is frozen yet; use freeze_golden for the first cut")
    current = load_golden(paths)          # verifies the old checksum before we move it
    turns = CorpusStore(paths).read(version)

    assisted = sum(1 for t in turns if t.assisted)
    if assisted:
        raise GoldenPostDeployment(assisted)

    old = {(t.source, t.source_id): t.text
           for t in CorpusStore(paths).read(current.version)}
    new = {(t.source, t.source_id): t.text for t in turns}
    dropped = sum(1 for k in old if k not in new)
    rewritten = sum(1 for k, v in old.items() if k in new and new[k] != v)
    if dropped or rewritten:
        raise GoldenNotASuperset(dropped, rewritten)

    archive = Path(paths.golden) / f"golden-superseded-{current.version}.json"
    archive.write_text(meta.read_text(encoding="utf-8"), encoding="utf-8")
    archive.chmod(0o444)
    meta.chmod(0o644)
    meta.unlink()
    snap = freeze_golden(paths, version)
    ledger.append("golden_supersede", paths.subject_id,
                  {"from_version": current.version, "to_version": version,
                   "from_sha256": current.sha256, "to_sha256": snap.sha256,
                   "from_turns": current.turn_count, "to_turns": snap.turn_count,
                   "turns_added": snap.turn_count - current.turn_count,
                   "archived_meta": str(archive), "reason": reason})
    return snap
