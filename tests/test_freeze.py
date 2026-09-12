from datetime import datetime, timezone
import pytest
from persona_twin.paths import SubjectPaths
from persona_twin.schema import Turn
from persona_twin.corpus.store import CorpusStore
from persona_twin.corpus.freeze import freeze_golden, load_golden, GoldenCorpusTampered

def _t(i):
    return Turn(subject_id="alice", source="imessage", source_id=str(i), thread_id="c1",
                ts=datetime.now(timezone.utc), author_id="alice", is_subject=True,
                text=f"m{i}", assisted=False)

def _seeded(tmp_path):
    p = SubjectPaths("alice", tmp_path); p.ensure()
    CorpusStore(p).write("v1", [_t(1), _t(2)])
    return p

def test_freeze_writes_snapshot_with_checksum(tmp_path):
    snap = freeze_golden(_seeded(tmp_path), "v1")
    assert snap.turn_count == 2 and len(snap.sha256) == 64 and snap.path.exists()

def test_golden_is_immutable_once_frozen(tmp_path):
    p = _seeded(tmp_path)
    freeze_golden(p, "v1")
    with pytest.raises(ValueError):
        freeze_golden(p, "v1")

def test_load_golden_returns_snapshot(tmp_path):
    p = _seeded(tmp_path)
    frozen = freeze_golden(p, "v1")
    assert load_golden(p).sha256 == frozen.sha256

def test_load_golden_is_none_before_freezing(tmp_path):
    p = SubjectPaths("alice", tmp_path); p.ensure()
    assert load_golden(p) is None

def test_load_golden_verifies_checksum_on_untampered_snapshot(tmp_path):
    # The normal path: verification must not break loading an untouched
    # snapshot, and the digest returned must match what freeze_golden gave.
    p = _seeded(tmp_path)
    frozen = freeze_golden(p, "v1")
    loaded = load_golden(p)
    assert loaded.sha256 == frozen.sha256

def test_load_golden_raises_when_snapshot_file_is_tampered(tmp_path):
    p = _seeded(tmp_path)
    frozen = freeze_golden(p, "v1")
    frozen.path.chmod(0o644)
    body = frozen.path.read_text(encoding="utf-8")
    frozen.path.write_text(body.replace("m1", "TAMPERED CONTENT"), encoding="utf-8")
    frozen.path.chmod(0o444)
    with pytest.raises(GoldenCorpusTampered):
        load_golden(p)

def test_load_golden_raises_when_recorded_digest_is_tampered(tmp_path):
    # A mismatch is a mismatch regardless of which side of the comparison
    # was altered: corrupting golden.json's recorded sha256 (rather than
    # the snapshot itself) must also be caught.
    p = _seeded(tmp_path)
    freeze_golden(p, "v1")
    meta = p.golden / "golden.json"
    import json
    d = json.loads(meta.read_text(encoding="utf-8"))
    d["sha256"] = "0" * 64
    meta.chmod(0o644)
    meta.write_text(json.dumps(d, indent=2), encoding="utf-8")
    meta.chmod(0o444)
    with pytest.raises(GoldenCorpusTampered):
        load_golden(p)
