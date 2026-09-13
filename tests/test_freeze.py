from datetime import datetime, timezone
from pathlib import Path
import pytest
from persona_twin.ledger import LearningLedger
from persona_twin.paths import SubjectPaths
from persona_twin.schema import Turn
from persona_twin.corpus.store import CorpusStore
from persona_twin.corpus.freeze import (
    freeze_golden, load_golden, supersede_golden,
    GoldenCorpusTampered, GoldenNotASuperset, GoldenPostDeployment)

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


# --- CC2 re-cut (supersede) -------------------------------------------------

def _turn(source_id, text="hello", source="imessage", ts="2024-01-01T00:00:00+00:00",
          assisted=False):
    from datetime import datetime
    return Turn(subject_id="alice", source=source, source_id=source_id,
                thread_id="t1", ts=datetime.fromisoformat(ts), author_id="alice",
                is_subject=True, text=text, assisted=assisted)


def _frozen(tmp_path, turns, version="v1"):
    paths = SubjectPaths("alice", tmp_path); paths.ensure()
    CorpusStore(paths).write(version, turns)
    freeze_golden(paths, version)
    return paths


def test_supersede_repoints_the_baseline_at_a_superset_build(tmp_path):
    base = [_turn("a"), _turn("b")]
    paths = _frozen(tmp_path, base)
    CorpusStore(paths).write("v2", base + [_turn("c")])
    led = LearningLedger(paths.ledger)
    snap = supersede_golden(paths, "v2", led, reason="two new sources landed")
    assert snap.version == "v2" and snap.turn_count == 3
    assert load_golden(paths).version == "v2"


def test_supersede_keeps_the_previous_snapshot_on_disk(tmp_path):
    base = [_turn("a"), _turn("b")]
    paths = _frozen(tmp_path, base)
    old = Path(paths.golden) / "golden-v1.jsonl"
    old_bytes = old.read_bytes()
    CorpusStore(paths).write("v2", base + [_turn("c")])
    supersede_golden(paths, "v2", LearningLedger(paths.ledger), reason="r")
    assert old.exists() and old.read_bytes() == old_bytes
    assert (Path(paths.golden) / "golden-superseded-v1.json").exists()


def test_supersede_refuses_to_drop_a_baseline_turn(tmp_path):
    paths = _frozen(tmp_path, [_turn("a"), _turn("b")])
    CorpusStore(paths).write("v2", [_turn("a"), _turn("c")])   # "b" lost
    with pytest.raises(GoldenNotASuperset):
        supersede_golden(paths, "v2", LearningLedger(paths.ledger), reason="r")
    assert load_golden(paths).version == "v1"


def test_supersede_refuses_when_baseline_text_was_rewritten(tmp_path):
    paths = _frozen(tmp_path, [_turn("a", "original wording")])
    CorpusStore(paths).write("v2", [_turn("a", "rewritten wording"), _turn("b")])
    with pytest.raises(GoldenNotASuperset):
        supersede_golden(paths, "v2", LearningLedger(paths.ledger), reason="r")
    assert load_golden(paths).version == "v1"


def test_supersede_refuses_an_assisted_turn(tmp_path):
    """CC2 is a PRE-deployment baseline; assisted text is what it exists to exclude."""
    paths = _frozen(tmp_path, [_turn("a")])
    CorpusStore(paths).write("v2", [_turn("a"), _turn("b", assisted=True)])
    with pytest.raises(GoldenPostDeployment):
        supersede_golden(paths, "v2", LearningLedger(paths.ledger), reason="r")
    assert load_golden(paths).version == "v1"


def test_supersede_records_provenance_in_the_ledger(tmp_path):
    base = [_turn("a")]
    paths = _frozen(tmp_path, base)
    CorpusStore(paths).write("v2", base + [_turn("b")])
    led = LearningLedger(paths.ledger)
    supersede_golden(paths, "v2", led, reason="two new sources landed")
    e = [x for x in led.read_all() if x.kind == "golden_supersede"][-1]
    assert e.payload["from_version"] == "v1" and e.payload["to_version"] == "v2"
    assert e.payload["reason"] == "two new sources landed"
    assert e.payload["turns_added"] == 1


def test_supersede_without_an_existing_freeze_is_an_error(tmp_path):
    paths = SubjectPaths("alice", tmp_path); paths.ensure()
    CorpusStore(paths).write("v1", [_turn("a")])
    with pytest.raises(ValueError):
        supersede_golden(paths, "v1", LearningLedger(paths.ledger), reason="r")
