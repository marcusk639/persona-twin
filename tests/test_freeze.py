from datetime import datetime, timezone
import pytest
from persona_twin.paths import SubjectPaths
from persona_twin.schema import Turn
from persona_twin.corpus.store import CorpusStore
from persona_twin.corpus.freeze import freeze_golden, load_golden

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
