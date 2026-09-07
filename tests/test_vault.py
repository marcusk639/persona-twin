from datetime import datetime, timezone
from persona_twin.paths import SubjectPaths
from persona_twin.schema import RawEnvelope
from persona_twin.vault import VaultWriter

def _env(sid="1", source="imessage"):
    now = datetime.now(timezone.utc)
    return RawEnvelope(subject_id="alice", source=source, source_id=sid,
                       ts=now, payload={"text": "hi"}, ingested_at=now)

def test_write_returns_true_once_then_false(tmp_path):
    w = VaultWriter(SubjectPaths("alice", tmp_path))
    assert w.write(_env()) is True
    assert w.write(_env()) is False, "re-running a connector must not duplicate"

def test_count_per_source(tmp_path):
    w = VaultWriter(SubjectPaths("alice", tmp_path))
    w.write(_env("1")); w.write(_env("2")); w.write(_env("1", source="git"))
    assert w.count("imessage") == 2 and w.count("git") == 1

def test_overlapping_window_is_safe(tmp_path):
    """Re-ingesting an overlapping range must not reweight the corpus."""
    w = VaultWriter(SubjectPaths("alice", tmp_path))
    for i in range(5):
        w.write(_env(str(i)))
    for i in range(3, 8):
        w.write(_env(str(i)))
    assert w.count("imessage") == 8
