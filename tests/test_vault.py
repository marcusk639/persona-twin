from datetime import datetime, timedelta, timezone
from persona_twin.paths import SubjectPaths
from persona_twin.schema import RawEnvelope
from persona_twin.vault import VaultWriter

def _env(sid="1", source="imessage", ts=None):
    now = ts or datetime.now(timezone.utc)
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

def test_iter_source_orders_by_ts(tmp_path):
    """iter_source must yield envelopes in chronological order regardless of write order."""
    w = VaultWriter(SubjectPaths("alice", tmp_path))
    base = datetime.now(timezone.utc)
    w.write(_env("mid", ts=base + timedelta(seconds=1)))
    w.write(_env("first", ts=base))
    w.write(_env("last", ts=base + timedelta(seconds=2)))
    assert [e.source_id for e in w.iter_source("imessage")] == ["first", "mid", "last"]

def test_iter_source_roundtrips_envelope(tmp_path):
    """An envelope read back via iter_source must equal the one written, payload included."""
    w = VaultWriter(SubjectPaths("alice", tmp_path))
    written = _env("1")
    w.write(written)
    (read_back,) = list(w.iter_source("imessage"))
    assert read_back == written
    assert read_back.payload == written.payload

def test_iter_source_empty_for_unknown_source(tmp_path):
    w = VaultWriter(SubjectPaths("alice", tmp_path))
    w.write(_env("1"))
    assert list(w.iter_source("nonexistent")) == []

def test_iter_source_excludes_other_sources(tmp_path):
    w = VaultWriter(SubjectPaths("alice", tmp_path))
    w.write(_env("1", source="imessage"))
    w.write(_env("1", source="git"))
    ids = [e.source_id for e in w.iter_source("imessage")]
    assert ids == ["1"]
    assert all(e.source == "imessage" for e in w.iter_source("imessage"))
