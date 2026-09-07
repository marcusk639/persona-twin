from datetime import datetime, timezone
import pytest
from pydantic import ValidationError
from persona_twin.schema import RawEnvelope, Turn


def _turn(**kw):
    base = dict(subject_id="alice", source="imessage", source_id="1",
                thread_id="t1", ts=datetime.now(timezone.utc),
                author_id="alice", is_subject=True, text="hi", assisted=False)
    base.update(kw)
    return Turn(**base)


def test_assisted_has_no_default_cc1():
    """CC1: omitting `assisted` must be an error, not a silent False."""
    with pytest.raises(ValidationError):
        Turn(subject_id="alice", source="imessage", source_id="1", thread_id="t1",
             ts=datetime.now(timezone.utc), author_id="alice", is_subject=True, text="hi")


def test_turn_roundtrips():
    t = _turn()
    assert Turn.model_validate_json(t.model_dump_json()) == t


def test_envelope_key_is_stable_and_scoped():
    e = RawEnvelope(subject_id="alice", source="imessage", source_id="42",
                    ts=datetime.now(timezone.utc), payload={"a": 1},
                    ingested_at=datetime.now(timezone.utc))
    assert e.key() == ("alice", "imessage", "42")


def test_naive_timestamps_rejected():
    with pytest.raises(ValidationError):
        _turn(ts=datetime(2020, 1, 1))
