from datetime import datetime, timezone, timedelta
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


def test_envelope_naive_ts_rejected():
    """Reject naive datetime for RawEnvelope.ts."""
    with pytest.raises(ValidationError):
        RawEnvelope(subject_id="alice", source="imessage", source_id="42",
                    ts=datetime(2020, 1, 1), payload={},
                    ingested_at=datetime.now(timezone.utc))


def test_envelope_naive_ingested_at_rejected():
    """Reject naive datetime for RawEnvelope.ingested_at."""
    with pytest.raises(ValidationError):
        RawEnvelope(subject_id="alice", source="imessage", source_id="42",
                    ts=datetime.now(timezone.utc), payload={},
                    ingested_at=datetime(2020, 1, 1))


def test_turn_ts_normalized_to_utc():
    """Turn.ts is normalized to UTC; instant is preserved."""
    tz_plus5 = timezone(timedelta(hours=5))
    ts_in_tz = datetime(2024, 1, 1, 12, 0, tzinfo=tz_plus5)
    t = _turn(ts=ts_in_tz)

    # Should be normalized to UTC
    assert t.ts.tzinfo == timezone.utc
    # Instant should be preserved: 12:00+05:00 == 07:00 UTC
    assert t.ts.hour == 7
    assert t.ts.day == 1


def test_envelope_ts_normalized_to_utc():
    """RawEnvelope.ts is normalized to UTC; instant is preserved."""
    tz_plus5 = timezone(timedelta(hours=5))
    ts_in_tz = datetime(2024, 1, 1, 12, 0, tzinfo=tz_plus5)
    e = RawEnvelope(subject_id="alice", source="imessage", source_id="42",
                    ts=ts_in_tz, payload={},
                    ingested_at=datetime.now(timezone.utc))

    assert e.ts.tzinfo == timezone.utc
    assert e.ts.hour == 7
    assert e.ts.day == 1


def test_envelope_ingested_at_normalized_to_utc():
    """RawEnvelope.ingested_at is normalized to UTC; instant is preserved."""
    tz_plus5 = timezone(timedelta(hours=5))
    ingested_in_tz = datetime(2024, 1, 1, 12, 0, tzinfo=tz_plus5)
    e = RawEnvelope(subject_id="alice", source="imessage", source_id="42",
                    ts=datetime.now(timezone.utc), payload={},
                    ingested_at=ingested_in_tz)

    assert e.ingested_at.tzinfo == timezone.utc
    assert e.ingested_at.hour == 7
    assert e.ingested_at.day == 1
