from datetime import datetime, timezone
import pytest
from persona_twin.schema import RawEnvelope
from persona_twin.normalize.turns import normalize


def _env(source, payload):
    now = datetime.now(timezone.utc)
    return RawEnvelope(subject_id="alice", source=source, source_id="1",
                       ts=now, payload=payload, ingested_at=now)


def test_imessage_from_me_is_subject():
    t = normalize(_env("imessage", {"text": "hi", "is_from_me": True,
                                    "handle": "+1555", "chat_guid": "c1"}))
    assert t.is_subject is True and t.thread_id == "c1" and t.author_id == "alice"


def test_imessage_from_other_is_not_subject():
    t = normalize(_env("imessage", {"text": "yo", "is_from_me": False,
                                    "handle": "+1555", "chat_guid": "c1"}))
    assert t.is_subject is False and t.author_id == "+1555"


def test_claude_code_is_always_subject():
    t = normalize(_env("claude_code", {"text": "ship it", "file": "/x/a.jsonl"}))
    assert t.is_subject is True and t.thread_id == "/x/a.jsonl"


def test_historical_turns_are_unassisted_cc1():
    t = normalize(_env("git_repos", {"text": "feat: x", "repo": "/r"}))
    assert t.assisted is False


def test_unknown_source_raises():
    with pytest.raises(KeyError):
        normalize(_env("mystery", {"text": "x"}))
