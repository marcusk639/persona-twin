from datetime import datetime, timezone
import pytest
from persona_twin.schema import RawEnvelope
from persona_twin.normalize.turns import normalize


def _env(source, payload, source_id="1"):
    now = datetime.now(timezone.utc)
    return RawEnvelope(subject_id="alice", source=source, source_id=source_id,
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


def test_mail_normalizes_correctly():
    t = normalize(_env("mail", {"text": "let's sync", "subject": "hi", "to": "bob@example.com"}))
    assert (t.text == "let's sync" and t.thread_id == "bob@example.com"
            and t.is_subject is True and t.assisted is False)


def test_missing_thread_key_gets_unique_orphan_id():
    # mail's "to" key can be absent from the payload entirely.
    t1 = normalize(_env("mail", {"text": "a"}, source_id="s1"))
    t2 = normalize(_env("mail", {"text": "b"}, source_id="s2"))
    assert t1.thread_id != "unknown" and t2.thread_id != "unknown"
    assert t1.thread_id != t2.thread_id


def test_empty_thread_key_gets_unique_orphan_id():
    # mail's connector emits "to": msg.get("To", "") -- present but empty,
    # not absent -- when a message has no To header.
    t1 = normalize(_env("mail", {"text": "a", "to": ""}, source_id="s1"))
    t2 = normalize(_env("mail", {"text": "b", "to": ""}, source_id="s2"))
    assert t1.thread_id != "" and t2.thread_id != ""
    assert t1.thread_id != t2.thread_id


def test_literal_unknown_thread_key_gets_unique_orphan_id():
    # iMessage's SQL COALESCEs an unresolved chat_guid to the literal
    # string "unknown" -- that literal must not become a shared thread_id
    # either, or every such message merges into one fabricated thread.
    t1 = normalize(_env("imessage", {"text": "a", "is_from_me": True,
                                     "handle": "unknown", "chat_guid": "unknown"},
                        source_id="s1"))
    t2 = normalize(_env("imessage", {"text": "b", "is_from_me": True,
                                     "handle": "unknown", "chat_guid": "unknown"},
                        source_id="s2"))
    assert t1.thread_id != "unknown" and t2.thread_id != "unknown"
    assert t1.thread_id != t2.thread_id
