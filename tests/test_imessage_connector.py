import sqlite3
import pytest
from datetime import datetime, timezone
from persona_twin.config import SubjectConfig
from persona_twin.paths import SubjectPaths
from persona_twin.connectors.base import SubjectContext
from persona_twin.connectors.imessage import IMessageConnector, APPLE_EPOCH

def _blob(text: str) -> bytes:
    b = text.encode("utf-8")
    return b"\x04\x0bstreamtyped\x84\x84\x08NSString\x01\x94\x84\x01+" + bytes([len(b)]) + b

def _db(tmp_path, rows):
    p = tmp_path / "chat.db"
    con = sqlite3.connect(p)
    con.executescript("""
        create table message (ROWID integer primary key, guid text, text text,
            attributedBody blob, is_from_me integer, date integer, handle_id integer);
        create table handle (ROWID integer primary key, id text);
        create table chat (ROWID integer primary key, guid text);
        create table chat_message_join (chat_id integer, message_id integer);
    """)
    con.execute("insert into handle (ROWID, id) values (1, '+15551234567')")
    con.execute("insert into chat (ROWID, guid) values (1, 'chat-1')")
    for rid, text, blob, from_me in rows:
        con.execute("insert into message (ROWID, guid, text, attributedBody, is_from_me,"
                    " date, handle_id) values (?,?,?,?,?,?,1)",
                    (rid, f"g{rid}", text, blob, from_me, 700000000 * 1_000_000_000))
        con.execute("insert into chat_message_join values (1, ?)", (rid,))
    con.commit(); con.close()
    return p

def _ctx(tmp_path):
    p = SubjectPaths("alice", tmp_path); p.ensure()
    return SubjectContext(config=SubjectConfig(subject_id="alice", display_name="A"), paths=p)

def test_reads_plain_text_rows(tmp_path):
    db = _db(tmp_path, [(1, "hello", None, 1)])
    envs = [e for e, _ in IMessageConnector(db).fetch(_ctx(tmp_path), None)]
    assert envs[0].payload["text"] == "hello" and envs[0].payload["is_from_me"] is True

def test_recovers_null_text_from_attributed_body(tmp_path):
    db = _db(tmp_path, [(1, None, _blob("recovered text"), 0)])
    envs = [e for e, _ in IMessageConnector(db).fetch(_ctx(tmp_path), None)]
    assert envs[0].payload["text"] == "recovered text"

def test_skips_rows_with_no_recoverable_text(tmp_path):
    db = _db(tmp_path, [(1, None, b"\x00garbage", 1)])
    assert list(IMessageConnector(db).fetch(_ctx(tmp_path), None)) == []

def test_cursor_is_max_rowid_and_resumes(tmp_path):
    db = _db(tmp_path, [(1, "a", None, 1), (2, "b", None, 1)])
    ctx = _ctx(tmp_path)
    c = IMessageConnector(db)
    cur = [x for _, x in c.fetch(ctx, None)][-1]
    assert cur == "2"
    assert list(c.fetch(ctx, "2")) == []

def test_apple_epoch_converts_to_utc(tmp_path):
    db = _db(tmp_path, [(1, "a", None, 1)])
    env = next(e for e, _ in IMessageConnector(db).fetch(_ctx(tmp_path), None))
    assert env.ts.year == 2023 and env.ts.tzinfo is timezone.utc

def test_apple_epoch_seconds_magnitude_matches_nanoseconds(tmp_path):
    # message.date is nanoseconds on modern macOS but seconds on older
    # databases (_to_utc switches on magnitude at 10**12). _db()'s fixture
    # rows always write a nanosecond-scale `date`, so the seconds branch
    # (`else float(apple_ts)`) has zero coverage without this test. Insert a
    # row with a raw seconds-scale `date` and assert it converts to the same
    # instant as the nanosecond fixture used by test_apple_epoch_converts_to_utc.
    # Fails if the magnitude switch in _to_utc is removed.
    (tmp_path / "seconds").mkdir()
    (tmp_path / "nanos").mkdir()
    p_seconds = tmp_path / "seconds" / "chat.db"
    con = sqlite3.connect(p_seconds)
    con.executescript("""
        create table message (ROWID integer primary key, guid text, text text,
            attributedBody blob, is_from_me integer, date integer, handle_id integer);
        create table handle (ROWID integer primary key, id text);
        create table chat (ROWID integer primary key, guid text);
        create table chat_message_join (chat_id integer, message_id integer);
    """)
    con.execute("insert into handle (ROWID, id) values (1, '+15551234567')")
    con.execute("insert into chat (ROWID, guid) values (1, 'chat-1')")
    con.execute("insert into message (ROWID, guid, text, attributedBody, is_from_me,"
                " date, handle_id) values (1, 'g1', 'a', NULL, 1, 700000000, 1)")
    con.execute("insert into chat_message_join values (1, 1)")
    con.commit(); con.close()

    p_nanos = _db(tmp_path / "nanos", [(1, "a", None, 1)])

    env_seconds = next(e for e, _ in IMessageConnector(p_seconds).fetch(_ctx(tmp_path), None))
    env_nanos = next(e for e, _ in IMessageConnector(p_nanos).fetch(_ctx(tmp_path), None))
    assert env_seconds.ts == env_nanos.ts
    assert env_seconds.ts.year == 2023 and env_seconds.ts.tzinfo is timezone.utc

def test_distinct_source_ids_for_identical_text(tmp_path):
    # source_id must come from ROWID, not a hash of text -- two rows with
    # identical text must not collide in the vault's (subject_id, source,
    # source_id) dedup key (lesson paid for by the claude_code connector).
    db = _db(tmp_path, [(1, "same text", None, 1), (2, "same text", None, 1)])
    ids = [x for _, x in IMessageConnector(db).fetch(_ctx(tmp_path), None)]
    assert ids == ["1", "2"]
    assert len(set(ids)) == 2

def test_emits_both_sides_of_conversation(tmp_path):
    db = _db(tmp_path, [(1, "from them", None, 0), (2, "from me", None, 1)])
    envs = [e for e, _ in IMessageConnector(db).fetch(_ctx(tmp_path), None)]
    assert [e.payload["is_from_me"] for e in envs] == [False, True]

def test_missing_database_raises(tmp_path):
    # Narrowed to the specific exception sqlite raises for a missing
    # read-only-mode file, so this proves the actual failure mode (loud,
    # not silent) rather than merely that some exception occurred.
    missing = tmp_path / "does-not-exist.db"
    with pytest.raises(sqlite3.OperationalError):
        list(IMessageConnector(missing).fetch(_ctx(tmp_path), None))

def test_multi_chat_message_emitted_once(tmp_path):
    # macOS chat merge/split can leave a message with more than one
    # chat_message_join row. Without `group by m.ROWID` in _SQL, the LEFT
    # JOIN fans out and the connector would emit the same message twice,
    # sharing one source_id -- violating the uniqueness-within-a-run
    # contract that run_connector's dedup relies on. This test fails
    # without the GROUP BY.
    db = _db(tmp_path, [(1, "hello", None, 1)])
    con = sqlite3.connect(db)
    con.execute("insert into chat (ROWID, guid) values (2, 'chat-2')")
    con.execute("insert into chat_message_join values (2, 1)")  # 2nd join row for ROWID 1
    con.commit(); con.close()
    envs = [e for e, _ in IMessageConnector(db).fetch(_ctx(tmp_path), None)]
    assert len(envs) == 1
    assert envs[0].source_id == "1"
