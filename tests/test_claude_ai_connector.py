import json
import zipfile
from pathlib import Path

import pytest

from persona_twin.config import SubjectConfig
from persona_twin.connectors.base import SubjectContext
from persona_twin.connectors.claude_ai import ClaudeAiConnector
from persona_twin.normalize.turns import normalize
from persona_twin.paths import SubjectPaths
from persona_twin.vault import VaultWriter


def _ctx(tmp_path):
    paths = SubjectPaths("alice", tmp_path)
    paths.ensure()
    return SubjectContext(
        config=SubjectConfig(subject_id="alice", display_name="A"), paths=paths)


def _msg(uuid, sender, text, *, attachments=None, files=None,
         created_at="2026-06-03T06:14:11.670509Z"):
    return {"uuid": uuid, "sender": sender, "text": text,
            "content": [{"type": "text", "text": text}] if text else [],
            "created_at": created_at, "updated_at": created_at,
            "attachments": attachments or [], "files": files or [],
            "parent_message_uuid": "00000000-0000-4000-8000-000000000000"}


def _conv(uuid, messages, name="a chat"):
    return {"uuid": uuid, "name": name, "summary": "",
            "created_at": "2026-06-03T06:14:11.670509Z",
            "updated_at": "2026-06-03T06:14:11.670509Z",
            "account": {"uuid": "acct"}, "chat_messages": messages}


def _export(path: Path, conversations: list[dict]) -> Path:
    """Write a claude.ai-shaped export zip (conversations.json inside)."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("conversations.json", json.dumps(conversations))
    return path


DEFAULT = [
    _conv("conv-1", [
        _msg("m-1", "human", "why does this test hang?"),
        _msg("m-2", "assistant", "because the lock is held"),
        _msg("m-3", "human", "ship it"),
    ]),
    _conv("conv-2", [_msg("m-4", "human", "draft the release note")]),
]


def _zip(tmp_path, conversations=DEFAULT, name="conversations-000.zip"):
    return _export(tmp_path / name, conversations)


def test_extracts_only_human_messages(tmp_path):
    c = ClaudeAiConnector([_zip(tmp_path)])
    texts = [e.payload["text"] for e, _ in c.fetch(_ctx(tmp_path), None)]
    assert texts == ["why does this test hang?", "ship it",
                     "draft the release note"]


def test_reads_a_bare_conversations_json_too(tmp_path):
    p = tmp_path / "conversations.json"
    p.write_text(json.dumps(DEFAULT))
    c = ClaudeAiConnector([p])
    assert [e.payload["text"] for e, _ in c.fetch(_ctx(tmp_path), None)] == [
        "why does this test hang?", "ship it", "draft the release note"]


def test_missing_export_raises_rather_than_yielding_nothing(tmp_path):
    c = ClaudeAiConnector([tmp_path / "not-there.zip"])
    with pytest.raises(FileNotFoundError):
        list(c.fetch(_ctx(tmp_path), None))


def test_source_id_is_the_message_uuid_and_stable_across_runs(tmp_path):
    c = ClaudeAiConnector([_zip(tmp_path)])
    first = [e.source_id for e, _ in c.fetch(_ctx(tmp_path), None)]
    second = [e.source_id for e, _ in c.fetch(_ctx(tmp_path), None)]
    assert first == ["m-1", "m-3", "m-4"] == second


def test_message_uuids_survive_a_re_export_under_a_new_filename(tmp_path):
    """A later export re-exports the same history; the vault must not re-ingest it."""
    ctx = _ctx(tmp_path)
    writer = VaultWriter(ctx.paths)
    first = _zip(tmp_path, name="conversations-000.zip")
    for env, _ in ClaudeAiConnector([first]).fetch(ctx, None):
        assert writer.write(env)
    later = _zip(tmp_path, DEFAULT + [_conv("conv-3", [_msg("m-5", "human", "new one")])],
                 name="conversations-001.zip")
    wrote = [writer.write(env)
             for env, _ in ClaudeAiConnector([later]).fetch(ctx, None)]
    assert wrote == [False, False, False, True]


def test_attachment_payload_is_dropped_but_counted(tmp_path):
    pasted = "x" * 50_000
    conv = [_conv("conv-1", [
        _msg("m-1", "human", "what breaks here?",
             attachments=[{"file_name": "log.txt", "file_size": len(pasted),
                           "file_type": "text/plain",
                           "extracted_content": pasted}],
             files=[{"file_name": "shot.png", "file_uuid": "f-1"}])])]
    (env, _), = ClaudeAiConnector([_zip(tmp_path, conv)]).fetch(_ctx(tmp_path), None)
    assert pasted not in json.dumps(env.payload)
    assert env.payload["text"] == "what breaks here?"
    assert env.payload["attachment_count"] == 1
    assert env.payload["attachment_chars"] == len(pasted)
    assert env.payload["file_count"] == 1


def test_an_attachment_does_not_by_itself_mark_the_prose_pasted(tmp_path):
    """The payload is dropped, so what remains is genuinely typed prose."""
    conv = [_conv("conv-1", [
        _msg("m-1", "human", "what breaks here?",
             attachments=[{"file_name": "log.txt", "file_size": 50_000,
                           "file_type": "text/plain",
                           "extracted_content": "x" * 50_000}])])]
    (env, _), = ClaudeAiConnector([_zip(tmp_path, conv)]).fetch(_ctx(tmp_path), None)
    assert env.payload["looks_pasted"] is False


def test_inline_paste_is_flagged(tmp_path):
    conv = [_conv("conv-1", [
        _msg("m-1", "human", "here is the file:\n" + "line of log\n" * 1200),
        _msg("m-2", "human", "and here:\n```\n" + "code\n" * 2000 + "```\nthoughts?"),
    ])]
    flags = [e.payload["looks_pasted"]
             for e, _ in ClaudeAiConnector([_zip(tmp_path, conv)]).fetch(_ctx(tmp_path), None)]
    assert flags == [True, True]


def test_skips_messages_with_no_typed_text(tmp_path):
    conv = [_conv("conv-1", [
        _msg("m-1", "human", "",
             attachments=[{"file_name": "a.txt", "file_size": 3,
                           "file_type": "text/plain", "extracted_content": "abc"}]),
        _msg("m-2", "human", "   "),
        _msg("m-3", "human", "real question"),
    ])]
    ids = [e.source_id
           for e, _ in ClaudeAiConnector([_zip(tmp_path, conv)]).fetch(_ctx(tmp_path), None)]
    assert ids == ["m-3"]


def test_timestamps_are_utc_aware(tmp_path):
    (env, _) = next(iter(ClaudeAiConnector([_zip(tmp_path)]).fetch(_ctx(tmp_path), None)))
    assert env.ts.tzinfo is not None
    assert env.ts.isoformat() == "2026-06-03T06:14:11.670509+00:00"


def test_cursor_skips_an_export_already_read_whole(tmp_path):
    c = ClaudeAiConnector([_zip(tmp_path)])
    ctx = _ctx(tmp_path)
    last = None
    for _, cur in c.fetch(ctx, None):
        last = cur
    assert list(c.fetch(ctx, last)) == []


def test_cursor_picks_up_a_newly_added_export(tmp_path):
    ctx = _ctx(tmp_path)
    first = _zip(tmp_path, name="conversations-000.zip")
    last = None
    for _, cur in ClaudeAiConnector([first]).fetch(ctx, None):
        last = cur
    second = _zip(tmp_path, [_conv("conv-9", [_msg("m-9", "human", "later export")])],
                  name="conversations-001.zip")
    texts = [e.payload["text"]
             for e, _ in ClaudeAiConnector([first, second]).fetch(ctx, last)]
    assert texts == ["later export"]


def test_cursor_is_not_advanced_past_a_file_that_failed_midway(tmp_path):
    """A crash inside one export must leave that export re-scannable."""
    broken = tmp_path / "conversations-000.zip"
    _export(broken, DEFAULT)
    ctx = _ctx(tmp_path)
    cursors = []
    for _, cur in ClaudeAiConnector([broken]).fetch(ctx, None):
        cursors.append(cur)
    # Only the final message of the export carries the "this file is done"
    # marker; abandoning the run earlier leaves the file unrecorded.
    assert json.loads(cursors[0]) == {}
    assert json.loads(cursors[-1]) != {}


def test_normalizes_into_a_subject_turn_threaded_by_conversation(tmp_path):
    (env, _) = next(iter(ClaudeAiConnector([_zip(tmp_path)]).fetch(_ctx(tmp_path), None)))
    turn = normalize(env)
    assert turn is not None
    assert turn.thread_id == "conv-1"
    assert turn.is_subject is True
    assert turn.author_id == "alice"
    assert turn.text == "why does this test hang?"


def test_source_is_in_the_corpus_build_sources():
    from persona_twin.corpus.build import SOURCES
    assert "claude_ai" in SOURCES
