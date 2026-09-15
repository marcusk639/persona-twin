import json
from pathlib import Path

import pytest

from persona_twin.config import SubjectConfig
from persona_twin.connectors.base import SubjectContext
from persona_twin.connectors.chatgpt import ChatGPTConnector
from persona_twin.normalize.turns import normalize
from persona_twin.paths import SubjectPaths
from persona_twin.vault import VaultWriter


def _ctx(tmp_path):
    paths = SubjectPaths("alice", tmp_path)
    paths.ensure()
    return SubjectContext(
        config=SubjectConfig(subject_id="alice", display_name="A"), paths=paths)


def _node(mid, role, text, *, parent, create_time=1700000000.0, ct="text",
          model="gpt-4o", sysmsg=False, parts=None):
    meta = {"model_slug": model}
    if sysmsg:
        meta["is_user_system_message"] = True
    return {"id": mid, "parent": parent,
            "message": {"id": mid, "author": {"role": role, "name": None},
                        "content": {"content_type": ct,
                                    "parts": [text] if parts is None else parts},
                        "create_time": create_time, "metadata": meta}}


def _conv(conv_id, turns, *, title="a thread", create_time=1699000000.0,
          extra=(), current=None):
    """Build a conversation whose `turns` form one linear parent chain."""
    mapping = {"root": {"id": "root", "parent": None, "message": None}}
    parent, last = "root", None
    for i, (mid, role, text) in enumerate(turns):
        mapping[mid] = _node(mid, role, text, parent=parent,
                             create_time=create_time + i)
        parent, last = mid, mid
    for node in extra:
        mapping[node["id"]] = node
    return {"conversation_id": conv_id, "id": conv_id, "title": title,
            "create_time": create_time, "update_time": create_time + 100,
            "current_node": current or last, "mapping": mapping,
            "default_model_slug": "gpt-4o"}


DEFAULT = [
    _conv("conv-1", [("u-1", "user", "why does this test hang?"),
                     ("a-1", "assistant", "because of the lock"),
                     ("u-2", "user", "and how do I fix it")]),
    _conv("conv-2", [("u-3", "user", "draft the release note")]),
]


def _export(tmp_path, conversations=DEFAULT, name="conversations-000.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(conversations))
    return path


def _texts(tmp_path, conversations=DEFAULT):
    c = ChatGPTConnector([_export(tmp_path, conversations)])
    return [e.payload["text"] for e, _ in c.fetch(_ctx(tmp_path), None)]


def test_extracts_user_turns_in_conversation_order(tmp_path):
    assert _texts(tmp_path) == ["why does this test hang?",
                                "and how do I fix it",
                                "draft the release note"]


def test_the_assistant_reply_never_reaches_the_payload(tmp_path):
    conv = [_conv("conv-1", [("u-1", "user", "short question"),
                             ("a-1", "assistant", "a very long model reply " * 200)])]
    (env, _), = ChatGPTConnector([_export(tmp_path, conv)]).fetch(_ctx(tmp_path), None)
    assert "a very long model reply" not in json.dumps(env.payload)


def test_abandoned_regeneration_branch_is_excluded(tmp_path):
    """An edited prompt leaves a sibling node that `current_node` does not reach."""
    conv = _conv("conv-1", [("u-1", "user", "the kept phrasing"),
                            ("a-1", "assistant", "ok")])
    conv["mapping"]["u-1b"] = _node("u-1b", "user", "the retracted phrasing",
                                    parent="root")
    ids = [e.source_id for e, _ in
           ChatGPTConnector([_export(tmp_path, [conv])]).fetch(_ctx(tmp_path), None)]
    assert ids == ["u-1"]


def test_a_parent_cycle_terminates_instead_of_hanging(tmp_path):
    conv = _conv("conv-1", [("u-1", "user", "a question")])
    conv["mapping"]["root"]["parent"] = "u-1"     # root -> u-1 -> root
    ids = [e.source_id for e, _ in
           ChatGPTConnector([_export(tmp_path, [conv])]).fetch(_ctx(tmp_path), None)]
    assert ids == ["u-1"]


def test_custom_instructions_are_excluded(tmp_path):
    """`is_user_system_message` is account settings, not the subject writing."""
    conv = _conv("conv-1", [("u-1", "user", "a real question")])
    sysnode = _node("u-0", "user", "You are talking to an accountant. Be terse.",
                    parent="root", sysmsg=True)
    conv["mapping"]["u-1"]["parent"] = "u-0"
    conv["mapping"]["u-0"] = sysnode
    ids = [e.source_id for e, _ in
           ChatGPTConnector([_export(tmp_path, [conv])]).fetch(_ctx(tmp_path), None)]
    assert ids == ["u-1"]


def test_multimodal_caption_is_kept_and_images_are_counted(tmp_path):
    conv = _conv("conv-1", [("u-1", "user", "")])
    conv["mapping"]["u-1"]["message"]["content"] = {
        "content_type": "multimodal_text",
        "parts": [{"content_type": "image_asset_pointer", "asset_pointer": "file-a"},
                  {"content_type": "image_asset_pointer", "asset_pointer": "file-b"},
                  "what is wrong with this chart?"]}
    (env, _), = ChatGPTConnector([_export(tmp_path, [conv])]).fetch(_ctx(tmp_path), None)
    assert env.payload["text"] == "what is wrong with this chart?"
    assert env.payload["image_count"] == 2


def test_an_image_with_no_caption_is_skipped(tmp_path):
    conv = _conv("conv-1", [("u-1", "user", ""), ("u-2", "user", "real text")])
    conv["mapping"]["u-1"]["message"]["content"] = {
        "content_type": "multimodal_text",
        "parts": [{"content_type": "image_asset_pointer", "asset_pointer": "file-a"}]}
    ids = [e.source_id for e, _ in
           ChatGPTConnector([_export(tmp_path, [conv])]).fetch(_ctx(tmp_path), None)]
    assert ids == ["u-2"]


def test_skips_empty_and_whitespace_turns(tmp_path):
    conv = [_conv("conv-1", [("u-1", "user", ""), ("u-2", "user", "   "),
                             ("u-3", "user", "real question")])]
    ids = [e.source_id for e, _ in
           ChatGPTConnector([_export(tmp_path, conv)]).fetch(_ctx(tmp_path), None)]
    assert ids == ["u-3"]


def test_non_text_content_types_are_skipped(tmp_path):
    """Tool calls and code interpreter output are not the subject's writing."""
    conv = _conv("conv-1", [("u-1", "user", "a question"),
                            ("u-2", "user", "print(1)")])
    conv["mapping"]["u-2"]["message"]["content"]["content_type"] = "code"
    ids = [e.source_id for e, _ in
           ChatGPTConnector([_export(tmp_path, [conv])]).fetch(_ctx(tmp_path), None)]
    assert ids == ["u-1"]


def test_source_id_is_the_message_id_and_stable_across_runs(tmp_path):
    c = ChatGPTConnector([_export(tmp_path)])
    first = [e.source_id for e, _ in c.fetch(_ctx(tmp_path), None)]
    second = [e.source_id for e, _ in c.fetch(_ctx(tmp_path), None)]
    assert first == ["u-1", "u-2", "u-3"] == second


def test_message_ids_survive_a_re_export_under_a_new_filename(tmp_path):
    ctx = _ctx(tmp_path)
    writer = VaultWriter(ctx.paths)
    first = _export(tmp_path, name="conversations-000.json")
    for env, _ in ChatGPTConnector([first]).fetch(ctx, None):
        assert writer.write(env)
    later = _export(tmp_path, DEFAULT + [_conv("conv-3", [("u-9", "user", "new one")])],
                    name="conversations-001.json")
    wrote = [writer.write(env)
             for env, _ in ChatGPTConnector([later]).fetch(ctx, None)]
    assert wrote == [False, False, False, True]


def test_payload_records_the_model_slug_without_filtering_on_it(tmp_path):
    conv = _conv("conv-1", [("u-1", "user", "a question")])
    conv["mapping"]["u-1"]["message"]["metadata"]["model_slug"] = "o3-mini"
    (env, _), = ChatGPTConnector([_export(tmp_path, [conv])]).fetch(_ctx(tmp_path), None)
    assert env.payload["model_slug"] == "o3-mini"


def test_unfenced_markdown_document_is_flagged_pasted(tmp_path):
    doc = "# Architecture\n\n" + ("A paragraph of design notes.\n" * 400)
    assert "```" not in doc and len(doc) > 8000
    conv = [_conv("conv-1", [("u-1", "user", doc), ("u-2", "user", "a normal question")])]
    flags = [e.payload["looks_pasted"]
             for e, _ in ChatGPTConnector([_export(tmp_path, conv)]).fetch(_ctx(tmp_path), None)]
    assert flags == [True, False]


def test_fenced_paste_is_flagged(tmp_path):
    conv = [_conv("conv-1", [("u-1", "user", "look:\n```\n" + "code\n" * 2000 + "```\nwhy?")])]
    (env, _), = ChatGPTConnector([_export(tmp_path, conv)]).fetch(_ctx(tmp_path), None)
    assert env.payload["looks_pasted"] is True


def test_timestamps_are_utc_aware_from_the_message_epoch(tmp_path):
    (env, _) = next(iter(ChatGPTConnector([_export(tmp_path)]).fetch(_ctx(tmp_path), None)))
    assert env.ts.tzinfo is not None
    assert env.ts.isoformat() == "2023-11-03T08:26:40+00:00"


def test_a_message_without_a_timestamp_falls_back_to_the_conversation(tmp_path):
    """Never `now()` -- a shared sentinel would merge unrelated records.

    The conversation's own create_time is deliberately far from the default
    fixture epoch, so this assertion fails if the fallback reaches for
    anything other than the conversation it belongs to.
    """
    conv = _conv("conv-1", [("u-1", "user", "a question")], create_time=1690000000.0)
    conv["mapping"]["u-1"]["message"]["create_time"] = None
    (env, _), = ChatGPTConnector([_export(tmp_path, [conv])]).fetch(_ctx(tmp_path), None)
    assert env.ts.isoformat() == "2023-07-22T04:26:40+00:00"


def test_missing_export_raises_rather_than_yielding_nothing(tmp_path):
    c = ChatGPTConnector([tmp_path / "not-there.json"])
    with pytest.raises(FileNotFoundError):
        list(c.fetch(_ctx(tmp_path), None))


def test_an_export_that_is_not_a_list_raises(tmp_path):
    path = tmp_path / "conversations-000.json"
    path.write_text(json.dumps({"conversations": []}))
    with pytest.raises(ValueError):
        list(ChatGPTConnector([path]).fetch(_ctx(tmp_path), None))


def test_cursor_skips_an_export_already_read_whole(tmp_path):
    c = ChatGPTConnector([_export(tmp_path)])
    ctx = _ctx(tmp_path)
    last = None
    for _, cur in c.fetch(ctx, None):
        last = cur
    assert list(c.fetch(ctx, last)) == []


def test_cursor_is_not_advanced_past_a_file_that_failed_midway(tmp_path):
    ctx = _ctx(tmp_path)
    cursors = [cur for _, cur in ChatGPTConnector([_export(tmp_path)]).fetch(ctx, None)]
    assert json.loads(cursors[0]) == {}
    assert json.loads(cursors[-1]) != {}


def test_normalizes_into_a_subject_turn_threaded_by_conversation(tmp_path):
    (env, _) = next(iter(ChatGPTConnector([_export(tmp_path)]).fetch(_ctx(tmp_path), None)))
    turn = normalize(env)
    assert turn is not None
    assert turn.thread_id == "conv-1"
    assert turn.is_subject is True
    assert turn.author_id == "alice"
    assert turn.text == "why does this test hang?"


def test_source_is_in_the_corpus_build_sources():
    from persona_twin.corpus.build import SOURCES
    assert "chatgpt" in SOURCES


def test_a_message_without_an_id_gets_a_position_bearing_hash(tmp_path):
    """Content alone would collide on repeated short turns and lose them."""
    conv = _conv("conv-1", [("u-1", "user", "ok"), ("a-1", "assistant", "yes"),
                            ("u-2", "user", "ok")])
    for nid in ("u-1", "u-2"):
        conv["mapping"][nid]["message"]["id"] = None
    ids = [e.source_id for e, _ in
           ChatGPTConnector([_export(tmp_path, [conv])]).fetch(_ctx(tmp_path), None)]
    assert len(set(ids)) == 2, "identical text at different positions collided"
