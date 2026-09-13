import json
from pathlib import Path

import pytest

from persona_twin.config import SubjectConfig
from persona_twin.connectors.base import SubjectContext
from persona_twin.connectors.perplexity import PerplexityConnector
from persona_twin.normalize.turns import normalize
from persona_twin.paths import SubjectPaths
from persona_twin.vault import VaultWriter


def _ctx(tmp_path):
    paths = SubjectPaths("alice", tmp_path)
    paths.ensure()
    return SubjectContext(
        config=SubjectConfig(subject_id="alice", display_name="A"), paths=paths)


def _entry(uuid, query, *, answer="an answer", status="COMPLETED",
           mode="pro", created_at="2026-08-30T16:34:45.680477Z"):
    return {"entry_uuid": uuid, "query": query, "answer": answer,
            "created_at": created_at, "label": "None",
            "query_status": status, "engine_mode": mode}


def _conv(uuid, entries, title="a thread"):
    return {"context_uuid": uuid, "context_title": title,
            "created_at": "2026-08-30T16:34:45.629681Z",
            "updated_at": "2026-08-30T19:10:32.460693Z",
            "mode": "COPILOT", "collection_uuid": "None", "entries": entries}


def _export(path: Path, conversations: list[dict]) -> Path:
    path.write_text(json.dumps({"conversations": conversations}))
    return path


DEFAULT = [
    _conv("ctx-1", [_entry("e-1", "why does this test hang?"),
                    _entry("e-2", "and how do I fix it")]),
    _conv("ctx-2", [_entry("e-3", "draft the release note")]),
]


def _json(tmp_path, conversations=DEFAULT, name="conversations-1.json"):
    return _export(tmp_path / name, conversations)


def test_extracts_queries_only(tmp_path):
    c = PerplexityConnector([_json(tmp_path)])
    texts = [e.payload["text"] for e, _ in c.fetch(_ctx(tmp_path), None)]
    assert texts == ["why does this test hang?", "and how do I fix it",
                     "draft the release note"]


def test_the_answer_never_reaches_the_payload(tmp_path):
    conv = [_conv("ctx-1", [_entry("e-1", "short question",
                                   answer="a very long model reply " * 200)])]
    (env, _), = PerplexityConnector([_json(tmp_path, conv)]).fetch(_ctx(tmp_path), None)
    assert "a very long model reply" not in json.dumps(env.payload)


def test_missing_export_raises_rather_than_yielding_nothing(tmp_path):
    c = PerplexityConnector([tmp_path / "not-there.json"])
    with pytest.raises(FileNotFoundError):
        list(c.fetch(_ctx(tmp_path), None))


def test_source_id_is_the_entry_uuid_and_stable_across_runs(tmp_path):
    c = PerplexityConnector([_json(tmp_path)])
    first = [e.source_id for e, _ in c.fetch(_ctx(tmp_path), None)]
    second = [e.source_id for e, _ in c.fetch(_ctx(tmp_path), None)]
    assert first == ["e-1", "e-2", "e-3"] == second


def test_entry_uuids_survive_a_re_export_under_a_new_filename(tmp_path):
    ctx = _ctx(tmp_path)
    writer = VaultWriter(ctx.paths)
    first = _json(tmp_path, name="conversations-1.json")
    for env, _ in PerplexityConnector([first]).fetch(ctx, None):
        assert writer.write(env)
    later = _json(tmp_path, DEFAULT + [_conv("ctx-3", [_entry("e-9", "new one")])],
                  name="conversations-2.json")
    wrote = [writer.write(env)
             for env, _ in PerplexityConnector([later]).fetch(ctx, None)]
    assert wrote == [False, False, False, True]


def test_skips_empty_queries(tmp_path):
    conv = [_conv("ctx-1", [_entry("e-1", ""), _entry("e-2", "   "),
                            _entry("e-3", "real question")])]
    ids = [e.source_id
           for e, _ in PerplexityConnector([_json(tmp_path, conv)]).fetch(_ctx(tmp_path), None)]
    assert ids == ["e-3"]


def test_keeps_blocked_and_failed_queries_and_records_the_status(tmp_path):
    """A blocked query is still something the subject typed."""
    conv = [_conv("ctx-1", [_entry("e-1", "a blocked question", status="BLOCKED"),
                            _entry("e-2", "a failed one", status="FAILED")])]
    got = [(e.source_id, e.payload["query_status"])
           for e, _ in PerplexityConnector([_json(tmp_path, conv)]).fetch(_ctx(tmp_path), None)]
    assert got == [("e-1", "BLOCKED"), ("e-2", "FAILED")]


def test_unfenced_markdown_document_is_flagged_pasted(tmp_path):
    """The dominant paste shape here is a raw markdown doc, not a fenced block."""
    doc = "# Architecture\n\n" + ("A paragraph of design notes.\n" * 400)
    assert "```" not in doc and len(doc) > 8000
    conv = [_conv("ctx-1", [_entry("e-1", doc), _entry("e-2", "a normal question")])]
    flags = [e.payload["looks_pasted"]
             for e, _ in PerplexityConnector([_json(tmp_path, conv)]).fetch(_ctx(tmp_path), None)]
    assert flags == [True, False]


def test_fenced_paste_is_flagged(tmp_path):
    conv = [_conv("ctx-1", [_entry("e-1", "look:\n```\n" + "code\n" * 2000 + "```\nwhy?")])]
    (env, _), = PerplexityConnector([_json(tmp_path, conv)]).fetch(_ctx(tmp_path), None)
    assert env.payload["looks_pasted"] is True


def test_timestamps_are_utc_aware(tmp_path):
    (env, _) = next(iter(PerplexityConnector([_json(tmp_path)]).fetch(_ctx(tmp_path), None)))
    assert env.ts.tzinfo is not None
    assert env.ts.isoformat() == "2026-08-30T16:34:45.680477+00:00"


def test_cursor_skips_an_export_already_read_whole(tmp_path):
    c = PerplexityConnector([_json(tmp_path)])
    ctx = _ctx(tmp_path)
    last = None
    for _, cur in c.fetch(ctx, None):
        last = cur
    assert list(c.fetch(ctx, last)) == []


def test_cursor_is_not_advanced_past_a_file_that_failed_midway(tmp_path):
    ctx = _ctx(tmp_path)
    cursors = [cur for _, cur in PerplexityConnector([_json(tmp_path)]).fetch(ctx, None)]
    assert json.loads(cursors[0]) == {}
    assert json.loads(cursors[-1]) != {}


def test_normalizes_into_a_subject_turn_threaded_by_context(tmp_path):
    (env, _) = next(iter(PerplexityConnector([_json(tmp_path)]).fetch(_ctx(tmp_path), None)))
    turn = normalize(env)
    assert turn is not None
    assert turn.thread_id == "ctx-1"
    assert turn.is_subject is True
    assert turn.author_id == "alice"
    assert turn.text == "why does this test hang?"


def test_source_is_in_the_corpus_build_sources():
    from persona_twin.corpus.build import SOURCES
    assert "perplexity" in SOURCES
