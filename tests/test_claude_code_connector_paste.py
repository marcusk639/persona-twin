"""Fix round 1 (§4.4 M1 follow-up): paste-contamination detection.

A 520 MB real-corpus run showed 9.04% extraction against a predicted 1.46%
because pasted file/log/stack-trace content rides inside genuine `type ==
"user"` turns. These tests use a separate fixture directory
(tests/fixtures_paste/) rather than extending tests/fixtures/transcript.jsonl,
because ClaudeCodeConnector._files() recursively globs its root directory —
adding paste-scenario lines under tests/fixtures/ would leak into the
existing exact-match assertions in test_claude_code_connector.py.
"""
from pathlib import Path
from persona_twin.config import SubjectConfig
from persona_twin.paths import SubjectPaths
from persona_twin.connectors.base import SubjectContext
from persona_twin.connectors.claude_code import ClaudeCodeConnector

FIX_PASTE = Path(__file__).parent / "fixtures_paste"


def _ctx(tmp_path):
    paths = SubjectPaths("alice", tmp_path)
    paths.ensure()
    return SubjectContext(
        config=SubjectConfig(subject_id="alice", display_name="A"), paths=paths)


def _envelopes(tmp_path):
    c = ClaudeCodeConnector(roots=[FIX_PASTE])
    return [e for e, _ in c.fetch(_ctx(tmp_path), None)]


def test_fenced_code_block_is_stripped_from_text(tmp_path):
    envs = _envelopes(tmp_path)
    entry1 = envs[0]
    assert "here's the fix" in entry1.payload["text"]
    assert "thanks" in entry1.payload["text"]
    assert "def foo" not in entry1.payload["text"]
    assert entry1.payload["raw_chars"] > entry1.payload["prose_chars"]


def test_mostly_fenced_message_is_flagged_pasted(tmp_path):
    envs = _envelopes(tmp_path)
    entry2 = envs[1]
    assert entry2.payload["looks_pasted"] is True


def test_long_unfenced_message_is_flagged_pasted(tmp_path):
    envs = _envelopes(tmp_path)
    entry3 = envs[2]
    assert entry3.payload["prose_chars"] > 8000
    assert entry3.payload["looks_pasted"] is True


def test_short_ordinary_message_is_not_flagged(tmp_path):
    envs = _envelopes(tmp_path)
    entry4 = envs[3]
    assert entry4.payload["looks_pasted"] is False
    assert entry4.payload["raw_chars"] == entry4.payload["prose_chars"]


def test_source_ids_stable_across_runs_with_paste_content(tmp_path):
    ctx = _ctx(tmp_path)
    c = ClaudeCodeConnector(roots=[FIX_PASTE])
    first = [e.source_id for e, _ in c.fetch(ctx, None)]
    second = [e.source_id for e, _ in c.fetch(ctx, None)]
    assert first == second and len(set(first)) == len(first)


def test_nothing_is_dropped_for_being_pasted(tmp_path):
    # All 4 fixture messages must be emitted, including the flagged ones —
    # filtering out pasted content is the corpus builder's job, not the
    # connector's (§4.4 M1 follow-up: silent dropping would be unrecoverable).
    envs = _envelopes(tmp_path)
    assert len(envs) == 4
