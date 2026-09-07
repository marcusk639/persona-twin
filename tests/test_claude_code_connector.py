from pathlib import Path
from persona_twin.config import SubjectConfig
from persona_twin.paths import SubjectPaths
from persona_twin.connectors.base import SubjectContext
from persona_twin.connectors.claude_code import ClaudeCodeConnector

FIX = Path(__file__).parent / "fixtures"

def _ctx(tmp_path):
    paths = SubjectPaths("alice", tmp_path); paths.ensure()
    return SubjectContext(
        config=SubjectConfig(subject_id="alice", display_name="A"), paths=paths)

def test_extracts_only_genuine_user_text(tmp_path):
    c = ClaudeCodeConnector(roots=[FIX])
    texts = [e.payload["text"] for e, _ in c.fetch(_ctx(tmp_path), None)]
    assert texts == ["fix the failing test in auth.py", "ship it"]

def test_skips_tool_results_and_system_reminders(tmp_path):
    c = ClaudeCodeConnector(roots=[FIX])
    joined = " ".join(e.payload["text"] for e, _ in c.fetch(_ctx(tmp_path), None))
    assert "1 passed" not in joined and "ignore me" not in joined

def test_source_ids_are_stable_across_runs(tmp_path):
    c = ClaudeCodeConnector(roots=[FIX])
    first = [e.source_id for e, _ in c.fetch(_ctx(tmp_path), None)]
    second = [e.source_id for e, _ in c.fetch(_ctx(tmp_path), None)]
    assert first == second and len(set(first)) == len(first)

def test_cursor_resume_skips_already_read_bytes(tmp_path):
    c = ClaudeCodeConnector(roots=[FIX])
    ctx = _ctx(tmp_path)
    last_cursor = None
    for _, cur in c.fetch(ctx, None):
        last_cursor = cur
    assert list(c.fetch(ctx, last_cursor)) == []
