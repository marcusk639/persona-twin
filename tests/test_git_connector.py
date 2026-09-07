import subprocess
from persona_twin.config import SubjectConfig
from persona_twin.paths import SubjectPaths
from persona_twin.connectors.base import SubjectContext
from persona_twin.connectors.git_repos import GitConnector

def _repo(tmp_path, msgs, email="alice@example.com"):
    r = tmp_path / "repo"; r.mkdir()
    run = lambda *a: subprocess.run(a, cwd=r, check=True, capture_output=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", email)
    run("git", "config", "user.name", "Alice")
    for i, m in enumerate(msgs):
        (r / f"f{i}.txt").write_text(str(i))
        run("git", "add", "-A")
        run("git", "commit", "-q", "-m", m)
    return r

def _ctx(tmp_path):
    p = SubjectPaths("alice", tmp_path); p.ensure()
    return SubjectContext(config=SubjectConfig(subject_id="alice", display_name="A"), paths=p)

def test_extracts_commit_messages(tmp_path):
    repo = _repo(tmp_path, ["feat: add thing", "fix: broken thing"])
    c = GitConnector([repo], ["alice@example.com"])
    msgs = [e.payload["text"] for e, _ in c.fetch(_ctx(tmp_path), None)]
    assert "feat: add thing" in msgs and "fix: broken thing" in msgs

def test_filters_by_author_email(tmp_path):
    repo = _repo(tmp_path, ["by someone else"], email="bob@example.com")
    c = GitConnector([repo], ["alice@example.com"])
    assert list(c.fetch(_ctx(tmp_path), None)) == []

def test_resume_returns_nothing_new(tmp_path):
    repo = _repo(tmp_path, ["one", "two"])
    c = GitConnector([repo], ["alice@example.com"])
    ctx = _ctx(tmp_path)
    cursor = None
    for _, cur in c.fetch(ctx, None):
        cursor = cur
    assert list(c.fetch(ctx, cursor)) == []
