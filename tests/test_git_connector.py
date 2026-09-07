import json
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

def test_identical_messages_get_distinct_source_ids(tmp_path):
    # Two commits with byte-identical messages must still get distinct
    # source_ids: the connector uses the commit SHA, not a hash of the
    # message text, so this is expected to pass trivially — it pins the
    # property against a future change to the id scheme (see the prior
    # connector's collision bug, which hashed message text and silently
    # dropped 4.4% of messages at ingest).
    repo = _repo(tmp_path, ["wip", "wip"])
    c = GitConnector([repo], ["alice@example.com"])
    ids = [e.source_id for e, _ in c.fetch(_ctx(tmp_path), None)]
    assert len(ids) == 2
    assert len(set(ids)) == 2

def test_pruned_cursor_falls_back_to_full_log(tmp_path):
    # A stored cursor SHA that no longer exists in the repo (e.g. pruned
    # by a hard rebase) must not silently look like "no new commits
    # forever" -- the connector must retry with a full log and recover
    # the repo's commits.
    repo = _repo(tmp_path, ["one", "two"])
    c = GitConnector([repo], ["alice@example.com"])
    fabricated_sha = "a" * 40
    cursor = json.dumps({str(repo): fabricated_sha})
    msgs = [e.payload["text"] for e, _ in c.fetch(_ctx(tmp_path), cursor)]
    assert "one" in msgs and "two" in msgs

def test_non_git_directory_is_skipped_without_raising(tmp_path):
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    good_repo = _repo(tmp_path, ["only commit"])
    c = GitConnector([not_a_repo, good_repo], ["alice@example.com"])
    msgs = [e.payload["text"] for e, _ in c.fetch(_ctx(tmp_path), None)]
    assert msgs == ["only commit"]
