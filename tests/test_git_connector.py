import json
import subprocess
import pytest
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

def test_broken_repo_raises_instead_of_reporting_false_zero(tmp_path):
    # A directory that isn't a git repo at all (also covers ".git" pruned,
    # path unmounted, typo'd repo root -- all surface as the same
    # CalledProcessError). This must abort the run, not be silently
    # skipped: a silent skip is indistinguishable, once it reaches the
    # ledger, from "this repo legitimately had zero new commits" -- and a
    # false success in the ledger is unrecoverable after the fact.
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    good_repo = _repo(tmp_path, ["only commit"])
    c = GitConnector([not_a_repo, good_repo], ["alice@example.com"])
    with pytest.raises(RuntimeError) as exc_info:
        list(c.fetch(_ctx(tmp_path), None))
    assert str(not_a_repo) in str(exc_info.value)

def test_legitimately_empty_repo_still_succeeds_with_zero_envelopes(tmp_path):
    # The case the broken-repo raise above must NOT be conflated with: a
    # real, healthy repo that simply has no new commits since the cursor
    # (git log succeeds with empty stdout, not a CalledProcessError).
    repo = _repo(tmp_path, ["one", "two"])
    c = GitConnector([repo], ["alice@example.com"])
    ctx = _ctx(tmp_path)
    cursor = None
    for _, cur in c.fetch(ctx, None):
        cursor = cur
    assert list(c.fetch(ctx, cursor)) == []

def test_fresh_repo_with_no_commits_yet_returns_empty_without_raising(tmp_path):
    # A third distinct situation, which must not collapse into either of the
    # other two: `git log` fails here exactly as it does for a broken repo
    # (no HEAD to walk), but the repo itself is real and readable -- adding
    # a brand-new repo to a subject's config is a normal operator action,
    # and this must not abort the ingest run.
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=fresh, check=True, capture_output=True)
    c = GitConnector([fresh], ["alice@example.com"])
    assert list(c.fetch(_ctx(tmp_path), None)) == []
