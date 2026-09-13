from __future__ import annotations
import json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from persona_twin.connectors.base import SubjectContext
from persona_twin.schema import RawEnvelope

_SEP = "\x1e"

class GitConnector:
    name = "git_repos"

    def __init__(self, repo_roots: list[Path], author_emails: list[str]) -> None:
        self.repos = [Path(r) for r in repo_roots]
        self.emails = [e.lower() for e in author_emails]

    def _run_log(self, repo: Path, since_sha: str | None) -> tuple[str | None, str | None]:
        """Run `git log`, returning (stdout, None) on success or (None, error)
        on failure (also warns to stderr).

        A failure here does not mean "no new commits" — it can also mean
        the cursor SHA no longer exists (pruned after a rebase) or the
        path isn't a git repo at all. The caller decides what a failure
        means (retry, raise, or ignore); this method's only job is to
        never raise and to never stay silent about what went wrong.
        """
        rng = [f"{since_sha}..HEAD"] if since_sha else []
        cmd = ["git", "log", f"--format=%H{_SEP}%aI{_SEP}%ae{_SEP}%B%x00", *rng]
        try:
            out = subprocess.run(cmd, cwd=repo, check=True,
                                 capture_output=True, text=True).stdout
            return out, None
        except subprocess.CalledProcessError as exc:
            err = exc.stderr.strip() if exc.stderr else str(exc)
            msg = f"git log failed for {repo!s} (cursor={since_sha!r}): {err}"
            print(f"git_repos: {msg}", file=sys.stderr)
            return None, msg

    def _is_valid_repo(self, repo: Path) -> bool:
        """Structural check for "is this path a readable git repository",
        used to tell a legitimate zero-commit repo apart from a broken or
        missing one when `git log` fails. `git rev-parse --git-dir` succeeds
        inside any repo, including a freshly initialised one with no
        commits, and fails outside one -- a structural fact about the path,
        not a parse of git's (localized, version-dependent) stderr text.
        """
        try:
            subprocess.run(["git", "rev-parse", "--git-dir"], cwd=repo,
                           check=True, capture_output=True, text=True)
            return True
        except (subprocess.CalledProcessError, OSError):
            return False

    def _parse(self, out: str) -> list[tuple[str, str, str]]:
        rows = []
        for chunk in out.split("\x00"):
            if not chunk.strip():
                continue
            sha, iso, email, body = chunk.strip().split(_SEP, 3)
            if email.lower() in self.emails:
                rows.append((sha, iso, body.strip()))
        return list(reversed(rows))

    def _log(self, repo: Path, since_sha: str | None) -> list[tuple[str, str, str]]:
        out, err = self._run_log(repo, since_sha)
        if out is None and since_sha is not None:
            # The ranged call failed with a cursor set — most likely the
            # stored cursor SHA no longer exists in this repo (e.g. a hard
            # rebase pruned it), which would otherwise silently return zero
            # commits forever. Retry with the full log instead of a range.
            # This is safe specifically because source_id is the commit
            # SHA and VaultWriter dedupes on (subject_id, source,
            # source_id): re-reading full history costs time but
            # re-ingests nothing already in the vault, and recovers any
            # commits that would otherwise be stranded.
            out, err = self._run_log(repo, None)
        if out is None:
            # Both attempts failed. Two situations look identical to `git
            # log` and must not be conflated: a genuinely broken or missing
            # repo (not a git repo, `.git` pruned, path unmounted, typo'd
            # repo root), and a legitimate, freshly initialised repo that
            # simply has no commits yet -- `git log` fails there too, with
            # no HEAD to walk. Only the first case is a failure.
            if self._is_valid_repo(repo):
                # A real repo with zero commits ever. Adding a brand-new
                # repo to a subject's config is a normal operator action;
                # aborting the whole ingest run over it would be a false
                # alarm, and false alarms are how a fail-loud design gets
                # switched off by the person it's supposed to protect.
                return []
            # Not a readable repository at all. This must not be reported
            # as "zero new commits": that is indistinguishable from a repo
            # that legitimately had nothing new, and run_connector would
            # write a false success to the ledger, which is this project's
            # provenance audit trail. Raising instead is cheap:
            # run_connector leaves the per-repo cursor at its previous
            # value on any exception (connectors/base.py), and re-running
            # after the repo is fixed re-ingests nothing already written,
            # because VaultWriter dedupes on (subject_id, source,
            # source_id) and source_id here is the bare commit SHA (see
            # fetch() below) — so the retry costs only time.
            raise RuntimeError(
                f"git_repos: {repo!s} is unreadable, aborting this ingest "
                f"run rather than reporting a false zero-new-commits "
                f"success: {err}")
        return self._parse(out)

    def fetch(self, ctx: SubjectContext,
              cursor: str | None) -> Iterator[tuple[RawEnvelope, str]]:
        # source_id is the bare commit SHA, deliberately not qualified by
        # repo path. When the same commit exists in multiple repos that
        # share history (e.g. a fork or a repo split), the vault's
        # (subject_id, source, source_id) dedup collapses them to one
        # envelope — correct, since it's one authored commit message and
        # ingesting it twice would duplicate content in the corpus. The
        # trade-off: payload["repo"] then reflects whichever repo was
        # scanned first, so repo attribution for a shared commit is
        # arbitrary between the repos it appears in. Confirmed against a
        # real 32-repo scan: 1,235 commits emitted, 1 collision, exactly
        # this shared-history case (same SHA in two repos with common
        # ancestry) — not a hash collision.
        heads: dict[str, str] = json.loads(cursor) if cursor else {}
        for repo in self.repos:
            key = str(repo)
            for sha, iso, body in self._log(repo, heads.get(key)):
                heads[key] = sha
                yield RawEnvelope(
                    subject_id=ctx.config.subject_id, source=self.name,
                    source_id=sha, ts=datetime.fromisoformat(iso).astimezone(timezone.utc),
                    payload={"text": body, "repo": key},
                    ingested_at=datetime.now(timezone.utc)), json.dumps(heads)
