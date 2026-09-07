from __future__ import annotations
import json, subprocess
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

    def _log(self, repo: Path, since_sha: str | None) -> list[tuple[str, str, str]]:
        rng = [f"{since_sha}..HEAD"] if since_sha else []
        cmd = ["git", "log", f"--format=%H{_SEP}%aI{_SEP}%ae{_SEP}%B%x00", *rng]
        try:
            out = subprocess.run(cmd, cwd=repo, check=True,
                                 capture_output=True, text=True).stdout
        except subprocess.CalledProcessError:
            return []
        rows = []
        for chunk in out.split("\x00"):
            if not chunk.strip():
                continue
            sha, iso, email, body = chunk.strip().split(_SEP, 3)
            if email.lower() in self.emails:
                rows.append((sha, iso, body.strip()))
        return list(reversed(rows))

    def fetch(self, ctx: SubjectContext,
              cursor: str | None) -> Iterator[tuple[RawEnvelope, str]]:
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
