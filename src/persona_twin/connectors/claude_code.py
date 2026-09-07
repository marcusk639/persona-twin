from __future__ import annotations
import hashlib, json, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from persona_twin.connectors.base import SubjectContext
from persona_twin.schema import RawEnvelope

_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)
_COMMAND = re.compile(r"<(command-name|command-message|local-command-stdout)>.*?</\1>", re.S)

def _user_text(entry: dict) -> str | None:
    """Return genuine typed prose from a user entry, or None (spec §4.4 M1)."""
    if entry.get("type") != "user":
        return None
    content = (entry.get("message") or {}).get("content")
    if isinstance(content, str):
        parts = [content]
    elif isinstance(content, list):
        parts = [p.get("text", "") for p in content
                 if isinstance(p, dict) and p.get("type") == "text"]
    else:
        return None
    text = "\n".join(parts)
    text = _COMMAND.sub("", _REMINDER.sub("", text)).strip()
    return text or None

class ClaudeCodeConnector:
    name = "claude_code"

    def __init__(self, roots: list[Path]) -> None:
        self.roots = [Path(r) for r in roots]

    def _files(self) -> list[Path]:
        out: list[Path] = []
        for root in self.roots:
            out.extend(sorted(root.rglob("*.jsonl")))
        return out

    def fetch(self, ctx: SubjectContext,
              cursor: str | None) -> Iterator[tuple[RawEnvelope, str]]:
        offsets: dict[str, int] = json.loads(cursor) if cursor else {}
        for path in self._files():
            key = str(path)
            start = offsets.get(key, 0)
            size = path.stat().st_size
            if start >= size:
                continue
            with path.open("rb") as fh:
                fh.seek(start)
                for raw in fh:
                    start += len(raw)
                    try:
                        entry = json.loads(raw.decode("utf-8", errors="ignore"))
                    except json.JSONDecodeError:
                        continue
                    text = _user_text(entry)
                    if not text:
                        continue
                    digest = hashlib.sha256(
                        f"{key}:{text}".encode("utf-8")).hexdigest()[:24]
                    ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                    offsets[key] = start
                    yield RawEnvelope(
                        subject_id=ctx.config.subject_id, source=self.name,
                        source_id=digest, ts=ts,
                        payload={"text": text, "file": key},
                        ingested_at=datetime.now(timezone.utc)), json.dumps(offsets)
            offsets[key] = size
