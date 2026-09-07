from __future__ import annotations
import hashlib, json, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from persona_twin.connectors.base import SubjectContext
from persona_twin.schema import RawEnvelope

_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)
_COMMAND = re.compile(r"<(command-name|command-message|local-command-stdout)>.*?</\1>", re.S)
_FENCE = re.compile(r"```.*?```", re.S)

# Paste-detection thresholds (fix round 1, §4.4 M1 follow-up): a real 520 MB
# corpus run showed 9.04% extraction against a predicted 1.46%, because a
# third of "user" messages are pasted file/log/stack-trace content rather
# than typed prose. These blocks are stripped from `text`, but the message
# is still emitted with raw/prose char counts and a `looks_pasted` flag so
# the corpus builder — not this connector — decides what to keep.
_PASTE_CHAR_THRESHOLD = 8000

def _user_text(entry: dict) -> dict | None:
    """Return genuine typed prose info from a user entry, or None (spec §4.4 M1).

    Keys: "raw" (post reminder/command strip, pre fence strip — used for
    source_id so the id stays stable if the fence heuristic changes later),
    "text" (post fence strip — the subject's prose, what downstream reads),
    "raw_chars", "prose_chars", "looks_pasted".
    """
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
    raw = _COMMAND.sub("", _REMINDER.sub("", text)).strip()
    if not raw:
        return None
    prose = _FENCE.sub("", raw).strip()
    raw_chars = len(raw)
    prose_chars = len(prose)
    looks_pasted = (prose_chars > _PASTE_CHAR_THRESHOLD
                     or (raw_chars - prose_chars) > raw_chars / 2)
    return {"raw": raw, "text": prose, "raw_chars": raw_chars,
            "prose_chars": prose_chars, "looks_pasted": looks_pasted}

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
                    info = _user_text(entry)
                    if info is None:
                        continue
                    # Hash the pre-fence-strip "raw" text, not the emitted
                    # "text": raw is the stable identity of the source
                    # message, so source_id survives future changes to the
                    # paste heuristic without creating vault duplicates.
                    digest = hashlib.sha256(
                        f"{key}:{info['raw']}".encode("utf-8")).hexdigest()[:24]
                    ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                    offsets[key] = start
                    yield RawEnvelope(
                        subject_id=ctx.config.subject_id, source=self.name,
                        source_id=digest, ts=ts,
                        payload={"text": info["text"], "file": key,
                                 "raw_chars": info["raw_chars"],
                                 "prose_chars": info["prose_chars"],
                                 "looks_pasted": info["looks_pasted"]},
                        ingested_at=datetime.now(timezone.utc)), json.dumps(offsets)
            offsets[key] = size
