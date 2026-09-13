"""Claude.ai connector (spec §5.3): the subject's chat-assistant voice.

Reads the Settings -> Privacy -> Export data archive. Only `sender == "human"`
messages are emitted -- the assistant's half is not the subject's writing, and
unlike iMessage there is no reply-pair value in keeping a model's replies.

Attachments: 82.8% of the export's human-side character volume lives in
`attachments[].extracted_content` (1.9M chars against 394K of inline prose on
the real subject-01 export) -- pasted files, logs and stack traces the subject
uploaded, not wrote. That payload is dropped here rather than ingested, so it
can never reach the corpus by a route the `looks_pasted` heuristic does not
watch: that heuristic reads inline characters only, and would have scored an
attachment-heavy message as 100% clean prose. What is recorded instead is the
*shape* of the drop (`attachment_count`, `attachment_chars`, `file_count`), so
the decision is visible downstream rather than invisible.

Note the deliberate asymmetry with `looks_pasted`: an attachment does NOT set
the flag. Once the payload is gone, what remains inline ("what breaks here?")
is genuinely typed prose and belongs in the corpus. Flagging it would discard
the subject's writing to punish an upload that was already discarded.
"""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from persona_twin.connectors.base import SubjectContext, scan_snapshot_exports
from persona_twin.schema import RawEnvelope

_FENCE = re.compile(r"```.*?```", re.S)

# Same threshold and formula as the claude_code connector, deliberately: both
# sources feed one corpus, and two different definitions of "pasted" would
# make the corpus's composition depend on which tool the subject happened to
# be using.
_PASTE_CHAR_THRESHOLD = 8000

_CONVERSATIONS = "conversations.json"


def _load(path: Path) -> list[dict]:
    """Read conversations from an export zip or a bare conversations.json."""
    if not path.exists():
        # A missing export must not look like "zero new messages" -- that is
        # indistinguishable from a fully-caught-up cursor and would silently
        # drop an entire export (same reasoning as the mbox connector).
        raise FileNotFoundError(f"claude_ai connector: export not found: {path}")
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if Path(n).name == _CONVERSATIONS]
            if not names:
                raise ValueError(
                    f"claude_ai connector: no {_CONVERSATIONS} in {path}")
            raw = z.read(names[0])
    else:
        raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"claude_ai connector: {path} is not a conversation list")
    return data


def _text(msg: dict) -> str:
    """The typed prose of a message.

    `text` and the concatenation of `content[].text` blocks agree on every one
    of the 2,069 human messages in the real export, so `text` is used and the
    blocks are the fallback for a shape change rather than the primary read.
    """
    inline = msg.get("text")
    if isinstance(inline, str) and inline.strip():
        return inline.strip()
    blocks = [b.get("text", "") for b in (msg.get("content") or [])
              if isinstance(b, dict) and b.get("type") == "text"]
    return "\n".join(blocks).strip()


def _ts(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


class ClaudeAiConnector:
    name = "claude_ai"

    def __init__(self, export_paths: list[Path]) -> None:
        self.paths = [Path(p) for p in export_paths]

    def fetch(self, ctx: SubjectContext,
              cursor: str | None) -> Iterator[tuple[RawEnvelope, str]]:
        """Whole-file snapshot cursor; see base.scan_snapshot_exports.

        Safe here because `source_id` is the server-assigned message uuid,
        which is stable across re-exports, so a re-scan writes zero new rows.
        """
        yield from scan_snapshot_exports(
            self.name, self.paths, cursor,
            lambda path: self._envelopes(ctx, path))

    def _envelopes(self, ctx: SubjectContext, path: Path) -> Iterator[RawEnvelope]:
        key = str(path)
        for conv in _load(path):
            conv_uuid = str(conv.get("uuid") or "")
            for idx, msg in enumerate(conv.get("chat_messages") or []):
                if msg.get("sender") != "human":
                    continue
                text = _text(msg)
                if not text:
                    continue
                prose = _FENCE.sub("", text).strip()
                raw_chars, prose_chars = len(text), len(prose)
                attachments = msg.get("attachments") or []
                # `uuid` is server-assigned, globally unique and stable across
                # re-exports, which makes it a better key than any content
                # hash -- content hashing collides on repeated short messages
                # ("yes", "ship it") and a positional fallback would shift
                # whenever the export regenerates. The hash below is only for
                # an export shaped unexpectedly enough to omit the uuid.
                source_id = str(msg.get("uuid") or "").strip() or hashlib.sha256(
                    f"{key}:{conv_uuid}:{idx}:{text}".encode("utf-8")
                ).hexdigest()[:24]
                yield RawEnvelope(
                    subject_id=ctx.config.subject_id, source=self.name,
                    source_id=source_id, ts=_ts(msg.get("created_at")),
                    payload={
                        "text": text,
                        "conversation_uuid": conv_uuid,
                        "raw_chars": raw_chars,
                        "prose_chars": prose_chars,
                        "looks_pasted": (
                            prose_chars > _PASTE_CHAR_THRESHOLD
                            or (raw_chars - prose_chars) > raw_chars / 2),
                        "attachment_count": len(attachments),
                        "attachment_chars": sum(
                            len(a.get("extracted_content") or "")
                            for a in attachments),
                        "file_count": len(msg.get("files") or []),
                    },
                    ingested_at=datetime.now(timezone.utc))
