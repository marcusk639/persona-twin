"""ChatGPT connector (spec §5.3): the subject's question-and-brief voice.

Reads the OpenAI account export (`conversations-*.json`). Only the subject's
`user` turns are emitted -- the assistant half is a model's writing, dropped
for the same reason the claude.ai and perplexity connectors drop it.

Unlike every other export this project reads, ChatGPT does not store a
conversation as a list. It stores a DAG: `mapping` is keyed by node id, each
node carries a `parent` pointer, and `current_node` names the leaf of the
branch the conversation actually ended on. Editing a prompt or regenerating a
reply forks the tree and leaves the abandoned nodes in `mapping` forever, so
iterating `mapping.values()` yields turns the subject retracted alongside the
ones he kept. This connector walks `current_node` back through `parent` and
emits only that chain.

Measured on the real subject-01 export: 5,870 user text turns exist in
`mapping`, of which 5,603 are on a live branch -- the 267 excluded turns
(30,672 chars, 2% of volume) are overwhelmingly edited-prompt variants, near
duplicates of a turn that IS emitted. Keeping them would put pairs of
almost-identical sentences in the corpus that no downstream dedupe catches,
because each carries a distinct node id.

Register note: this source is short-form, like perplexity and claude.ai
(median 63 chars here vs 84 and 75 there). It adds volume to a register
iMessage already dominates rather than opening a new one, and should not be
weighted as if it were prose.

Paste shape: the threshold and formula are copied verbatim from the
claude_code connector, NOT re-tuned here, because two definitions of "pasted"
would make the corpus's composition depend on which tool the subject happened
to have open. On the real export it flags 24 turns carrying 493K chars --
32% of the source's volume in 0.4% of its turns.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from persona_twin.connectors.base import SubjectContext, scan_snapshot_exports
from persona_twin.schema import RawEnvelope

_FENCE = re.compile(r"```.*?```", re.S)

# Identical to the claude_code / perplexity threshold, by design -- see module
# docstring. Changing it here without changing it there splits the corpus.
_PASTE_CHAR_THRESHOLD = 8000

# Everything else a user node can hold is a tool call, an execution result, or
# a system hint -- none of it typed by the subject.
_TEXT_TYPES = frozenset({"text", "multimodal_text"})


def _load(path: Path) -> list[dict]:
    data = json.loads(path.read_bytes())
    if not isinstance(data, list):
        raise ValueError(
            f"chatgpt connector: {path} is not a list of conversations")
    return data


def _ts(value: Any, fallback: Any) -> datetime:
    """Epoch seconds -> aware UTC, falling back to the conversation's own time.

    Deliberately never `datetime.now()`: a wall-clock default is a shared
    sentinel that lands every undated record in the same arbitrary instant,
    scrambling ordering for anything that reads `ts`. The conversation's
    create_time is a real, per-record value from the same export.
    """
    for candidate in (value, fallback):
        if isinstance(candidate, (int, float)):
            return datetime.fromtimestamp(float(candidate), tz=timezone.utc)
    return datetime.fromtimestamp(0, tz=timezone.utc)


def _live_chain(conv: dict) -> list[str]:
    """Node ids from `current_node` back to the root, in conversation order.

    The `seen` set is not defensive decoration: a malformed or hand-edited
    export with a parent cycle would otherwise spin forever inside a
    generator, which presents as a hung ingest with no error to read.
    """
    mapping = conv.get("mapping") or {}
    node, seen, chain = conv.get("current_node"), set(), []
    while node and node in mapping:
        if node in seen:
            break
        seen.add(node)
        chain.append(node)
        node = mapping[node].get("parent")
    chain.reverse()
    return chain


def _text_and_images(content: dict) -> tuple[str, int]:
    """Join the typed parts; count the attachments without reading them.

    A multimodal turn is a caption plus image pointers. The caption is the
    subject's writing and is kept; the pointers are recorded only as a count,
    so the corpus notes that context was attached without this connector
    taking on binary assets.
    """
    parts = content.get("parts") or []
    text = "".join(p for p in parts if isinstance(p, str))
    images = sum(1 for p in parts
                 if isinstance(p, dict)
                 and p.get("content_type") == "image_asset_pointer")
    return text, images


class ChatGPTConnector:
    name = "chatgpt"

    def __init__(self, export_paths: list[Path]) -> None:
        self.paths = [Path(p) for p in export_paths]

    def fetch(self, ctx: SubjectContext,
              cursor: str | None) -> Iterator[tuple[RawEnvelope, str]]:
        """Whole-file snapshot cursor; see base.scan_snapshot_exports.

        Safe here because `source_id` is the server-assigned message uuid,
        verified unique across all 5,627 emitted turns of the real export and
        stable across re-exports, so a re-scan writes zero new rows.
        """
        yield from scan_snapshot_exports(
            self.name, self.paths, cursor,
            lambda path: self._envelopes(ctx, path))

    def _envelopes(self, ctx: SubjectContext, path: Path) -> Iterator[RawEnvelope]:
        key = str(path)
        for conv in _load(path):
            conv_id = str(conv.get("conversation_id") or conv.get("id") or "")
            mapping = conv.get("mapping") or {}
            for idx, node_id in enumerate(_live_chain(conv)):
                msg = mapping[node_id].get("message")
                if not msg:
                    continue
                if (msg.get("author") or {}).get("role") != "user":
                    continue
                meta = msg.get("metadata") or {}
                # Custom instructions are account settings the server replays
                # as a user turn. They are not something the subject wrote in
                # this conversation, and they repeat verbatim across every
                # conversation opened while they were set.
                if meta.get("is_user_system_message"):
                    continue
                content = msg.get("content") or {}
                if content.get("content_type") not in _TEXT_TYPES:
                    continue
                text, images = _text_and_images(content)
                if not text.strip():
                    continue
                prose = _FENCE.sub("", text).strip()
                raw_chars, prose_chars = len(text), len(prose)
                # `message.id` is server-assigned and unique across the real
                # export; the hash is only a fallback for an export shaped
                # unexpectedly enough to omit it. It includes the position,
                # because a content-only hash collides on repeated short
                # messages and silently loses them to the vault's dedupe.
                source_id = str(msg.get("id") or "").strip() or hashlib.sha256(
                    f"{key}:{conv_id}:{idx}:{text}".encode("utf-8")
                ).hexdigest()[:24]
                yield RawEnvelope(
                    subject_id=ctx.config.subject_id, source=self.name,
                    source_id=source_id,
                    ts=_ts(msg.get("create_time"), conv.get("create_time")),
                    payload={
                        "text": text,
                        "conversation_id": conv_id,
                        "title": str(conv.get("title") or ""),
                        "raw_chars": raw_chars,
                        "prose_chars": prose_chars,
                        "looks_pasted": (
                            prose_chars > _PASTE_CHAR_THRESHOLD
                            or (raw_chars - prose_chars) > raw_chars / 2),
                        # Recorded, never used as a filter -- which model the
                        # subject happened to be talking to says nothing about
                        # whether he wrote the turn.
                        "model_slug": str(meta.get("model_slug") or ""),
                        "image_count": images,
                    },
                    ingested_at=datetime.now(timezone.utc))
