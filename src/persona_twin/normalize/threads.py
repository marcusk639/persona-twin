from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from persona_twin.schema import Turn


@dataclass(frozen=True)
class ReplyPair:
    context: list[Turn]
    reply: Turn


def reply_pairs(
    turns: list[Turn], max_context: int = 6, max_gap_minutes: int = 180
) -> list[ReplyPair]:
    """Group turns into (inbound context -> subject reply) pairs within a thread."""
    by_thread: dict[str, list[Turn]] = defaultdict(list)
    for t in turns:
        by_thread[t.thread_id].append(t)

    gap = timedelta(minutes=max_gap_minutes)
    pairs: list[ReplyPair] = []
    for thread_turns in by_thread.values():
        ordered = sorted(thread_turns, key=lambda t: t.ts)
        for i, turn in enumerate(ordered):
            if not turn.is_subject:
                continue
            context = [p for p in ordered[:i] if turn.ts - p.ts <= gap]
            if not any(not p.is_subject for p in context):
                continue
            pairs.append(ReplyPair(context=context[-max_context:], reply=turn))
    return pairs
