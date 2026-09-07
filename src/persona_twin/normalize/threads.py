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
        # `ts` is non-decreasing across `ordered`, so for any subject turn at
        # index i the set of preceding turns within `gap` of its timestamp is
        # always a contiguous suffix of ordered[:i] (if p qualifies and q is
        # between p and i, q qualifies too, since q.ts >= p.ts). That means
        # the gap boundary `left` only ever moves forward as i increases, so
        # a single two-pointer sweep over the thread finds it in O(n) instead
        # of rescanning ordered[:i] from scratch for every subject turn.
        left = 0
        for i, turn in enumerate(ordered):
            if not turn.is_subject:
                continue
            while left < i and turn.ts - ordered[left].ts > gap:
                left += 1
            # Truncate to the last `max_context` turns *before* checking for
            # an inbound turn: the non-subject check must run on the context
            # that actually ships, not the untruncated gap-filtered window --
            # otherwise an inbound turn outside the cap can satisfy the check
            # while the shipped context is entirely the subject's own turns.
            context = ordered[max(left, i - max_context) : i]
            if not any(not p.is_subject for p in context):
                continue
            pairs.append(ReplyPair(context=list(context), reply=turn))
    return pairs
