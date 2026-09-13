"""Deterministic held-out split (spec §7).

The split is a pure function of a turn's own thread_id and timestamp. It never
depends on corpus size, ordering, or version — so adding a new source later
cannot move an existing turn across the boundary, and baselines measured today
stay comparable to results measured after the next export lands.

Whole threads are assigned, never individual turns: adjacent turns in one
conversation leak, and a model that memorised the first half of a thread would
score well on the second half for the wrong reason.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

from persona_twin.corpus.quarantine import DEFAULT_WEEKS, is_quarantined
from persona_twin.schema import Turn

HELDOUT_FRACTION = 0.15
_BUCKETS = 100
_HELDOUT_BUCKETS = int(HELDOUT_FRACTION * _BUCKETS)


def thread_bucket(thread_id: str) -> int:
    """Stable 0-99 bucket for a thread. sha256, not hash() — the builtin is
    salted per process and would reshuffle the split on every run."""
    digest = hashlib.sha256(thread_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % _BUCKETS


def is_heldout(turn: Turn) -> bool:
    return thread_bucket(turn.thread_id) < _HELDOUT_BUCKETS


@dataclass(frozen=True)
class CorpusSplit:
    train: list[Turn]
    heldout: list[Turn]
    quarantined: list[Turn]


def split_corpus(turns: list[Turn], now: datetime,
                 weeks: int = DEFAULT_WEEKS) -> CorpusSplit:
    """Partition turns into train / heldout / quarantined.

    Quarantine (CC3) takes precedence: a recent turn is quarantined regardless
    of which side of the held-out split its thread falls on, because it must be
    neither trained on nor scored against until it ages out.
    """
    train: list[Turn] = []
    heldout: list[Turn] = []
    quarantined: list[Turn] = []
    for t in turns:
        if is_quarantined(t, now, weeks):
            quarantined.append(t)
        elif is_heldout(t):
            heldout.append(t)
        else:
            train.append(t)
    return CorpusSplit(train=train, heldout=heldout, quarantined=quarantined)
