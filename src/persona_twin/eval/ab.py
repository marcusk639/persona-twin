"""Blind A/B trials (spec §7, criterion S1).

A judge sees the subject's real reply and the system's candidate reply to the
same context, in random order, and picks the one they believe is real. 50% is
a perfect score for the system: it means the two are indistinguishable. S1
passes at <= 60% accuracy with n >= 300 and a reported 95% interval.

Two properties keep the result honest rather than merely looking clean:

* Per-trial randomised A/B ordering, seeded and reproducible, is what stops a
  judge -- human or model -- from learning that the real reply always lands
  on the same side. The seed controls reproducibility across runs, not
  predictability to the judge: nothing about a trial's presentation (its
  context, its trial_id) reveals which side the random draw picked.
* At most one trial per conversation thread. A thread can yield several
  ReplyPairs -- one per subject turn -- whose contexts overlap heavily
  (each is a prefix-extended version of the last). Scoring every one of them
  would let a single conversation dominate the sample with correlated
  outcomes: that both double-counts the conversation's evidence and breaks
  the independent-trials assumption `wilson_interval` relies on, so the
  reported margin would look narrower than the independent evidence behind
  it actually supports.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from persona_twin.normalize.threads import ReplyPair
from persona_twin.schema import Turn


@dataclass(frozen=True)
class ABTrial:
    trial_id: str
    context: list[Turn]
    option_a: str
    option_b: str
    real_is_a: bool


def build_trials(
    pairs: list[ReplyPair],
    candidate_replies: dict[str, str],
    seed: int = 0,
) -> list[ABTrial]:
    """Pair each real reply with a candidate keyed by the reply's source_id.

    Pairs with no candidate are skipped rather than filled, so a generation
    failure shrinks the sample visibly instead of silently biasing it. A
    reply that appears more than once in `pairs` (the same source_id) is only
    ever eligible once, the same defensive posture as the thread cap below.

    Eligible pairs are grouped by thread_id and capped at one trial per
    thread (see module docstring). When a thread offers more than one
    eligible pair, the one that becomes a trial is chosen from the seeded RNG
    rather than always the first one encountered -- otherwise the choice
    would systematically favour whichever reply happens to sort first, e.g.
    the earliest reply in a conversation, which is often shorter or more
    formulaic than later ones. Threads are visited in sorted order so the
    trial list, and every downstream `rng` draw, is independent of whatever
    order `pairs` happened to arrive in.
    """
    rng = random.Random(seed)
    by_thread: dict[str, list[ReplyPair]] = {}
    seen_source_ids: set[str] = set()
    for p in pairs:
        sid = p.reply.source_id
        if sid in seen_source_ids or sid not in candidate_replies:
            continue
        seen_source_ids.add(sid)
        by_thread.setdefault(p.reply.thread_id, []).append(p)

    trials: list[ABTrial] = []
    for thread_id in sorted(by_thread):
        candidates = sorted(by_thread[thread_id], key=lambda p: p.reply.source_id)
        chosen = candidates[rng.randrange(len(candidates))]
        candidate = candidate_replies[chosen.reply.source_id]
        real_is_a = rng.random() < 0.5
        reply = chosen.reply
        trials.append(ABTrial(
            trial_id=f"{reply.subject_id}:{reply.source}:{reply.source_id}",
            context=list(chosen.context),
            option_a=reply.text if real_is_a else candidate,
            option_b=candidate if real_is_a else reply.text,
            real_is_a=real_is_a,
        ))
    return trials


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (default z=1.96).

    n=0 is mathematically undefined -- the closed-form formula divides by n
    -- and is handled explicitly rather than propagating a ZeroDivisionError
    or, worse, quietly returning something that looks like a real interval.
    A zero-width interval such as (0.0, 0.0) at zero observations would read
    as a confident result on no data, exactly the class of flattering-looking
    defect this harness exists to catch. (0.0, 1.0) is the honest answer:
    with no observations, any true proportion from 0 to 1 is equally
    consistent with what was seen.
    """
    if n < 0 or successes < 0 or successes > n:
        raise ValueError(f"invalid trial counts: successes={successes}, n={n}")
    if n == 0:
        return (0.0, 1.0)
    phat = successes / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass(frozen=True)
class ABResult:
    n: int
    correct: int
    accuracy: float
    ci_low: float
    ci_high: float


def score_trials(trials: list[ABTrial], guesses: dict[str, str]) -> ABResult:
    """`guesses` maps trial_id -> "a" | "b" (case-insensitive). A missing or
    unrecognised guess counts as wrong, which is conservative: it can only
    make the system look more detectable, never less.

    Trial ids are required to be unique within `trials` -- a collision would
    let one trial's guess silently stand in for another's outcome, the same
    class of defect as two different configurations sharing one fingerprint.
    `build_trials` already guarantees this; the check here protects callers
    that assemble a trial list by hand (e.g. in tests) or merge trial lists
    from more than one `build_trials` call.
    """
    seen_ids: set[str] = set()
    correct = 0
    for t in trials:
        if t.trial_id in seen_ids:
            raise ValueError(f"duplicate trial_id: {t.trial_id!r}")
        seen_ids.add(t.trial_id)
        picked = (guesses.get(t.trial_id) or "").strip().lower()
        if picked == ("a" if t.real_is_a else "b"):
            correct += 1
    n = len(trials)
    acc = correct / n if n else 0.0
    lo, hi = wilson_interval(correct, n)
    return ABResult(n=n, correct=correct, accuracy=acc, ci_low=lo, ci_high=hi)
