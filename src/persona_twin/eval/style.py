"""Computed style fingerprint (spec §7, criterion S2).

Every value here is arithmetic over the text. No model judgement is involved,
so the metric cannot drift when a model version changes, and the S2 target is
the subject's own self-distance band rather than a number someone picked.

Three properties have to hold for that band to mean anything, and each is
enforced by a named mechanism below:

* No single conversation may dominate a fingerprint — `thread_mass_cap` and
  `mass_balanced_sample`. A corpus where two threads hold 65% of the turns
  otherwise produces a "style" that is really the style of those two threads.
* The two things being compared must be estimated from the same amount of
  text — `_balanced_halves`. A distance between a 179-turn half and a
  3,275-turn half measures small-sample estimation error, not style.
* Each field must be weighed in units of how much the subject's own writing
  varies on that field — `SelfDistanceBand.scales` and `distance`. Otherwise
  a field's weight is an accident of its numeric range.
"""
from __future__ import annotations

import math
import random
import statistics
import unicodedata
from dataclasses import dataclass, field

from persona_twin.schema import Turn

_PUNCT = set(".,;:!?—-()[]\"'")
_FUNCTION_WORDS = ("the", "a", "to", "and", "i", "it", "that", "is", "of", "you",
                   "for", "in", "but", "so", "just", "not", "have", "do")


def _is_emoji(ch: str) -> bool:
    return unicodedata.category(ch) == "So"


@dataclass(frozen=True)
class StyleFingerprint:
    n: int
    mean_len: float
    median_len: float
    len_stdev: float
    burstiness: float
    punct_rate: float
    emoji_rate: float
    upper_rate: float
    type_token_ratio: float
    function_word_rates: dict[str, float] = field(default_factory=dict)


def fingerprint(turns: list[Turn]) -> StyleFingerprint:
    texts = [t.text for t in turns]
    if not texts:
        return StyleFingerprint(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {})
    lens = [len(x) for x in texts]
    mean_len = statistics.fmean(lens)
    stdev = statistics.pstdev(lens) if len(lens) > 1 else 0.0
    chars = sum(lens) or 1
    all_text = "".join(texts)
    words = [w.lower().strip("".join(_PUNCT)) for w in all_text.split()]
    words = [w for w in words if w]
    total_words = len(words) or 1
    return StyleFingerprint(
        n=len(texts),
        mean_len=mean_len,
        median_len=float(statistics.median(lens)),
        len_stdev=stdev,
        burstiness=stdev / mean_len if mean_len else 0.0,
        punct_rate=sum(1 for c in all_text if c in _PUNCT) / chars,
        emoji_rate=sum(1 for c in all_text if _is_emoji(c)) / chars,
        upper_rate=sum(1 for c in all_text if c.isupper()) / chars,
        type_token_ratio=len(set(words)) / total_words,
        function_word_rates={w: words.count(w) / total_words for w in _FUNCTION_WORDS},
    )


_SCALAR_FIELDS = ("mean_len", "median_len", "len_stdev", "burstiness",
                  "punct_rate", "emoji_rate", "upper_rate", "type_token_ratio")

# Every comparable quantity, flattened to one namespace. The function-word
# rates are prefixed so a word can never collide with a scalar field name.
_FIELD_NAMES = _SCALAR_FIELDS + tuple(f"fw:{w}" for w in _FUNCTION_WORDS)


def field_values(fp: StyleFingerprint) -> dict[str, float]:
    """The fingerprint as a flat name -> value mapping, in _FIELD_NAMES order."""
    values = {name: float(getattr(fp, name)) for name in _SCALAR_FIELDS}
    for w in _FUNCTION_WORDS:
        values[f"fw:{w}"] = fp.function_word_rates.get(w, 0.0)
    return values


def distance(a: StyleFingerprint, b: StyleFingerprint,
             scales: dict[str, float]) -> float:
    """L1 distance with every field expressed in units of the subject's own
    variation on that field.

    `scales[name]` is how far apart two samples of the subject's own writing
    typically land on that field (see `SelfDistanceBand.scales`). Dividing by
    it makes the terms commensurable and gives the sum a meaning: "how many
    times the subject's own between-conversation variation apart are these
    two bodies of text".

    This replaces a relative denominator (`abs(x - y) / max(abs(x), abs(y))`),
    which was wrong for two compounding reasons. It is bounded by 1.0, so every
    field's contribution was capped at the same value no matter how decisive
    the field was; and it saturates on near-zero quantities, so `emoji_rate` —
    which lives at the 1e-4 scale and whose own sampling noise is the same size
    as the gap between two different people — spent almost the whole of that
    budget on noise while all eighteen function-word rates together contributed
    a fiftieth of it.

    When a field has no measured variation (`scale == 0`, i.e. the subject was
    identical on it in every trial) there is no unit to divide by, and treating
    any difference as infinitely significant would be a fabrication. The term
    falls back to the bounded relative comparison, which contributes at most
    1.0 — the honest statement that the field registers a difference but the
    corpus cannot say how large a difference that is.
    """
    av, bv = field_values(a), field_values(b)
    total = 0.0
    for name in _FIELD_NAMES:
        x, y = av[name], bv[name]
        gap = abs(x - y)
        scale = scales.get(name, 0.0)
        if scale > 0.0:
            total += gap / scale
        else:
            magnitude = max(abs(x), abs(y))
            total += gap / magnitude if magnitude > 0.0 else 0.0
    return total


def thread_mass_cap(turns: list[Turn]) -> int:
    """The most turns any one thread may contribute: one average thread's worth.

    Derived from the data (ceil of the mean thread size), never chosen. The
    corpus this was written against has 99 held-out subject threads whose
    median size is 2 and whose two largest hold 65% of all turns; without a cap
    every "style" measurement is really a measurement of those two
    conversations, and every bootstrap draw is decided by which side they land
    on rather than by style.
    """
    threads = {t.thread_id for t in turns}
    if not threads:
        return 0
    return max(1, math.ceil(len(turns) / len(threads)))


def _by_thread(turns: list[Turn]) -> dict[str, list[Turn]]:
    grouped: dict[str, list[Turn]] = {}
    for t in turns:
        grouped.setdefault(t.thread_id, []).append(t)
    return grouped


def mass_balanced_sample(turns: list[Turn], rng: random.Random) -> list[Turn]:
    """Draw a sample in which no thread exceeds `thread_mass_cap`.

    Every fingerprint that takes part in an S2 comparison is built from one of
    these, the band's halves included — a bar measured on mass-equalised text
    and a distance measured on raw text are not comparable numbers.

    Threads are visited in sorted id order and over-sized ones are subsampled,
    so the result depends only on the turns and the rng, not on the order the
    corpus happened to arrive in.
    """
    cap = thread_mass_cap(turns)
    sample: list[Turn] = []
    for _tid, ts in sorted(_by_thread(turns).items()):
        sample.extend(ts if len(ts) <= cap else rng.sample(ts, cap))
    return sample


def _balanced_halves(by_thread: dict[str, list[Turn]],
                     rng: random.Random) -> tuple[list[Turn], list[Turn]]:
    """Split whole threads into two halves carrying near-equal turn counts.

    Splitting on whole threads, not individual turns, is what makes the band a
    between-conversation measurement: turns from one thread share a
    conversational register, so shuffling turns puts both halves in the same
    registers and measures only within-sample sampling noise.

    Splitting on thread *count* rather than turn count was the defect this
    replaces: with a skewed thread-size distribution the halves differed by an
    order of magnitude in size, and the resulting distance was dominated by how
    badly the smaller half's fingerprint was estimated. Threads are shuffled
    (so each draw mixes conversations differently) and then assigned to
    whichever side is currently lighter, which bounds the final imbalance by
    the largest single thread.
    """
    ids = list(by_thread)
    rng.shuffle(ids)
    left: list[Turn] = []
    right: list[Turn] = []
    for tid in ids:
        target = left if len(left) <= len(right) else right
        target.extend(by_thread[tid])
    return left, right


# A fingerprint of fewer than four turns has no usable spread: pstdev over
# three values, and therefore burstiness, is noise. Both halves of every trial
# must clear it or the trial is discarded rather than counted.
_MIN_SIDE = 4


@dataclass(frozen=True)
class SelfDistanceBand:
    """The S2 bar, or an explicit refusal to state one.

    `median` and `p95` are None exactly when `reason` is non-empty. Callers
    must check `measured` — there is no degenerate numeric band, because a
    self-distance of 0.0000 renders as an unusually tight bar, which is the
    flattering direction.

    `trials` is how many bootstrap trials actually produced a distance, not how
    many were requested, and `partitions` is how many distinct thread
    partitions those trials realised. Both are reported because both have been
    silently degenerate in this module's history: 800 turns in a single thread
    completed zero trials and still printed a confident 0.0000, and turns in
    exactly two threads admit exactly one partition, so 200 trials were one
    measurement repeated 200 times and printed as a bootstrap.
    """
    median: float | None
    p95: float | None
    trials: int
    partitions: int
    scales: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    @property
    def measured(self) -> bool:
        return self.p95 is not None

    def distance(self, a: StyleFingerprint, b: StyleFingerprint) -> float:
        """`distance` against this band's scales — the only comparable form."""
        if not self.measured:
            raise ValueError(f"band was not measured: {self.reason}")
        return distance(a, b, self.scales)


def _partition_key(left: list[Turn], right: list[Turn]) -> frozenset[str]:
    """Which *split* this trial made, independent of which side was called left.

    `distance` is symmetric, so {A} | {B} and {B} | {A} are one measurement.
    Keying on the left side alone would count them as two and hide exactly the
    degeneracy `partitions` exists to expose — a two-thread corpus has one
    possible split and would otherwise look like a bootstrap over two.
    """
    a = frozenset(t.thread_id for t in left)
    b = frozenset(t.thread_id for t in right)
    return a if min(a) <= min(b) else b


def _refusal(reason: str, trials: int = 0, partitions: int = 0) -> SelfDistanceBand:
    return SelfDistanceBand(median=None, p95=None, trials=trials,
                            partitions=partitions, scales={}, reason=reason)


def self_distance_band(turns: list[Turn], trials: int = 200,
                       seed: int = 0) -> SelfDistanceBand:
    """Bootstrap the subject against themselves: repeatedly draw a mass-balanced
    sample, split its threads into two equal-mass halves, and measure how far
    apart the halves land.

    This is the S2 bar. A system is indistinguishable in style when its
    distance from the subject sits inside the range the subject occupies
    against themselves — an empirical target rather than a chosen threshold.

    The per-field `scales` come out of the same trials, and have to: the scale
    of a field *is* the subject's own variation on it, so it cannot be known
    before the self-comparison has been run. Trials are therefore scored in two
    passes — collect the fingerprint pairs, average the per-field gaps into
    scales, then score every pair against those scales.
    """
    if len(turns) < 2 * _MIN_SIDE:
        return _refusal(
            f"only {len(turns)} turn(s); a band needs at least {2 * _MIN_SIDE} "
            f"so both halves can clear {_MIN_SIDE}")

    rng = random.Random(seed)
    pairs: list[tuple[StyleFingerprint, StyleFingerprint]] = []
    partitions: set[frozenset[str]] = set()
    for _ in range(trials):
        sample = mass_balanced_sample(turns, rng)
        left, right = _balanced_halves(_by_thread(sample), rng)
        if len(left) < _MIN_SIDE or len(right) < _MIN_SIDE:
            continue
        partitions.add(_partition_key(left, right))
        pairs.append((fingerprint(left), fingerprint(right)))

    if not pairs:
        return _refusal(
            f"no bootstrap trial produced two halves of at least {_MIN_SIDE} "
            f"turns across {len({t.thread_id for t in turns})} thread(s); the "
            "thread structure cannot be split, so there is no band")
    if len(partitions) < 2:
        return _refusal(
            f"the {len({t.thread_id for t in turns})} thread(s) admit only one "
            "distinct split, so every trial re-measured the same pair of halves; "
            "that is a single measurement, not a bootstrap",
            trials=len(pairs), partitions=len(partitions))

    gaps = [{name: abs(av[name] - bv[name]) for name in _FIELD_NAMES}
            for av, bv in ((field_values(a), field_values(b)) for a, b in pairs)]
    scales = {name: statistics.fmean(g[name] for g in gaps) for name in _FIELD_NAMES}

    dists = sorted(distance(a, b, scales) for a, b in pairs)
    return SelfDistanceBand(median=dists[len(dists) // 2],
                            p95=dists[int(len(dists) * 0.95)],
                            trials=len(dists), partitions=len(partitions),
                            scales=scales)


# How many independent mass-balanced draws `corpus_distance` averages. A
# Monte-Carlo budget, not a threshold: raising it narrows the spread of the
# answer without moving what the answer converges to. It exists because a
# single draw subsamples every over-long thread once, and one unlucky draw
# should not decide a gate.
_COMPARISON_DRAWS = 25


def corpus_distance(a_turns: list[Turn], b_turns: list[Turn],
                    band: SelfDistanceBand, draws: int = _COMPARISON_DRAWS,
                    seed: int = 0) -> float:
    """Distance between two corpora, on the same footing as `band`.

    Both sides are mass-balanced exactly as the band's halves were, and the
    result is averaged over independent draws. Comparing a raw pooled
    fingerprint against a band built from mass-balanced halves would put a
    number produced one way next to a bar produced another.
    """
    if not band.measured:
        raise ValueError(f"band was not measured: {band.reason}")
    if not a_turns or not b_turns:
        raise ValueError("corpus_distance needs turns on both sides")
    rng = random.Random(seed)
    return statistics.fmean(
        distance(fingerprint(mass_balanced_sample(a_turns, rng)),
                 fingerprint(mass_balanced_sample(b_turns, rng)),
                 band.scales)
        for _ in range(draws))
