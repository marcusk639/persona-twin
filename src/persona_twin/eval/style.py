"""Computed style fingerprint (spec §7, criterion S2).

Every value here is arithmetic over the text. No model judgement is involved,
so the metric cannot drift when a model version changes, and the S2 target is
the subject's own self-distance band rather than a number someone picked.
"""
from __future__ import annotations

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


def distance(a: StyleFingerprint, b: StyleFingerprint) -> float:
    """Normalised L1 distance. Length-scale fields are compared relatively so a
    corpus of long messages does not dominate the rate-based fields."""
    total = 0.0
    for name in _SCALAR_FIELDS:
        x, y = getattr(a, name), getattr(b, name)
        denom = max(abs(x), abs(y), 1e-9)
        total += abs(x - y) / denom
    for w in _FUNCTION_WORDS:
        x = a.function_word_rates.get(w, 0.0)
        y = b.function_word_rates.get(w, 0.0)
        total += abs(x - y)
    return total


def self_distance_band(turns: list[Turn], trials: int = 200,
                       seed: int = 0) -> tuple[float, float]:
    """Bootstrap the subject against themselves: repeatedly split the sample in
    half and measure the distance between halves. Returns (median, p95).

    This is the S2 bar. A system is indistinguishable in style when its distance
    from the subject sits inside the range the subject occupies against
    themselves — an empirical target rather than a chosen threshold.
    """
    if len(turns) < 4:
        return (0.0, 0.0)
    rng = random.Random(seed)
    dists: list[float] = []
    for _ in range(trials):
        pool = list(turns)
        rng.shuffle(pool)
        mid = len(pool) // 2
        dists.append(distance(fingerprint(pool[:mid]), fingerprint(pool[mid:])))
    dists.sort()
    return (dists[len(dists) // 2], dists[int(len(dists) * 0.95)])
