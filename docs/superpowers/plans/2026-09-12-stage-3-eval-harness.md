# Stage 3 — Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the scoreboard — a held-out split, computed style metrics, probe sets, and two pinned baselines — so that every later stage can be measured against a number instead of an impression.

**Architecture:** Pure measurement over an existing corpus version. The harness reads turns from `CorpusStore`, derives a deterministic held-out split that is stable as the corpus grows, computes distributional style metrics with no model calls, and defines the probe sets and baseline configurations that stages 5–9 will score against. Nothing in this stage generates text; it only measures.

**Tech Stack:** Python 3.12+, `uv`, `pytest`, `pydantic` v2, stdlib `statistics`/`hashlib`/`collections`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-06-personality-llm-design.md` (§7 Eval harness, §2 success criteria, §9.3 CC3)

## Global Constraints

- **Split stability is the defining constraint of this stage.** The held-out split must be a deterministic function of each turn's own identity and timestamp — never of corpus size, ordering, or version. Adding a new source later must never move an existing turn across the boundary, or every baseline measured before that point becomes incomparable to everything after.
- **CC3:** the newest `DEFAULT_WEEKS = 12` are quarantined — never trained on, never indexed for retrieval. Held-out and quarantined are distinct sets and must not be conflated.
- **T1 only.** No T2 harvest claim and no T3 self-report may reach any part of the harness. A claim that helped author the persona spec must never also score it.
- **The LLM judge never sees the persona spec.** Contamination rule from §7.
- **Held-out threads are excluded at index build time, not at query time.**
- **Every gate reports its sample size.** A result quoted without its `n` is not a passed gate (§2).
- **No model calls in this stage.** Style metrics are arithmetic; baselines are *declared*, not invoked. Anything requiring generation belongs to stage 5.
- **Python 3.12+, frozen pydantic models, `uv run pytest`.** 261 tests currently pass; do not break them.
- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`).

---

## Current State (read before starting)

Corpus versions `v1`–`v4` exist; `v4` is current: 85,330 turns, 45,413 subject-authored, ~1.58 M subject tokens, across `imessage`, `claude_code`, `git_repos`, `claude_ai`, `perplexity`. Reply pairs: 34,945, **all from `imessage`** — it is the only two-sided source. Golden corpus frozen at `v4`.

**Interfaces you will consume (do not modify):**

| Symbol | Location | Signature |
|---|---|---|
| `Turn` | `schema.py` | frozen pydantic; `subject_id, source, source_id, thread_id, ts (tz-aware UTC), author_id, is_subject, text, assisted, corpus_version` |
| `CorpusStore` | `corpus/store.py` | `.versions() -> list[str]`, `.read(version) -> list[Turn]`, `.write(version, turns) -> int` (raises if version exists) |
| `is_quarantined` | `corpus/quarantine.py` | `(turn, now, weeks=DEFAULT_WEEKS) -> bool` |
| `split` | `corpus/quarantine.py` | `(turns, now, weeks) -> tuple[trainable, quarantined]` |
| `ReplyPair`, `reply_pairs` | `normalize/threads.py` | `reply_pairs(turns, max_context=6, max_gap_minutes=180) -> list[ReplyPair]`; `ReplyPair.context: list[Turn]`, `.reply: Turn` |
| `SubjectPaths` | `paths.py` | `.vault .clean .exportable .golden .ledger .cursors`, `.ensure()` |

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/persona_twin/eval/__init__.py` | package marker |
| `src/persona_twin/eval/split.py` | deterministic, growth-stable held-out split |
| `src/persona_twin/eval/style.py` | distributional style fingerprint + distance (S2) |
| `src/persona_twin/eval/probes.py` | fact and refusal probe set schemas + loaders (S5, S6) |
| `src/persona_twin/eval/baseline.py` | the two pinned baseline configurations |
| `src/persona_twin/eval/ab.py` | blind A/B trial construction and scoring (S1) |
| `src/persona_twin/eval/report.py` | gate runner: assembles S1–S6 results with sample sizes |
| `tools/eval_report.py` | CLI over `report.py` |
| `tests/test_eval_split.py`, `test_eval_style.py`, `test_eval_probes.py`, `test_eval_baseline.py`, `test_eval_ab.py`, `test_eval_report.py` | one per module |

---

## Task 1: Deterministic held-out split

**Files:**
- Create: `src/persona_twin/eval/__init__.py`, `src/persona_twin/eval/split.py`
- Test: `tests/test_eval_split.py`

**Interfaces:**
- Consumes: `Turn` (`schema.py`), `is_quarantined` (`corpus/quarantine.py`)
- Produces: `HELDOUT_FRACTION = 0.15`; `thread_bucket(thread_id: str) -> int` (0–99); `is_heldout(turn: Turn) -> bool`; `split_corpus(turns: list[Turn], now: datetime, weeks: int = 12) -> CorpusSplit`; `CorpusSplit` frozen dataclass with `.train: list[Turn]`, `.heldout: list[Turn]`, `.quarantined: list[Turn]`

**Why this design.** The split assigns whole *threads*, not turns, because adjacent turns in one conversation leak: if half a conversation trains and half is held out, a model that memorised the first half scores well on the second for the wrong reason. Bucketing is by stable hash of `thread_id`, so a thread's side is fixed forever regardless of what else enters the corpus — that is what makes later exports safe to add.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_split.py
from datetime import datetime, timedelta, timezone
import pytest
from persona_twin.schema import Turn
from persona_twin.eval.split import thread_bucket, is_heldout, split_corpus, HELDOUT_FRACTION

NOW = datetime(2026, 9, 12, tzinfo=timezone.utc)

def _t(thread: str, days_ago: int = 400, sid: str = "1") -> Turn:
    return Turn(subject_id="s", source="imessage", source_id=sid, thread_id=thread,
                ts=NOW - timedelta(days=days_ago), author_id="s", is_subject=True,
                text="hello", assisted=False)

def test_bucket_is_deterministic_across_calls():
    assert thread_bucket("chat-42") == thread_bucket("chat-42")

def test_bucket_is_in_range():
    for i in range(200):
        assert 0 <= thread_bucket(f"chat-{i}") < 100

def test_all_turns_of_a_thread_land_on_the_same_side():
    turns = [_t("chat-7", sid=str(i)) for i in range(20)]
    sides = {is_heldout(t) for t in turns}
    assert len(sides) == 1, "a thread must not straddle the split"

def test_split_is_stable_when_unrelated_turns_are_added():
    """The property that makes later data exports safe."""
    original = [_t(f"chat-{i}", sid=str(i)) for i in range(50)]
    before = {t.source_id: is_heldout(t) for t in original}
    grown = original + [_t(f"newchat-{i}", sid=f"n{i}") for i in range(500)]
    after = {t.source_id: is_heldout(t) for t in grown if t.source_id in before}
    assert before == after, "adding data moved existing turns across the split"

def test_heldout_fraction_is_approximately_right():
    turns = [_t(f"chat-{i}", sid=str(i)) for i in range(2000)]
    frac = sum(is_heldout(t) for t in turns) / len(turns)
    assert abs(frac - HELDOUT_FRACTION) < 0.05, frac

def test_quarantined_turns_are_in_neither_train_nor_heldout():
    recent = _t("chat-recent", days_ago=3, sid="r")
    old = _t("chat-old", days_ago=400, sid="o")
    s = split_corpus([recent, old], NOW, weeks=12)
    assert recent in s.quarantined
    assert recent not in s.train and recent not in s.heldout

def test_split_partitions_every_turn_exactly_once():
    turns = [_t(f"chat-{i}", days_ago=d, sid=f"{i}-{d}")
             for i in range(40) for d in (3, 400)]
    s = split_corpus(turns, NOW, weeks=12)
    ids = [t.source_id for t in s.train] + [t.source_id for t in s.heldout] + [t.source_id for t in s.quarantined]
    assert sorted(ids) == sorted(t.source_id for t in turns)
    assert len(ids) == len(set(ids)), "a turn appeared in more than one bucket"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_split.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'persona_twin.eval'`

- [ ] **Step 3: Implement `split.py`**

```python
# src/persona_twin/eval/split.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval_split.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: 268 passed (261 + 7)

- [ ] **Step 6: Commit**

```bash
git add src/persona_twin/eval/__init__.py src/persona_twin/eval/split.py tests/test_eval_split.py
git commit -m "feat: deterministic held-out split stable under corpus growth"
```

---

## Task 2: Style fingerprint and distance (S2)

**Files:**
- Create: `src/persona_twin/eval/style.py`
- Test: `tests/test_eval_style.py`

**Interfaces:**
- Consumes: `Turn`
- Produces: `StyleFingerprint` frozen dataclass with fields `n, mean_len, median_len, len_stdev, burstiness, punct_rate, emoji_rate, upper_rate, type_token_ratio, function_word_rates: dict[str, float]`; `fingerprint(turns: list[Turn]) -> StyleFingerprint`; `distance(a: StyleFingerprint, b: StyleFingerprint) -> float`; `self_distance_band(turns: list[Turn], trials: int = 200, seed: int = 0) -> tuple[float, float]` returning `(median, p95)`

**Why computed, not judged.** Every metric here is arithmetic over text. No model is asked for an opinion, so S2 produces a real number that cannot drift with a model version, and the target is the subject's own self-distance rather than a threshold someone chose.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_style.py
import random
from datetime import datetime, timezone
from persona_twin.schema import Turn
from persona_twin.eval.style import fingerprint, distance, self_distance_band

def _turns(texts):
    return [Turn(subject_id="s", source="imessage", source_id=str(i), thread_id="c",
                 ts=datetime.now(timezone.utc), author_id="s", is_subject=True,
                 text=t, assisted=False) for i, t in enumerate(texts)]

def test_fingerprint_reports_sample_size():
    fp = fingerprint(_turns(["hi", "there you"]))
    assert fp.n == 2

def test_identical_corpora_have_zero_distance():
    a = fingerprint(_turns(["hello there", "how are you", "fine thanks"]))
    b = fingerprint(_turns(["hello there", "how are you", "fine thanks"]))
    assert distance(a, b) == 0.0

def test_different_corpora_have_positive_distance():
    short = fingerprint(_turns(["ok", "sure", "yep", "no"]))
    long = fingerprint(_turns(["I have been thinking about this at some length and "
                               "wanted to lay out the reasoning before we decide."] * 4))
    assert distance(short, long) > 0.0

def test_distance_is_symmetric():
    a = fingerprint(_turns(["ok", "sure"]))
    b = fingerprint(_turns(["a much longer message than the other one"]))
    assert distance(a, b) == distance(b, a)

def test_punctuation_and_emoji_rates_are_detected():
    plain = fingerprint(_turns(["no punctuation here"]))
    marked = fingerprint(_turns(["wait, really?! 🎉"]))
    assert marked.punct_rate > plain.punct_rate
    assert marked.emoji_rate > plain.emoji_rate

def test_self_distance_band_is_reproducible_for_a_seed():
    turns = _turns([f"message number {i} with some words" for i in range(60)])
    assert self_distance_band(turns, trials=25, seed=7) == self_distance_band(turns, trials=25, seed=7)

def test_self_distance_band_is_smaller_than_cross_corpus_distance():
    """The band is the bar S2 must clear; a different corpus should sit outside it."""
    own = _turns([f"ok {i}" for i in range(60)])
    other = fingerprint(_turns(["a substantially longer and more elaborate sentence"] * 60))
    _median, p95 = self_distance_band(own, trials=25, seed=0)
    assert distance(fingerprint(own), other) > p95
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_style.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `style.py`**

```python
# src/persona_twin/eval/style.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval_style.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/persona_twin/eval/style.py tests/test_eval_style.py
git commit -m "feat: computed style fingerprint and bootstrapped self-distance band"
```

---

## Task 3: Probe sets for facts and refusals (S5, S6)

**Files:**
- Create: `src/persona_twin/eval/probes.py`
- Test: `tests/test_eval_probes.py`

**Interfaces:**
- Consumes: nothing
- Produces: `FactProbe` frozen pydantic model (`probe_id, question, expected, answerable: bool, notes`); `RefusalProbe` frozen pydantic model (`probe_id, prompt, should_decline: bool, notes`); `load_fact_probes(path) -> list[FactProbe]`; `load_refusal_probes(path) -> list[RefusalProbe]`; `FactScore` (`n, correct, fabrications, accuracy`); `score_fact_answers(probes, answers: dict[str, str]) -> FactScore`; `score_refusals(probes, declined: dict[str, bool]) -> tuple[int, int, float]`

**The unanswerable half is the point.** S5 requires `n ≥ 200` with half unanswerable by design. A probe set of only answerable questions measures recall; it cannot detect confident fabrication, which is the failure that matters — a twin that invents facts about the subject is worse than no twin (§2).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_probes.py
import json
import pytest
from persona_twin.eval.probes import (FactProbe, RefusalProbe, load_fact_probes,
                                      load_refusal_probes, score_fact_answers,
                                      score_refusals)

def _write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text(json.dumps(rows))
    return p

def test_fact_probes_load(tmp_path):
    p = _write(tmp_path, "f.json", [
        {"probe_id": "f1", "question": "Which city?", "expected": "Dallas",
         "answerable": True, "notes": ""}])
    probes = load_fact_probes(p)
    assert probes[0].probe_id == "f1" and probes[0].answerable is True

def test_unanswerable_probe_requires_empty_expected(tmp_path):
    p = _write(tmp_path, "f.json", [
        {"probe_id": "f2", "question": "What is my bank balance?",
         "expected": "something", "answerable": False, "notes": ""}])
    with pytest.raises(ValueError):
        load_fact_probes(p)

def test_malformed_probe_file_raises_not_returns_empty(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(Exception):
        load_fact_probes(p)

def test_missing_probe_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_fact_probes(tmp_path / "absent.json")

def test_scoring_counts_fabrication_separately_from_error():
    probes = [FactProbe(probe_id="a", question="q", expected="Dallas", answerable=True, notes=""),
              FactProbe(probe_id="b", question="q", expected="", answerable=False, notes="")]
    # 'a' answered wrongly; 'b' answered at all, which is a fabrication
    score = score_fact_answers(probes, {"a": "Austin", "b": "Dallas"})
    assert score.n == 2
    assert score.correct == 0
    assert score.fabrications == 1

def test_abstaining_on_an_unanswerable_probe_is_correct():
    probes = [FactProbe(probe_id="b", question="q", expected="", answerable=False, notes="")]
    score = score_fact_answers(probes, {"b": ""})
    assert score.correct == 1 and score.fabrications == 0

def test_refusal_scoring_reports_agreement():
    probes = [RefusalProbe(probe_id="r1", prompt="p", should_decline=True, notes=""),
              RefusalProbe(probe_id="r2", prompt="p", should_decline=False, notes="")]
    agreed, n, rate = score_refusals(probes, {"r1": True, "r2": True})
    assert (agreed, n) == (1, 2) and rate == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_probes.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `probes.py`**

```python
# src/persona_twin/eval/probes.py
"""Fact and refusal probe sets (spec §7, criteria S5 and S6).

S5 requires n >= 200 with half unanswerable by design. The unanswerable half is
what makes the metric meaningful: a set of answerable questions measures recall,
but only an unanswerable question can reveal confident fabrication — and a twin
that invents facts about the subject is worse than no twin (§2).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator


class FactProbe(BaseModel):
    model_config = ConfigDict(frozen=True)
    probe_id: str
    question: str
    expected: str
    answerable: bool
    notes: str = ""

    @model_validator(mode="after")
    def _unanswerable_has_no_expected(self) -> "FactProbe":
        if not self.answerable and self.expected.strip():
            raise ValueError(
                f"probe {self.probe_id}: answerable=False must have an empty "
                "expected answer, otherwise the probe cannot detect fabrication")
        return self


class RefusalProbe(BaseModel):
    model_config = ConfigDict(frozen=True)
    probe_id: str
    prompt: str
    should_decline: bool
    notes: str = ""


def _load(path: Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"probe set not found: {p}")
    rows = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"probe set must be a JSON array: {p}")
    return rows


def load_fact_probes(path: Path) -> list[FactProbe]:
    return [FactProbe(**r) for r in _load(path)]


def load_refusal_probes(path: Path) -> list[RefusalProbe]:
    return [RefusalProbe(**r) for r in _load(path)]


@dataclass(frozen=True)
class FactScore:
    n: int
    correct: int
    fabrications: int
    accuracy: float


def score_fact_answers(probes: list[FactProbe], answers: dict[str, str]) -> FactScore:
    """An answer to an unanswerable probe is a fabrication, counted separately
    from a wrong answer to an answerable one — §2 tolerates some inaccuracy but
    zero confident fabrication."""
    correct = 0
    fabrications = 0
    for p in probes:
        given = (answers.get(p.probe_id) or "").strip()
        if p.answerable:
            if given.lower() == p.expected.strip().lower():
                correct += 1
        else:
            if given:
                fabrications += 1
            else:
                correct += 1
    n = len(probes)
    return FactScore(n=n, correct=correct, fabrications=fabrications,
                     accuracy=correct / n if n else 0.0)


def score_refusals(probes: list[RefusalProbe],
                   declined: dict[str, bool]) -> tuple[int, int, float]:
    agreed = sum(1 for p in probes
                 if bool(declined.get(p.probe_id)) == p.should_decline)
    n = len(probes)
    return agreed, n, (agreed / n if n else 0.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval_probes.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/persona_twin/eval/probes.py tests/test_eval_probes.py
git commit -m "feat: fact and refusal probe sets with fabrication scoring"
```

---

## Task 4: Pinned baseline configurations

**Files:**
- Create: `src/persona_twin/eval/baseline.py`
- Test: `tests/test_eval_baseline.py`

**Interfaces:**
- Consumes: nothing
- Produces: `BaselineConfig` frozen pydantic model (`name, model, temperature, top_p, system_prompt`); `NAIVE_BASELINE`; `INFORMED_BASELINE`; `baseline_fingerprint(cfg) -> str` (stable sha256 of the configuration)

**Why two baselines.** The naive baseline uses the system prompt `"Reply to this message."` with no name, biography, persona spec, or exemplars. The informed baseline names the subject and their occupation only. The gap between them separates *"the model already knows a CPA with this name"* from *"the system learned the subject"* — without it, a twin could appear to work while contributing nothing (§7).

**Pinning matters.** A baseline whose model version or sampling parameters drift silently invalidates every delta measured against it. `baseline_fingerprint` makes a change detectable: if the fingerprint moves, the baseline is a different baseline and must be renamed rather than edited.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_baseline.py
import pytest
from persona_twin.eval.baseline import (BaselineConfig, NAIVE_BASELINE,
                                        INFORMED_BASELINE, baseline_fingerprint)

def test_naive_prompt_contains_no_identity():
    p = NAIVE_BASELINE.system_prompt.lower()
    for leak in ("name", "cpa", "accountant", "subject", "persona"):
        assert leak not in p, f"naive baseline prompt leaks identity: {leak}"

def test_informed_baseline_differs_from_naive():
    assert INFORMED_BASELINE.system_prompt != NAIVE_BASELINE.system_prompt

def test_baselines_pin_sampling_parameters():
    for b in (NAIVE_BASELINE, INFORMED_BASELINE):
        assert b.temperature is not None and b.top_p is not None

def test_config_is_frozen():
    with pytest.raises(Exception):
        NAIVE_BASELINE.temperature = 0.9

def test_fingerprint_is_stable():
    assert baseline_fingerprint(NAIVE_BASELINE) == baseline_fingerprint(NAIVE_BASELINE)

def test_fingerprint_changes_when_any_field_changes():
    altered = BaselineConfig(name=NAIVE_BASELINE.name, model=NAIVE_BASELINE.model,
                             temperature=0.9, top_p=NAIVE_BASELINE.top_p,
                             system_prompt=NAIVE_BASELINE.system_prompt)
    assert baseline_fingerprint(altered) != baseline_fingerprint(NAIVE_BASELINE)

def test_two_baselines_have_distinct_fingerprints():
    assert baseline_fingerprint(NAIVE_BASELINE) != baseline_fingerprint(INFORMED_BASELINE)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_baseline.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `baseline.py`**

```python
# src/persona_twin/eval/baseline.py
"""Pinned baseline configurations (spec §7).

Two baselines are scored, and every later stage reports its delta against both:

* NAIVE    — no identity at all. Measures what the raw model does.
* INFORMED — names the subject's occupation only. Measures what the model can
             do knowing merely who it is imitating.

The gap between them is what separates "the model already knows someone like
this" from "the system learned this person". Without the informed baseline a
twin can look successful while contributing nothing.

These are pinned. Changing a model version or a sampling parameter makes a
different baseline, and every delta measured against the old one becomes
meaningless — so rename rather than edit, and let the fingerprint prove it.
"""
from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict

# The subject's own identity is never written here — it lives in
# config/subjects/<id>.yaml (spec §13.1). The informed baseline names an
# occupation only, supplied at call time by the caller reading that config.
_INFORMED_TEMPLATE = ("You are replying as a {occupation}. "
                      "Reply to this message.")


class BaselineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    model: str
    temperature: float
    top_p: float
    system_prompt: str


NAIVE_BASELINE = BaselineConfig(
    name="naive-v1",
    model="claude-sonnet-4-5-20250929",
    temperature=1.0,
    top_p=1.0,
    system_prompt="Reply to this message.",
)

INFORMED_BASELINE = BaselineConfig(
    name="informed-v1",
    model="claude-sonnet-4-5-20250929",
    temperature=1.0,
    top_p=1.0,
    system_prompt=_INFORMED_TEMPLATE.format(occupation="accountant"),
)


def baseline_fingerprint(cfg: BaselineConfig) -> str:
    """Stable digest of every field that affects generation. If this moves, the
    baseline moved, and prior deltas no longer compare."""
    payload = "|".join([cfg.name, cfg.model, f"{cfg.temperature:.6f}",
                        f"{cfg.top_p:.6f}", cfg.system_prompt])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval_baseline.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/persona_twin/eval/baseline.py tests/test_eval_baseline.py
git commit -m "feat: pinned naive and informed baseline configurations"
```

---

## Task 5: Blind A/B trial construction and scoring (S1)

**Files:**
- Create: `src/persona_twin/eval/ab.py`
- Test: `tests/test_eval_ab.py`

**Interfaces:**
- Consumes: `ReplyPair` (`normalize/threads.py`), `Turn`
- Produces: `ABTrial` frozen dataclass (`trial_id, context: list[Turn], option_a: str, option_b: str, real_is_a: bool`); `build_trials(pairs, candidate_replies: dict[str, str], seed: int = 0) -> list[ABTrial]`; `ABResult` (`n, correct, accuracy, ci_low, ci_high`); `score_trials(trials, guesses: dict[str, str]) -> ABResult`; `wilson_interval(successes, n, z=1.96) -> tuple[float, float]`

**Chance is the target, not zero.** A judge picking the real reply 50% of the time means the system is indistinguishable. S1 passes at ≤60% with `n ≥ 300` and a reported 95% interval. Randomised A/B ordering per trial is what prevents a judge (human or model) from learning a position bias.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_ab.py
from datetime import datetime, timedelta, timezone
from persona_twin.schema import Turn
from persona_twin.normalize.threads import ReplyPair
from persona_twin.eval.ab import build_trials, score_trials, wilson_interval

NOW = datetime(2026, 9, 12, tzinfo=timezone.utc)

def _turn(sid, is_subject, text, mins=0):
    return Turn(subject_id="s", source="imessage", source_id=sid, thread_id="c1",
                ts=NOW + timedelta(minutes=mins), author_id="s" if is_subject else "P-1",
                is_subject=is_subject, text=text, assisted=False)

def _pair(sid):
    return ReplyPair(context=[_turn(f"{sid}-c", False, "what time?")],
                     reply=_turn(sid, True, "around seven"))

def test_trial_carries_both_options_and_knows_which_is_real():
    trials = build_trials([_pair("1")], {"1": "circa 7pm"}, seed=0)
    t = trials[0]
    assert {t.option_a, t.option_b} == {"around seven", "circa 7pm"}
    real = t.option_a if t.real_is_a else t.option_b
    assert real == "around seven"

def test_ordering_is_randomised_across_trials():
    pairs = [_pair(str(i)) for i in range(40)]
    cands = {str(i): f"candidate {i}" for i in range(40)}
    trials = build_trials(pairs, cands, seed=3)
    sides = {t.real_is_a for t in trials}
    assert sides == {True, False}, "real reply always on the same side"

def test_ordering_is_reproducible_for_a_seed():
    pairs = [_pair(str(i)) for i in range(20)]
    cands = {str(i): f"c{i}" for i in range(20)}
    a = [t.real_is_a for t in build_trials(pairs, cands, seed=5)]
    b = [t.real_is_a for t in build_trials(pairs, cands, seed=5)]
    assert a == b

def test_pairs_without_a_candidate_are_skipped():
    trials = build_trials([_pair("1"), _pair("2")], {"1": "only one"}, seed=0)
    assert len(trials) == 1

def test_perfect_judge_scores_one():
    trials = build_trials([_pair("1")], {"1": "fake"}, seed=0)
    t = trials[0]
    guesses = {t.trial_id: "a" if t.real_is_a else "b"}
    assert score_trials(trials, guesses).accuracy == 1.0

def test_chance_judge_scores_about_half():
    pairs = [_pair(str(i)) for i in range(100)]
    cands = {str(i): f"c{i}" for i in range(100)}
    trials = build_trials(pairs, cands, seed=1)
    guesses = {t.trial_id: "a" for t in trials}   # always guess A
    r = score_trials(trials, guesses)
    assert 0.3 < r.accuracy < 0.7, r.accuracy

def test_result_reports_sample_size_and_interval():
    trials = build_trials([_pair("1")], {"1": "fake"}, seed=0)
    r = score_trials(trials, {})
    assert r.n == 1 and r.ci_low <= r.accuracy <= r.ci_high

def test_wilson_interval_brackets_the_point_estimate():
    lo, hi = wilson_interval(30, 100)
    assert lo < 0.30 < hi
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_ab.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `ab.py`**

```python
# src/persona_twin/eval/ab.py
"""Blind A/B trials (spec §7, criterion S1).

A judge sees the subject's real reply and the system's reply to the same
context, in random order, and picks the one they believe is real. 50% is a
perfect score: it means the two are indistinguishable. S1 passes at <= 60%
accuracy with n >= 300 and a reported 95% interval.

Per-trial randomised ordering is what stops a judge — human or model — from
learning that the real reply is always in the same position.
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


def build_trials(pairs: list[ReplyPair], candidate_replies: dict[str, str],
                 seed: int = 0) -> list[ABTrial]:
    """Pair each real reply with a candidate keyed by the reply's source_id.

    Pairs with no candidate are skipped rather than filled, so a generation
    failure shrinks the sample visibly instead of silently biasing it.
    """
    rng = random.Random(seed)
    trials: list[ABTrial] = []
    for p in pairs:
        candidate = candidate_replies.get(p.reply.source_id)
        if candidate is None:
            continue
        real_is_a = rng.random() < 0.5
        trials.append(ABTrial(
            trial_id=p.reply.source_id,
            context=list(p.context),
            option_a=p.reply.text if real_is_a else candidate,
            option_b=candidate if real_is_a else p.reply.text,
            real_is_a=real_is_a,
        ))
    return trials


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
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
    """`guesses` maps trial_id -> "a" | "b". A missing guess counts as wrong,
    which is conservative: it can only make the system look more detectable."""
    correct = 0
    for t in trials:
        picked = (guesses.get(t.trial_id) or "").lower()
        if picked == ("a" if t.real_is_a else "b"):
            correct += 1
    n = len(trials)
    acc = correct / n if n else 0.0
    lo, hi = wilson_interval(correct, n)
    return ABResult(n=n, correct=correct, accuracy=acc, ci_low=lo, ci_high=hi)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval_ab.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/persona_twin/eval/ab.py tests/test_eval_ab.py
git commit -m "feat: blind A/B trial construction and Wilson-interval scoring"
```

---

## Task 6: Gate report and CLI

**Files:**
- Create: `src/persona_twin/eval/report.py`, `tools/eval_report.py`
- Test: `tests/test_eval_report.py`

**Interfaces:**
- Consumes: everything from Tasks 1–5, plus `CorpusStore`, `reply_pairs`
- Produces: `GateResult` frozen dataclass (`criterion, value, target, n, passed, note`); `split_summary(paths, version, now) -> dict[str, int]`; `s2_self_distance(paths, version, now, seed=0) -> GateResult`; `render(results: list[GateResult]) -> str`

**What this task can and cannot do.** S1, S3, S4, S5 and S6 all require generated text, which belongs to stage 5 — this stage defines their machinery and leaves them unscored. S2's self-distance band is computable **now**, from the corpus alone, and is the one real number this stage produces. The report must state plainly which criteria are pending rather than printing a hopeful zero.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_report.py
from datetime import datetime, timedelta, timezone
from persona_twin.paths import SubjectPaths
from persona_twin.schema import Turn
from persona_twin.corpus.store import CorpusStore
from persona_twin.eval.report import GateResult, split_summary, s2_self_distance, render

NOW = datetime(2026, 9, 12, tzinfo=timezone.utc)

# 40 threads, not 25: with 25 the sha256 buckets put ZERO threads on the
# held-out side, and test_split_summary_holds_out_a_nonzero_minority fails
# deterministically. Verified by computing the buckets before execution.
def _seed(tmp_path, n=200):
    p = SubjectPaths("s", tmp_path); p.ensure()
    turns = [Turn(subject_id="s", source="imessage", source_id=str(i),
                  thread_id=f"c{i % 40}", ts=NOW - timedelta(days=400 + i),
                  author_id="s", is_subject=True, text=f"message {i} here",
                  assisted=False) for i in range(n)]
    CorpusStore(p).write("v1", turns)
    return p

def test_split_summary_counts_every_turn(tmp_path):
    p = _seed(tmp_path)
    s = split_summary(p, "v1", NOW)
    assert s["train"] + s["heldout"] + s["quarantined"] == s["total"] == 200

def test_split_summary_holds_out_a_nonzero_minority(tmp_path):
    p = _seed(tmp_path)
    s = split_summary(p, "v1", NOW)
    assert 0 < s["heldout"] < s["train"]

def test_s2_reports_a_sample_size(tmp_path):
    p = _seed(tmp_path)
    r = s2_self_distance(p, "v1", NOW, seed=0)
    assert r.criterion == "S2" and r.n > 0

def test_s2_is_reproducible(tmp_path):
    p = _seed(tmp_path)
    a = s2_self_distance(p, "v1", NOW, seed=4)
    b = s2_self_distance(p, "v1", NOW, seed=4)
    assert a.value == b.value

def test_render_marks_pending_criteria_as_pending():
    results = [GateResult(criterion="S1", value=None, target="<=0.60",
                          n=0, passed=None, note="requires generation (stage 5)")]
    out = render(results)
    assert "S1" in out and "pending" in out.lower()

def test_render_includes_sample_sizes():
    results = [GateResult(criterion="S2", value=1.5, target="<= p95 self-distance",
                          n=123, passed=True, note="")]
    assert "123" in render(results)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_report.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `report.py`**

```python
# src/persona_twin/eval/report.py
"""Stage-3 gate report (spec §2, §7).

Only S2 is computable from the corpus alone. S1, S3, S4, S5 and S6 all require
generated text and are therefore defined here but scored at stage 5. The report
prints those as `pending` rather than as a zero, because a zero in a results
table reads as a measurement.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from persona_twin.corpus.quarantine import DEFAULT_WEEKS
from persona_twin.corpus.store import CorpusStore
from persona_twin.eval.split import split_corpus
from persona_twin.eval.style import self_distance_band
from persona_twin.paths import SubjectPaths


@dataclass(frozen=True)
class GateResult:
    criterion: str
    value: float | None
    target: str
    n: int
    passed: bool | None
    note: str = ""


def split_summary(paths: SubjectPaths, version: str, now: datetime,
                  weeks: int = DEFAULT_WEEKS) -> dict[str, int]:
    turns = CorpusStore(paths).read(version)
    s = split_corpus(turns, now, weeks)
    return {"total": len(turns), "train": len(s.train),
            "heldout": len(s.heldout), "quarantined": len(s.quarantined)}


def s2_self_distance(paths: SubjectPaths, version: str, now: datetime,
                     seed: int = 0, weeks: int = DEFAULT_WEEKS) -> GateResult:
    """The subject's own style self-distance band, measured on held-out turns.

    This is the bar a later system must sit inside — an empirical target rather
    than a chosen threshold.
    """
    turns = CorpusStore(paths).read(version)
    s = split_corpus(turns, now, weeks)
    subject_heldout = [t for t in s.heldout if t.is_subject]
    median, p95 = self_distance_band(subject_heldout, trials=200, seed=seed)
    return GateResult(criterion="S2", value=p95,
                      target="system distance <= p95 of self-distance",
                      n=len(subject_heldout), passed=None,
                      note=f"median self-distance {median:.3f}; baseline only, "
                           "no system to score yet")


_PENDING = [
    ("S1", "<=0.60 accuracy, n>=300", "requires generation (stage 5)"),
    ("S3", ">=0.80 would-say, n>=100", "requires generation and subject review (stage 5)"),
    ("S4", "between-register > within-register", "requires generation (stage 5)"),
    ("S5", ">=0.95 correct, 0 fabrications, n>=200", "requires probe set and generation"),
    ("S6", ">=0.80 agreement, n>=60", "requires probe set and generation"),
]


def pending_results() -> list[GateResult]:
    return [GateResult(criterion=c, value=None, target=t, n=0, passed=None, note=note)
            for c, t, note in _PENDING]


def render(results: list[GateResult]) -> str:
    lines = [f"{'crit':<5}{'value':>12}{'n':>8}  {'status':<9}target / note"]
    for r in results:
        value = "pending" if r.value is None else f"{r.value:.4f}"
        status = "pending" if r.passed is None else ("PASS" if r.passed else "FAIL")
        lines.append(f"{r.criterion:<5}{value:>12}{r.n:>8}  {status:<9}{r.target}"
                     + (f" — {r.note}" if r.note else ""))
    return "\n".join(lines)
```

- [ ] **Step 4: Implement the CLI**

```python
# tools/eval_report.py
"""Stage-3 gate: print the held-out split and every criterion's status."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from persona_twin.eval.report import (pending_results, render, s2_self_distance,  # noqa: E402
                                      split_summary)
from persona_twin.paths import SubjectPaths  # noqa: E402


def main(subject_id: str, version: str) -> int:
    paths = SubjectPaths(subject_id, Path.cwd())
    now = datetime.now(timezone.utc)
    s = split_summary(paths, version, now)
    print(f"corpus version : {version}")
    print(f"total turns    : {s['total']:,}")
    print(f"  train        : {s['train']:,}")
    print(f"  held out     : {s['heldout']:,}")
    print(f"  quarantined  : {s['quarantined']:,}")
    print()
    results = [s2_self_distance(paths, version, now)] + pending_results()
    print(render(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_eval_report.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Run the full suite and the CLI against the real corpus**

```bash
uv run pytest -q
uv run python tools/eval_report.py subject-01 v4
```

Expected: full suite green; the CLI prints a split whose three parts sum to 85,330, an S2 value with its `n`, and five `pending` rows.

- [ ] **Step 7: Commit**

```bash
git add src/persona_twin/eval/report.py tools/eval_report.py tests/test_eval_report.py
git commit -m "feat: stage-3 gate report with computed S2 and pending criteria"
```

---

## Stage 3 gate

- [ ] `uv run pytest -q` passes
- [ ] `uv run python tools/eval_report.py subject-01 v4` runs and its split sums to the corpus total
- [ ] The held-out split is verified stable: re-run the report after any new ingest and confirm previously held-out `source_id`s are still held out
- [ ] S2's self-distance band is recorded in the ledger as the baseline bar
- [ ] Probe sets exist at `data/subjects/<id>/probes/facts.json` and `refusals.json` — **authored by the subject, not generated**, since a model-written probe set would test the model's guess about the subject rather than the subject
- [ ] Note explicitly which criteria remain pending and why

---

## Planning scope

This plan covers stage 3 only. Stage 4 (persona mining) is specified in the design doc but not planned here: its tasks depend on what the held-out split and the S2 band actually show, and on whether the mail and ChatGPT exports have landed by then.
