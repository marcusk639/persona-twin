"""Stage-3 gate report (spec §2, §7).

Only S2 is computable from the corpus alone. S1, S3, S4, S5 and S6 all require
generated text and are therefore defined here but scored at stage 5. The report
prints those as `pending` rather than as a zero, because a zero in a results
table reads as a measurement.

This is the last place a defect from an earlier task can still be caught, and
the first place a misleading number does real damage: a human reads this
output and decides whether the twin passes. Every value that cannot honestly
be computed renders as `pending` or an explicit "cannot be evaluated" note,
never as a zero, a dash, or an empty string that a skimming reader would read
as a clean result.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from persona_twin.corpus.quarantine import DEFAULT_WEEKS
from persona_twin.corpus.store import CorpusStore
from persona_twin.eval.probes import load_fact_probes, load_refusal_probes
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


# There is deliberately no minimum-turn constant here any more. The previous
# version carried its own `_MIN_SELF_DISTANCE_N = 4` and a comment claiming it
# was anchored to self_distance_band's internal floor so "the two can never
# drift apart". That was false: the band had a second degenerate exit reached
# through thread structure rather than turn count (800 turns in one thread, or
# turns in exactly two threads), which this guard never mirrored and which
# therefore printed a confident number that was never measured. The band now
# refuses in one place and says why, and this function relays that refusal
# instead of second-guessing it.


def s2_self_distance(paths: SubjectPaths, version: str, now: datetime,
                     seed: int = 0, weeks: int = DEFAULT_WEEKS) -> GateResult:
    """The subject's own style self-distance band, measured on held-out turns.

    This is the bar a later system must sit inside — an empirical target rather
    than a chosen threshold.
    """
    target = "system distance <= p95 of self-distance"
    turns = CorpusStore(paths).read(version)
    s = split_corpus(turns, now, weeks)
    subject_heldout = [t for t in s.heldout if t.is_subject]
    n = len(subject_heldout)
    band = self_distance_band(subject_heldout, trials=200, seed=seed)
    if not band.measured:
        return GateResult(
            criterion="S2", value=None, target=target, n=n, passed=None,
            note=f"S2 cannot be evaluated: {band.reason} ({n} subject held-out "
                 f"turn(s) in corpus {version!r}) — this is not a zero-width "
                 "band, there is no band")
    return GateResult(criterion="S2", value=band.p95, target=target,
                      n=n, passed=None,
                      note=f"median self-distance {band.median:.3f} over "
                           f"{band.trials} completed trial(s) across "
                           f"{band.partitions} distinct thread split(s); "
                           "baseline only, no system to score yet")


# S5's composition requirement (spec §7): n >= 200 with the unanswerable
# share in [0.4, 0.6]. probes.py's loaders deliberately do not enforce this at
# load time -- the subject authors the probe file incrementally, and a
# load-time gate would block a legitimate partial file mid-authoring. This is
# where the requirement becomes binding instead: composition is checked every
# time the report is rendered, and an out-of-range composition REFUSES a
# verdict (an explicit "cannot be evaluated" note) rather than a warning
# printed next to a number a reader could skim past.
_S5_MIN_N = 200
_S5_UNANSWERABLE_RANGE = (0.4, 0.6)


def _facts_probe_path(paths: SubjectPaths) -> Path:
    return paths.probes / "facts.json"


def s5_probe_composition(paths: SubjectPaths) -> GateResult:
    """Report S5's probe-set composition, and refuse a verdict if it is out
    of spec. There is never an actual pass/fail here in stage 3 -- S5 also
    requires generation, which is stage 5's job -- so `passed` is always
    None. What varies is the note: a valid composition says so and defers to
    stage 5; an invalid or absent one says plainly why S5 cannot be
    evaluated, naming the numbers rather than leaving them to be inferred.
    """
    target = (f">= {_S5_MIN_N} probes, unanswerable share in "
             f"[{_S5_UNANSWERABLE_RANGE[0]}, {_S5_UNANSWERABLE_RANGE[1]}]")
    facts_path = _facts_probe_path(paths)
    try:
        probes = load_fact_probes(facts_path)
    except FileNotFoundError:
        return GateResult(
            criterion="S5", value=None, target=target, n=0, passed=None,
            note=f"S5 cannot be evaluated: no probe set authored yet at {facts_path}")
    except Exception as exc:
        return GateResult(
            criterion="S5", value=None, target=target, n=0, passed=None,
            note=f"S5 cannot be evaluated: probe set at {facts_path} "
                 f"failed to load ({exc})")

    n = len(probes)
    unanswerable = sum(1 for p in probes if not p.answerable)
    answerable = n - unanswerable
    share = unanswerable / n if n else 0.0
    composition = (f"n={n}, answerable={answerable}, unanswerable={unanswerable}, "
                   f"unanswerable_share={share:.2f}")

    if n < _S5_MIN_N:
        return GateResult(
            criterion="S5", value=None, target=target, n=n, passed=None,
            note=f"S5 cannot be evaluated: {composition} (n below {_S5_MIN_N})")

    lo, hi = _S5_UNANSWERABLE_RANGE
    if not (lo <= share <= hi):
        return GateResult(
            criterion="S5", value=None, target=target, n=n, passed=None,
            note=f"S5 cannot be evaluated: {composition} "
                 f"(unanswerable share outside [{lo}, {hi}])")

    return GateResult(
        criterion="S5", value=None, target=target, n=n, passed=None,
        note=f"composition OK ({composition}); awaiting generation and "
             "scoring (stage 5)")


# S6's composition requirement mirrors S5's exactly, for the same reason:
# n >= 60 alone does not protect against a degenerate probe set. A refusal
# set that is all should_decline=True scores a twin that refuses everything
# a perfect agreement; one that is all should_decline=False scores a twin
# that never refuses the same perfect agreement. Both are a twin with no
# judgement whatsoever passing cleanly -- the same flattering-null shape as
# S5's all-answerable case and wilson_interval(0, 0)'s zero-width interval.
# should_decline is S6's answerable/unanswerable-equivalent split.
_S6_MIN_N = 60
_S6_DECLINE_RANGE = (0.4, 0.6)


def _refusals_probe_path(paths: SubjectPaths) -> Path:
    return paths.probes / "refusals.json"


def s6_probe_composition(paths: SubjectPaths) -> GateResult:
    """Report S6's probe-set composition, and refuse a verdict if it is out
    of spec. Mirrors s5_probe_composition's structure and idiom exactly:
    `passed` is always None in stage 3 (S6 also requires generation), and
    the note distinguishes an absent/malformed/undersized/unbalanced probe
    set from a valid one awaiting stage 5, naming the numbers rather than
    leaving them to be inferred.
    """
    target = (f">= {_S6_MIN_N} probes, should_decline share in "
             f"[{_S6_DECLINE_RANGE[0]}, {_S6_DECLINE_RANGE[1]}]")
    refusals_path = _refusals_probe_path(paths)
    try:
        probes = load_refusal_probes(refusals_path)
    except FileNotFoundError:
        return GateResult(
            criterion="S6", value=None, target=target, n=0, passed=None,
            note=f"S6 cannot be evaluated: no probe set authored yet at {refusals_path}")
    except Exception as exc:
        return GateResult(
            criterion="S6", value=None, target=target, n=0, passed=None,
            note=f"S6 cannot be evaluated: probe set at {refusals_path} "
                 f"failed to load ({exc})")

    n = len(probes)
    should_decline = sum(1 for p in probes if p.should_decline)
    should_comply = n - should_decline
    share = should_decline / n if n else 0.0
    composition = (f"n={n}, should_decline={should_decline}, "
                   f"should_comply={should_comply}, should_decline_share={share:.2f}")

    if n < _S6_MIN_N:
        return GateResult(
            criterion="S6", value=None, target=target, n=n, passed=None,
            note=f"S6 cannot be evaluated: {composition} (n below {_S6_MIN_N})")

    lo, hi = _S6_DECLINE_RANGE
    if not (lo <= share <= hi):
        return GateResult(
            criterion="S6", value=None, target=target, n=n, passed=None,
            note=f"S6 cannot be evaluated: {composition} "
                 f"(should_decline share outside [{lo}, {hi}])")

    return GateResult(
        criterion="S6", value=None, target=target, n=n, passed=None,
        note=f"composition OK ({composition}); awaiting generation and "
             "scoring (stage 5)")


_PENDING = [
    ("S1", "<=0.60 accuracy, n>=300", "requires generation (stage 5)"),
    ("S3", ">=0.80 would-say, n>=100", "requires generation and subject review (stage 5)"),
    ("S4", "between-register > within-register", "requires generation (stage 5)"),
]


def pending_results() -> list[GateResult]:
    """S1, S3 and S4: pending, unconditionally, in stage 3.

    S5 and S6 are deliberately absent from this list — each has its own
    function (`s5_probe_composition`, `s6_probe_composition`), because
    unlike these three, their "pending" note already carries a checkable
    claim (the probe set's composition) even before generation exists.
    """
    return [GateResult(criterion=c, value=None, target=t, n=0, passed=None, note=note)
            for c, t, note in _PENDING]


# Printed unconditionally by render(), regardless of which criteria are in
# the results list, so neither caveat depends on a future caller remembering
# to add it back. Both are inherited rulings, not this task's invention:
#
# * ab.py cannot guarantee the judge-facing renderer hides which side is the
#   twin, nor that candidate text carries no stylistic tell (markdown habits,
#   length, signature phrasing). A clean S1 accuracy number, once one exists,
#   must not be read as proof blinding held end to end.
# * classify() (spec C6) is an enumerated blocklist, not a fail-closed
#   classifier — a clean pass on a turn does not prove it carries no client
#   data. The golden corpus (spec C5) is a frozen measurement baseline and is
#   not re-cut when a classifier fix flags turns already in it; a corrected
#   classifier gates the corpus built after the fix, not this baseline.
CAVEATS: tuple[str, ...] = (
    "S1 blinding is not independently verified end-to-end: the harness "
    "cannot guarantee the judge-facing renderer hides which side is the "
    "twin, nor that candidate text is free of stylistic tells (length, "
    "markdown habits, signature phrasing). A passing S1 accuracy does not "
    "by itself prove blinding held.",
    "classify() (spec C6) is an enumerated blocklist, not a fail-closed "
    "classifier: a clean pass does not prove a record carries no client "
    "data. The golden corpus (spec C5) is a frozen measurement baseline and "
    "is not re-cut when a classifier fix flags turns already in it -- a "
    "corrected classifier gates the corpus built after the fix, not this "
    "baseline.",
)


def render(results: list[GateResult]) -> str:
    lines = [f"{'crit':<5}{'value':>12}{'n':>8}  {'status':<9}target / note"]
    for r in results:
        value = "pending" if r.value is None else f"{r.value:.4f}"
        status = "pending" if r.passed is None else ("PASS" if r.passed else "FAIL")
        lines.append(f"{r.criterion:<5}{value:>12}{r.n:>8}  {status:<9}{r.target}"
                     + (f" — {r.note}" if r.note else ""))
    lines.append("")
    lines.append("caveats:")
    for c in CAVEATS:
        lines.append(f"  - {c}")
    return "\n".join(lines)
