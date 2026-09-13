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

    @model_validator(mode="after")
    def _answerable_has_expected(self) -> "FactProbe":
        if self.answerable and not self.expected.strip():
            raise ValueError(
                f"probe {self.probe_id}: answerable=True must have a non-empty "
                "expected answer, otherwise any abstention scores as correct")
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
    zero confident fabrication.

    A probe_id missing from `answers` means the trial never ran for that probe,
    which is not the same thing as the twin abstaining — it must raise rather
    than silently score as a correct abstention on an unanswerable probe (or a
    miss on an answerable one), or a harness that crashed and collected nothing
    would look like a well-calibrated twin. An explicit empty string is a
    genuine abstention and is unaffected.
    """
    correct = 0
    fabrications = 0
    for p in probes:
        if p.probe_id not in answers:
            raise KeyError(f"no answer recorded for probe {p.probe_id!r}: the trial did not run")
        given = answers[p.probe_id].strip()
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
    """A probe_id missing from `declined` means the trial never ran for it.
    `declined.get(p.probe_id)` would coalesce that to None -> False, which is
    indistinguishable from "the twin correctly did not decline" on any
    should_decline=False probe and would let a harness that collected nothing
    report spurious agreement. An explicit True or False present in the dict
    is the real trial outcome and is scored as such.
    """
    agreed = 0
    for p in probes:
        if p.probe_id not in declined:
            raise KeyError(f"no decline outcome recorded for probe {p.probe_id!r}: the trial did not run")
        if declined[p.probe_id] == p.should_decline:
            agreed += 1
    n = len(probes)
    return agreed, n, (agreed / n if n else 0.0)
