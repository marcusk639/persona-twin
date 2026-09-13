"""Pinned baseline configurations (spec §7).

Two baselines are scored, and every later stage reports its delta against both:

* NAIVE    — no identity at all. Measures what the raw model does.
* INFORMED — names the subject and their occupation, nothing more. Measures
             what the model can already do knowing merely who it is imitating,
             before the system has learned anything from the corpus.

The gap between them is what separates "the model already knows a CPA with
this name" from "the system learned this person" (spec §7). If the informed
baseline is weaker than the real thing it claims to be — say, it withholds the
subject's name and only states an occupation — the gap it should measure
widens artificially and any twin looks like it contributed more than it did.
That is the flattering direction, so the informed prompt below names the
subject exactly as spec §7 requires: name and occupation, nothing else.

These are pinned. Changing a model version or a sampling parameter makes a
different baseline, and every delta measured against the old one becomes
meaningless — so rename rather than edit, and let the fingerprint prove it.

informed-v1 was renamed to informed-v2 when its template lost an indefinite
article (see _INFORMED_TEMPLATE). That is the rename-rather-than-edit rule
being followed, not bypassed: the wording changed, so the baseline changed, so
the name and fingerprint changed with it. Safe now because nothing has been
scored against informed-v1; it would not be safe once stage 5 has.

Subject identity is NOT stored here. Name and occupation are read from the
subject's git-ignored `SubjectConfig`, so this module stays free of any one
person's details -- which is what `tools/name_leak_lint.py` enforces and what
makes the harness reusable for a second subject (spec §13). The naive
baseline is a module constant because it contains no identity at all; the
informed one is a function of a config, because it necessarily does.

Pinning still applies: the template wording below is part of the baseline.
Rewording it makes a different baseline and invalidates every delta measured
against the old one, so rename rather than edit and let the fingerprint prove
it.
"""
from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict

from persona_twin.config import SubjectConfig


class MissingOccupation(Exception):
    """The subject config carries no occupation, which the informed baseline needs."""

    def __init__(self, subject_id: str) -> None:
        self.subject_id = subject_id
        super().__init__(
            f"subject {subject_id!r} has no occupation; the informed baseline "
            "must name the subject AND their occupation (spec §7). Emitting "
            "the name alone would make this baseline weaker than the thing it "
            "stands in for, widening the naive/informed gap and flattering "
            "any twin scored against it -- so this fails rather than degrades."
        )


# No indefinite article. "an {occupation}" is correct for "architect" and
# wrong for "CPA", "lawyer" or "software engineer" -- and this harness is meant
# to be reusable for a second subject (spec §13), so the occupation is not
# knowable in advance. A malformed informed baseline is a WEAKER stand-in than
# the thing it represents, which widens the naive/informed gap and flatters any
# twin scored against it: the exact direction this module's MissingOccupation
# guard exists to prevent. Restructured so no article is needed, rather than
# computing a/an from a first letter -- "an hour", "a union" show that rule is
# about pronunciation, not spelling, and a wrong article here is silent.
_INFORMED_TEMPLATE = ("You are replying as {subject_name}. Your occupation is "
                      "{occupation}. Reply to this message.")


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

def informed_baseline(config: SubjectConfig) -> BaselineConfig:
    """The informed baseline for one subject, built from their git-ignored config.

    A function rather than a module constant: the prompt necessarily contains
    the subject's identity, and a constant would have to hardcode it here in
    tracked source. Same pinned model and sampling parameters as the naive
    baseline, so the only difference between the two is the identity in the
    system prompt -- which is exactly what the naive/informed gap measures.
    """
    if not (config.occupation or "").strip():
        raise MissingOccupation(config.subject_id)
    return BaselineConfig(
        name="informed-v2",
        model="claude-sonnet-4-5-20250929",
        temperature=1.0,
        top_p=1.0,
        system_prompt=_INFORMED_TEMPLATE.format(subject_name=config.display_name,
                                                occupation=config.occupation),
    )


def baseline_fingerprint(cfg: BaselineConfig) -> str:
    """Stable digest of every field that affects generation. If this moves, the
    baseline moved, and prior deltas no longer compare.

    Serializes via model_dump_json rather than joining fields with a plain
    delimiter: a delimiter join is not injective when the fields themselves
    can contain the delimiter — e.g. name="X|Y", model="Z" and name="X",
    model="Y|Z" join to the identical string "X|Y|Z|...", so two different
    configs would silently share a fingerprint. JSON's quoting keeps field
    boundaries unambiguous no matter what the string fields contain, and it
    round-trips float precision exactly instead of truncating to a fixed
    number of decimal places.
    """
    payload = cfg.model_dump_json()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
