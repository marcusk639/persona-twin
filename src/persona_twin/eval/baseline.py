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

Single-subject scope note: config/subjects/subject-01.yaml is the only
subject record this project has, and it carries no occupation field, so the
subject's name and occupation are pinned here as literals rather than read
from config at call time. When this harness grows a second subject, this
module should take both as parameters instead of hardcoding them.
"""
from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict

# Matches config/subjects/subject-01.yaml's display_name. See the module
# docstring's single-subject scope note.
_SUBJECT_NAME = "Marcus Klein"
_SUBJECT_OCCUPATION = "accountant"

_INFORMED_TEMPLATE = ("You are replying as {subject_name}, an {occupation}. "
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
    system_prompt=_INFORMED_TEMPLATE.format(subject_name=_SUBJECT_NAME,
                                            occupation=_SUBJECT_OCCUPATION),
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
