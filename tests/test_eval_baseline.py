"""Tests for pinned baseline configurations (spec §7)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from persona_twin.config import SubjectConfig
from persona_twin.eval.baseline import (
    NAIVE_BASELINE, BaselineConfig, MissingOccupation, baseline_fingerprint,
    informed_baseline,
)


def _cfg(name="Jordan Rivers", occupation="architect"):
    return SubjectConfig(subject_id="subject-test", display_name=name,
                         occupation=occupation)


INFORMED_BASELINE = informed_baseline(_cfg())


def test_naive_prompt_contains_no_identity():
    p = NAIVE_BASELINE.system_prompt.lower()
    for leak in ("name", "cpa", "accountant", "subject", "persona"):
        assert leak not in p, f"naive baseline prompt leaks identity: {leak}"


def test_informed_baseline_differs_from_naive():
    assert INFORMED_BASELINE.system_prompt != NAIVE_BASELINE.system_prompt


def test_informed_baseline_names_the_subject_and_occupation():
    # Spec §7: the informed baseline must name the subject AND their
    # occupation, not occupation alone -- an informed baseline missing the
    # name is weaker than the real thing it stands in for, which widens the
    # naive/informed gap artificially and flatters any twin scored against it.
    p = informed_baseline(_cfg()).system_prompt
    assert "Jordan Rivers" in p
    assert "architect" in p.lower()


def test_informed_prompt_derives_from_config_not_a_literal():
    """Two subjects must yield two prompts -- the guard against re-hardcoding.

    A hardcoded name passes every assertion about one subject's prompt; only
    varying the config catches it. This is what keeps subject identity in
    git-ignored config, where tools/name_leak_lint.py expects it.
    """
    a = informed_baseline(_cfg("Ada Fielding", "botanist")).system_prompt
    b = informed_baseline(_cfg("Rex Calloway", "welder")).system_prompt
    assert a != b
    assert "Ada Fielding" in a and "Ada Fielding" not in b
    assert "botanist" in a and "welder" in b


def test_informed_baseline_refuses_a_subject_with_no_occupation():
    """Fail loudly rather than emit a weaker informed baseline.

    Silently dropping the occupation makes the informed baseline weaker than
    the thing it stands in for, which widens the naive/informed gap and
    flatters any twin scored against it -- the flattering direction, so it
    must raise instead of degrade.
    """
    with pytest.raises(MissingOccupation):
        informed_baseline(_cfg(occupation=None))


def test_informed_prompt_keeps_its_pinned_wording():
    """The template is pinned; rewording it makes a different baseline.

    Asserted on a placeholder subject so no real identity enters tracked
    tests. The real subject's fingerprint continuity is verified against
    git-ignored config, not here.
    """
    assert (informed_baseline(_cfg("Ada Fielding", "botanist")).system_prompt
            == "You are replying as Ada Fielding. Your occupation is botanist. "
               "Reply to this message.")


def test_informed_prompt_is_grammatical_for_any_occupation():
    """The template carries no indefinite article, so it cannot be wrong for
    one. "an {occupation}" reads correctly for "architect" and wrongly for
    "CPA", "lawyer" or "software engineer" -- and the harness is meant to be
    reusable for a second subject whose occupation is not knowable here. A
    malformed informed baseline is a weaker stand-in than the thing it
    represents, which widens the naive/informed gap in the flattering
    direction.

    Checked against occupations that break BOTH article choices: "an" is wrong
    for lawyer, "a" is wrong for architect and editor, and the a/an rule is
    about pronunciation rather than spelling ("an hour", "a union"), so
    computing the article would be a second defect rather than a fix.
    """
    for occupation in ("CPA", "lawyer", "architect", "software engineer",
                       "editor", "hour-billing consultant", "underwriter"):
        prompt = informed_baseline(_cfg("Ada Fielding", occupation)).system_prompt
        assert f"occupation is {occupation}." in prompt, prompt
        for article in (" a ", " an ", " A ", " An "):
            assert article not in prompt, (occupation, prompt)


def test_renaming_the_informed_baseline_tracked_its_reworded_template():
    """Rename-rather-than-edit, observed: the template lost its article, so
    the baseline is a different baseline and says so. A reworded prompt still
    carrying the old name would silently invalidate every delta measured
    against it."""
    assert informed_baseline(_cfg()).name == "informed-v2"


def test_baselines_pin_sampling_parameters():
    for b in (NAIVE_BASELINE, INFORMED_BASELINE):
        assert b.temperature is not None and b.top_p is not None


def test_config_is_frozen():
    with pytest.raises(Exception):
        NAIVE_BASELINE.temperature = 0.9


def test_config_requires_every_field():
    # A field that can be silently omitted is a field that can silently drift
    # without construction failing -- confirm every field is mandatory.
    with pytest.raises(ValidationError):
        BaselineConfig(name="x", model="y", temperature=1.0, top_p=1.0)  # missing system_prompt


def test_fingerprint_is_stable():
    assert baseline_fingerprint(NAIVE_BASELINE) == baseline_fingerprint(NAIVE_BASELINE)


def test_fingerprint_is_equal_for_separately_constructed_identical_configs():
    # Two distinct objects with identical field values, not the same object
    # twice -- the fingerprint must be a function of value, not identity.
    a = BaselineConfig(name="n", model="m", temperature=0.5, top_p=0.9,
                       system_prompt="s")
    b = BaselineConfig(name="n", model="m", temperature=0.5, top_p=0.9,
                       system_prompt="s")
    assert a is not b
    assert baseline_fingerprint(a) == baseline_fingerprint(b)


def test_two_baselines_have_distinct_fingerprints():
    assert baseline_fingerprint(NAIVE_BASELINE) != baseline_fingerprint(INFORMED_BASELINE)


# --- One test per field: each must move the fingerprint on its own. A
# fingerprint that silently drops a field lets a baseline drift on that field
# while keeping its old identity, which quietly invalidates every delta
# measured against it. ---

def test_fingerprint_changes_when_name_changes():
    altered = NAIVE_BASELINE.model_copy(update={"name": "naive-v2"})
    assert baseline_fingerprint(altered) != baseline_fingerprint(NAIVE_BASELINE)


def test_fingerprint_changes_when_model_changes():
    altered = NAIVE_BASELINE.model_copy(update={"model": "claude-opus-5"})
    assert baseline_fingerprint(altered) != baseline_fingerprint(NAIVE_BASELINE)


def test_fingerprint_changes_when_temperature_changes():
    altered = NAIVE_BASELINE.model_copy(update={"temperature": 0.9})
    assert baseline_fingerprint(altered) != baseline_fingerprint(NAIVE_BASELINE)


def test_fingerprint_changes_when_top_p_changes():
    altered = NAIVE_BASELINE.model_copy(update={"top_p": 0.9})
    assert baseline_fingerprint(altered) != baseline_fingerprint(NAIVE_BASELINE)


def test_fingerprint_changes_when_system_prompt_changes():
    altered = NAIVE_BASELINE.model_copy(update={"system_prompt": "Reply."})
    assert baseline_fingerprint(altered) != baseline_fingerprint(NAIVE_BASELINE)


# --- Null-system / edge-case checks on the fingerprint itself. ---

def test_fingerprint_handles_empty_system_prompt():
    cfg = BaselineConfig(name="n", model="m", temperature=1.0, top_p=1.0,
                         system_prompt="")
    fp = baseline_fingerprint(cfg)
    assert fp and isinstance(fp, str)
    other = cfg.model_copy(update={"system_prompt": "x"})
    assert baseline_fingerprint(other) != fp


def test_fingerprint_handles_empty_name():
    cfg = BaselineConfig(name="", model="m", temperature=1.0, top_p=1.0,
                         system_prompt="s")
    fp = baseline_fingerprint(cfg)
    assert fp and isinstance(fp, str)
    other = cfg.model_copy(update={"name": "x"})
    assert baseline_fingerprint(other) != fp


def test_fingerprint_does_not_collide_across_a_field_boundary():
    # Regression for a plain "|".join(...) fingerprint: name="X|Y", model="Z"
    # and name="X", model="Y|Z" produce the identical joined string
    # "X|Y|Z|1.000000|1.000000|s", so two DIFFERENT configs would fingerprint
    # identically. JSON-quoted serialization keeps field boundaries
    # unambiguous regardless of what characters the string fields contain.
    a = BaselineConfig(name="X|Y", model="Z", temperature=1.0, top_p=1.0,
                       system_prompt="s")
    b = BaselineConfig(name="X", model="Y|Z", temperature=1.0, top_p=1.0,
                       system_prompt="s")
    assert baseline_fingerprint(a) != baseline_fingerprint(b)
