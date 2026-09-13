"""Tests for pinned baseline configurations (spec §7)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from persona_twin.eval.baseline import (
    NAIVE_BASELINE, INFORMED_BASELINE, BaselineConfig, baseline_fingerprint,
)


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
    p = INFORMED_BASELINE.system_prompt
    assert "Marcus Klein" in p
    assert "accountant" in p.lower()


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
