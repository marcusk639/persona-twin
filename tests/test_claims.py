import json
from difflib import SequenceMatcher
import pydantic
import pytest
from persona_twin.harvest.claims import load_report, mark_recurrence

REPORT = [
    {"text": "Opens terse when annoyed", "evidence_quote": "just fix it",
     "source_confidence": "high", "disconfirming_case": "sometimes writes long"},
    {"text": "Avoids exclamation marks", "evidence_quote": "ok sounds good",
     "source_confidence": "medium", "disconfirming_case": "none found"},
]

def test_claims_load_as_unverified_t2(tmp_path):
    p = tmp_path / "r.json"; p.write_text(json.dumps(REPORT))
    claims = load_report(p, source_model="chatgpt", run=1)
    assert len(claims) == 2
    assert all(c.tier == "T2" and c.verification_status == "unverified" for c in claims)

def test_recurring_claims_are_flagged(tmp_path):
    a = tmp_path / "a.json"; a.write_text(json.dumps(REPORT))
    b = tmp_path / "b.json"; b.write_text(json.dumps(REPORT))
    run_a = load_report(a, "chatgpt", 1)
    run_b = load_report(b, "chatgpt", 2)
    assert all(c.recurrence for c in mark_recurrence(run_a, run_b))

def test_drifting_claims_are_not_flagged(tmp_path):
    a = tmp_path / "a.json"; a.write_text(json.dumps(REPORT))
    b = tmp_path / "b.json"
    b.write_text(json.dumps([{"text": "Loves long philosophical asides",
                              "evidence_quote": "q", "source_confidence": "low",
                              "disconfirming_case": "n/a"}]))
    merged = mark_recurrence(load_report(a, "chatgpt", 1), load_report(b, "chatgpt", 2))
    assert not any(c.recurrence for c in merged)

def test_empty_report_is_valid(tmp_path):
    p = tmp_path / "r.json"; p.write_text("[]")
    assert load_report(p, "claude", 1) == []

def test_missing_file_raises_not_empty(tmp_path):
    p = tmp_path / "missing.json"
    try:
        load_report(p, "claude", 1)
        assert False, "expected an exception for a missing report file"
    except OSError:
        pass

def test_malformed_report_raises_not_empty(tmp_path):
    p = tmp_path / "bad.json"; p.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(ValueError):
        load_report(p, "claude", 1)

def test_claim_fields_are_immutable(tmp_path):
    p = tmp_path / "r.json"; p.write_text(json.dumps(REPORT))
    claim = load_report(p, "chatgpt", 1)[0]
    with pytest.raises(pydantic.ValidationError):
        claim.text = "mutated"

def test_recurrence_boundary_just_above_threshold_is_flagged(tmp_path):
    # SequenceMatcher ratio computed and asserted explicitly so the test
    # documents where the 0.6 cutoff actually falls, rather than hoping a
    # hand-picked string lands on the intended side.
    text_a = "Opens terse when annoyed"
    text_b = "Opens terse when frustrated and a bit busy"
    ratio = SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()
    assert ratio > 0.6, ratio

    a = tmp_path / "a.json"; a.write_text(json.dumps([{
        "text": text_a, "evidence_quote": "q",
        "source_confidence": "high", "disconfirming_case": "n/a"}]))
    b = tmp_path / "b.json"; b.write_text(json.dumps([{
        "text": text_b, "evidence_quote": "q",
        "source_confidence": "high", "disconfirming_case": "n/a"}]))
    merged = mark_recurrence(load_report(a, "chatgpt", 1), load_report(b, "chatgpt", 2))
    assert merged[0].recurrence is True

def test_recurrence_boundary_just_below_threshold_is_not_flagged(tmp_path):
    text_a = "Opens terse when annoyed"
    text_b = "Opens terse when frustrated and rather busy"
    ratio = SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()
    assert ratio < 0.6, ratio

    a = tmp_path / "a.json"; a.write_text(json.dumps([{
        "text": text_a, "evidence_quote": "q",
        "source_confidence": "high", "disconfirming_case": "n/a"}]))
    b = tmp_path / "b.json"; b.write_text(json.dumps([{
        "text": text_b, "evidence_quote": "q",
        "source_confidence": "high", "disconfirming_case": "n/a"}]))
    merged = mark_recurrence(load_report(a, "chatgpt", 1), load_report(b, "chatgpt", 2))
    assert merged[0].recurrence is False
