import json
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
    import pytest
    with pytest.raises(ValueError):
        load_report(p, "claude", 1)
