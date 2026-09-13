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

def test_missing_answer_raises_keyerror_naming_the_probe():
    """A probe_id absent from answers means the trial did not run for it — this
    must not silently score as a correct abstention (or an incorrect answer)."""
    probes = [FactProbe(probe_id="a", question="q", expected="Dallas", answerable=True, notes=""),
              FactProbe(probe_id="b", question="q", expected="", answerable=False, notes="")]
    with pytest.raises(KeyError, match="b"):
        score_fact_answers(probes, {"a": "Dallas"})

def test_explicit_empty_string_is_still_a_valid_abstention():
    """Distinguishing the controller ruling from the brief's original behaviour:
    a probe_id that IS present with an explicit empty/whitespace answer is a
    genuine abstention, not a missing trial, and must not raise."""
    probes = [FactProbe(probe_id="b", question="q", expected="", answerable=False, notes="")]
    score = score_fact_answers(probes, {"b": "   "})
    assert score.correct == 1 and score.fabrications == 0
