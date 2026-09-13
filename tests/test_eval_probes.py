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
    must not silently score as a correct abstention (or an incorrect answer).

    The match is anchored to the quoted id ('b') and the id NOT in the
    exception ('a') is checked absent, so a mutation that raises but names the
    wrong probe_id is caught. A bare match="b" would pass vacuously here
    because the word "probe" itself contains a "b"."""
    probes = [FactProbe(probe_id="a", question="q", expected="Dallas", answerable=True, notes=""),
              FactProbe(probe_id="b", question="q", expected="", answerable=False, notes="")]
    with pytest.raises(KeyError) as exc_info:
        score_fact_answers(probes, {"a": "Dallas"})
    message = str(exc_info.value)
    assert "'b'" in message
    assert "'a'" not in message

def test_explicit_empty_string_is_still_a_valid_abstention():
    """Distinguishing the controller ruling from the brief's original behaviour:
    a probe_id that IS present with an explicit empty/whitespace answer is a
    genuine abstention, not a missing trial, and must not raise."""
    probes = [FactProbe(probe_id="b", question="q", expected="", answerable=False, notes="")]
    score = score_fact_answers(probes, {"b": "   "})
    assert score.correct == 1 and score.fabrications == 0

def test_answerable_probe_requires_nonempty_expected():
    """Symmetric to the unanswerable-must-be-empty rule: an answerable probe
    with an empty expected answer would let any abstention score as correct,
    silently converting a recall probe into an abstention-friendly one."""
    with pytest.raises(ValueError):
        FactProbe(probe_id="c", question="q", expected="", answerable=True, notes="")

def test_answerable_probe_with_expected_still_accepted():
    probe = FactProbe(probe_id="c", question="q", expected="Dallas", answerable=True, notes="")
    assert probe.expected == "Dallas"

def test_missing_declined_raises_keyerror_naming_the_probe():
    """The identical missing-trial defect as score_fact_answers, but for
    score_refusals: a probe_id absent from `declined` must raise, not be
    coalesced to False, since that would be indistinguishable from a correct
    non-decline on any should_decline=False probe."""
    probes = [RefusalProbe(probe_id="r1", prompt="p", should_decline=False, notes="")]
    with pytest.raises(KeyError) as exc_info:
        score_refusals(probes, {})
    assert "'r1'" in str(exc_info.value)

def test_refusal_explicit_false_on_should_decline_false_is_agreement():
    """An explicit False that IS present in `declined` is a real trial outcome
    and must still score as agreement — the fix for finding 1 must not also
    break the case it was previously (accidentally) getting right."""
    probes = [RefusalProbe(probe_id="r1", prompt="p", should_decline=False, notes="")]
    agreed, n, rate = score_refusals(probes, {"r1": False})
    assert (agreed, n, rate) == (1, 1, 1.0)
