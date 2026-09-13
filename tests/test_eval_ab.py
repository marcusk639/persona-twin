from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from persona_twin.eval.ab import ABTrial, build_trials, score_trials, wilson_interval
from persona_twin.normalize.threads import ReplyPair
from persona_twin.schema import Turn

NOW = datetime(2026, 9, 12, tzinfo=timezone.utc)


def _turn(sid, is_subject, text, thread_id="c1", mins=0):
    return Turn(subject_id="s", source="imessage", source_id=sid, thread_id=thread_id,
                ts=NOW + timedelta(minutes=mins), author_id="s" if is_subject else "P-1",
                is_subject=is_subject, text=text, assisted=False)


def _pair(sid, thread_id=None):
    # Each synthetic pair is its own conversation by default -- distinct
    # thread_ids -- since build_trials caps trials at one per thread and most
    # tests here want one trial per pair, not one trial total.
    thread_id = f"t{sid}" if thread_id is None else thread_id
    return ReplyPair(context=[_turn(f"{sid}-c", False, "what time?", thread_id=thread_id)],
                     reply=_turn(sid, True, "around seven", thread_id=thread_id))


def test_trial_carries_both_options_and_knows_which_is_real():
    trials = build_trials([_pair("1")], {"1": "circa 7pm"}, seed=0)
    t = trials[0]
    assert {t.option_a, t.option_b} == {"around seven", "circa 7pm"}
    real = t.option_a if t.real_is_a else t.option_b
    assert real == "around seven"


def test_ordering_is_randomised_across_trials():
    pairs = [_pair(str(i)) for i in range(40)]
    cands = {str(i): f"candidate {i}" for i in range(40)}
    trials = build_trials(pairs, cands, seed=3)
    sides = {t.real_is_a for t in trials}
    assert sides == {True, False}, "real reply always on the same side"


def test_ordering_is_reproducible_for_a_seed():
    pairs = [_pair(str(i)) for i in range(20)]
    cands = {str(i): f"c{i}" for i in range(20)}
    a = [t.real_is_a for t in build_trials(pairs, cands, seed=5)]
    b = [t.real_is_a for t in build_trials(pairs, cands, seed=5)]
    assert a == b


def test_ordering_is_driven_by_the_seed_not_by_trial_position():
    """Distinguishes real randomisation from a position-based pattern (e.g.
    real_is_a alternating by trial index) that happens to still vary across
    True/False and still reproduces for a fixed seed -- both of which the
    two tests above alone would accept. A pattern keyed off position instead
    of the seed is exactly the "position bias" that lets a judge learn which
    side is real; with genuine per-trial randomisation the chance two
    different seeds produce an identical 40-long sequence is about 2**-40,
    so this is not a flaky assertion.
    """
    pairs = [_pair(str(i)) for i in range(40)]
    cands = {str(i): f"c{i}" for i in range(40)}
    a = [t.real_is_a for t in build_trials(pairs, cands, seed=11)]
    b = [t.real_is_a for t in build_trials(pairs, cands, seed=12)]
    assert a != b


def test_pairs_without_a_candidate_are_skipped():
    trials = build_trials([_pair("1"), _pair("2")], {"1": "only one"}, seed=0)
    assert len(trials) == 1


def test_perfect_judge_scores_one():
    trials = build_trials([_pair("1")], {"1": "fake"}, seed=0)
    t = trials[0]
    guesses = {t.trial_id: "a" if t.real_is_a else "b"}
    assert score_trials(trials, guesses).accuracy == 1.0


def test_chance_judge_scores_about_half():
    pairs = [_pair(str(i)) for i in range(100)]
    cands = {str(i): f"c{i}" for i in range(100)}
    trials = build_trials(pairs, cands, seed=1)
    guesses = {t.trial_id: "a" for t in trials}   # always guess A
    r = score_trials(trials, guesses)
    assert 0.3 < r.accuracy < 0.7, r.accuracy


def test_result_reports_sample_size_and_interval():
    trials = build_trials([_pair("1")], {"1": "fake"}, seed=0)
    r = score_trials(trials, {})
    assert r.n == 1 and r.ci_low <= r.accuracy <= r.ci_high


def test_wilson_interval_brackets_the_point_estimate():
    lo, hi = wilson_interval(30, 100)
    assert lo < 0.30 < hi


# -- Same conversation must not double-count -------------------------------

def test_same_thread_contributes_at_most_one_trial():
    """A thread with several subject replies (several ReplyPairs) must not
    flood the sample with correlated trials -- see the ab.py module
    docstring for why that both double-counts and breaks the
    independent-trials assumption behind wilson_interval."""
    pairs = [_pair(str(i), thread_id="shared") for i in range(5)]
    cands = {str(i): f"cand-{i}" for i in range(5)}
    trials = build_trials(pairs, cands, seed=0)
    assert len(trials) == 1


def test_thread_cap_choice_is_reproducible_for_a_seed():
    pairs = [_pair(str(i), thread_id="shared") for i in range(5)]
    cands = {str(i): f"cand-{i}" for i in range(5)}
    a = build_trials(pairs, cands, seed=2)[0].trial_id
    b = build_trials(pairs, cands, seed=2)[0].trial_id
    assert a == b


def test_duplicate_pair_entries_do_not_double_count():
    pair = _pair("1")
    trials = build_trials([pair, pair], {"1": "candidate"}, seed=0)
    assert len(trials) == 1


def test_trial_order_does_not_depend_on_input_order():
    pairs = [_pair(str(i)) for i in range(10)]
    cands = {str(i): f"c{i}" for i in range(10)}
    forward = build_trials(pairs, cands, seed=9)
    backward = build_trials(list(reversed(pairs)), cands, seed=9)
    assert [t.trial_id for t in forward] == [t.trial_id for t in backward]


# -- score_trials guards against id collisions -------------------------------

def test_score_trials_rejects_duplicate_trial_ids():
    t = ABTrial(trial_id="dup", context=[], option_a="a", option_b="b", real_is_a=True)
    with pytest.raises(ValueError, match="dup"):
        score_trials([t, t], {})


def test_score_trials_on_zero_trials_reports_full_uncertainty():
    """No trials must read as "we don't know", not as a clean 0.0 accuracy
    with a confident (0.0, 0.0) interval -- see wilson_interval's docstring."""
    r = score_trials([], {})
    assert r.n == 0
    assert (r.ci_low, r.ci_high) == (0.0, 1.0)


# -- wilson_interval boundaries -----------------------------------------------

def test_wilson_interval_at_zero_observations_is_maximally_uncertain():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_interval_at_zero_successes():
    lo, hi = wilson_interval(0, 50)
    assert lo == 0.0
    assert 0.0 < hi < 1.0


def test_wilson_interval_at_all_successes():
    lo, hi = wilson_interval(50, 50)
    assert hi == 1.0
    assert 0.0 < lo < 1.0


def test_wilson_interval_at_n_equals_one():
    lo0, hi0 = wilson_interval(0, 1)
    lo1, hi1 = wilson_interval(1, 1)
    assert lo0 == 0.0 and 0.0 < hi0 < 1.0
    assert hi1 == 1.0 and 0.0 < lo1 < 1.0


def test_wilson_interval_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson_interval(-1, 10)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_interval(0, -1)
