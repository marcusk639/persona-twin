import random
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from persona_twin.schema import Turn
from persona_twin.eval.style import (
    SelfDistanceBand, corpus_distance, distance, field_values, fingerprint,
    mass_balanced_sample, self_distance_band, thread_mass_cap,
    _FIELD_NAMES, _balanced_halves, _by_thread,
)


def _turns(texts, thread_ids=None, is_subject=True, author="s"):
    if thread_ids is None:
        thread_ids = ["c"] * len(texts)
    return [Turn(subject_id="s", source="imessage",
                 source_id=f"{author}-{thread_ids[i]}-{i}", thread_id=thread_ids[i],
                 ts=datetime.now(timezone.utc), author_id=author,
                 is_subject=is_subject, text=t, assisted=False)
            for i, t in enumerate(texts)]


def _thread_ids(n, num_threads):
    return [f"t{i % num_threads}" for i in range(n)]


def _unit_scales():
    """Every field weighed at 1.0 — isolates `distance`'s field coverage from
    whichever weights a particular corpus happens to produce."""
    return {name: 1.0 for name in _FIELD_NAMES}


# --------------------------------------------------------------------------
# fingerprint / distance basics
# --------------------------------------------------------------------------

def test_fingerprint_reports_sample_size():
    fp = fingerprint(_turns(["hi", "there you"]))
    assert fp.n == 2

def test_identical_corpora_have_zero_distance():
    a = fingerprint(_turns(["hello there", "how are you", "fine thanks"]))
    b = fingerprint(_turns(["hello there", "how are you", "fine thanks"]))
    assert distance(a, b, _unit_scales()) == 0.0

def test_different_corpora_have_positive_distance():
    short = fingerprint(_turns(["ok", "sure", "yep", "no"]))
    long = fingerprint(_turns(["I have been thinking about this at some length and "
                               "wanted to lay out the reasoning before we decide."] * 4))
    assert distance(short, long, _unit_scales()) > 0.0

def test_distance_is_symmetric():
    a = fingerprint(_turns(["ok", "sure"]))
    b = fingerprint(_turns(["a much longer message than the other one"]))
    scales = _unit_scales()
    assert distance(a, b, scales) == distance(b, a, scales)

def test_distance_divides_by_the_scale_of_each_field():
    """The scales are the whole point of the metric: a gap counts in units of
    the subject's own variation on that field. Halving a field's scale must
    double what that field contributes."""
    a = fingerprint(_turns(["ok", "sure", "yep", "no"]))
    b = fingerprint(_turns(["okay!", "sure, thing.", "yes?", "nope;"]))
    wide = _unit_scales()
    narrow = dict(wide, punct_rate=0.5)
    contribution = abs(a.punct_rate - b.punct_rate)
    assert contribution > 0.0   # otherwise the assertion below holds vacuously
    assert distance(a, b, narrow) == pytest.approx(
        distance(a, b, wide) + contribution)

def test_a_near_zero_field_cannot_saturate_the_distance():
    """The measured defect this metric replaces: `abs(x-y)/max(x,y,1e-9)` is
    bounded by 1.0 and saturates near zero, so an absolute gap of 0.0003
    emoji-per-character scored 0.856 of a roughly 8.0 budget — tied for the
    largest single term in the whole distance. Weighed against the subject's
    own variation on the field, the same gap must stay a small contribution.

    Isolated with `_only` so nothing but emoji_rate differs: two corpora built
    to differ in emoji also differ in length and punctuation, and the claim
    here is about one field."""
    base = fingerprint(_turns(["no emoji here at all", "still none", "nope", "nah"]))
    nudged = _only("emoji_rate", base, base.emoji_rate + 0.0003)
    realistic = dict(_unit_scales(), emoji_rate=0.02)
    assert distance(base, nudged, realistic) < 0.02
    old_relative_term = 0.0003 / max(base.emoji_rate, base.emoji_rate + 0.0003, 1e-9)
    assert old_relative_term > 0.9   # what the replaced denominator charged

def test_distance_falls_back_to_a_bounded_relative_term_when_a_scale_is_zero():
    """A field the subject never varied on has no unit to divide by. The term
    must stay finite and bounded rather than fabricating an infinite one."""
    base = fingerprint(_turns(["no emoji here", "none at all", "nope", "nah"]))
    moved = _only("emoji_rate", base, 0.5)
    only_emoji = {name: (0.0 if name == "emoji_rate" else 1e9)
                  for name in _FIELD_NAMES}
    assert 0.0 < distance(base, moved, only_emoji) <= 1.0
    assert distance(base, moved, dict(_unit_scales(), emoji_rate=0.0)) < float("inf")


# --------------------------------------------------------------------------
# every fingerprint field must be load-bearing (constant-folding must break)
#
# Each test asserts twice: that `fingerprint` actually varies the field
# between two corpora built to differ in it, and that `distance` charges for a
# difference in it. Between them they fail whether the field is folded out of
# the fingerprint or out of the distance.
# --------------------------------------------------------------------------

def _only(field_name, base, value):
    """A fingerprint identical to `base` except for one field."""
    if field_name.startswith("fw:"):
        rates = dict(base.function_word_rates)
        rates[field_name[3:]] = value
        return replace(base, function_word_rates=rates)
    return replace(base, **{field_name: value})


def _charges_for(field_name, base):
    """`distance` must charge for a difference in exactly this field, with
    every other field held identical."""
    moved = _only(field_name, base, field_values(base)[field_name] + 1.0)
    return distance(base, moved, _unit_scales()) > 0.0


_BASE = fingerprint(_turns(["hello there", "how are you", "fine thanks", "ok"]))


def test_mean_len_is_load_bearing():
    short = fingerprint(_turns(["aa", "aa", "aa", "aa"]))
    long = fingerprint(_turns(["a" * 50] * 4))
    assert short.mean_len != long.mean_len
    assert _charges_for("mean_len", _BASE)

def test_median_len_is_load_bearing():
    # Same mean length (2.0) either way; only the median moves, so a corpus
    # pair that differed in both would not prove median_len is read at all.
    even = fingerprint(_turns(["aa", "aa", "aa", "aa"]))
    skewed = fingerprint(_turns(["a", "a", "a", "aaaaa"]))
    assert even.mean_len == skewed.mean_len
    assert even.median_len != skewed.median_len
    assert _charges_for("median_len", _BASE)

def test_len_stdev_is_load_bearing():
    # Same mean and same median; only the spread moves.
    flat = fingerprint(_turns(["aa", "aa", "aa", "aa"]))
    spread = fingerprint(_turns(["a", "a", "aaa", "aaa"]))
    assert flat.mean_len == spread.mean_len
    assert flat.median_len == spread.median_len
    assert flat.len_stdev != spread.len_stdev
    assert _charges_for("len_stdev", _BASE)

def test_burstiness_is_load_bearing():
    # Same absolute spread (stdev 1.0), different mean, so burstiness — the
    # ratio — is the only thing that separates these two.
    small = fingerprint(_turns(["a", "a", "aaa", "aaa"]))
    large = fingerprint(_turns(["aaa", "aaa", "aaaaa", "aaaaa"]))
    assert small.len_stdev == large.len_stdev
    assert small.burstiness != large.burstiness
    assert _charges_for("burstiness", _BASE)

def test_punct_rate_is_load_bearing():
    plain = fingerprint(_turns(["no punctuation here", "none at all here"]))
    marked = fingerprint(_turns(["no, punctuation. here!", "none; at: all? here"]))
    assert marked.punct_rate > plain.punct_rate
    assert _charges_for("punct_rate", _BASE)

def test_emoji_rate_is_load_bearing():
    plain = fingerprint(_turns(["no emoji here", "none at all"]))
    marked = fingerprint(_turns(["no emoji here 🎉", "none at all 🎉"]))
    assert marked.emoji_rate > plain.emoji_rate
    assert _charges_for("emoji_rate", _BASE)

def test_upper_rate_is_load_bearing():
    lower = fingerprint(_turns(["all lower case here", "still lower"]))
    upper = fingerprint(_turns(["ALL UPPER CASE HERE", "STILL UPPER"]))
    assert upper.upper_rate > lower.upper_rate
    assert _charges_for("upper_rate", _BASE)

def test_type_token_ratio_is_load_bearing():
    repetitive = fingerprint(_turns(["same same same same"] * 4))
    varied = fingerprint(_turns(["alpha bravo charlie delta",
                                 "echo foxtrot golf hotel",
                                 "india juliet kilo lima",
                                 "mike november oscar papa"]))
    assert varied.type_token_ratio > repetitive.type_token_ratio
    assert _charges_for("type_token_ratio", _BASE)

def test_function_word_rates_are_load_bearing():
    """The eighteen function-word rates are one field of the fingerprint and
    are checked as one: deleting the loop that fills them, or the loop that
    reads them in `distance`, must fail here."""
    with_you = fingerprint(_turns(["you you you you", "you again you"]))
    without = fingerprint(_turns(["alpha bravo charlie delta", "echo foxtrot"]))
    assert with_you.function_word_rates["you"] > without.function_word_rates["you"]
    assert len(with_you.function_word_rates) == 18
    assert _charges_for("fw:you", _BASE)

def test_every_field_is_read_by_distance():
    """Belt and braces on the nine tests above: no field may be silently
    dropped from `_FIELD_NAMES` or from `distance`'s loop."""
    for name in _FIELD_NAMES:
        moved = _only(name, _BASE, field_values(_BASE)[name] + 1.0)
        assert distance(_BASE, moved, _unit_scales()) > 0.0, name


# --------------------------------------------------------------------------
# thread-mass capping
# --------------------------------------------------------------------------

def test_thread_mass_cap_is_one_average_threads_worth():
    turns = _turns(["x"] * 30, ["big"] * 24 + [f"t{i}" for i in range(6)])
    assert thread_mass_cap(turns) == 5   # ceil(30 / 7 threads)

def test_mass_balanced_sample_lets_no_thread_exceed_the_cap():
    turns = _turns(["x"] * 300, ["big"] * 280 + [f"t{i}" for i in range(20)])
    sample = mass_balanced_sample(turns, random.Random(0))
    cap = thread_mass_cap(turns)
    assert max(len(v) for v in _by_thread(sample).values()) <= cap

def test_mass_balanced_sample_stops_one_thread_from_defining_the_style():
    """The measured defect: two threads held 65% of the subject's held-out
    turns, so the 'style' of the corpus was the style of those threads. After
    capping, the fingerprint must reflect the many conversations rather than
    the one long one."""
    crowd = _turns(["ok", "sure", "yep", "no", "k"] * 4,
                   [f"t{i}" for i in range(20)])
    giant = _turns(["I WOULD LIKE TO SET OUT THE FULL REASONING IN DETAIL!!!"] * 400,
                   ["giant"] * 400)
    pooled = fingerprint(crowd + giant)
    capped = fingerprint(mass_balanced_sample(crowd + giant, random.Random(0)))
    crowd_only = fingerprint(crowd)
    scales = _unit_scales()
    assert (distance(capped, crowd_only, scales)
            < distance(pooled, crowd_only, scales))


# --------------------------------------------------------------------------
# balanced halves
# --------------------------------------------------------------------------

def test_balanced_halves_carry_near_equal_turn_counts():
    """The measured defect: splitting on thread COUNT put 179 turns on one
    side and 3,275 on the other, so the distance was dominated by how badly
    the small half's fingerprint was estimated. Greedy assignment to the
    lighter side bounds the imbalance by the largest single thread."""
    sizes = [40, 30, 20, 10] + [1] * 40
    turns = [t for i, size in enumerate(sizes)
             for t in _turns(["hello there"] * size, [f"t{i}"] * size)]
    for seed in range(20):
        left, right = _balanced_halves(_by_thread(turns), random.Random(seed))
        assert abs(len(left) - len(right)) <= max(sizes)

def test_balanced_halves_keep_whole_threads_together():
    turns = _turns(["hello there"] * 60, _thread_ids(60, 12))
    left, right = _balanced_halves(_by_thread(turns), random.Random(1))
    assert not ({t.thread_id for t in left} & {t.thread_id for t in right})


# --------------------------------------------------------------------------
# the band: reproducibility, refusals, trial accounting
# --------------------------------------------------------------------------

def test_self_distance_band_is_reproducible_for_a_seed():
    texts = [f"message number {i} with some words" for i in range(60)]
    turns = _turns(texts, _thread_ids(60, 6))
    a = self_distance_band(turns, trials=25, seed=7)
    b = self_distance_band(turns, trials=25, seed=7)
    assert (a.median, a.p95, a.trials) == (b.median, b.p95, b.trials)

def test_self_distance_band_reports_how_many_trials_completed():
    turns = _turns([f"message number {i} with words" for i in range(60)],
                   _thread_ids(60, 6))
    band = self_distance_band(turns, trials=25, seed=0)
    assert band.measured and band.trials == 25 and band.partitions > 1

def test_self_distance_band_refuses_on_too_few_turns():
    band = self_distance_band(_turns(["a", "b", "c"], ["t0", "t1", "t2"]))
    assert not band.measured and band.p95 is None and band.trials == 0
    assert "turn" in band.reason

def test_self_distance_band_refuses_when_one_thread_holds_everything():
    """Reproduced from the review: 800 subject turns in ONE thread completed
    zero trials and still rendered a confident S2 of 0.0000, because the
    degenerate exit returned (0.0, 0.0) instead of refusing."""
    turns = _turns([f"message {i} here" for i in range(800)], ["only"] * 800)
    band = self_distance_band(turns, trials=50, seed=0)
    assert not band.measured and band.p95 is None
    assert "thread" in band.reason

def test_self_distance_band_refuses_when_two_threads_admit_one_split():
    """Also reproduced from the review: with exactly two threads every trial
    re-measures the same pair of halves, so median equals p95 exactly and one
    measurement is printed as a 200-trial bootstrap."""
    turns = _turns([f"message {i} here" for i in range(80)], _thread_ids(80, 2))
    band = self_distance_band(turns, trials=200, seed=0)
    assert not band.measured and band.p95 is None
    assert band.partitions == 1
    assert "bootstrap" in band.reason

def test_an_unmeasured_band_refuses_to_score_anything():
    band = self_distance_band(_turns(["a", "b", "c"], ["t0", "t1", "t2"]))
    a = fingerprint(_turns(["hello there", "how are you"]))
    with pytest.raises(ValueError):
        band.distance(a, a)
    with pytest.raises(ValueError):
        corpus_distance(_turns(["x"] * 8), _turns(["y"] * 8), band)


# --------------------------------------------------------------------------
# what the band is FOR: separating the subject from everyone else
#
# The fixtures below are the missing direction. Every prior fixture in this
# file was 100% is_subject=True, so "does the bar tell the subject apart from
# the other people in his own threads" was structurally unaskable, and a bar
# that separated nobody from anybody would have passed the whole suite.
# --------------------------------------------------------------------------

_SUBJECT_LINES = ["yeah that works", "ok", "sounds good", "i can do that",
                  "will check and revert", "on it", "give me an hour",
                  "yep", "no thats fine", "lets do monday"]
_OTHER_LINES = ["Thanks so much!! 😊 Could you please take a look when you get a "
                "chance?? I really appreciate it!!",
                "Hi!! Just checking in on this one — do you have any update for "
                "me yet?? 🙏",
                "Perfect, thank you!!! You are the BEST. Have a great weekend!! 🎉",
                "Sorry to bother you again!! Did you get a chance to review what "
                "I sent over?? 😅",
                "That sounds wonderful, thank you!!! I will get you everything "
                "you need by Friday 😊"]
_BOILERPLATE = ["I'd be happy to help you with that. Here's a breakdown of the "
                "key considerations.",
                "Certainly! Let me walk you through the process step by step so "
                "it is clear.",
                "That's a great question. There are several factors worth taking "
                "into account here.",
                "Based on the information provided, I would recommend the "
                "following approach.",
                "It's important to note that individual circumstances may vary "
                "considerably."]


def _persona(lines, n, threads, is_subject, author, seed=11):
    """Lines are drawn at random rather than cycled: `_thread_ids` cycles too,
    and two cycles whose lengths share a factor put a fixed subset of the lines
    in each thread, so any thread-based split would compare different line sets
    rather than different samples of one voice."""
    rng = random.Random(seed)
    texts = [rng.choice(lines) for _ in range(n)]
    return _turns(texts, _thread_ids(n, threads), is_subject=is_subject,
                  author=author)


def _heldout(turn):
    """Half the threads, so both sides of every framing below are estimated
    from the same amount of text as the band's own halves are. A 1:3 split
    would make the small side noisier than anything the band ever measured,
    and the distance would report that noise rather than a difference in
    voice."""
    return int(turn.thread_id[1:]) % 2 == 0


def _mixed_corpus():
    """One corpus holding both sides of the same conversations, as a real
    iMessage export does: the subject's turns and his correspondents' turns."""
    return (_persona(_SUBJECT_LINES, 480, 24, True, "subject")
            + _persona(_OTHER_LINES, 480, 24, False, "other"))


def test_the_band_admits_a_second_sample_of_the_subjects_own_writing():
    """Direction one: the subject must clear his own bar. A bar nothing can
    clear discriminates perfectly and is useless."""
    corpus = _mixed_corpus()
    subject = [t for t in corpus if t.is_subject]
    band = self_distance_band(subject, trials=100, seed=0)
    held = [t for t in subject if _heldout(t)]
    rest = [t for t in subject if not _heldout(t)]
    assert band.measured
    assert corpus_distance(rest, held, band, draws=10, seed=1) <= band.p95

def test_the_band_rejects_the_other_people_in_the_subjects_own_threads():
    """Direction two, and the defect this replaces: on the real corpus the old
    bar could not tell the subject apart from his own correspondents. Three of
    four framings of that comparison passed. Nothing in this file asserted the
    bar SEPARATES anything, so the failure was invisible."""
    corpus = _mixed_corpus()
    subject = [t for t in corpus if t.is_subject]
    others = [t for t in corpus if not t.is_subject]
    band = self_distance_band(subject, trials=100, seed=0)
    assert band.measured
    assert corpus_distance(subject, others, band, draws=10, seed=1) > band.p95

def test_the_band_rejects_every_framing_of_the_subject_against_others():
    """All four train/held-out framings, not just the convenient one. The old
    bar failed exactly one of these four and passed the other three, so a test
    that checked a single framing would have called the defect fixed."""
    corpus = _mixed_corpus()

    def part(is_subject, heldout):
        return [t for t in corpus
                if t.is_subject == is_subject
                and _heldout(t) == heldout]

    band = self_distance_band([t for t in corpus if t.is_subject],
                              trials=100, seed=0)
    assert band.measured
    for subj_held in (True, False):
        for other_held in (True, False):
            d = corpus_distance(part(True, subj_held), part(False, other_held),
                                band, draws=10, seed=1)
            assert d > band.p95, f"subject heldout={subj_held} vs other heldout={other_held}"

def test_the_band_rejects_generic_assistant_boilerplate():
    corpus = _mixed_corpus()
    subject = [t for t in corpus if t.is_subject]
    band = self_distance_band(subject, trials=100, seed=0)
    boilerplate = _persona(_BOILERPLATE, 200, 20, False, "llm")
    assert corpus_distance(subject, boilerplate, band, draws=10, seed=1) > band.p95

def test_corpus_distance_mass_balances_both_sides_before_scoring():
    """A scored distance must be immune to one long conversation the way the
    band's halves are. Comparing raw pooled fingerprints instead would let a
    single thread decide the gate — the same defect as the band's, one level
    up — so the balanced answer has to sit strictly below the pooled one."""
    reference = _persona(_SUBJECT_LINES, 200, 20, True, "subject", seed=1)
    crowd = _persona(_SUBJECT_LINES, 200, 20, True, "subject", seed=2)
    giant = _turns(["I WOULD LIKE TO SET OUT THE FULL REASONING IN DETAIL!!! 🎉"] * 400,
                   ["giant"] * 400)
    band = self_distance_band(reference, trials=50, seed=0)
    balanced = corpus_distance(reference, crowd + giant, band, draws=10, seed=1)
    pooled = distance(fingerprint(reference), fingerprint(crowd + giant),
                      band.scales)
    assert balanced < pooled

def test_corpus_distance_is_comparable_to_the_band_it_is_scored_against():
    """Both sides of a scored distance are mass-balanced exactly as the band's
    halves were. Scoring a raw pooled fingerprint against a mass-balanced bar
    would put a number produced one way next to a bar produced another."""
    corpus = _mixed_corpus()
    subject = [t for t in corpus if t.is_subject]
    band = self_distance_band(subject, trials=50, seed=0)
    a = corpus_distance(subject, subject, band, draws=10, seed=2)
    assert a < band.p95

def test_self_distance_band_is_smaller_than_cross_corpus_distance():
    """The band is the bar S2 must clear; a different corpus should sit outside
    it. The fixture spans several threads so the split has something to
    divide — with a single thread the band would refuse outright."""
    own = _turns([f"ok {i}" for i in range(60)], _thread_ids(60, 6))
    other = _turns(["a substantially longer and more elaborate sentence"] * 60,
                   _thread_ids(60, 6))
    band = self_distance_band(own, trials=25, seed=0)
    assert band.measured
    assert corpus_distance(own, other, band, draws=10, seed=0) > band.p95

def test_self_distance_band_is_thread_aware():
    """Splitting on threads, not turns, is load-bearing: the same texts confined
    to one thread cannot be split at all (the band refuses), while the same
    texts spread across many threads produce a real band. A reversion to
    turn-level shuffling would ignore thread_id entirely and measure a band in
    both cases."""
    texts = [f"message number {i} with some words" for i in range(60)]
    single = self_distance_band(_turns(texts, ["only-thread"] * 60), trials=50, seed=3)
    many = self_distance_band(_turns(texts, _thread_ids(60, 12)), trials=50, seed=3)
    assert not single.measured
    assert many.measured


def test_band_scales_are_the_subjects_own_variation_per_field():
    """The scales are not a weighting someone chose; they are measured from the
    same trials that produce the band, one per comparable field."""
    turns = _turns([f"message number {i} with some words" for i in range(60)],
                   _thread_ids(60, 6))
    band = self_distance_band(turns, trials=25, seed=0)
    assert set(band.scales) == set(_FIELD_NAMES)
    assert all(v >= 0.0 for v in band.scales.values())
