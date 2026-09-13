import random
from datetime import datetime, timezone
from persona_twin.schema import Turn
from persona_twin.eval.style import fingerprint, distance, self_distance_band

def _turns(texts, thread_ids=None):
    if thread_ids is None:
        thread_ids = ["c"] * len(texts)
    return [Turn(subject_id="s", source="imessage", source_id=str(i), thread_id=thread_ids[i],
                 ts=datetime.now(timezone.utc), author_id="s", is_subject=True,
                 text=t, assisted=False) for i, t in enumerate(texts)]

def _thread_ids(n, num_threads):
    return [f"t{i % num_threads}" for i in range(n)]

def test_fingerprint_reports_sample_size():
    fp = fingerprint(_turns(["hi", "there you"]))
    assert fp.n == 2

def test_identical_corpora_have_zero_distance():
    a = fingerprint(_turns(["hello there", "how are you", "fine thanks"]))
    b = fingerprint(_turns(["hello there", "how are you", "fine thanks"]))
    assert distance(a, b) == 0.0

def test_different_corpora_have_positive_distance():
    short = fingerprint(_turns(["ok", "sure", "yep", "no"]))
    long = fingerprint(_turns(["I have been thinking about this at some length and "
                               "wanted to lay out the reasoning before we decide."] * 4))
    assert distance(short, long) > 0.0

def test_distance_is_symmetric():
    a = fingerprint(_turns(["ok", "sure"]))
    b = fingerprint(_turns(["a much longer message than the other one"]))
    assert distance(a, b) == distance(b, a)

def test_punctuation_and_emoji_rates_are_detected():
    plain = fingerprint(_turns(["no punctuation here"]))
    marked = fingerprint(_turns(["wait, really?! 🎉"]))
    assert marked.punct_rate > plain.punct_rate
    assert marked.emoji_rate > plain.emoji_rate

def test_self_distance_band_is_reproducible_for_a_seed():
    texts = [f"message number {i} with some words" for i in range(60)]
    turns = _turns(texts, _thread_ids(60, 6))
    assert self_distance_band(turns, trials=25, seed=7) == self_distance_band(turns, trials=25, seed=7)

def test_self_distance_band_is_smaller_than_cross_corpus_distance():
    """The band is the bar S2 must clear; a different corpus should sit outside it.

    The fixture spans 6 threads so the thread-stratified split has something to
    divide — with a single thread the split would be degenerate and the test
    would measure nothing.
    """
    own = _turns([f"ok {i}" for i in range(60)], _thread_ids(60, 6))
    other = fingerprint(_turns(["a substantially longer and more elaborate sentence"] * 60))
    _median, p95 = self_distance_band(own, trials=25, seed=0)
    assert distance(fingerprint(own), other) > p95

def test_self_distance_band_is_thread_aware():
    """Splitting on threads, not turns, is the whole point of the fix: the same
    texts confined to one thread can't be split meaningfully (degenerate band),
    while the same texts spread across many threads produce a real band. A
    reversion to turn-level shuffling would ignore thread_id entirely and make
    these two bands identical, since the underlying text is the same."""
    texts = [f"message number {i} with some words" for i in range(60)]
    single_thread = _turns(texts, ["only-thread"] * 60)
    many_threads = _turns(texts, _thread_ids(60, 12))
    band_single = self_distance_band(single_thread, trials=50, seed=3)
    band_many = self_distance_band(many_threads, trials=50, seed=3)
    assert band_single != band_many
