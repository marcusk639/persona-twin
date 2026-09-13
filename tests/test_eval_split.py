from datetime import datetime, timedelta, timezone
import pytest
from persona_twin.schema import Turn
from persona_twin.eval.split import thread_bucket, is_heldout, split_corpus, HELDOUT_FRACTION

NOW = datetime(2026, 9, 12, tzinfo=timezone.utc)

def _t(thread: str, days_ago: int = 400, sid: str = "1") -> Turn:
    return Turn(subject_id="s", source="imessage", source_id=sid, thread_id=thread,
                ts=NOW - timedelta(days=days_ago), author_id="s", is_subject=True,
                text="hello", assisted=False)

def test_bucket_is_deterministic_across_calls():
    assert thread_bucket("chat-42") == thread_bucket("chat-42")

def test_bucket_matches_known_sha256_digest():
    """Pins actual digest values so a switch to builtin hash() (which Python
    salts per process) or a change of input encoding would be caught, unlike
    a same-process comparison which passes under either scheme."""
    assert thread_bucket("chat-7") == 44
    assert thread_bucket("thread-0") == 19

def test_bucket_is_in_range():
    for i in range(200):
        assert 0 <= thread_bucket(f"chat-{i}") < 100

def test_all_turns_of_a_thread_land_on_the_same_side():
    turns = [_t("chat-7", sid=str(i)) for i in range(20)]
    sides = {is_heldout(t) for t in turns}
    assert len(sides) == 1, "a thread must not straddle the split"

def test_split_is_stable_when_unrelated_turns_are_added():
    """The property that makes later data exports safe.

    Round-trips through split_corpus (not is_heldout directly) so a
    regression introduced inside split_corpus itself -- e.g. a future
    stratification step that considers the whole corpus -- would be caught.
    """
    original = [_t(f"chat-{i}", sid=str(i)) for i in range(50)]
    before = {t.source_id for t in split_corpus(original, NOW).heldout}
    grown = original + [_t(f"newchat-{i}", sid=f"n{i}") for i in range(500)]
    after = {t.source_id for t in split_corpus(grown, NOW).heldout if t.source_id in {o.source_id for o in original}}
    assert before == after, "adding data moved existing turns across the split"

def test_heldout_fraction_is_approximately_right():
    turns = [_t(f"chat-{i}", sid=str(i)) for i in range(2000)]
    frac = sum(is_heldout(t) for t in turns) / len(turns)
    assert abs(frac - HELDOUT_FRACTION) < 0.05, frac

def test_quarantined_turns_are_in_neither_train_nor_heldout():
    recent = _t("chat-recent", days_ago=3, sid="r")
    old = _t("chat-old", days_ago=400, sid="o")
    s = split_corpus([recent, old], NOW, weeks=12)
    assert recent in s.quarantined
    assert recent not in s.train and recent not in s.heldout

def test_quarantine_takes_precedence_over_heldout_side():
    """held-5 hashes into the held-out band (thread_bucket == 1). A recent
    turn on that thread must still be quarantined, not heldout -- this fails
    if split_corpus checks is_heldout before is_quarantined."""
    assert thread_bucket("held-5") < 15  # sanity: thread really is heldout-band
    recent = _t("held-5", days_ago=3, sid="r")
    old = _t("held-5", days_ago=400, sid="o")
    s = split_corpus([recent, old], NOW, weeks=12)
    assert recent in s.quarantined
    assert recent not in s.heldout
    assert old in s.heldout

def test_split_partitions_every_turn_exactly_once():
    turns = [_t(f"chat-{i}", days_ago=d, sid=f"{i}-{d}")
             for i in range(40) for d in (3, 400)]
    s = split_corpus(turns, NOW, weeks=12)
    ids = [t.source_id for t in s.train] + [t.source_id for t in s.heldout] + [t.source_id for t in s.quarantined]
    assert sorted(ids) == sorted(t.source_id for t in turns)
    assert len(ids) == len(set(ids)), "a turn appeared in more than one bucket"
