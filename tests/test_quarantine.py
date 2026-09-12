from datetime import datetime, timedelta, timezone
from persona_twin.schema import Turn
from persona_twin.corpus.quarantine import is_quarantined, split

NOW = datetime(2026, 9, 7, tzinfo=timezone.utc)

def _t(days_ago, i=0):
    return Turn(subject_id="alice", source="imessage", source_id=str(i), thread_id="c1",
                ts=NOW - timedelta(days=days_ago), author_id="alice", is_subject=True,
                text="m", assisted=False)

def test_recent_turn_is_quarantined():
    assert is_quarantined(_t(7), NOW, weeks=12)

def test_old_turn_is_trainable():
    assert not is_quarantined(_t(365), NOW, weeks=12)

def test_boundary_is_inclusive_of_older_side():
    assert not is_quarantined(_t(85), NOW, weeks=12)   # 85 days > 84
    assert is_quarantined(_t(83), NOW, weeks=12)

def test_boundary_exact_is_not_quarantined():
    # exactly 12 weeks (84 days) ago: ts == now - timedelta(weeks=12), strict
    # comparison means this sits on the trainable side of the line.
    turn = Turn(subject_id="alice", source="imessage", source_id="edge", thread_id="c1",
                ts=NOW - timedelta(weeks=12), author_id="alice", is_subject=True,
                text="m", assisted=False)
    assert not is_quarantined(turn, NOW, weeks=12)

def test_split_partitions_all_turns():
    turns = [_t(5, 1), _t(400, 2), _t(10, 3)]
    trainable, quarantined = split(turns, NOW, weeks=12)
    assert len(trainable) == 1 and len(quarantined) == 2
    assert len(trainable) + len(quarantined) == len(turns)
    # identity, not just counts: every input turn lands in exactly one output
    # list, and nothing is lost or duplicated.
    for turn in turns:
        placements = int(any(turn is t for t in trainable)) + \
                     int(any(turn is t for t in quarantined))
        assert placements == 1
    assert {id(t) for t in trainable + quarantined} == {id(t) for t in turns}

def test_window_rolls_forward():
    turn = _t(90)
    assert not is_quarantined(turn, NOW, weeks=12)
    assert is_quarantined(turn, NOW - timedelta(days=30), weeks=12)
