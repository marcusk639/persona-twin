from datetime import datetime, timedelta, timezone
from persona_twin.schema import Turn
from persona_twin.normalize.threads import reply_pairs

T0 = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)


def _t(i, is_subject, minutes=0, thread="c1"):
    return Turn(
        subject_id="alice",
        source="imessage",
        source_id=str(i),
        thread_id=thread,
        ts=T0 + timedelta(minutes=minutes),
        author_id="alice" if is_subject else "bob",
        is_subject=is_subject,
        text=f"m{i}",
        assisted=False,
    )


def test_pairs_subject_reply_with_prior_context():
    pairs = reply_pairs([_t(1, False, 0), _t(2, True, 1)])
    assert len(pairs) == 1
    assert pairs[0].reply.source_id == "2" and pairs[0].context[0].source_id == "1"


def test_reply_with_no_inbound_context_is_dropped():
    assert reply_pairs([_t(1, True, 0), _t(2, True, 1)]) == []


def test_context_does_not_cross_threads():
    turns = [_t(1, False, 0, thread="c1"), _t(2, True, 1, thread="c2")]
    assert reply_pairs(turns) == []


def test_stale_context_is_excluded_by_gap():
    pairs = reply_pairs([_t(1, False, 0), _t(2, True, 600)], max_gap_minutes=180)
    assert pairs == []


def test_context_is_capped_and_ordered():
    turns = [_t(i, False, i) for i in range(10)] + [_t(99, True, 20)]
    pairs = reply_pairs(turns, max_context=3)
    ctx = pairs[0].context
    assert len(ctx) == 3 and [c.source_id for c in ctx] == ["7", "8", "9"]


def test_gap_at_exact_boundary_is_kept():
    # reply exactly max_gap_minutes after the inbound turn: not stale, must be kept
    pairs = reply_pairs([_t(1, False, 0), _t(2, True, 180)], max_gap_minutes=180)
    assert len(pairs) == 1
    assert pairs[0].context[0].source_id == "1"


def test_gap_one_minute_past_boundary_drops_context_turn():
    # 181 minutes: just past the cutoff, context turn must be excluded, leaving
    # no non-subject context turns, so the reply is dropped entirely
    pairs = reply_pairs([_t(1, False, 0), _t(2, True, 181)], max_gap_minutes=180)
    assert pairs == []


def test_context_cap_at_exact_max_context_keeps_all():
    # exactly max_context inbound turns precede the reply: all must be kept
    turns = [_t(i, False, i) for i in range(6)] + [_t(99, True, 10)]
    pairs = reply_pairs(turns, max_context=6)
    assert len(pairs) == 1
    assert [c.source_id for c in pairs[0].context] == ["0", "1", "2", "3", "4", "5"]


def test_context_cap_one_over_max_context_drops_oldest():
    # max_context + 1 inbound turns precede the reply: the oldest one must be dropped
    turns = [_t(i, False, i) for i in range(7)] + [_t(99, True, 10)]
    pairs = reply_pairs(turns, max_context=6)
    assert len(pairs) == 1
    assert [c.source_id for c in pairs[0].context] == ["1", "2", "3", "4", "5", "6"]


def test_shipped_context_after_truncation_must_still_contain_inbound_turn():
    # One inbound turn, then more than max_context (6) consecutive subject
    # turns, all inside the gap window. The non-subject check must run on the
    # *shipped* (post-truncation) context, not the untruncated gap-filtered
    # list -- otherwise a reply whose nearest max_context turns are entirely
    # the subject's own messages ships with a context that never prompted it.
    turns = [_t(0, False, 0)] + [_t(i, True, i) for i in range(1, 8)]
    pairs = reply_pairs(turns, max_context=6)

    # The reply at index 7 has exactly the last 6 turns (indices 1-6, all
    # subject) as its truncated window -- the inbound turn 0 falls outside
    # the cap, so this reply must be dropped rather than shipped with an
    # all-subject context.
    assert not any(p.reply.source_id == "7" for p in pairs)
    for pair in pairs:
        assert any(not c.is_subject for c in pair.context)


def test_multiple_subject_replies_in_same_thread_each_get_own_context():
    turns = [
        _t(1, False, 0),
        _t(2, True, 1),
        _t(3, False, 2),
        _t(4, True, 3),
    ]
    pairs = reply_pairs(turns)
    assert len(pairs) == 2
    assert [c.source_id for c in pairs[0].context] == ["1"]
    assert [c.source_id for c in pairs[1].context] == ["1", "2", "3"]


def test_out_of_order_input_is_sorted_by_ts_before_pairing():
    turns = [_t(2, True, 1), _t(1, False, 0)]
    pairs = reply_pairs(turns)
    assert len(pairs) == 1
    assert pairs[0].context[0].source_id == "1"


def test_orphan_thread_ids_never_merge_into_pairs():
    # Simulates per-envelope orphan thread ids from turns.py: each turn has a
    # unique thread_id, so no reply can ever have a same-thread inbound turn.
    turns = [
        _t(1, False, 0, thread="imessage:orphan:1"),
        _t(2, True, 1, thread="imessage:orphan:2"),
    ]
    assert reply_pairs(turns) == []
