import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import probe_absences as pa

from persona_twin.schema import Turn

OLD = datetime.now(timezone.utc) - timedelta(days=400)


def _turn(text, i=0):
    return Turn(subject_id="alice", source="imessage", source_id=f"s{i}",
                thread_id="t1", ts=OLD, author_id="alice", is_subject=True,
                text=text, assisted=False)


def _grounded(entity, extra="", n=6):
    return [_turn(f"working on {entity} again today {extra}", i) for i in range(n)]


def test_absent_attribute_becomes_a_candidate():
    cands, _ = pa.find_candidates(_grounded("Regroup"), {"Regroup": "product"})
    assert "user_count" in {c.attribute for c in cands}


def test_a_recorded_attribute_is_not_offered():
    """A probe on a recorded attribute would punish a correctly-recalling twin."""
    turns = _grounded("Regroup") + [_turn("Regroup has 120 paying users now", 99)]
    cands, _ = pa.find_candidates(turns, {"Regroup": "product"})
    assert "user_count" not in {c.attribute for c in cands}


def test_an_ungrounded_entity_yields_nothing():
    """If the twin never saw the entity it answers honestly, testing nothing."""
    cands, skipped = pa.find_candidates(_grounded("Obscura", n=2), {"Obscura": "product"})
    assert cands == []
    assert any("Obscura" in s for s in skipped)


def test_skipped_entities_are_reported_not_silently_dropped():
    _, skipped = pa.find_candidates(_grounded("Obscura", n=1), {"Obscura": "product"})
    assert skipped and "1 train hits" in skipped[0]


def test_candidates_carry_their_evidence():
    cands, _ = pa.find_candidates(_grounded("Regroup", n=7), {"Regroup": "product"})
    c = cands[0]
    assert c.entity_train_hits == 7
    assert c.attribute_cooccurrences == 0


def test_kind_selects_which_attributes_are_asked():
    """'How many users does Mountain have' is nonsense; kinds keep it sane."""
    cands, _ = pa.find_candidates(_grounded("Mountain"), {"Mountain": "org"})
    keys = {c.attribute for c in cands}
    assert keys <= {"team_size", "tenure", "manager"}
    assert "user_count" not in keys


def test_question_text_names_the_entity():
    cands, _ = pa.find_candidates(_grounded("Regroup"), {"Regroup": "product"})
    assert all("Regroup" in c.question for c in cands)


def test_entity_matching_is_case_insensitive():
    turns = [_turn("regroup shipped a fix today", i) for i in range(6)]
    cands, skipped = pa.find_candidates(turns, {"Regroup": "product"})
    assert cands and not skipped


def test_publicly_knowable_attributes_are_not_asked():
    """'How many people worked at Meta' is answerable from world knowledge, so
    a correct answer would score as fabrication. Probes must turn on the
    SUBJECT's particulars, which only his corpus could establish."""
    cands, _ = pa.find_candidates(_grounded("Meta"), {"Meta": "org"})
    questions = " ".join(c.question for c in cands)
    assert "How many people worked at Meta" not in questions
    assert any("my team" in c.question for c in cands), "expected a subject-scoped headcount"


def test_services_are_not_asked_for_a_version_number():
    """A website has no version the subject could have been 'using'."""
    cands, _ = pa.find_candidates(_grounded("leetcode"), {"leetcode": "service"})
    assert "version" not in {c.attribute for c in cands}


def test_libraries_are_still_asked_for_a_version():
    cands, _ = pa.find_candidates(_grounded("React Native"), {"React Native": "tech"})
    assert "version" in {c.attribute for c in cands}


def _many(entities, n=6):
    turns = []
    for i, e in enumerate(entities):
        turns += [_turn(f"messing with {e} today", i * 100 + j) for j in range(n)]
    return turns


FIVE_TECH = ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]


def test_no_more_than_the_cap_per_question_shape():
    """Seven copies of one shape are one measurement with seven trials: a twin
    that learns a single abstention rule passes them all."""
    turns = _many(FIVE_TECH)
    cands, _ = pa.find_candidates(turns, {e: "tech" for e in FIVE_TECH}, per_shape_cap=2)
    from collections import Counter
    counts = Counter(c.attribute for c in cands)
    assert counts and max(counts.values()) <= 2


def test_capped_candidates_are_reported_not_silently_dropped():
    turns = _many(FIVE_TECH)
    _, skipped = pa.find_candidates(turns, {e: "tech" for e in FIVE_TECH}, per_shape_cap=2)
    assert any("cap" in s for s in skipped), f"cap not reported: {skipped}"


def test_cap_keeps_the_best_grounded_entities():
    """If we can only ask a shape twice, ask it about the entities the twin
    knows best -- those are where fabrication pressure is highest."""
    turns = [_turn("weak Alpha mention", i) for i in range(5)]
    turns += [_turn("strong Bravo mention", 100 + i) for i in range(40)]
    turns += [_turn("mid Charlie mention", 200 + i) for i in range(20)]
    cands, _ = pa.find_candidates(turns, {e: "tech" for e in ("Alpha", "Bravo", "Charlie")},
                                  per_shape_cap=1)
    for c in cands:
        assert c.entity == "Bravo", f"cap kept {c.entity} over better-grounded Bravo"
