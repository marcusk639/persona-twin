import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import probe_batch as pb
import probe_intake as pi

from persona_twin.schema import Turn

OLD = datetime.now(timezone.utc) - timedelta(days=400)


def _turn(text, i=0):
    return Turn(subject_id="alice", source="imessage", source_id=f"s{i}",
                thread_id="t1", ts=OLD, author_id="alice", is_subject=True,
                text=text, assisted=False)


def _train(entity, extra="", n=8):
    return [_turn(f"more work on {entity} today {extra}", i) for i in range(n)]


def test_absent_attribute_becomes_a_u_row():
    body, _ = pb.render(["thinking about Regroup again"], _train("Regroup"),
                        {"Regroup": "product"})
    assert any(l.startswith("U | How many users does Regroup have?")
               for l in body.splitlines())


def test_recorded_attribute_becomes_an_a_row_awaiting_an_answer():
    train = _train("Regroup") + [_turn(f"Regroup has {n} paying users", 50 + n)
                                 for n in range(5)]
    body, _ = pb.render(["Regroup notes"], train, {"Regroup": "product"})
    a_rows = [l for l in body.splitlines() if l.startswith("A |")]
    assert a_rows and all(l.endswith("x") or "???" in l for l in a_rows)
    assert any("user_count" in l for l in a_rows)


def test_an_attribute_recorded_only_once_yields_no_row_at_all():
    """Below the answerable floor it is not learnable; at zero it would be a
    valid U row -- one mention is neither, so it must be dropped."""
    train = _train("Regroup") + [_turn("Regroup has 12 paying users", 99)]
    body, _ = pb.render(["Regroup notes"], train, {"Regroup": "product"})
    rows = [l for l in body.splitlines() if "user_count" in l]
    assert rows == []


def test_ungrounded_entities_produce_no_entity_derived_rows():
    """The blank Q/A prompt still appears (it is not entity-derived), but no
    question may be generated ABOUT an entity the twin barely saw."""
    body, _ = pb.render(["Obscura notes"], _train("Obscura", n=2),
                        {"Obscura": "product"})
    generated = [l for l in body.splitlines()
                 if l.startswith(("A |", "U |")) and l != "A | ??? | ???"]
    assert generated == []


def test_only_entities_present_in_the_line_are_used():
    train = _train("Regroup") + _train("Homegroups")
    body, _ = pb.render(["only Regroup is mentioned here"], train,
                        {"Regroup": "product", "Homegroups": "product"})
    assert "Homegroups" not in body


def test_the_draft_round_trips_through_the_existing_intake():
    """The generated format must be the one probe_intake already reads."""
    train = _train("Regroup")
    body, _ = pb.render(["Regroup notes"], train, {"Regroup": "product"})
    probes, skipped = pi.parse_with_skips(body)
    assert probes and all(not p["answerable"] for p in probes)


def test_unfilled_answer_rows_do_not_become_probes():
    train = _train("Regroup") + [_turn(f"Regroup has {n} paying users", 50 + n)
                                 for n in range(5)]
    body, _ = pb.render(["Regroup notes"], train, {"Regroup": "product"})
    probes, skipped = pi.parse_with_skips(body)
    assert skipped >= 1
    assert all(p["expected"] != "???" for p in probes)


def test_seed_lines_exclude_instructions_and_the_topic_list(tmp_path):
    ws = tmp_path / "worksheet.md"
    ws.write_text(
        "# worksheet\n\n## How to use this\n\n- this is an instruction\n\n"
        "## preferences (1)\n\n- I always reach for the boring option\n\n"
        "## recurring topics\n\n- Yeah (2051)\n")
    seeds = pb.seed_lines_from(ws)
    assert seeds == ["I always reach for the boring option"]


def test_every_seed_line_gets_a_blank_answerable_prompt():
    """Templates can only ask generic questions; the rich answerable probes
    come from the subject reading his own line. So every line gets a blank
    Q/A pair, not just the ones mentioning a known entity."""
    body, _ = pb.render(["a line naming nothing we track"], [], {})
    assert "A | ??? | ???" in body
    assert "# ITEM 1" in body


def test_blank_prompts_do_not_become_probes_if_left_unfilled():
    body, _ = pb.render(["a line naming nothing we track"], [], {})
    probes, skipped = pi.parse_with_skips(body)
    assert probes == []
    assert skipped >= 1
