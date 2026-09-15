import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import probe_intake as pi


def test_parses_answerable_and_unanswerable_rows():
    probes = pi.parse("A | what framework | React Native\n"
                      "U | how many users\n")
    assert [p["answerable"] for p in probes] == [True, False]
    assert probes[0]["expected"] == "React Native"
    assert probes[1]["expected"] == ""


def test_unfilled_placeholder_rows_are_skipped_not_turned_into_probes():
    """A generated draft ships with ??? where an answer goes. Converting those
    verbatim would create probes whose expected answer is literally '???',
    which the gate counts toward n>=200 while measuring nothing."""
    text = ("A | What framework is Regroup built with? | ???\n"
            "A | What kind of app is Regroup? | sober living\n")
    probes, skipped = pi.parse_with_skips(text)
    assert [p["question"] for p in probes] == ["What kind of app is Regroup?"]
    assert skipped == 1


def test_unanswerable_rows_are_never_treated_as_unfilled():
    """U rows legitimately have an empty expected; that is not a placeholder."""
    probes, skipped = pi.parse_with_skips("U | How many users does Regroup have?\n")
    assert len(probes) == 1 and skipped == 0


def test_an_answerable_row_with_no_expected_still_raises():
    """Distinct from a placeholder: the author wrote the row and left it
    broken, rather than the generator leaving it to be filled."""
    with pytest.raises(pi.IntakeError):
        pi.parse("A | what framework |\n")


def test_comments_and_blank_lines_are_ignored():
    probes = pi.parse("# ITEM 12\n\nA | q | a\n")
    assert len(probes) == 1


def _write(p, rows):
    p.write_text(json.dumps(rows, indent=2))
    return p


EXISTING = [
    {"probe_id": "fact-001", "question": "What framework is Regroup built with?",
     "expected": "React Native", "answerable": True, "notes": "hand-authored"},
    {"probe_id": "fact-002", "question": "How many users does Regroup have?",
     "expected": "", "answerable": False, "notes": "verified absent"},
]


def test_merge_keeps_existing_probes(tmp_path):
    """Overwriting would silently discard hand-authored work."""
    dest = _write(tmp_path / "facts.json", EXISTING)
    src = tmp_path / "draft.txt"
    src.write_text("A | What kind of app is Regroup? | sober living\n")
    assert pi.main(str(src), str(dest), merge=True) == 0
    rows = json.loads(dest.read_text())
    assert [r["probe_id"] for r in rows][:2] == ["fact-001", "fact-002"]
    assert rows[0]["notes"] == "hand-authored"
    assert len(rows) == 3


def test_merged_probes_continue_the_id_sequence(tmp_path):
    dest = _write(tmp_path / "facts.json", EXISTING)
    src = tmp_path / "draft.txt"
    src.write_text("A | What kind of app is Regroup? | sober living\n")
    pi.main(str(src), str(dest), merge=True)
    assert json.loads(dest.read_text())[-1]["probe_id"] == "fact-003"


def test_merge_skips_a_near_duplicate_question(tmp_path):
    """Substring matching missed 'How many users' vs 'How many PAYING users';
    token overlap catches it."""
    dest = _write(tmp_path / "facts.json", EXISTING)
    src = tmp_path / "draft.txt"
    src.write_text("U | How many paying users does Regroup have?\n")
    pi.main(str(src), str(dest), merge=True)
    assert len(json.loads(dest.read_text())) == 2


def test_merge_reports_what_it_skipped(tmp_path, capsys):
    """Silent skips read as a successful import of everything."""
    dest = _write(tmp_path / "facts.json", EXISTING)
    src = tmp_path / "draft.txt"
    src.write_text("U | How many users does Regroup have?\n"
                   "A | Brand new question here? | ???\n"
                   "A | Another new one? | yes\n")
    pi.main(str(src), str(dest), merge=True)
    out = capsys.readouterr().out
    assert "1 duplicate" in out and "1 unfilled" in out


def test_without_merge_the_destination_is_replaced(tmp_path):
    """The old behaviour must stay available and stay explicit."""
    dest = _write(tmp_path / "facts.json", EXISTING)
    src = tmp_path / "draft.txt"
    src.write_text("A | Only this one? | yes\n")
    pi.main(str(src), str(dest), merge=False)
    assert len(json.loads(dest.read_text())) == 1


def test_merging_into_an_absent_file_starts_the_sequence_at_one(tmp_path):
    """First-ever merge: the id counter has no existing rows to continue from."""
    dest = tmp_path / "facts.json"
    src = tmp_path / "draft.txt"
    src.write_text("A | First question? | yes\n")
    pi.main(str(src), str(dest), merge=True)
    assert json.loads(dest.read_text())[0]["probe_id"] == "fact-001"


def test_merging_into_a_file_whose_ids_are_not_numeric(tmp_path):
    """Hand-written probe files may use non-numeric ids; the counter must not
    crash or collide with them."""
    dest = _write(tmp_path / "facts.json",
                  [{"probe_id": "custom-a", "question": "Old one?", "expected": "x",
                    "answerable": True, "notes": ""}])
    src = tmp_path / "draft.txt"
    src.write_text("A | Brand new distinct question? | yes\n")
    pi.main(str(src), str(dest), merge=True)
    ids = [r["probe_id"] for r in json.loads(dest.read_text())]
    assert ids == ["custom-a", "fact-001"]
