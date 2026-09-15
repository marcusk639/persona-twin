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
