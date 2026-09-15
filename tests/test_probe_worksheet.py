import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import probe_worksheet as pw

from persona_twin.corpus.store import CorpusStore
from persona_twin.eval.split import is_heldout
from persona_twin.paths import SubjectPaths
from persona_twin.schema import Turn

OLD = datetime.now(timezone.utc) - timedelta(days=400)


def _turn(thread_id, text, *, is_subject=True, source_id=None):
    return Turn(subject_id="alice", source="imessage",
                source_id=source_id or f"{thread_id}-{abs(hash(text)) % 10**8}",
                thread_id=thread_id, ts=OLD, author_id="alice",
                is_subject=is_subject, text=text, assisted=False)


def _thread_on_side(heldout: bool) -> str:
    """Find a thread_id whose deterministic bucket lands on the wanted side."""
    for i in range(10_000):
        tid = f"t{i}"
        if is_heldout(_turn(tid, "x" * 100)) is heldout:
            return tid
    raise AssertionError("no thread found")


SENTENCE = ("I always reach for the boring option first when a deadline is "
            "involved, because clever tends to cost more later on.")


def test_heldout_turns_never_reach_the_worksheet(tmp_path):
    """Probes derived from held-out turns would be scored against themselves."""
    paths = SubjectPaths("alice", tmp_path)
    paths.ensure()
    train_tid, held_tid = _thread_on_side(False), _thread_on_side(True)
    secret = ("I always reach for SECRETHELDOUTMARKER first on a new project, and "
              "have done for years, because the alternatives all disappointed me.")
    CorpusStore(paths).write("v1", [
        _turn(train_tid, SENTENCE),
        _turn(held_tid, secret),
    ])
    assert pw.main(["alice", "v1", "--root", str(tmp_path)]) == 0
    body = (Path(paths.probes) / "worksheet.md").read_text(encoding="utf-8")
    assert "SECRETHELDOUTMARKER" not in body
    assert "boring option" in body


def test_turns_written_by_other_people_are_excluded(tmp_path):
    paths = SubjectPaths("alice", tmp_path)
    paths.ensure()
    tid = _thread_on_side(False)
    other = ("I always insist on a second reviewer with OTHERPERSONMARKER before "
             "anything ships, because one pair of eyes has never been enough here.")
    CorpusStore(paths).write("v1", [
        _turn(tid, SENTENCE),
        _turn(tid, other, is_subject=False),
    ])
    pw.main(["alice", "v1", "--root", str(tmp_path)])
    body = (Path(paths.probes) / "worksheet.md").read_text(encoding="utf-8")
    assert "OTHERPERSONMARKER" not in body


def test_near_identical_turns_collapse_to_one_line():
    """The subject repeats phrasings; forty variants of one opinion is not
    forty probes."""
    turns = [_turn("t", SENTENCE, source_id=f"s{i}") for i in range(5)]
    turns.append(_turn("t", "I prefer to write the test first, always, no matter "
                            "how small the change actually turns out to be.",
                       source_id="other"))
    picked = pw.select(turns, pw.CATEGORIES["preferences"])
    assert len(picked) == 2


def test_short_turns_are_skipped():
    """The corpus median is ~42 chars; 'yeah ok' carries no checkable claim."""
    picked = pw.select([_turn("t", "i always do")], pw.CATEGORIES["preferences"])
    assert picked == []


def test_a_turn_matching_no_category_is_not_selected():
    picked = pw.select([_turn("t", "The build finished and the artifacts were "
                                   "uploaded to the bucket without incident.")],
                       pw.CATEGORIES["preferences"])
    assert picked == []


def test_exits_nonzero_when_the_corpus_version_is_missing(tmp_path):
    paths = SubjectPaths("alice", tmp_path)
    paths.ensure()
    assert pw.main(["alice", "nope", "--root", str(tmp_path)]) == 1


def test_text_duplicated_into_a_heldout_thread_is_excluded(tmp_path):
    """The split assigns whole threads, but the subject broadcasts the same
    message to several people -- so identical content can sit on both sides."""
    paths = SubjectPaths("alice", tmp_path)
    paths.ensure()
    train_tid, held_tid = _thread_on_side(False), _thread_on_side(True)
    broadcast = ("I always send the same note to everyone BROADCASTMARKER when a "
                 "quarter closes, because repeating myself beats being misquoted.")
    CorpusStore(paths).write("v1", [
        _turn(train_tid, broadcast, source_id="a"),
        _turn(held_tid, broadcast, source_id="b"),
        _turn(train_tid, SENTENCE, source_id="c"),
    ])
    pw.main(["alice", "v1", "--root", str(tmp_path)])
    body = (Path(paths.probes) / "worksheet.md").read_text(encoding="utf-8")
    assert "BROADCASTMARKER" not in body
    assert "boring option" in body
