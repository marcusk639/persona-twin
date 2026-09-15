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


SENSITIVE_CASES = [
    ("substance dosing", "I always tell people to taper down to 2.5mg oxy before "
                         "jumping off, because otherwise withdrawal is brutal for weeks."),
    ("mental health crisis", "I always feel like I have to find my own way out or I'll "
                             "make an early exit from this life, and that scares me badly."),
    ("intimate relationship", "I love you more than anything and I always feel invalidated "
                              "when I try to defend myself in our arguments about this."),
    ("third-party benefits claim", "I always tell him to write in 10B that his EPRs from "
                                   "2013 show declines anticipated for the disability claim."),
    ("medical", "I always put off calling the doctor about the prescription until the "
                "diagnosis actually starts interfering with my work somehow."),
]


def test_sensitive_lines_are_excluded_from_the_worksheet():
    """Probes built on these would put third parties' private data into an
    eval artifact, and probe files are not classifier-gated."""
    for label, text in SENSITIVE_CASES:
        picked = pw.select([_turn("t", text)], pw.CATEGORIES["preferences"])
        assert picked == [], f"{label} line was not excluded: {text[:50]}"


def test_ordinary_work_lines_still_survive_the_sensitivity_filter():
    """Over-exclusion costs probe candidates; the filter must not eat the corpus."""
    picked = pw.select([_turn("t", SENTENCE)], pw.CATEGORIES["preferences"])
    assert len(picked) == 1


def test_excluded_count_is_reported_not_silently_dropped(tmp_path):
    """A filter that hides how much it removed reads as full coverage."""
    paths = SubjectPaths("alice", tmp_path)
    paths.ensure()
    tid = _thread_on_side(False)
    turns = [_turn(tid, SENTENCE, source_id="keep")]
    turns += [_turn(tid, t, source_id=f"drop{i}")
              for i, (_, t) in enumerate(SENSITIVE_CASES)]
    CorpusStore(paths).write("v1", turns)
    pw.main(["alice", "v1", "--root", str(tmp_path)])
    body = (Path(paths.probes) / "worksheet.md").read_text(encoding="utf-8")
    assert "withheld as sensitive" in body


# Family coverage: each category is probed with SEVERAL phrasings, because a
# filter built from remembered instances leaves siblings unswept -- "attorney"
# without "lawyer", "can't go on" without "can't live like this".
SENSITIVE_FAMILIES = {
    "legal counsel": ["spent nearly 4k on this lawyer and it went badly",
                      "my attorney said to wait", "the legal counsel we retained",
                      "we are heading into arbitration over it",
                      "they offered a severance package", "a settlement was proposed"],
    "crisis": ["I can't live like this anymore", "I can't go on like this",
               "make an early exit from this life", "thinking about ending it"],
    "substance": ["down to 2.5mg oxy", "he is tapering off suboxone",
                  "worried about a relapse", "the withdrawal was brutal"],
    "medical": ["waiting on the diagnosis", "she changed my prescription",
                "my therapist suggested", "the medication makes me foggy"],
    "intimate": ["I love you more than anything", "we talked about divorce",
                 "the custody arrangement", "I feel invalidated when we argue"],
    "third-party claim": ["his EPRs from 2013", "the VA disability claim",
                          "filing a disability claim for him"],
}


def test_every_sensitive_family_is_swept_not_just_one_phrasing():
    import probe_worksheet as m
    for family, phrasings in SENSITIVE_FAMILIES.items():
        for phrase in phrasings:
            assert m._SENSITIVE.search(phrase), f"{family}: unswept phrasing {phrase!r}"


def test_the_filter_does_not_eat_ordinary_work_language():
    import probe_worksheet as m
    for ok in ["I always reach for the boring option first when a deadline looms",
               "we settled on Postgres for the billing service after benchmarking",
               "the interview process at that company was nonsensical",
               "I prefer to write the migration before the endpoint"]:
        assert not m._SENSITIVE.search(ok), f"false positive on work text: {ok!r}"


def test_product_domain_vocabulary_is_not_treated_as_sensitive():
    """The subject builds sober-living and recovery software; excluding that
    vocabulary would eat his main project's entire corpus."""
    import probe_worksheet as m
    for ok in ["the sober living house setup screen is the finicky part",
               "Regroup is a sober living app for house managers",
               "the recovery platform needs a detox intake flow",
               "house managers configure their house in the app"]:
        assert not m._SENSITIVE.search(ok), f"product-domain false positive: {ok!r}"
