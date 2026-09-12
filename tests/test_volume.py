from datetime import datetime, timezone
from persona_twin.paths import SubjectPaths
from persona_twin.schema import Turn
from persona_twin.corpus.store import CorpusStore
from persona_twin.corpus.volume import measure

def _t(i, is_subject, source="imessage", text="hello world"):
    return Turn(subject_id="alice", source=source, source_id=str(i), thread_id="c1",
                ts=datetime.now(timezone.utc),
                author_id="alice" if is_subject else "P-1",
                is_subject=is_subject, text=text, assisted=False)

def _seeded(tmp_path, turns):
    p = SubjectPaths("alice", tmp_path); p.ensure()
    CorpusStore(p).write("v1", turns)
    return p

def test_counts_only_subject_authored_chars(tmp_path):
    p = _seeded(tmp_path, [_t(1, True, text="abcde"), _t(2, False, text="zzzzzzzzzz")])
    r = measure(p, "v1")
    assert r.subject_turns == 1 and r.subject_chars == 5 and r.total_turns == 2

def test_token_estimate_uses_four_chars_per_token(tmp_path):
    p = _seeded(tmp_path, [_t(1, True, text="a" * 400)])
    assert measure(p, "v1").est_tokens == 100

def test_cost_projection_scales_with_rate(tmp_path):
    p = _seeded(tmp_path, [_t(1, True, text="a" * 4_000_000)])
    r = measure(p, "v1", usd_per_mtok=3.0)
    assert round(r.projected_usd, 2) == 3.0

def test_breakdown_by_source(tmp_path):
    p = _seeded(tmp_path, [_t(1, True, source="imessage"),
                           _t(2, True, source="git_repos")])
    assert set(measure(p, "v1").by_source) == {"imessage", "git_repos"}

def test_all_subject_turns_still_counted_when_no_non_subject_turns_present(tmp_path):
    """Non-vacuous companion to the mixed-authorship test above: with an
    all-subject corpus, subject_turns/subject_chars must equal the totals
    rather than merely "not obviously wrong" -- a broken filter that let
    everything through would also pass a fixture that only ever had
    subject turns."""
    p = _seeded(tmp_path, [_t(1, True, text="abcde"), _t(2, True, text="fghij")])
    r = measure(p, "v1")
    assert r.subject_turns == r.total_turns == 2
    assert r.subject_chars == 10

def test_non_subject_only_corpus_yields_zero_subject_volume(tmp_path):
    """Companion at the other extreme: an all-non-subject corpus must
    measure zero subject volume, not silently fall back to total counts."""
    p = _seeded(tmp_path, [_t(1, False, text="abcde"), _t(2, False, text="fghij")])
    r = measure(p, "v1")
    assert r.total_turns == 2
    assert r.subject_turns == 0
    assert r.subject_chars == 0
    assert r.est_tokens == 0
    assert r.projected_usd == 0.0
    assert r.by_source == {}
