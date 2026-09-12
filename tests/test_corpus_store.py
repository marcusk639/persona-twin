from datetime import datetime, timezone
import pytest
from persona_twin.paths import SubjectPaths
from persona_twin.schema import Turn
from persona_twin.corpus.store import CorpusStore

def _t(i):
    return Turn(subject_id="alice", source="imessage", source_id=str(i), thread_id="c1",
                ts=datetime.now(timezone.utc), author_id="alice", is_subject=True,
                text=f"m{i}", assisted=False)

def test_write_and_read_roundtrip(tmp_path):
    s = CorpusStore(SubjectPaths("alice", tmp_path))
    assert s.write("v1", [_t(1), _t(2)]) == 2
    assert [t.source_id for t in s.read("v1")] == ["1", "2"]

def test_versions_are_isolated(tmp_path):
    s = CorpusStore(SubjectPaths("alice", tmp_path))
    s.write("v1", [_t(1)]); s.write("v2", [_t(1), _t(2)])
    assert len(s.read("v1")) == 1 and len(s.read("v2")) == 2
    assert set(s.versions()) == {"v1", "v2"}

def test_rewriting_a_version_raises(tmp_path):
    s = CorpusStore(SubjectPaths("alice", tmp_path))
    s.write("v1", [_t(1)])
    with pytest.raises(ValueError):
        s.write("v1", [_t(2)])

def test_written_turns_carry_their_corpus_version(tmp_path):
    s = CorpusStore(SubjectPaths("alice", tmp_path))
    s.write("v1", [_t(1)])
    assert s.read("v1")[0].corpus_version == "v1"
