import pytest
from persona_twin.ledger import LearningLedger

def test_appends_and_reads_back(tmp_path):
    led = LearningLedger(tmp_path / "l.jsonl")
    a = led.append("ingest", "alice", {"source": "imessage", "n": 10})
    b = led.append("corpus_build", "alice", {"version": "v1"}, parents=[a.entry_id])
    got = led.read_all()
    assert [e.entry_id for e in got] == [a.entry_id, b.entry_id]
    assert got[1].parents == [a.entry_id]

def test_entries_are_never_rewritten(tmp_path):
    p = tmp_path / "l.jsonl"
    led = LearningLedger(p)
    led.append("ingest", "alice", {"n": 1})
    first = p.read_text()
    led.append("ingest", "alice", {"n": 2})
    assert p.read_text().startswith(first), "existing lines must not change"

def test_provenance_walks_parents(tmp_path):
    led = LearningLedger(tmp_path / "l.jsonl")
    a = led.append("ingest", "alice", {})
    b = led.append("scrub", "alice", {}, parents=[a.entry_id])
    c = led.append("adapter", "alice", {}, parents=[b.entry_id])
    chain = [e.entry_id for e in led.provenance(c.entry_id)]
    assert chain == [c.entry_id, b.entry_id, a.entry_id]

def test_unknown_provenance_raises(tmp_path):
    led = LearningLedger(tmp_path / "l.jsonl")
    with pytest.raises(KeyError):
        led.provenance("nope")
