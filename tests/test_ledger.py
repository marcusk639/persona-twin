import pytest
from datetime import datetime
from pydantic import ValidationError
from persona_twin.ledger import LearningLedger, LedgerEntry

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

def test_entry_is_frozen(tmp_path):
    led = LearningLedger(tmp_path / "l.jsonl")
    a = led.append("ingest", "alice", {})
    with pytest.raises(ValidationError):
        a.kind = "modified"

def test_naive_datetime_raises(tmp_path):
    led = LearningLedger(tmp_path / "l.jsonl")
    naive_ts = datetime(2026, 9, 7, 12, 0, 0)  # no timezone
    with pytest.raises(ValidationError):
        LedgerEntry(
            entry_id="test123", ts=naive_ts, kind="ingest",
            subject_id="alice", payload={})

def test_provenance_diamond_graph(tmp_path):
    led = LearningLedger(tmp_path / "l.jsonl")
    a = led.append("ingest", "alice", {})
    b = led.append("scrub", "alice", {}, parents=[a.entry_id])
    c = led.append("scrub", "alice", {}, parents=[a.entry_id])
    d = led.append("build", "alice", {}, parents=[b.entry_id, c.entry_id])
    chain = [e.entry_id for e in led.provenance(d.entry_id)]
    assert chain[0] == d.entry_id
    assert chain.count(a.entry_id) == 1, "a should appear exactly once despite two paths"
    assert set(chain) == {a.entry_id, b.entry_id, c.entry_id, d.entry_id}

def test_provenance_with_revisited_parent(tmp_path):
    led = LearningLedger(tmp_path / "l.jsonl")
    a = led.append("ingest", "alice", {})
    b = led.append("scrub", "alice", {}, parents=[a.entry_id])
    c = led.append("build", "alice", {}, parents=[b.entry_id, a.entry_id])
    chain = led.provenance(c.entry_id)
    assert len(chain) == 3
    assert chain[0].entry_id == c.entry_id
    assert chain[-1].entry_id == a.entry_id


def test_entry_containing_a_unicode_line_separator_round_trips(tmp_path):
    """U+2028 is a line break to str.splitlines() but NOT escaped by JSON.

    run_connector records str(exc) on a partial ingest, so a connector failing
    while handling corpus text can put this character into the append-only
    chain. Splitting on it shears one entry into two unparseable halves and
    makes the whole provenance chain unreadable -- fail-closed in the wrong
    direction, on the mechanism that makes consent revocation executable.
    """
    led = LearningLedger(tmp_path / "ledger.jsonl")
    led.append("ingest", "alice", {"error": "bad turn: line break"})
    led.append("ingest", "alice", {"error": "plain"})
    entries = led.read_all()
    assert len(entries) == 2
    assert entries[0].payload["error"] == "bad turn: line break"
