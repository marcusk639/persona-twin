import pytest
from datetime import datetime, timezone
from persona_twin.config import SubjectConfig
from persona_twin.cursors import CursorStore
from persona_twin.paths import SubjectPaths
from persona_twin.schema import RawEnvelope
from persona_twin.ledger import LearningLedger
from persona_twin.connectors.base import SubjectContext, run_connector

class FakeConnector:
    name = "fake"
    def __init__(self, ids): self.ids = ids; self.seen_cursor = "unset"
    def fetch(self, ctx, cursor):
        self.seen_cursor = cursor
        now = datetime.now(timezone.utc)
        for i in self.ids:
            yield RawEnvelope(subject_id=ctx.config.subject_id, source=self.name,
                              source_id=i, ts=now, payload={"i": i}, ingested_at=now), i

class CrashingConnector:
    """Yields some envelopes, then raises partway through the stream."""
    name = "fake"
    def __init__(self, ids, fail_at):
        self.ids = ids; self.fail_at = fail_at
    def fetch(self, ctx, cursor):
        now = datetime.now(timezone.utc)
        for i in self.ids:
            if i == self.fail_at:
                raise RuntimeError(f"boom at {i}")
            yield RawEnvelope(subject_id=ctx.config.subject_id, source=self.name,
                              source_id=i, ts=now, payload={"i": i}, ingested_at=now), i

def _ctx(tmp_path):
    paths = SubjectPaths("alice", tmp_path); paths.ensure()
    cfg = SubjectConfig(subject_id="alice", display_name="A", aliases=[], enabled_sources=["fake"])
    return SubjectContext(config=cfg, paths=paths)

def test_first_run_starts_with_no_cursor(tmp_path):
    ctx = _ctx(tmp_path); led = LearningLedger(ctx.paths.ledger)
    c = FakeConnector(["1", "2"])
    res = run_connector(c, ctx, led)
    assert c.seen_cursor is None and res.new == 2 and res.cursor == "2"

def test_second_run_resumes_from_cursor(tmp_path):
    ctx = _ctx(tmp_path); led = LearningLedger(ctx.paths.ledger)
    run_connector(FakeConnector(["1", "2"]), ctx, led)
    c2 = FakeConnector(["3"])
    res = run_connector(c2, ctx, led)
    assert c2.seen_cursor == "2" and res.new == 1

def test_rerun_of_same_ids_records_duplicates(tmp_path):
    ctx = _ctx(tmp_path); led = LearningLedger(ctx.paths.ledger)
    run_connector(FakeConnector(["1", "2"]), ctx, led)
    res = run_connector(FakeConnector(["1", "2"]), ctx, led)
    assert res.new == 0 and res.duplicates == 2

def test_ingest_is_recorded_in_ledger(tmp_path):
    ctx = _ctx(tmp_path); led = LearningLedger(ctx.paths.ledger)
    run_connector(FakeConnector(["1"]), ctx, led)
    kinds = [e.kind for e in led.read_all()]
    assert "ingest" in kinds

def test_cursor_not_advanced_when_connector_yields_nothing(tmp_path):
    """A run that yields nothing must not rewrite the cursor to the same value."""
    ctx = _ctx(tmp_path); led = LearningLedger(ctx.paths.ledger)
    cursors = CursorStore(ctx.paths.cursors)
    assert cursors.get("fake") is None
    run_connector(FakeConnector([]), ctx, led)
    assert cursors.get("fake") is None

    run_connector(FakeConnector(["1", "2"]), ctx, led)
    assert cursors.get("fake") == "2"

    run_connector(FakeConnector([]), ctx, led)
    assert cursors.get("fake") == "2"

def test_crash_mid_stream_still_records_partial_ledger_entry(tmp_path):
    """Rows written before a crash must remain traceable via a partial ingest entry."""
    ctx = _ctx(tmp_path); led = LearningLedger(ctx.paths.ledger)
    cursors = CursorStore(ctx.paths.cursors)
    conn = CrashingConnector(["1", "2", "3", "4", "5"], fail_at="4")

    with pytest.raises(RuntimeError, match="boom at 4"):
        run_connector(conn, ctx, led)

    assert cursors.get("fake") is None  # cursor not advanced past the failure

    entries = [e for e in led.read_all() if e.kind == "ingest"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry.payload["partial"] is True
    assert "boom at 4" in entry.payload["error"]
    assert entry.payload["new"] == 3
    assert entry.payload["duplicates"] == 0
    assert entry.payload["cursor_to"] == "3"
