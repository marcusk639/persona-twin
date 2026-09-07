from datetime import datetime, timezone
from persona_twin.config import SubjectConfig
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
