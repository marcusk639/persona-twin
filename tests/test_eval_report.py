import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

from persona_twin.paths import SubjectPaths
from persona_twin.schema import Turn
from persona_twin.corpus.store import CorpusStore
from persona_twin.eval.report import (
    GateResult, split_summary, s2_self_distance, s5_probe_composition,
    s6_probe_composition, render, pending_results, CAVEATS,
)

NOW = datetime(2026, 9, 12, tzinfo=timezone.utc)


# 40 threads, not 25: with 25 the sha256 buckets put ZERO threads on the
# held-out side, and test_split_summary_holds_out_a_nonzero_minority fails
# deterministically. Verified by computing the buckets before execution.
def _seed(tmp_path, n=200):
    p = SubjectPaths("s", tmp_path); p.ensure()
    turns = [Turn(subject_id="s", source="imessage", source_id=str(i),
                  thread_id=f"c{i % 40}", ts=NOW - timedelta(days=400 + i),
                  author_id="s", is_subject=True, text=f"message {i} here",
                  assisted=False) for i in range(n)]
    CorpusStore(p).write("v1", turns)
    return p


# --- base behaviour (spec's own test set) ---

def test_split_summary_counts_every_turn(tmp_path):
    p = _seed(tmp_path)
    s = split_summary(p, "v1", NOW)
    assert s["train"] + s["heldout"] + s["quarantined"] == s["total"] == 200

def test_split_summary_holds_out_a_nonzero_minority(tmp_path):
    p = _seed(tmp_path)
    s = split_summary(p, "v1", NOW)
    assert 0 < s["heldout"] < s["train"]

def test_s2_reports_a_sample_size(tmp_path):
    p = _seed(tmp_path)
    r = s2_self_distance(p, "v1", NOW, seed=0)
    assert r.criterion == "S2" and r.n > 0

def test_s2_is_reproducible(tmp_path):
    p = _seed(tmp_path)
    a = s2_self_distance(p, "v1", NOW, seed=4)
    b = s2_self_distance(p, "v1", NOW, seed=4)
    assert a.value == b.value

def test_render_marks_pending_criteria_as_pending():
    """Checks the status column, not just the value column. The original
    version of this test (checking only "pending" in out.lower()) is the
    14th test-that-cannot-fail found in this project: the value column
    prints the literal string "pending" regardless of what the status
    column says, so a status-derivation bug that renders every passed=None
    row as PASS still leaves this assertion green. Confirmed by mutation:
    see task-6-report.md."""
    results = [GateResult(criterion="S1", value=None, target="<=0.60",
                          n=0, passed=None, note="requires generation (stage 5)")]
    out = render(results)
    assert "S1" in out and "pending" in out.lower()
    for line in out.splitlines():
        if line.startswith("S1"):
            assert "PASS" not in line
            assert "FAIL" not in line

def test_render_includes_sample_sizes():
    results = [GateResult(criterion="S2", value=1.5, target="<= p95 self-distance",
                          n=123, passed=True, note="")]
    assert "123" in render(results)


# --- NO DATA / PARTIAL DATA: S2 must not report a flattering zero ---

def test_s2_on_nonexistent_corpus_version_does_not_report_a_flattering_zero(tmp_path):
    """NO DATA: reading a version that was never written returns an empty
    corpus. Unguarded, self_distance_band's own degenerate (0.0, 0.0) on <4
    turns would surface as GateResult(value=0.0, n=0) -- a self-distance p95
    of 0.0000 reads as an unusually TIGHT bar, exactly the flattering
    direction this project keeps finding defects in. It must render as
    unevaluable instead."""
    p = SubjectPaths("s", tmp_path); p.ensure()
    CorpusStore(p)  # creates the db; writes no version
    r = s2_self_distance(p, "does-not-exist", NOW)
    assert r.value is None
    assert r.n == 0
    assert "cannot be evaluated" in r.note.lower()

def test_s2_on_too_few_heldout_turns_does_not_report_a_flattering_zero(tmp_path):
    """PARTIAL DATA: a real corpus, but too thin (on any split of 3 turns,
    the subject-heldout count is at most 3) to compute a meaningful band.
    Same guard as the no-data case, exercised on partial rather than absent
    data, and independent of how the sha256 buckets happen to split these 3."""
    p = SubjectPaths("s", tmp_path); p.ensure()
    turns = [Turn(subject_id="s", source="imessage", source_id=str(i),
                  thread_id=f"c{i}", ts=NOW - timedelta(days=400 + i),
                  author_id="s", is_subject=True, text=f"message {i}",
                  assisted=False) for i in range(3)]
    CorpusStore(p).write("v1", turns)
    r = s2_self_distance(p, "v1", NOW)
    assert r.value is None
    assert r.n < 4
    assert "cannot be evaluated" in r.note.lower()


# --- S5 probe-set composition gate (inherited obligation 1) ---

def _write_facts(paths, n_answerable, n_unanswerable):
    probes_dir = Path(paths.root) / "data" / "subjects" / paths.subject_id / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"probe_id": f"a{i}", "question": "q", "expected": "x",
             "answerable": True, "notes": ""} for i in range(n_answerable)]
    rows += [{"probe_id": f"u{i}", "question": "q", "expected": "",
              "answerable": False, "notes": ""} for i in range(n_unanswerable)]
    (probes_dir / "facts.json").write_text(json.dumps(rows))


def test_s5_refuses_when_no_probe_set_authored_yet(tmp_path):
    # UNEVALUABLE, reason 1: the subject hasn't authored the file yet -- the
    # common case in stage 3, since probes.py's loaders are machinery only.
    p = SubjectPaths("s", tmp_path); p.ensure()
    r = s5_probe_composition(p)
    assert r.criterion == "S5"
    assert r.value is None and r.passed is None and r.n == 0
    assert "cannot be evaluated" in r.note.lower()
    assert "no probe set" in r.note.lower()

def test_s5_refuses_when_n_below_200(tmp_path):
    p = SubjectPaths("s", tmp_path); p.ensure()
    _write_facts(p, n_answerable=80, n_unanswerable=80)  # n=160, ratio fine, n too low
    r = s5_probe_composition(p)
    assert r.value is None and r.passed is None
    assert r.n == 160
    assert "cannot be evaluated" in r.note.lower()
    assert "160" in r.note

def test_s5_refuses_when_unanswerable_share_out_of_range(tmp_path):
    p = SubjectPaths("s", tmp_path); p.ensure()
    _write_facts(p, n_answerable=180, n_unanswerable=20)  # n=200, share=0.1
    r = s5_probe_composition(p)
    assert r.value is None and r.passed is None
    assert r.n == 200
    assert "cannot be evaluated" in r.note.lower()
    assert "outside" in r.note.lower()

def test_s5_reports_composition_when_valid_and_still_defers_to_stage_5(tmp_path):
    p = SubjectPaths("s", tmp_path); p.ensure()
    _write_facts(p, n_answerable=100, n_unanswerable=100)  # n=200, share=0.5
    r = s5_probe_composition(p)
    assert r.value is None      # composition being valid still isn't a verdict --
    assert r.passed is None     # generation (stage 5) hasn't happened yet.
    assert r.n == 200
    assert "cannot be evaluated" not in r.note.lower()
    assert "100" in r.note      # both answerable and unanswerable counts present
    assert "stage 5" in r.note.lower()

def test_s5_refuses_on_malformed_probe_file(tmp_path):
    p = SubjectPaths("s", tmp_path); p.ensure()
    probes_dir = Path(p.root) / "data" / "subjects" / p.subject_id / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)
    (probes_dir / "facts.json").write_text("{not json")
    r = s5_probe_composition(p)
    assert r.value is None and r.passed is None
    assert "cannot be evaluated" in r.note.lower()

def test_render_of_composition_gated_s5_never_shows_pass_or_fail(tmp_path):
    p = SubjectPaths("s", tmp_path); p.ensure()
    _write_facts(p, n_answerable=100, n_unanswerable=100)  # valid composition
    r = s5_probe_composition(p)
    out = render([r])
    assert "pending" in out.lower()
    assert "PASS" not in out and "FAIL" not in out


# --- S6 probe-set composition gate (mirrors S5's, inherited obligation 1 --
# extended to S6 per team-lead ruling: n>=60 alone does not protect against
# a should_decline-degenerate set, the same flattering-null shape as S5's
# all-answerable case) ---

def _write_refusals(paths, n_decline, n_comply):
    probes_dir = Path(paths.root) / "data" / "subjects" / paths.subject_id / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"probe_id": f"d{i}", "prompt": "p", "should_decline": True, "notes": ""}
            for i in range(n_decline)]
    rows += [{"probe_id": f"c{i}", "prompt": "p", "should_decline": False, "notes": ""}
             for i in range(n_comply)]
    (probes_dir / "refusals.json").write_text(json.dumps(rows))


def test_s6_refuses_when_no_probe_set_authored_yet(tmp_path):
    p = SubjectPaths("s", tmp_path); p.ensure()
    r = s6_probe_composition(p)
    assert r.criterion == "S6"
    assert r.value is None and r.passed is None and r.n == 0
    assert "cannot be evaluated" in r.note.lower()
    assert "no probe set" in r.note.lower()

def test_s6_refuses_when_n_below_60(tmp_path):
    p = SubjectPaths("s", tmp_path); p.ensure()
    _write_refusals(p, n_decline=20, n_comply=20)  # n=40, share fine, n too low
    r = s6_probe_composition(p)
    assert r.value is None and r.passed is None
    assert r.n == 40
    assert "cannot be evaluated" in r.note.lower()
    assert "40" in r.note

def test_s6_refuses_when_all_probes_should_decline(tmp_path):
    # The degenerate case that motivated this gate: a twin that refuses
    # everything scores a perfect 1.0 agreement against an all-decline set,
    # with no judgement involved at all.
    p = SubjectPaths("s", tmp_path); p.ensure()
    _write_refusals(p, n_decline=60, n_comply=0)
    r = s6_probe_composition(p)
    assert r.value is None and r.passed is None
    assert r.n == 60
    assert "cannot be evaluated" in r.note.lower()
    assert "outside" in r.note.lower()

def test_s6_refuses_when_all_probes_should_comply(tmp_path):
    # The mirror-image degenerate case: a twin that never refuses scores a
    # perfect 1.0 agreement against an all-comply set.
    p = SubjectPaths("s", tmp_path); p.ensure()
    _write_refusals(p, n_decline=0, n_comply=60)
    r = s6_probe_composition(p)
    assert r.value is None and r.passed is None
    assert r.n == 60
    assert "cannot be evaluated" in r.note.lower()
    assert "outside" in r.note.lower()

def test_s6_reports_composition_when_valid_and_still_defers_to_stage_5(tmp_path):
    p = SubjectPaths("s", tmp_path); p.ensure()
    _write_refusals(p, n_decline=30, n_comply=30)  # n=60, share=0.5
    r = s6_probe_composition(p)
    assert r.value is None      # composition being valid still isn't a verdict --
    assert r.passed is None     # generation (stage 5) hasn't happened yet.
    assert r.n == 60
    assert "cannot be evaluated" not in r.note.lower()
    assert "30" in r.note       # both should_decline and should_comply counts present
    assert "stage 5" in r.note.lower()

def test_s6_refuses_on_malformed_probe_file(tmp_path):
    p = SubjectPaths("s", tmp_path); p.ensure()
    probes_dir = Path(p.root) / "data" / "subjects" / p.subject_id / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)
    (probes_dir / "refusals.json").write_text("{not json")
    r = s6_probe_composition(p)
    assert r.value is None and r.passed is None
    assert "cannot be evaluated" in r.note.lower()

def test_render_of_composition_gated_s6_never_shows_pass_or_fail(tmp_path):
    p = SubjectPaths("s", tmp_path); p.ensure()
    _write_refusals(p, n_decline=30, n_comply=30)  # valid composition
    r = s6_probe_composition(p)
    out = render([r])
    assert "pending" in out.lower()
    assert "PASS" not in out and "FAIL" not in out


# --- caveats (inherited obligations 2, 3, 4): always present, unconditionally ---

def test_render_always_includes_s1_blinding_caveat():
    out = render(pending_results())
    assert "blinding" in out.lower()

def test_render_always_includes_corpus_safety_caveat_even_with_no_criteria():
    # Not conditioned on any specific criterion being present in the
    # results list -- this is a standing limitation of the harness, not
    # something tied to one row, so it must survive an empty report too.
    out = render([])
    assert "c5" in out.lower() or "golden corpus" in out.lower()
    assert "c6" in out.lower() or "classify" in out.lower()

def test_caveats_constant_names_both_inherited_limits():
    joined = " ".join(CAVEATS).lower()
    assert "blinding" in joined
    assert "classify" in joined
    assert "golden" in joined


# --- an unevaluable criterion must never render as passing ---

def test_render_never_shows_a_computed_value_for_pending_criteria():
    """Checks both the value column ("pending", not a number) AND the status
    column (never PASS/FAIL) independently. Checking only the value column
    is not enough on its own: a status-derivation bug that renders every
    passed=None row as PASS still leaves the value column reading "pending",
    so a test that only checks the value column passes right through that
    exact defect (confirmed by mutation -- see task-6-report.md)."""
    out = render(pending_results())
    for line in out.splitlines():
        if any(line.startswith(c) for c in ("S1", "S3", "S4")):
            assert "pending" in line
            assert "0.0000" not in line
            assert "PASS" not in line
            assert "FAIL" not in line
