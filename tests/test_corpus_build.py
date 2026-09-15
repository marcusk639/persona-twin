from datetime import datetime, timezone
import pytest
from persona_twin.config import SubjectConfig
from persona_twin.paths import SubjectPaths
from persona_twin.ledger import LearningLedger
from persona_twin.schema import RawEnvelope
from persona_twin.vault import VaultWriter
from persona_twin.connectors.base import SubjectContext
from persona_twin.corpus.build import build_corpus
from persona_twin.corpus.store import CorpusStore
from persona_twin.scrub.secrets import ScrubError

def _ctx(tmp_path):
    p = SubjectPaths("alice", tmp_path); p.ensure()
    return SubjectContext(config=SubjectConfig(subject_id="alice", display_name="A"), paths=p)

def _seed_source(ctx, source, payloads):
    w = VaultWriter(ctx.paths)
    now = datetime.now(timezone.utc)
    for i, pl in enumerate(payloads):
        w.write(RawEnvelope(subject_id="alice", source=source, source_id=str(i),
                            ts=now, payload=pl, ingested_at=now))

def _seed(ctx, payloads):
    _seed_source(ctx, "imessage", payloads)

def test_build_normalizes_scrubs_and_stores(tmp_path):
    ctx = _ctx(tmp_path)
    _seed(ctx, [{"text": "hello there", "is_from_me": True, "handle": "+1", "chat_guid": "c1"}])
    rep = build_corpus(ctx, LearningLedger(ctx.paths.ledger), "v1")
    assert rep.written == 1
    assert CorpusStore(ctx.paths).read("v1")[0].text == "hello there"

def test_secrets_are_redacted_in_the_corpus(tmp_path):
    ctx = _ctx(tmp_path)
    _seed(ctx, [{"text": "key AKIAIOSFODNN7EXAMPLE ok", "is_from_me": True,
                 "handle": "+1", "chat_guid": "c1"}])
    build_corpus(ctx, LearningLedger(ctx.paths.ledger), "v1")
    assert "AKIA" not in CorpusStore(ctx.paths).read("v1")[0].text

def test_confidential_turns_are_excluded(tmp_path):
    ctx = _ctx(tmp_path)
    _seed(ctx, [{"text": "SSN 123-45-6789", "is_from_me": True,
                 "handle": "+1", "chat_guid": "c1"}])
    rep = build_corpus(ctx, LearningLedger(ctx.paths.ledger), "v1")
    assert rep.excluded_confidential == 1 and rep.written == 0

def test_counterparties_are_pseudonymized(tmp_path):
    ctx = _ctx(tmp_path)
    _seed(ctx, [{"text": "yo", "is_from_me": False,
                 "handle": "+15551234567", "chat_guid": "c1"}])
    build_corpus(ctx, LearningLedger(ctx.paths.ledger), "v1")
    assert CorpusStore(ctx.paths).read("v1")[0].author_id.startswith("P-")

def test_build_is_recorded_in_ledger(tmp_path):
    ctx = _ctx(tmp_path)
    _seed(ctx, [{"text": "hi", "is_from_me": True, "handle": "+1", "chat_guid": "c1"}])
    led = LearningLedger(ctx.paths.ledger)
    build_corpus(ctx, led, "v1")
    assert any(e.kind == "corpus_build" for e in led.read_all())

def test_pasted_claude_code_turns_are_excluded(tmp_path):
    ctx = _ctx(tmp_path)
    _seed_source(ctx, "claude_code",
                 [{"text": "traceback (most recent call last): boom", "file": "f1",
                   "looks_pasted": True}])
    rep = build_corpus(ctx, LearningLedger(ctx.paths.ledger), "v1")
    assert rep.excluded_pasted == 1 and rep.written == 0

def test_non_pasted_claude_code_turns_are_included(tmp_path):
    ctx = _ctx(tmp_path)
    _seed_source(ctx, "claude_code",
                 [{"text": "typed prose, not a paste", "file": "f1", "looks_pasted": False}])
    rep = build_corpus(ctx, LearningLedger(ctx.paths.ledger), "v1")
    assert rep.excluded_pasted == 0 and rep.written == 1

def test_sources_without_the_looks_pasted_key_are_included(tmp_path):
    ctx = _ctx(tmp_path)
    _seed(ctx, [{"text": "hi", "is_from_me": True, "handle": "+1", "chat_guid": "c1"}])
    rep = build_corpus(ctx, LearningLedger(ctx.paths.ledger), "v1")
    assert rep.excluded_pasted == 0 and rep.written == 1

def test_scrub_error_propagates_and_leaves_no_partial_corpus(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    _seed(ctx, [{"text": "hi", "is_from_me": True, "handle": "+1", "chat_guid": "c1"}])

    def boom(text):
        raise ScrubError("simulated scrub failure")

    monkeypatch.setattr("persona_twin.corpus.build.redact", boom)
    with pytest.raises(ScrubError):
        build_corpus(ctx, LearningLedger(ctx.paths.ledger), "v1")
    assert CorpusStore(ctx.paths).versions() == []


# --- text-level identity scrub ---

def _ctx_named(tmp_path, *, pseudonym_name):
    p = SubjectPaths("alice", tmp_path); p.ensure()
    cfg = SubjectConfig(subject_id="alice", display_name="Marcus Klein",
                        aliases=["Marcus", "Klein"], pseudonym_name=pseudonym_name)
    return SubjectContext(config=cfg, paths=p)


def _msg(text, from_me=True):
    return {"text": text, "is_from_me": from_me, "handle": "+1", "chat_guid": "c1"}


def test_build_scrubs_the_subject_name_from_turn_text(tmp_path):
    """Pseudonymizer.apply rewrites author_id only, so a name in a message body
    reached the clean corpus and the frozen baseline untouched."""
    ctx = _ctx_named(tmp_path, pseudonym_name="Dylan")
    _seed(ctx, [_msg("Hey Marcus, you around?", from_me=False),
                _msg("yeah just got back")])
    build_corpus(ctx, LearningLedger(ctx.paths.ledger), "v1")
    texts = [t.text for t in CorpusStore(ctx.paths).read("v1")]
    assert "Marcus" not in " ".join(texts)
    assert "Hey Dylan, you around?" in texts


def test_build_refuses_rather_than_shipping_an_unscrubbed_corpus(tmp_path):
    """Failing loudly beats silence: the leak survived precisely because every
    check passed while asking a narrower question."""
    ctx = _ctx_named(tmp_path, pseudonym_name=None)
    _seed(ctx, [_msg("Hey Marcus, you around?", from_me=False)])
    with pytest.raises(ValueError, match="pseudonym_name"):
        build_corpus(ctx, LearningLedger(ctx.paths.ledger), "v1")


def test_the_requirement_is_conditional_on_the_name_appearing(tmp_path):
    ctx = _ctx_named(tmp_path, pseudonym_name=None)
    _seed(ctx, [_msg("the billing flow is finicky on complex views")])
    assert build_corpus(ctx, LearningLedger(ctx.paths.ledger), "v1").written == 1


def test_the_subjects_own_turns_are_scrubbed_too(tmp_path):
    """Signature lines and self-references are still the name in the corpus."""
    ctx = _ctx_named(tmp_path, pseudonym_name="Dylan")
    _seed(ctx, [_msg("thanks, Marcus")])
    assert CorpusStore(ctx.paths).read("v1") == [] or True
    build_corpus(ctx, LearningLedger(ctx.paths.ledger), "v1")
    assert "Marcus" not in CorpusStore(ctx.paths).read("v1")[0].text


def test_identity_scrubs_are_counted_separately_from_secret_redactions(tmp_path):
    """One number for two operations hides which one ran. A spike in
    'redacted' after adding the identity scrub would otherwise read as a
    secrets regression."""
    ctx = _ctx_named(tmp_path, pseudonym_name="Dylan")
    _seed(ctx, [_msg("Hey Marcus, you around?", from_me=False),
                _msg("key AKIAIOSFODNN7EXAMPLE ok"),
                _msg("nothing sensitive here")])
    r = build_corpus(ctx, LearningLedger(ctx.paths.ledger), "v1")
    assert r.redacted == 1, "only the secret turn is a redaction"
    assert r.identity_scrubbed == 1, "only the named turn is an identity scrub"


def test_a_turn_that_is_both_redacted_and_scrubbed_counts_in_both(tmp_path):
    """The two operations are independent; one turn can need both."""
    ctx = _ctx_named(tmp_path, pseudonym_name="Dylan")
    _seed(ctx, [_msg("Marcus here, key AKIAIOSFODNN7EXAMPLE ok")])
    r = build_corpus(ctx, LearningLedger(ctx.paths.ledger), "v1")
    assert r.redacted == 1 and r.identity_scrubbed == 1
    txt = CorpusStore(ctx.paths).read("v1")[0].text
    assert "Marcus" not in txt and "AKIA" not in txt
