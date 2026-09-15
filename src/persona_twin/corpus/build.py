from __future__ import annotations
from dataclasses import dataclass
from persona_twin.connectors.base import SubjectContext
from persona_twin.corpus.store import CorpusStore
from persona_twin.ledger import LearningLedger
from persona_twin.normalize.identity import Pseudonymizer
from persona_twin.normalize.turns import normalize
from persona_twin.schema import Turn
from persona_twin.scrub.classify import classify
from persona_twin.scrub.secrets import redact
from persona_twin.vault import VaultWriter

SOURCES = ("imessage", "claude_code", "claude_ai", "perplexity", "git_repos",
           "mail", "chatgpt")

@dataclass(frozen=True)
class BuildReport:
    version: str
    written: int
    redacted: int
    excluded_confidential: int
    excluded_pasted: int

def build_corpus(ctx: SubjectContext, ledger: LearningLedger, version: str) -> BuildReport:
    """Vault -> normalize -> scrub -> versioned clean corpus. Fails closed (spec C4).

    Classification runs before redaction: a confidential turn is excluded
    entirely rather than scrubbed, so redacting it first would both waste
    work and risk making a client record look retainable once its markers
    were replaced. The `looks_pasted` check sits alongside it for the same
    reason: a pasted file/log/stack-trace turn is dropped outright rather
    than scrubbed, so there is no point redacting something about to be
    discarded. The flag itself is set at extraction (claude_code connector)
    and deliberately never consumed there or in normalize() — filtering is
    a corpus-build decision, kept out of the vault so it can be revisited
    without re-ingesting.
    """
    vault = VaultWriter(ctx.paths)
    pseudo = Pseudonymizer(ctx.paths)
    store = CorpusStore(ctx.paths)

    kept: list[Turn] = []
    redacted = excluded = excluded_pasted = 0
    for source in SOURCES:
        for env in vault.iter_source(source):
            turn = normalize(env)
            if turn is None:
                continue
            if classify(turn) == "confidential":
                excluded += 1
                continue
            if env.payload.get("looks_pasted"):
                excluded_pasted += 1
                continue
            clean_text = redact(turn.text)      # raises ScrubError rather than half-scrubbing
            if clean_text != turn.text:
                redacted += 1
            kept.append(pseudo.apply(turn.model_copy(
                update={"text": clean_text, "corpus_version": version})))

    written = store.write(version, kept)
    ledger.append("corpus_build", ctx.config.subject_id,
                  {"version": version, "written": written,
                   "redacted": redacted, "excluded_confidential": excluded,
                   "excluded_pasted": excluded_pasted})
    return BuildReport(version, written, redacted, excluded, excluded_pasted)
