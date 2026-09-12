from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from persona_twin.corpus.store import CorpusStore
from persona_twin.paths import SubjectPaths

CHARS_PER_TOKEN = 4

@dataclass(frozen=True)
class VolumeReport:
    total_turns: int
    subject_turns: int
    subject_chars: int
    est_tokens: int
    projected_usd: float
    by_source: dict[str, int] = field(default_factory=dict)

def measure(paths: SubjectPaths, version: str,
            usd_per_mtok: float = 3.0) -> VolumeReport:
    """Replace the §4.4 estimate with a measurement before authorizing stage-4 mining."""
    turns = CorpusStore(paths).read(version)
    subject = [t for t in turns if t.is_subject]
    chars = sum(len(t.text) for t in subject)
    tokens = chars // CHARS_PER_TOKEN
    by_source = Counter(t.source for t in subject)
    return VolumeReport(
        total_turns=len(turns), subject_turns=len(subject), subject_chars=chars,
        est_tokens=tokens, projected_usd=(tokens / 1_000_000) * usd_per_mtok,
        by_source=dict(by_source))
