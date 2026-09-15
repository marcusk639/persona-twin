from __future__ import annotations
from pathlib import Path
import yaml
from pydantic import BaseModel

class SubjectConfig(BaseModel):
    subject_id: str
    display_name: str
    aliases: list[str] = []
    enabled_sources: list[str] = []
    # Optional because it is needed only by the informed eval baseline; a
    # subject can be ingested and scrubbed without one. Consumers that
    # require it must fail loudly rather than substitute a default -- see
    # eval.baseline.informed_baseline.
    occupation: str | None = None
    # Replacement for the subject's own name in turn TEXT. Optional because a
    # corpus whose bodies never contain the name needs no scrub; build_corpus
    # raises if the name IS present and this is unset, rather than shipping an
    # unscrubbed corpus -- see corpus.build.
    pseudonym_name: str | None = None

def load_subject(subject_id: str, root: Path) -> SubjectConfig:
    path = Path(root) / "config" / "subjects" / f"{subject_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no config for subject {subject_id!r} at {path}")
    cfg = SubjectConfig(**yaml.safe_load(path.read_text()))
    if cfg.subject_id != subject_id:
        raise ValueError(f"config declares {cfg.subject_id!r}, loaded as {subject_id!r}")
    return cfg
