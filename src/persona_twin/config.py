from __future__ import annotations
from pathlib import Path
import yaml
from pydantic import BaseModel

class SubjectConfig(BaseModel):
    subject_id: str
    display_name: str
    aliases: list[str] = []
    enabled_sources: list[str] = []

def load_subject(subject_id: str, root: Path) -> SubjectConfig:
    path = Path(root) / "config" / "subjects" / f"{subject_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no config for subject {subject_id!r} at {path}")
    cfg = SubjectConfig(**yaml.safe_load(path.read_text()))
    if cfg.subject_id != subject_id:
        raise ValueError(f"config declares {cfg.subject_id!r}, loaded as {subject_id!r}")
    return cfg
