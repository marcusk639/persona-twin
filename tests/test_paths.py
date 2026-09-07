from pathlib import Path
import pytest
from persona_twin.paths import SubjectPaths

def test_every_tier_is_subject_scoped(tmp_path):
    p = SubjectPaths("alice", root=tmp_path)
    for tier in (p.vault, p.clean, p.exportable, p.golden, p.cursors, p.ledger):
        assert "alice" in str(tier), f"{tier} is not subject-scoped"

def test_two_subjects_never_share_a_directory(tmp_path):
    a, b = SubjectPaths("alice", root=tmp_path), SubjectPaths("bob", root=tmp_path)
    assert a.vault != b.vault
    assert not str(a.vault).startswith(str(b.vault))

def test_ensure_creates_all_tiers(tmp_path):
    p = SubjectPaths("alice", root=tmp_path)
    p.ensure()
    assert p.vault.is_dir() and p.clean.is_dir() and p.golden.is_dir()

def test_subject_id_must_be_slug(tmp_path):
    with pytest.raises(ValueError):
        SubjectPaths("../escape", root=tmp_path)
