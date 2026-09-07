import pytest, yaml
from persona_twin.config import load_subject

def _write(root, sid, data):
    d = root / "config" / "subjects"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{sid}.yaml").write_text(yaml.safe_dump(data))

def test_loads_subject(tmp_path):
    _write(tmp_path, "alice", {
        "subject_id": "alice", "display_name": "Alice Example",
        "aliases": ["ali"], "enabled_sources": ["claude_code"]})
    cfg = load_subject("alice", root=tmp_path)
    assert cfg.display_name == "Alice Example"
    assert cfg.enabled_sources == ["claude_code"]

def test_missing_subject_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_subject("nobody", root=tmp_path)

def test_mismatched_id_raises(tmp_path):
    _write(tmp_path, "alice", {
        "subject_id": "bob", "display_name": "B", "aliases": [], "enabled_sources": []})
    with pytest.raises(ValueError):
        load_subject("alice", root=tmp_path)
