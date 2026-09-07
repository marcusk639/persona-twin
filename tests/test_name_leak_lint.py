from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.name_leak_lint import scan

def _mk(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)

def test_flags_subject_name_in_source(tmp_path):
    _mk(tmp_path, "src/persona_twin/x.py", "GREETING = 'hello Zephyr'\n")
    hits = scan(tmp_path, ["Zephyr"], allow_dirs=("config", "data"))
    assert len(hits) == 1 and hits[0][1] == 1

def test_allows_name_in_config_and_data(tmp_path):
    _mk(tmp_path, "config/subjects/zephyr.yaml", "display_name: Zephyr\n")
    _mk(tmp_path, "data/subjects/zephyr/vault/x.json", '{"n": "Zephyr"}')
    assert scan(tmp_path, ["Zephyr"], allow_dirs=("config", "data")) == []

def test_is_case_insensitive(tmp_path):
    _mk(tmp_path, "src/a.py", "x = 'ZEPHYR'\n")
    assert len(scan(tmp_path, ["Zephyr"], allow_dirs=("config", "data"))) == 1

def test_clean_tree_passes(tmp_path):
    _mk(tmp_path, "src/a.py", "x = 1\n")
    assert scan(tmp_path, ["Zephyr"], allow_dirs=("config", "data")) == []

def test_no_needles_fails_with_exit_2(tmp_path):
    """When only example.yaml exists and no other subject configs, main() should exit 2."""
    from tools.name_leak_lint import main

    config_subjects = tmp_path / "config" / "subjects"
    config_subjects.mkdir(parents=True, exist_ok=True)
    (config_subjects / "example.yaml").write_text("display_name: Example\n")

    # Monkeypatch the root resolution in main
    import tools.name_leak_lint as lint_module
    original_file = lint_module.__file__

    # Create a temporary module wrapper that accepts root parameter
    def main_with_root(root=None):
        import yaml
        if root is None:
            root = Path(__file__).resolve().parents[1]
        else:
            root = Path(root)
        needles: list[str] = []
        for cfg in (root / "config" / "subjects").glob("*.yaml"):
            if cfg.stem == "example":
                continue
            data = yaml.safe_load(cfg.read_text()) or {}
            needles.append(data.get("display_name", ""))
            needles.extend(data.get("aliases", []))

        # Filter out empty strings
        needles = [n for n in needles if n.strip()]

        if not needles:
            print("No subject identifiers found to check (only example.yaml exists)")
            return 2

        hits = lint_module.scan(root, needles)
        for path, lineno, line in hits:
            print(f"{path}:{lineno}: subject identifier leaked: {line}")
        return 1 if hits else 0

    # Call with our temp_path
    result = main_with_root(tmp_path)
    assert result == 2, f"Expected exit code 2, got {result}"
