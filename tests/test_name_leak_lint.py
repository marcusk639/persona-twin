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

    result = main(tmp_path)
    assert result == 2, f"Expected exit code 2 when no needles, got {result}"


def test_main_exits_1_when_leak_found(tmp_path):
    """main() should exit 1 when a subject needle is found in source code."""
    from tools.name_leak_lint import main

    config_subjects = tmp_path / "config" / "subjects"
    config_subjects.mkdir(parents=True, exist_ok=True)
    (config_subjects / "zephyr.yaml").write_text("display_name: Zephyr\naliases:\n  - Z\n")

    src = tmp_path / "src" / "app.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("# Hello Zephyr\n")

    result = main(tmp_path)
    assert result == 1, f"Expected exit code 1 when leak found, got {result}"


def test_main_exits_0_when_clean(tmp_path):
    """main() should exit 0 when no leaks are found."""
    from tools.name_leak_lint import main

    config_subjects = tmp_path / "config" / "subjects"
    config_subjects.mkdir(parents=True, exist_ok=True)
    (config_subjects / "zephyr.yaml").write_text("display_name: Zephyr\n")

    src = tmp_path / "src" / "app.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("# Just some code\nx = 1\n")

    result = main(tmp_path)
    assert result == 0, f"Expected exit code 0 when clean, got {result}"
