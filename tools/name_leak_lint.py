"""Fail the build if subject identifiers appear outside config/ and data/ (spec §13.1)."""
from __future__ import annotations
from pathlib import Path

SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules", ".superpowers", ".token-optimizer"}
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".json", ".txt", ".sh"}

def scan(root: Path, needles: list[str],
         allow_dirs: tuple[str, ...] = ("config", "data")) -> list[tuple[Path, int, str]]:
    """Scan for subject identifiers outside allowed directories.

    Uses case-insensitive substring matching with no word boundaries, so short aliases
    may match inside ordinary words (e.g., "Jo" in "Major"). This errs toward false
    positives, which is correct for a leak-prevention lint: catching a real leak is
    more important than avoiding false alarms.

    Args:
        root: Root directory to scan
        needles: Subject identifiers to search for (display_name and aliases)
        allow_dirs: Directory prefixes where identifiers are allowed (default: config, data)

    Returns:
        List of (path, line_number, line_text) tuples for each match outside allowed dirs.
    """
    root = Path(root)
    lowered = [n.lower() for n in needles if n.strip()]
    hits: list[tuple[Path, int, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        rel = path.relative_to(root)
        if set(rel.parts) & SKIP_DIRS or (rel.parts and rel.parts[0] in allow_dirs):
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            low = line.lower()
            if any(n in low for n in lowered):
                hits.append((path, i, line.strip()))
    return hits

def main(root: Path | None = None) -> int:
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

    # Filter out empty strings and check if any needles remain
    needles = [n for n in needles if n.strip()]

    if not needles:
        print("No subject identifiers found to check (only example.yaml exists)")
        return 2

    hits = scan(root, needles)
    for path, lineno, line in hits:
        print(f"{path}:{lineno}: subject identifier leaked: {line}")
    return 1 if hits else 0

if __name__ == "__main__":
    raise SystemExit(main())
