"""Convert a plain-text probe list into facts.json.

Writing 200 probes directly as JSON is tedious enough that it discourages
writing them at all. This takes a line-oriented format instead:

    A | Which firm did I work at before TWK? | Baker Tilly
    U | What was my grandmother's maiden name?
    U | Which of my brothers came to the 2019 conference? | | no brothers

Fields are pipe-separated: kind, question, expected, notes.
  A = answerable   -> expected is REQUIRED
  U = unanswerable -> expected must be empty; anything after is notes
Blank lines and lines starting with # are ignored, so you can group and
annotate freely while drafting.

Usage:
    uv run python tools/probe_intake.py probes.txt data/subjects/subject-01/probes/facts.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


class IntakeError(Exception):
    """A line could not be parsed. Reported with its line number, and nothing
    is written -- a partial probe file is worse than none, because the gate
    would report a composition that does not match what was intended."""


def parse(text: str) -> list[dict]:
    probes: list[dict] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        kind = parts[0].upper()
        if kind not in {"A", "U"}:
            raise IntakeError(f"line {lineno}: kind must be A or U, got {parts[0]!r}")
        if len(parts) < 2 or not parts[1]:
            raise IntakeError(f"line {lineno}: question is empty")
        question = parts[1]
        expected = parts[2] if len(parts) > 2 else ""
        notes = parts[3] if len(parts) > 3 else ""
        if kind == "A" and not expected:
            raise IntakeError(
                f"line {lineno}: answerable probe needs an expected answer")
        if kind == "U" and expected:
            raise IntakeError(
                f"line {lineno}: unanswerable probe must have no expected answer "
                f"(got {expected!r}); put commentary in the 4th field")
        probes.append({
            "probe_id": f"f{len(probes) + 1:03d}",
            "question": question,
            "expected": expected,
            "answerable": kind == "A",
            "notes": notes,
        })
    return probes


def report(probes: list[dict]) -> str:
    n = len(probes)
    unans = sum(1 for p in probes if not p["answerable"])
    share = unans / n if n else 0.0
    lines = [f"{n} probes: {n - unans} answerable, {unans} unanswerable "
             f"(share {share:.2f})"]
    if n < 200:
        lines.append(f"  need {200 - n} more to reach the S5 minimum of 200")
    if n and not 0.4 <= share <= 0.6:
        need = "unanswerable" if share < 0.4 else "answerable"
        lines.append(f"  share is outside [0.40, 0.60] -- write more {need} probes")
    if n >= 200 and 0.4 <= share <= 0.6:
        lines.append("  composition satisfies the S5 gate")
    return "\n".join(lines)


def main(src: str, dest: str) -> int:
    try:
        probes = parse(Path(src).read_text(encoding="utf-8"))
    except IntakeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    Path(dest).write_text(json.dumps(probes, indent=2), encoding="utf-8")
    print(f"wrote {dest}")
    print(report(probes))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
