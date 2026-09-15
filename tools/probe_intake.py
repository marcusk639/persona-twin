"""Convert a plain-text probe list into facts.json.

Writing 200 probes directly as JSON is tedious enough that it discourages
writing them at all. This takes a line-oriented format instead:

    A | What framework is Regroup built with? | React Native
    U | How many paying users does Regroup have?
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
import re
import sys
from pathlib import Path


class IntakeError(Exception):
    """A line could not be parsed. Reported with its line number, and nothing
    is written -- a partial probe file is worse than none, because the gate
    would report a composition that does not match what was intended."""


# A generated draft ships with this where an answer belongs. Converting it
# verbatim would produce a probe whose expected answer is literally "???" --
# counted toward S5's n>=200 while measuring nothing, and scored as a miss
# against every twin forever.
PLACEHOLDER = "???"


def parse_with_skips(text: str) -> tuple[list[dict], int]:
    """parse(), but unfilled placeholder rows are skipped and counted.

    Returned rather than logged: a draft that silently converted 40 of its 130
    rows would report a composition the author never intended.
    """
    kept, skipped = [], 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [p.strip() for p in stripped.split("|")]
        if parts[0].strip().upper() == "A" and len(parts) > 2 and parts[2] == PLACEHOLDER:
            skipped += 1
            continue
        kept.extend(parse(line))
    return kept, skipped


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


def _tokens(question: str) -> set[str]:
    return set(re.sub(r"[^a-z ]", "", question.lower()).split())


def _is_near_duplicate(question: str, existing: list[dict], threshold: float = 0.8) -> bool:
    """Token-set overlap, not substring.

    Substring matching let "How many users does Regroup have?" through against
    an existing "How many PAYING users does Regroup have?" -- neither contains
    the other, because the extra word sits in the middle.
    """
    t = _tokens(question)
    if not t:
        return False
    for row in existing:
        e = _tokens(row.get("question", ""))
        if e and len(t & e) / len(t | e) >= threshold:
            return True
    return False


def main(src: str, dest: str, merge: bool = False) -> int:
    try:
        probes, unfilled = parse_with_skips(Path(src).read_text(encoding="utf-8"))
    except IntakeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    existing: list[dict] = []
    if merge and Path(dest).exists():
        existing = json.loads(Path(dest).read_text(encoding="utf-8"))

    kept = list(existing)
    next_n = max((int(r["probe_id"].split("-")[-1]) for r in existing
                  if r["probe_id"].split("-")[-1].isdigit()), default=0)
    added = duplicates = 0
    for probe in probes:
        if _is_near_duplicate(probe["question"], kept):
            duplicates += 1
            continue
        next_n += 1
        added += 1
        kept.append({**probe, "probe_id": f"fact-{next_n:03d}"} if merge else probe)

    Path(dest).write_text(json.dumps(kept, indent=2) + "\n", encoding="utf-8")
    verb = "merged into" if merge else "wrote"
    print(f"{verb} {dest}: {added} added, {duplicates} duplicate(s) skipped, "
          f"{unfilled} unfilled (???) skipped")
    print(report(kept))
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--merge"]
    if len(args) != 2:
        print(__doc__)
        raise SystemExit(1)
    raise SystemExit(main(args[0], args[1], merge="--merge" in sys.argv))
