"""Stage-3 gate: print the held-out split and every criterion's status."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from persona_twin.eval.report import (pending_results, render, s2_self_distance,  # noqa: E402
                                      s5_probe_composition, s6_probe_composition,
                                      split_summary)
from persona_twin.paths import SubjectPaths  # noqa: E402


def main(subject_id: str, version: str) -> int:
    paths = SubjectPaths(subject_id, Path.cwd())
    now = datetime.now(timezone.utc)
    s = split_summary(paths, version, now)
    print(f"corpus version : {version}")
    print(f"total turns    : {s['total']:,}")
    print(f"  train        : {s['train']:,}")
    print(f"  held out     : {s['heldout']:,}")
    print(f"  quarantined  : {s['quarantined']:,}")
    print()
    results = [s2_self_distance(paths, version, now),
              s5_probe_composition(paths),
              s6_probe_composition(paths)] + pending_results()
    print(render(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
