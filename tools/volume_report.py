"""Stage-2 gate: measured subject-authored volume and projected mining cost (§4.4)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_twin.paths import SubjectPaths
from persona_twin.corpus.volume import measure

def main(subject_id: str, version: str) -> int:
    r = measure(SubjectPaths(subject_id, Path.cwd()), version)
    print(f"corpus version      : {version}")
    print(f"total turns         : {r.total_turns:,}")
    print(f"subject-authored    : {r.subject_turns:,}")
    print(f"subject characters  : {r.subject_chars:,}  ({r.subject_chars/1e6:.1f} MB)")
    print(f"estimated tokens    : {r.est_tokens:,}")
    print(f"projected one pass  : ${r.projected_usd:,.2f}")
    print("by source:")
    for source, n in sorted(r.by_source.items(), key=lambda kv: -kv[1]):
        print(f"  {source:<14} {n:,}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
