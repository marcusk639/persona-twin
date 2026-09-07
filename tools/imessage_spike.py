"""Stage-0 spike (spec §5.6): report iMessage readability and decode recovery.

Prints aggregate counts only -- never message content.
"""
from __future__ import annotations
import sqlite3, sys
from dataclasses import dataclass
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_twin.imessage.typedstream import decode_attributed_body

# Default to vault snapshot relative to repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "data" / "subjects" / "subject-01" / "vault" / "imessage" / "chat.db"

@dataclass
class ProbeReport:
    total: int
    from_me: int
    null_text: int
    recovered: int
    blobs_with_content: int = 0

    @property
    def recovery_rate(self) -> float:
        return 1.0 if self.blobs_with_content == 0 else self.recovered / self.blobs_with_content

def probe(db_path: Path = DEFAULT_DB) -> ProbeReport:
    uri = f"file:{db_path}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        total, from_me, null_text = con.execute(
            "select count(*), sum(is_from_me=1), sum(text is null or text='') from message"
        ).fetchone()

        # Count blobs that can be recovered and how many actually decode
        blobs_with_content = 0
        recovered = 0
        for (blob,) in con.execute(
            "select attributedBody from message where (text is null or text='') "
            "and attributedBody is not null"
        ):
            blobs_with_content += 1
            if decode_attributed_body(blob):
                recovered += 1
    finally:
        con.close()
    return ProbeReport(total or 0, from_me or 0, null_text or 0, recovered, blobs_with_content)

def main() -> int:
    # Parse optional command-line argument
    db_path = DEFAULT_DB
    if len(sys.argv) > 1:
        db_path = Path(sys.argv[1])

    # Check if database exists
    if not db_path.exists():
        if db_path == DEFAULT_DB:
            # Default path missing: print clear message
            print(f"ERROR: Default vault snapshot not found at {db_path}")
            print(f"To use the live iMessage database, copy it first:")
            print(f"  cp ~/Library/Messages/chat.db {db_path}")
            print(f"Or pass a different path:")
            print(f"  uv run python tools/imessage_spike.py /path/to/chat.db")
            return 2
        else:
            # User-specified path missing
            print(f"ERROR: Database not found at {db_path}")
            return 2

    try:
        r = probe(db_path)
    except sqlite3.OperationalError as exc:
        print(f"FAIL: cannot open chat.db ({exc}). Full Disk Access is not granted.")
        return 2
    print(f"total messages     : {r.total}")
    print(f"authored by subject: {r.from_me}")
    print(f"null/empty text    : {r.null_text}")
    print(f"recovered by decode: {r.recovered}")
    print(f"recovery rate      : {r.recovery_rate:.1%}")
    ok = r.recovery_rate >= 0.95
    print("GATE: PASS" if ok else "GATE: FAIL (<95%) -- change extraction approach before Task 10")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
