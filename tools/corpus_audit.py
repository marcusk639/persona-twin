"""Independent post-build audit of a stored corpus version (stage-2 gate).

Added by a controller ruling: the plan's own gate command only printed
BuildReport's counts, which report what the scrubber *caught* during the
build -- not what remains in the corpus afterward. A scrubber that silently
missed something produces a report identical to a clean run. This tool
re-derives the "zero secrets, zero confidential records" claim
independently: it reads the corpus back out of the store and re-runs the
scrubber and classifier against the stored text, rather than trusting the
builder's self-report.

Checks, each independent and each sufficient on its own to fail the audit:
  1. secrets.scan() returns zero spans on every stored turn's text.
  2. classify.classify() returns "open" for every stored turn.
  3. no stored turn with is_subject == False has an author_id outside the
     pseudonym shape "P-" + 12 hex characters (identity.Pseudonymizer's
     output format) -- anything else on a non-subject turn means an
     un-pseudonymized third-party identifier reached the corpus.

HONEST LIMIT: this audit re-scans stored text with the *same* scanner that
performed the redaction at build time. That proves no code path bypassed
the scrubber -- exactly the fail-closed guarantee this design claims -- but
it cannot prove the scanner's detection rules are complete. A secret shape
or confidentiality marker the scanner has never been taught will pass both
the original build and this audit identically. Rule completeness is a
separate concern, addressed by calibration against real data (see the
tuning notes in scrub/secrets.py and scrub/classify.py), not by this tool.

Never prints a matched secret span or a raw third-party identifier: findings
are reported as turn positions (source/source_id), match positions, and
labels, with any identifier truncated before printing.
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from persona_twin.paths import SubjectPaths
from persona_twin.corpus.store import CorpusStore
from persona_twin.scrub.secrets import scan
from persona_twin.scrub.classify import classify

# Must match identity.Pseudonymizer.pseudonym's output format exactly:
# "P-" followed by 12 lowercase hex characters (an HMAC-SHA256 prefix).
PSEUDONYM = re.compile(r"^P-[0-9a-f]{12}$")

MAX_EXAMPLES = 5


def _truncate_id(identifier: str, keep: int = 4) -> str:
    """Truncate a raw identifier so a finding never leaks it in full."""
    if len(identifier) <= keep:
        return "*" * len(identifier)
    return f"{identifier[:keep]}…({len(identifier)} chars)"


def _report(label: str, findings: list[str]) -> bool:
    print(f"{label:<24}: {len(findings):,}  (expect 0)")
    for line in findings[:MAX_EXAMPLES]:
        print(f"  - {line}")
    if len(findings) > MAX_EXAMPLES:
        print(f"  ... and {len(findings) - MAX_EXAMPLES} more")
    return not findings


def audit(paths: SubjectPaths, version: str) -> int:
    turns = CorpusStore(paths).read(version)

    secret_findings: list[str] = []
    confidential_findings: list[str] = []
    identifier_findings: list[str] = []

    for t in turns:
        for start, end, secret_label in scan(t.text):
            secret_findings.append(
                f"{t.source}:{t.source_id} pos={start}-{end} label={secret_label}")
        if classify(t) != "open":
            confidential_findings.append(f"{t.source}:{t.source_id}")
        if not t.is_subject and not PSEUDONYM.match(t.author_id):
            identifier_findings.append(
                f"{t.source}:{t.source_id} author={_truncate_id(t.author_id)}")

    print(f"corpus version          : {version}")
    print(f"turns audited           : {len(turns):,}")

    ok = True
    ok &= _report("secret spans found", secret_findings)
    ok &= _report("confidential turns", confidential_findings)
    ok &= _report("un-pseudonymized ids", identifier_findings)

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="corpus_audit.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Independently verify a built corpus has zero secrets, zero "
            "confidential records, and no un-pseudonymized third-party "
            "identifiers, by re-reading it from the store and re-running "
            "the scrubber/classifier against the stored text -- rather "
            "than trusting BuildReport's self-reported counts.\n\n"
            "LIMIT: re-scanning with the same scanner used at build time "
            "proves no code path bypassed the scrubber (the fail-closed "
            "design's core claim), but cannot prove the scanner's "
            "detection rules are complete. Rule completeness is a "
            "separate concern, addressed by calibration, not by this "
            "audit."))
    parser.add_argument("subject_id")
    parser.add_argument("version")
    args = parser.parse_args(argv)
    paths = SubjectPaths(args.subject_id, Path.cwd())
    return audit(paths, args.version)


if __name__ == "__main__":
    raise SystemExit(main())
