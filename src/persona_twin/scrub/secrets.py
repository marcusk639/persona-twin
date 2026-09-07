from __future__ import annotations

import math
import re
from collections import Counter


class ScrubError(Exception):
    """Raised when scrubbing cannot complete. Never return partial output (spec C4).

    Never include the matched span in the message — an exception that echoes
    the secret defeats the point of this module.
    """


REDACTION = "[REDACTED]"

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}")),
    ("openai_key", re.compile(r"sk-[A-Za-z0-9]{32,}")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("google_key", re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("bearer", re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]{20,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9\-_]{10,}\.[A-Za-z0-9\-_]{10,}\.[A-Za-z0-9\-_]{10,}")),
]

# `-` and `_` were dropped and the minimum length raised 24 -> 32. Kebab-case
# identifiers and file paths are genuinely high-entropy, so no entropy
# threshold alone can separate them from real secrets (measured: at
# threshold 5.2, 100% of surviving flags were still path-like). Narrowing the
# character class removes most path/identifier tokens outright, since real
# paths and identifiers are usually hyphen- or underscore-joined. `/` is kept
# because dropping it fragments base64 blobs into thousands of short,
# non-matching pieces (measured on real data).
_CANDIDATE = re.compile(r"[A-Za-z0-9+/=]{32,}")

# Calibrated against 2,683 real transcript envelopes (see task-15-report.md
# for the full before/after numbers). At the brief's original default (3.9)
# the entropy route flagged 61.5% of messages; 4.5 combined with the
# narrowed _CANDIDATE pattern and _looks_like_path exclusion below brings
# that to 6.6%, with zero remaining path-like false positives. Re-check this
# threshold if a materially different data source is added to the corpus.
_ENTROPY_THRESHOLD = 4.5


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _looks_like_path(token: str) -> bool:
    """True if every '/'-separated segment of token is lowercase-and-path-shaped.

    Filesystem paths are genuinely high-entropy strings and are abundant in
    developer transcripts (file paths, module names), which is why raising
    the entropy threshold alone can't separate them from real secrets.
    Base64 secrets are mixed-case, so this exclusion never matches them even
    though '/' stays in the candidate character class for their sake.
    """
    segments = token.split("/")
    if len(segments) < 2:
        return False
    return all(re.match(r"^[a-z0-9._-]+$", seg) for seg in segments)


def scan(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, label) spans. Deny-by-default on high entropy (spec C2)."""
    spans: list[tuple[int, int, str]] = []
    for label, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end(), label))
    for m in _CANDIDATE.finditer(text):
        token = m.group(0)
        if _looks_like_path(token):
            continue
        if shannon_entropy(token) >= _ENTROPY_THRESHOLD:
            spans.append((m.start(), m.end(), "high_entropy"))
    return sorted(set(spans))


def redact(text: str) -> str:
    """Replace every detected secret span with REDACTION.

    Fails closed (spec C4): re-scans its own output after substitution and
    raises ScrubError if anything still matches, rather than ever returning
    partially-scrubbed text.
    """
    if not isinstance(text, str):
        raise ScrubError(f"expected str, got {type(text).__name__}")
    spans = scan(text)
    if not spans:
        return text
    merged: list[tuple[int, int, str]] = []
    for start, end, label in spans:
        if merged and start <= merged[-1][1]:
            prev = merged[-1]
            merged[-1] = (prev[0], max(prev[1], end), prev[2])
        else:
            merged.append((start, end, label))
    out, cursor = [], 0
    for start, end, _ in merged:
        out.append(text[cursor:start])
        out.append(REDACTION)
        cursor = end
    out.append(text[cursor:])
    result = "".join(out)
    if scan(result):
        raise ScrubError("residual secrets after redaction; refusing partial output")
    return result
