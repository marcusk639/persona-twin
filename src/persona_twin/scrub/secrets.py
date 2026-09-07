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

_CANDIDATE = re.compile(r"[A-Za-z0-9+/=_\-]{24,}")
# Provisional default from the task brief. NOT calibrated against real data —
# Step 5 of the brief (tuning this against real transcripts) was deliberately
# skipped per scope; a separate pass must tune this before it is trusted.
_ENTROPY_THRESHOLD = 3.9


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def scan(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, label) spans. Deny-by-default on high entropy (spec C2)."""
    spans: list[tuple[int, int, str]] = []
    for label, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end(), label))
    for m in _CANDIDATE.finditer(text):
        token = m.group(0)
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
