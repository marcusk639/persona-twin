from __future__ import annotations
import re
from persona_twin.schema import Turn

# Sources whose every turn is client-derived data, regardless of content.
# Currently the only Karbon export ingested is practice-management data
# (client tax files, engagement notes) — never the subject's own writing.
CONFIDENTIAL_SOURCES = {"karbon"}

_MARKERS: list[re.Pattern[str]] = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),            # SSN
    re.compile(r"\b\d{2}-\d{7}\b"),                  # EIN
    re.compile(r"\b\d{9,17}\b"),                     # bank/account numbers
    re.compile(r"\brouting\s*(number|#)", re.I),
    re.compile(r"\baccount\s*(number|#)", re.I),
]


def classify(turn: Turn) -> str:
    """Client data is confidential; the subject's domain knowledge is not (spec §1, C1).

    Two independent routes to "confidential", either sufficient on its own:
    source (any turn from CONFIDENTIAL_SOURCES, regardless of content) and
    content markers (SSN/EIN/account-number patterns, regardless of source).
    The account-number marker (\\d{9,17}) is intentionally broad — see
    task-16-report.md for what it does to ordinary numbers (years, phone
    numbers, timestamps) and why that breadth is accepted rather than tuned
    away: a false "confidential" only costs a little corpus fidelity, while a
    false "open" leaks a third party's regulated financial data into a
    corpus that may reach a hosted model. The two costs are not symmetric,
    so this classifier errs toward "confidential" wherever a pattern's
    breadth is a judgement call.
    """
    if turn.source in CONFIDENTIAL_SOURCES:
        return "confidential"
    if any(p.search(turn.text) for p in _MARKERS):
        return "confidential"
    return "open"


def is_exportable(turn: Turn) -> bool:
    return classify(turn) == "open"
