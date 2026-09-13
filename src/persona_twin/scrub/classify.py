from __future__ import annotations
import re
import unicodedata
from persona_twin.schema import Turn

# Sources whose every turn is client-derived data, regardless of content.
# Currently the only Karbon export ingested is practice-management data
# (client tax files, engagement notes) — never the subject's own writing.
CONFIDENTIAL_SOURCES = {"karbon"}


class ClassifyError(Exception):
    """Raised when classification cannot proceed on malformed input.

    Mirrors scrub/secrets.py's ScrubError: a classifier that crashes loudly
    on input it cannot interpret is safer than one that silently falls
    through to "open" because an unexpected type made every marker's
    .search() a no-op.
    """


# Separators: people write the same identifier several ways. The canonical
# hyphenated forms and bare digit runs were matched from the start; the
# space- and dot-separated variants were added after a review found them
# missed, at a measured cost of 3 extra turns in 81,438 (0.55% -> 0.55%).
#
# The trailing \s{0,4} tolerates a hard line-wrap landing right after the
# separator (e.g. a chat client wrapping "123-45-" onto the next line before
# "6789") -- a real separator character is still required; this only asks it
# to not also demand that no incidental whitespace follow it. Bounded at 4
# characters (covers "\r\n" plus a couple of stray indent characters from a
# paste) rather than unbounded, so this can't bridge two genuinely unrelated
# numbers separated by a blank line or more.
_SEP = r"[-. ]\s{0,4}"

_MARKERS: list[re.Pattern[str]] = [
    re.compile(rf"\b\d{{3}}{_SEP}\d{{2}}{_SEP}\d{{4}}\b"),      # SSN, separated
    re.compile(rf"\b\d{{2}}{_SEP}\d{{7}}\b"),                  # EIN, separated
    re.compile(r"\b\d{9,}\b"),                               # bare digit run, unbounded above
    re.compile(rf"\b\d{{4}}{_SEP}\d{{4}}{_SEP}\d{{4}}(?:{_SEP}\d{{1,4}})?\b"),  # grouped account/card
    re.compile(r"\brouting\s*(number|#)", re.I),
    re.compile(r"\b(?:account|acct\.?|a/c)\s*(number|#)", re.I),
    # "account" alone missed the common CPA-office abbreviations "acct" and
    # "a/c" (e.g. "acct #12345678"). Deliberately NOT including bare "act":
    # this is a tax practice's data, where "the Act" (Tax Cuts and Jobs Act,
    # CARES Act, ...) and "Act #<form/section>" are ordinary domain
    # vocabulary, not account references -- adding it would trade a content
    # false-negative for a source-of-truth false-positive on real
    # discussion of legislation, not close the hole it's meant to close.
]

# Unicode look-alikes for the ASCII separators in _SEP. A non-breaking space
# or non-breaking hyphen from a pasted PDF/Word export renders identically to
# its ASCII twin but would otherwise hide an identifier from every marker
# above, since _SEP's character class is ASCII-only. This is intentionally a
# small, explicit alias table rather than a broad "fold everything unicode"
# pass -- NFKC alone won't do this (U+2011 has no compatibility decomposition
# to U+002D), so both need naming.
_SEPARATOR_ALIASES = str.maketrans({
    " ": " ",   # non-breaking space
    "‑": "-",   # non-breaking hyphen
})


def _for_matching(text: str) -> str:
    """A normalized VIEW of `text` used only to decide whether a marker
    fires -- never stored, never returned, and the caller's Turn is
    unmodified (it is frozen, so there is no way to mutate it in place).
    NFKC on top of the alias table folds broader compatibility variants
    (e.g. ideographic space, fullwidth forms) without enumerating every one
    of them by hand.
    """
    return unicodedata.normalize("NFKC", text.translate(_SEPARATOR_ALIASES))


def classify(turn: Turn) -> str:
    """Client data is confidential; the subject's domain knowledge is not (spec §1, C1).

    Two independent routes to "confidential", either sufficient on its own:
    source (any turn from CONFIDENTIAL_SOURCES, regardless of content) and
    content markers (SSN/EIN/account-number patterns, regardless of source).
    The account-number marker (\\d{9,}) is intentionally broad — see
    task-16-report.md for what it does to ordinary numbers (years, phone
    numbers, timestamps) and why that breadth is accepted rather than tuned
    away: a false "confidential" only costs a little corpus fidelity, while a
    false "open" leaks a third party's regulated financial data into a
    corpus that may reach a hosted model. The two costs are not symmetric,
    so this classifier errs toward "confidential" wherever a pattern's
    breadth is a judgement call.

    This remains an enumerated set of known shapes and phrases, not a
    deny-by-default classifier like secrets.py's entropy fallback: an
    identifier of a shape none of these markers anticipates is still missed.
    That residual risk is accepted and tracked, not silently designed away.
    """
    if not isinstance(turn.text, str):
        raise ClassifyError(f"expected str for turn.text, got {type(turn.text).__name__}")
    if turn.source in CONFIDENTIAL_SOURCES:
        return "confidential"
    text = _for_matching(turn.text)
    if any(p.search(text) for p in _MARKERS):
        return "confidential"
    return "open"


def is_exportable(turn: Turn) -> bool:
    return classify(turn) == "open"
