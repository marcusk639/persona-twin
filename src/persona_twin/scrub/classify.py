from __future__ import annotations
import re
import unicodedata
from persona_twin.schema import Turn

# Sources whose every turn is client-derived data, regardless of content.
# Currently the only Karbon export ingested is practice-management data
# (client tax files, engagement notes) — never the subject's own writing.
CONFIDENTIAL_SOURCES = {"karbon"}


def _is_confidential_source(source: str) -> bool:
    """Case- and naming-variant-tolerant match against CONFIDENTIAL_SOURCES.

    An exact, case-sensitive `in` check treats "Karbon", "KARBON" and
    "karbon_export" as ordinary open sources -- and this is the rule that
    protects records carrying NO content markers at all, so getting it
    wrong fails in the worst possible direction. No Karbon connector exists
    yet, which is exactly what makes it a trap: whoever writes one has no
    established casing or naming to match against. This codebase's own
    convention for connector names is a compound "<source>_<subtype>" (see
    git_repos, claude_ai, claude_code), so "karbon_export" is a plausible
    real name, not a hypothetical one -- matched here as a prefix, not a
    substring, so an unrelated source that merely contains "karbon"
    somewhere in its name would not.
    """
    normalized = source.strip().lower()
    return any(normalized == s or normalized.startswith(f"{s}_")
               for s in CONFIDENTIAL_SOURCES)


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

# The qualifier family that can follow "routing" / "account" / "acct" / "a/c":
# a bare "#", the full word, or the common written abbreviations "no[.]",
# "num[.]", "nbr[.]" -- a review found the phrase marker caught "account
# number" and "account #" but missed "acct no. 12345678", "account no
# 12345678" and "acct num 12345678", each a different spelling of the same
# qualifier, not a different marker. Sharing this fragment between the
# routing and account patterns means a future addition to the family (or a
# fix to one) can't be made to only one of them by accident.
_QUALIFIER = r"(?:number|no\.?|num\.?|nbr\.?|#)"

# No \b bookends on any of the four digit-shape patterns below: a boundary
# requires a transition between a "word" and a "non-word" character, and a
# letter or underscore is ALSO a word character, so \b\d{9,}\b cannot match
# "ID123456789", "123456789Z" or "acct_123456789" at all, \b\d{3}[-.
# ]\d{2}[-. ]\d{4}\b equally cannot match "ref123-45-6789" or
# "id_123-45-6789", and \b\d{4}[-. ]\d{4}[-. ]\d{4}\b equally cannot match
# "acct4093-8172-6354" or "x4093-8172-6354" -- the run is simply never
# bounded on that side, for a letter-glued grouped card number exactly as
# much as for a letter-glued SSN or a bare digit run. This was found and
# fixed one pattern at a time across three review rounds in this one
# module; each round's fix was correct on its own instance and each round
# left a sibling pattern with the identical defect unswept, which is why
# every remaining \b-bookended digit-shape pattern is fixed together here.
# Each pattern still finds its shape wherever it appears without the
# bookends (a specific-shape pattern doesn't need them for the same reason
# \d{9,} doesn't: the separator characters and group lengths already fix
# where the match starts and ends).
_MARKERS: list[re.Pattern[str]] = [
    re.compile(rf"\d{{3}}{_SEP}\d{{2}}{_SEP}\d{{4}}"),      # SSN, separated
    re.compile(rf"\d{{2}}{_SEP}\d{{7}}"),                  # EIN, separated
    re.compile(r"\d{9,}"),                                    # bare digit run, unbounded above, unbounded adjacency
    re.compile(rf"\d{{4}}{_SEP}\d{{4}}{_SEP}\d{{4}}(?:{_SEP}\d{{1,4}})?"),  # grouped account/card
    re.compile(rf"\brouting\s*{_QUALIFIER}", re.I),
    # "account" alone missed the common CPA-office abbreviations "acct" and
    # "a/c" (e.g. "acct #12345678"). Deliberately NOT including bare "act":
    # this is a tax practice's data, where "the Act" (Tax Cuts and Jobs Act,
    # CARES Act, ...) and "Act #<form/section>" are ordinary domain
    # vocabulary, not account references -- adding it would trade a content
    # false-negative for a source-of-truth false-positive on real
    # discussion of legislation, not close the hole it's meant to close.
    re.compile(rf"\b(?:account|acct\.?|a/c)\s*{_QUALIFIER}", re.I),
]

# Unicode characters that behave like an ASCII separator but are not one,
# handled as two FAMILIES rather than as a list of the specific codepoints a
# review happened to demonstrate -- a prior version of this table named only
# U+00A0 and U+2011 and was found, the next time it was reviewed, to have
# left every other member of both families unswept:
#
# * Cf (Format): zero-width joiners and soft hyphen. These render as
#   invisible, so the honest normalization is to remove them, not to
#   substitute a visible character -- "123​45​6789" is meant to be
#   read as one contiguous number with an invisible seam, not as
#   hyphen-separated.
# * Pd (Dash Punctuation): every dash variant -- en dash, em dash, figure
#   dash, horizontal bar, hyphen, non-breaking hyphen, and the fullwidth and
#   small compatibility forms -- folds to ASCII '-'. NFKC alone does not
#   reach this family: it maps U+2011 (non-breaking hyphen) to U+2010
#   (hyphen), not to U+002D, so the target of that mapping needs handling in
#   its own right, which the Pd sweep below gives it for free.
# * MINUS SIGN (U+2212) behaves like a dash but Unicode categorizes it as Sm
#   (Math Symbol), a category far too broad to sweep wholesale -- it also
#   contains '+', '=', '<', '>', which must never fold to '-'. Named
#   individually for that reason; this is the one deliberate exception to
#   "handle the category, not the character."
_DASH_LOOKALIKE = "−"  # MINUS SIGN — category Sm, not Pd; see above


def _for_matching(text: str) -> str:
    """A normalized VIEW of `text` used only to decide whether a marker
    fires -- never stored, never returned, and the caller's Turn is
    unmodified (it is frozen, so there is no way to mutate it in place).

    NFKC runs first and remains load-bearing on its own: it is what maps a
    non-breaking space and every other Unicode space variant to ASCII
    space, and U+2011 to U+2010, before the category sweep below ever sees
    them. The sweep after it folds the dash family to '-' and drops the
    invisible-format family entirely, neither of which NFKC reaches on its
    own (see the module-level comment above the two categories).
    """
    normalized = unicodedata.normalize("NFKC", text)
    out = []
    for ch in normalized:
        category = unicodedata.category(ch)
        if category == "Cf":
            continue
        if category == "Pd" or ch == _DASH_LOOKALIKE:
            out.append("-")
            continue
        out.append(ch)
    return "".join(out)


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
    if _is_confidential_source(turn.source):
        return "confidential"
    text = _for_matching(turn.text)
    if any(p.search(text) for p in _MARKERS):
        return "confidential"
    return "open"


def is_exportable(turn: Turn) -> bool:
    return classify(turn) == "open"
