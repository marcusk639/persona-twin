"""Generate an S5 authoring worksheet from the corpus (spec §7, S5).

This tool does NOT write probes. It surfaces material the subject can turn into
probes, because a model that authors fact probes makes S5 circular: the
answerable half would test retrieval of the same text the twin trains on, and
the unanswerable half would be calibrated to what a *model* finds plausibly
absent rather than what is genuinely unrecorded about a person.

Two constraints are enforced in code rather than left to the caller:

  * **Train split only.** Mining held-out turns for probe material would leak
    evaluation data into evaluation design -- the twin would then be scored on
    questions derived from the very turns reserved to score it.
  * **Subject-authored turns only.** A fact probe is about the subject; text
    written by someone else in their threads is not evidence about them.

Output goes under data/ (gitignored) because it is derived subject content.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from persona_twin.corpus.store import CorpusStore
from persona_twin.eval.split import split_corpus
from persona_twin.paths import SubjectPaths
from persona_twin.schema import Turn

# Each category names a kind of checkable claim the subject might turn into an
# answerable probe. The patterns are recall-oriented and deliberately loose --
# a false positive costs one skimmed line, a false negative hides material.
CATEGORIES: dict[str, str] = {
    "preferences": r"\b(i (prefer|like|love|hate|avoid|always|never|usually|tend to)|my (approach|rule|preference)|i'?d rather)\b",
    "opinions": r"\b(i think|i believe|in my (view|experience)|the (problem|issue) with)\b",
    "decisions": r"\b(we (decided|went with|chose|settled on)|i (decided|went with|chose|switched to)|going with)\b",
    "history": r"\b(back (when|then)|used to|years ago|when i (worked|was at)|before (that|i))\b",
}

# Short turns rarely carry a checkable claim ("yeah", "ok sounds good"). The
# corpus median is ~42 chars, so this keeps the substantive tail.
# Material that must not become probe content. Probe files are NOT gated by the
# confidentiality classifier and are handed to a scoring harness, so a probe
# built on any of this would put a third party's medical, legal, or intimate
# situation into an evaluation artifact. Deliberately recall-oriented: an
# over-broad pattern costs one probe candidate, an under-broad one spends
# somebody else's privacy. Checkable facts live in work and technical material
# anyway, so the expected loss is low.
_SENSITIVE = re.compile(
    # NOTE: re.X strips literal spaces, so every multi-word term uses \s+.
    # Organised by FAMILY, not by remembered instance: a list of specific
    # phrases leaves siblings unswept (attorney without lawyer, "can't go on"
    # without "can't live like this"). Each family is asserted with several
    # phrasings in the tests.
    # substances / recovery
    r"\b(oxy|opioid|suboxone|methadone|benzo|xanax|taper(ing|ed|s)?|withdrawal"
    r"|relaps\w*|detox|dose|dosage|\d+\s*mg)\b"
    # crisis / self-harm
    r"|\b(suicid\w*|kill\s+myself|end(ing)?\s+(my\s+life|it)|early\s+exit"
    r"|self.harm|overdose|hopeless)\b"
    r"|\bcan'?t\s+(go\s+on|live\s+like\s+this|keep\s+going)\b"
    # medical
    r"|\b(diagnos\w*|prescri\w*|therapist|psychiatrist|medication|meds)\b"
    # intimate / family law
    r"|\b(divorce|marriage\s+counsel\w*|custody|affair|invalidated)\b"
    r"|\bi\s+love\s+you\b"
    # legal / employment dispute / third-party claims
    r"|\b(lawyer|attorney|legal\s+counsel|law\s+firm|retained\s+counsel"
    r"|lawsuit|litigation|arbitration|deposition|settlement|severance"
    r"|wrongful\s+termination|retaliation|nda)\b"
    r"|\b(eprs?|va\s+(claim|disability)|disability\s+claim)\b",
    re.I)

_MIN_CHARS = 80
_MAX_CHARS = 600
_PER_CATEGORY = 40


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _dedupe_key(text: str) -> str:
    """Collapse near-identical turns: the subject repeats phrasings across
    threads, and forty variants of one opinion is not forty probes."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower())[:90]


def select(turns: list[Turn], pattern: str, limit: int = _PER_CATEGORY,
           exclude_texts: frozenset[str] = frozenset()) -> list[str]:
    """Longest-first, deduped, length-bounded matches for one category.

    `exclude_texts` carries normalized held-out text. Filtering on the train
    side alone is not enough: the split assigns whole THREADS, but the subject
    broadcasts the same message to several people, so identical content can sit
    on both sides. Measured on v5, exactly one such 450-char iMessage reached
    the worksheet -- selected legitimately from its train copy while an
    identical held-out copy existed in another thread. Excluding by content
    makes the guarantee "no held-out text", not merely "no held-out turn".
    """
    rx = re.compile(pattern, re.I)
    seen: set[str] = set()
    out: list[str] = []
    withheld = 0
    for t in sorted(turns, key=lambda t: -len(t.text)):
        text = _normalize(t.text)
        if not (_MIN_CHARS <= len(text) <= _MAX_CHARS) or not rx.search(text):
            continue
        if text in exclude_texts:
            continue
        if _SENSITIVE.search(text):
            withheld += 1
            continue
        key = _dedupe_key(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    select.last_withheld = withheld          # read by render(); see "no silent caps"
    return out


def topics(turns: list[Turn], limit: int = 40) -> list[tuple[str, int]]:
    """Recurring multi-word capitalized phrases, as topic anchors.

    Deliberately crude: this is a prompt for the subject's memory, not an
    extraction whose precision anything depends on.
    """
    counts: Counter[str] = Counter()
    rx = re.compile(r"\b([A-Z][A-Za-z0-9.+#-]{2,}(?: [A-Z][A-Za-z0-9.+#-]{2,}){0,2})\b")
    for t in turns:
        for m in rx.findall(t.text):
            if m.lower() in {"i", "the", "this", "that", "and"}:
                continue
            counts[m] += 1
    return [(k, v) for k, v in counts.most_common(limit * 3) if v >= 5][:limit]


def render(subject_id: str, version: str, train: list[Turn],
           heldout_n: int, quarantined_n: int,
           exclude_texts: frozenset[str] = frozenset()) -> str:
    lines: list[str] = []
    total_withheld = 0
    w = lines.append
    w(f"# S5 fact-probe worksheet — {subject_id} / {version}")
    w("")
    w("**This is not a probe set.** These are your own words, pulled from the")
    w("training split, to jog material loose. You write the probes; a model cannot")
    w("(see `docs/probe-authoring.md` for why).")
    w("")
    w(f"Drawn from {len(train):,} subject-authored train turns. "
      f"{heldout_n:,} held-out and {quarantined_n:,} quarantined turns were excluded — "
      "held-out deliberately, so probes are not derived from the turns that will score the twin.")
    w("")
    w("## How to use this")
    w("")
    w("- For each line below, ask: *what checkable fact does this imply?* Write that")
    w("  as a question with a short, canonical `expected` answer.")
    w("- Then, for the **unanswerable half**, use each line as a springboard to its")
    w("  neighbours: what closely-related thing is true about you but was **never**")
    w("  written down here? Those are the probes that catch fabrication, and they are")
    w("  the half only you can write — nothing in this file can generate them.")
    w("- Target: **≥200 probes, 40–60% unanswerable.** Roughly 100 of each.")
    w("")
    w("Draft in plain text and run `uv run python tools/probe_intake.py` to convert.")
    w("")
    for name, pattern in CATEGORIES.items():
        picked = select(train, pattern, exclude_texts=exclude_texts)
        withheld = getattr(select, "last_withheld", 0)
        total_withheld += withheld
        w(f"## {name} ({len(picked)})")
        w("")
        if not picked:
            w("_no matches_")
            w("")
            continue
        for text in picked:
            w(f"- {text}")
        w("")
    w(f"_{total_withheld} matching line(s) withheld as sensitive "
      "(health, substance, intimate, or third-party legal material). "
      "Probe files are not classifier-gated, so this material is kept out of "
      "probe authoring entirely._")
    w("")
    w("## recurring topics")
    w("")
    w("Frequency only — use these to check coverage, and to find territory where you")
    w("know things the corpus never recorded.")
    w("")
    for term, n in topics(train):
        w(f"- {term} ({n})")
    w("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("subject_id")
    ap.add_argument("version")
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)

    paths = SubjectPaths(args.subject_id, Path(args.root))
    turns = CorpusStore(paths).read(args.version)
    if not turns:
        print(f"no corpus at version {args.version!r}", file=sys.stderr)
        return 1
    sp = split_corpus(turns, datetime.now(timezone.utc))
    train = [t for t in sp.train if t.is_subject]

    # SubjectPaths.ensure() does not create probes/, so a subject that has
    # never had a probe set written would otherwise fail here.
    out = Path(paths.probes)
    out.mkdir(parents=True, exist_ok=True)
    out = out / "worksheet.md"
    # Content-level, not just thread-level -- see select().
    excluded = frozenset(_normalize(t.text) for t in sp.heldout)
    out.write_text(render(args.subject_id, args.version, train,
                          len(sp.heldout), len(sp.quarantined),
                          exclude_texts=excluded), encoding="utf-8")
    print(f"wrote {out} from {len(train):,} train subject turns "
          f"(held-out {len(sp.heldout):,} excluded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
