"""Find verified absences that make good S5 unanswerable probes (spec §7).

An unanswerable probe only works when the twin is *tempted*. That needs two
things at once, and this tool checks both against the corpus rather than
guessing:

  * the ENTITY is well grounded, so the twin knows it exists and feels it
    should know more (an entity the twin never saw produces an honest "I don't
    know", which tests nothing); and
  * the ATTRIBUTE is provably absent, so any specific answer is fabricated.

That combination is hard to find by reading, and it is exactly what a corpus
scan is good at. The tool proposes candidates with their evidence; the subject
vetoes any where he would genuinely expect the twin to know, and only then do
they become probes.

This does NOT make the probe set model-authored in the sense that matters:
absence here is measured, not imagined, so the unanswerable half is calibrated
to what the corpus lacks rather than to what a model finds plausibly missing.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from persona_twin.corpus.store import CorpusStore
from persona_twin.eval.split import split_corpus
from persona_twin.paths import SubjectPaths
from persona_twin.schema import Turn

# An entity below this many TRAIN mentions cannot tempt fabrication -- the twin
# simply never learned it, so a wrong answer is ignorance rather than invention.
_ENTITY_FLOOR = 5

# Above zero but small means the attribute IS recorded somewhere; a probe built
# on it would punish a twin that legitimately recalled the rare mention.
_ABSENCE_MAX = 0

# Cap per question SHAPE. N copies of "When did I start using {X}?" are one
# measurement with N trials, not N measurements: a twin that learns a single
# abstention rule passes all of them and scores as well calibrated. This is the
# same degeneracy the S6 balance gate exists to prevent, one level down.
_PER_SHAPE_CAP = 3


@dataclass(frozen=True)
class Attribute:
    key: str
    pattern: str          # what "the attribute is present" looks like
    template: str         # question, formatted with the entity


# Attribute families by entity kind. Kept small and concrete: each must produce
# a question with a short canonical answer if it WERE answerable, or the probe
# is untestable in both directions.
ATTRIBUTES: dict[str, list[Attribute]] = {
    "product": [
        Attribute("user_count", r"\b\d[\d,]*\s*(paying\s*)?(users?|subscribers?|customers?|downloads?)\b",
                  "How many users does {e} have?"),
        Attribute("price", r"(\$\s?\d|\b\d+\s*(dollars|bucks)\b|\bper\s+month\b|/mo\b|\bpricing\b)",
                  "What does {e} cost to use?"),
        Attribute("launch_date", r"\b(launched|released|shipped|went live|first version)\b",
                  "What date did {e} launch?"),
        Attribute("revenue", r"\b(revenue|mrr|arr|profit|earns?)\b",
                  "How much revenue does {e} make?"),
    ],
    "org": [
        # Scoped to the subject's own team, never the company: a company
        # headcount is answerable from world knowledge, so a correct answer
        # would score as fabrication.
        Attribute("team_size", r"\b\d[\d,]*\s*(people|engineers|devs)\s*(on|in)\s*(my|the)\s*team\b",
                  "How many people were on my team at {e}?"),
        Attribute("tenure", r"\b(\d+\s*(years?|months?)\s*(at|with)|joined in|started (there|at))\b",
                  "How long did I work at {e}?"),
        Attribute("manager", r"\b(my manager|reported to|my boss|skip.level)\b",
                  "Who was my manager at {e}?"),
    ],
    "tech": [
        Attribute("version", r"\bv?\d+\.\d+(\.\d+)?\b",
                  "What version of {e} was I using?"),
        Attribute("adoption_date", r"\b(started using|switched to|adopted|picked up)\b",
                  "When did I start using {e}?"),
    ],
    # A hosted service has no version the subject could have been "using".
    "service": [
        Attribute("adoption_date", r"\b(started using|switched to|adopted|picked up)\b",
                  "When did I start using {e}?"),
    ],
    "feature": [
        Attribute("build_duration", r"\b(took (me )?\d+|\d+\s*(days?|weeks?|months?)\s*(to|of))\b",
                  "How long did {e} take to build?"),
        Attribute("collaborator", r"\b(worked with|paired with|helped me|we built)\b",
                  "Who worked on {e} with me?"),
    ],
}


@dataclass(frozen=True)
class Candidate:
    entity: str
    kind: str
    attribute: str
    question: str
    entity_train_hits: int
    attribute_cooccurrences: int


def _count(turns: list[Turn], pattern: str) -> int:
    rx = re.compile(pattern, re.I)
    return sum(1 for t in turns if rx.search(t.text))


def find_candidates(train: list[Turn], entities: dict[str, str],
                    entity_floor: int = _ENTITY_FLOOR,
                    absence_max: int = _ABSENCE_MAX,
                    per_shape_cap: int = _PER_SHAPE_CAP) -> tuple[list[Candidate], list[str]]:
    """Return (candidates, skipped). Skips are returned, not dropped: a tool
    that silently ignores half its input reads as full coverage."""
    out: list[Candidate] = []
    skipped: list[str] = []
    for entity, kind in entities.items():
        rx_e = re.compile(re.escape(entity), re.I)
        hits = [t for t in train if rx_e.search(t.text)]
        if len(hits) < entity_floor:
            skipped.append(f"{entity} ({len(hits)} train hits < {entity_floor})")
            continue
        for attr in ATTRIBUTES.get(kind, []):
            co = _count(hits, attr.pattern)
            if co > absence_max:
                continue
            out.append(Candidate(entity=entity, kind=kind, attribute=attr.key,
                                 question=attr.template.format(e=entity),
                                 entity_train_hits=len(hits),
                                 attribute_cooccurrences=co))

    # Keep the best-grounded entities for each shape: fabrication pressure is
    # highest where the twin knows the entity best.
    kept: list[Candidate] = []
    by_shape: dict[str, list[Candidate]] = {}
    for c in sorted(out, key=lambda c: -c.entity_train_hits):
        by_shape.setdefault(c.attribute, []).append(c)
    for shape, group in by_shape.items():
        kept.extend(group[:per_shape_cap])
        if len(group) > per_shape_cap:
            dropped = ", ".join(c.entity for c in group[per_shape_cap:])
            skipped.append(f"{shape}: {len(group) - per_shape_cap} over cap "
                           f"of {per_shape_cap} ({dropped})")
    return kept, skipped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("subject_id")
    ap.add_argument("version")
    ap.add_argument("--root", default=".")
    ap.add_argument("--entities", required=True,
                    help='JSON map of entity -> kind, e.g. \'{"Regroup":"product"}\'')
    args = ap.parse_args(argv)

    entities = json.loads(args.entities)
    paths = SubjectPaths(args.subject_id, Path(args.root))
    turns = CorpusStore(paths).read(args.version)
    if not turns:
        print(f"no corpus at version {args.version!r}", file=sys.stderr)
        return 1
    train = [t for t in split_corpus(turns, datetime.now(timezone.utc)).train if t.is_subject]
    cands, skipped = find_candidates(train, entities)

    out = Path(paths.probes)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "absence-candidates.json"
    dest.write_text(json.dumps([asdict(c) for c in cands], indent=2) + "\n", encoding="utf-8")
    print(f"{len(cands)} candidate(s) -> {dest}")
    for s in skipped:
        print(f"  skipped: {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
