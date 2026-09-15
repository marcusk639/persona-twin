"""Generate a pre-filled probe draft the subject can complete in one sitting.

Authoring probes conversationally costs a full round-trip per item to produce
about two probes, and almost all of that round-trip is mechanical: finding
which entities in a line are grounded, checking which attributes are recorded,
and formatting. Only two things actually need the subject -- the short
canonical ANSWER to an answerable question, and a VETO on an unanswerable one.

So this emits the mechanical part already done, in the pipe format
tools/probe_intake.py already reads, with `???` where an answer belongs:

    # ITEM 12 | Regroup(45) React Native(92)
    # > the seed line this came from
    U | How many users does Regroup have? | | Regroup 45 train, user_count 0x
    A | What framework is Regroup built with? | ???

The subject fills each `???`, deletes any `U` row the twin should plausibly
know, and appends `# GAP: ...` lines for things that are true but absent from
the corpus. One editor pass over 130 items instead of 130 exchanges.

Answerable and unanswerable candidates come from the SAME scan, inverted: an
attribute that co-occurs with its entity is a question the corpus can answer,
and one that never co-occurs is a question it cannot. Both are measured.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_absences import ATTRIBUTES, _count  # noqa: E402

from persona_twin.corpus.store import CorpusStore  # noqa: E402
from persona_twin.eval.split import split_corpus  # noqa: E402
from persona_twin.paths import SubjectPaths  # noqa: E402
from persona_twin.schema import Turn  # noqa: E402

# Same floor as the absence finder: an entity the twin barely saw produces an
# honest "I don't know", which measures nothing in either direction.
_ENTITY_FLOOR = 5

# An answerable question needs its attribute RECORDED, not merely present once.
# Below this the twin never learned it, so a miss is corpus sparsity scored as
# a recall failure.
_ANSWERABLE_FLOOR = 5


def entities_in(line: str, entities: dict[str, str]) -> list[str]:
    return [e for e in entities if re.search(re.escape(e), line, re.I)]


def rows_for_entity(train: list[Turn], entity: str, kind: str) -> tuple[list[str], int]:
    """Return (draft rows, entity train hits) for one entity."""
    hits = [t for t in train if re.search(re.escape(entity), t.text, re.I)]
    if len(hits) < _ENTITY_FLOOR:
        return [], len(hits)
    # Unanswerable rows are NO LONGER generated. The absence detector behind
    # them checked one narrow vocabulary within a single turn, and a
    # wider-window re-check found 12 of 15 "verified absent" claims to be
    # false: "launched" / "went live" / "shipped" / "TestFlight" are one fact,
    # and a launch date is usually stated a sentence away from the product
    # name. Shipping those rows asks the subject to validate the tool's own
    # errors, and a false absence punishes a twin that answered correctly.
    #
    # Answerable rows are still offered: they require the attribute to be
    # PRESENT, and a false positive there merely proposes a question the
    # subject can decline.
    rows: list[str] = []
    for attr in ATTRIBUTES.get(kind, []):
        co = _count(hits, attr.pattern)
        if co >= _ANSWERABLE_FLOOR:
            rows.append(f"A | {attr.template.format(e=entity)} | ??? | "
                        f"{entity} {len(hits)}x train, {attr.key} {co}x")
    return rows, len(hits)


def render(seed_lines: list[str], train: list[Turn],
           entities: dict[str, str]) -> tuple[str, int]:
    out: list[str] = [
        "# =====================================================================",
        "#  S5 FACT-PROBE DRAFT",
        "# =====================================================================",
        "#",
        "#  These are your own words, pulled from the training split. Each block",
        "#  is one line you wrote, plus blanks for you to fill.",
        "#",
        "#  You do NOT need to complete every block. Many lines carry no",
        "#  checkable fact; skip those. Partial passes are fine -- the intake",
        "#  can be re-run as often as you like and will not double-add.",
        "#",
        "# ---------------------------------------------------------------------",
        "#  WHAT A BLOCK LOOKS LIKE",
        "# ---------------------------------------------------------------------",
        "#",
        "#    # ITEM 3 | Meta(241)          <- entity, and its training mentions",
        "#    # > Yeah Rudy hit me up ...   <- the line you wrote",
        "#    A | Who was my manager at Mountain? | ???   <- evidence on the line",
        "#    A | ??? | ???",
        "#    # GAP:",
        "#",
        "# ---------------------------------------------------------------------",
        "#  1. THE 'A' ROW  --  this is the one that matters",
        "# ---------------------------------------------------------------------",
        "#",
        "#  Read the line and ask: what fact does this imply that someone could",
        "#  check? Write the question, then a SHORT answer.",
        "#",
        "#    A | What framework is Regroup built with? | React Native",
        "#",
        "#  Scoring is exact match, case-insensitive. So:",
        "#",
        "#    React Native                          <- scores",
        "#    he used React Native for it           <- never matches",
        "#    React Native, though we tried Expo    <- never matches",
        "#",
        "#  One or two words is ideal. If the honest answer is a sentence or a",
        "#  list, it is not a fact probe -- leave the ??? and move on.",
        "#",
        "#  Prefer facts your archive states OFTEN. A fact mentioned once in",
        "#  40,000 turns cannot be recalled by the twin, so a miss there measures",
        "#  the archive, not the twin.",
        "#",
        "# ---------------------------------------------------------------------",
        "#  2. THE PRE-FILLED 'A' ROWS",
        "# ---------------------------------------------------------------------",
        "#",
        "#  Some blocks already carry a question, with evidence on the line",
        "#  ('Mountain 213x train, manager 9x'). Those attributes ARE recorded,",
        "#  so the answer exists in your archive -- just fill the ???.",
        "#",
        "#  Delete any you would rather not answer.",
        "#",
        "# ---------------------------------------------------------------------",
        "#  3. THE '# GAP:' LINE  --  optional, but valuable",
        "# ---------------------------------------------------------------------",
        "#",
        "#  Something true that your archive never recorded. Write it after the",
        "#  colon. These map what the twin cannot learn, and there will be more",
        "#  of them than you expect.",
        "#",
        "#    # GAP: the deep research skill runs on dynamic workflows",
        "#",
        "# ---------------------------------------------------------------------",
        "#  WHEN YOU ARE DONE (or partway)",
        "# ---------------------------------------------------------------------",
        "#",
        "#    uv run python tools/probe_intake.py --merge \\",
        "#      data/subjects/<id>/probes/draft.txt \\",
        "#      data/subjects/<id>/probes/facts.json",
        "#",
        "#  It keeps what is already authored, skips unfilled ??? rows, and",
        "#  reports where you stand against the gate.",
        "#",
        "#  The gate wants 200 probes, 40-60% unanswerable. The unanswerable",
        "#  half is being reworked -- an earlier auto-detector produced mostly",
        "#  false absences -- so concentrate on 'A' rows here.",
        "#",
        "# =====================================================================",
        "",
    ]
    n_rows = 0
    for i, line in enumerate(seed_lines, 1):
        present = entities_in(line, entities)
        rows: list[str] = []
        labels: list[str] = []
        for e in present:
            r, hits = rows_for_entity(train, e, entities[e])
            rows += r
            labels.append(f"{e}({hits})")
        # Every line gets a blank Q/A prompt, even with no known entity: the
        # templates below can only ask generic attribute questions, while the
        # probes with real content come from the subject reading his own words.
        rows.append("A | ??? | ???")
        snippet = re.sub(r"\s+", " ", line)[:150]
        out.append(f"# ITEM {i} | {' '.join(labels)}")
        out.append(f"# > {snippet}")
        out += rows
        out.append("# GAP: ")
        out.append("")
        n_rows += len(rows)
    return "\n".join(out) + "\n", n_rows


def seed_lines_from(worksheet: Path) -> list[str]:
    """Bullets under the CATEGORY headings only -- not the instructions at the
    top, and not the '- term (count)' topic list."""
    body = worksheet.read_text(encoding="utf-8")
    lines: list[str] = []
    for section in re.split(r"^## ", body, flags=re.M):
        name = section.split("\n", 1)[0].strip()
        if not re.match(r"(preferences|opinions|decisions|history)\b", name):
            continue
        lines += [l[2:] for l in section.splitlines() if l.startswith("- ")]
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("subject_id")
    ap.add_argument("version")
    ap.add_argument("--root", default=".")
    ap.add_argument("--entities", required=True, help="JSON map of entity -> kind")
    args = ap.parse_args(argv)

    import json
    entities = json.loads(args.entities)
    paths = SubjectPaths(args.subject_id, Path(args.root))
    turns = CorpusStore(paths).read(args.version)
    if not turns:
        print(f"no corpus at version {args.version!r}", file=sys.stderr)
        return 1
    train = [t for t in split_corpus(turns, datetime.now(timezone.utc)).train if t.is_subject]

    worksheet = Path(paths.probes) / "worksheet.md"
    if not worksheet.exists():
        print(f"no worksheet at {worksheet}; run tools/probe_worksheet.py first",
              file=sys.stderr)
        return 1
    seeds = seed_lines_from(worksheet)
    body, n_rows = render(seeds, train, entities)

    dest = Path(paths.probes) / "draft.txt"
    dest.write_text(body, encoding="utf-8")
    print(f"wrote {dest}: {n_rows} draft rows across {len(seeds)} seed lines")
    print(f"fill it in, then: uv run python tools/probe_intake.py {dest} "
          f"{Path(paths.probes) / 'facts.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
