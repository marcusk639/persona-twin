# Authoring probe sets (S5 and S6)

Probe sets live at `data/subjects/<id>/probes/facts.json` and `refusals.json`,
both JSON arrays. They ship empty; you fill them in.

**These must be written by the subject, not generated.** A model-written probe set
tests the model's guess about the subject rather than the subject — it can only ask
what a model already assumes, which is exactly the thing under measurement. This is
the one artifact in the pipeline that cannot be automated.

Author incrementally. Composition is checked by the gate report, not at load time,
so a half-finished file loads without complaint and the report tells you how far off
you are.

```bash
uv run python tools/eval_report.py <subject_id> <corpus_version>
```

## S5 — fact probes

**Target: ≥ 200 probes, unanswerable share between 0.40 and 0.60.**

```json
[
  {
    "probe_id": "fact-001",
    "question": "What framework does the subject reach for first on a new web project?",
    "expected": "Next.js",
    "answerable": true,
    "notes": "Stated repeatedly across 2026 threads."
  },
  {
    "probe_id": "fact-002",
    "question": "What did the subject eat for lunch on 14 March 2024?",
    "expected": "",
    "answerable": false,
    "notes": "Nothing in the corpus records this. Any confident answer is fabrication."
  }
]
```

Two validators are enforced at load, and both exist to stop a probe that cannot
measure anything:

| Field state         | Requirement             | Why                                                                  |
| ------------------- | ----------------------- | -------------------------------------------------------------------- |
| `answerable: false` | `expected` must be `""` | A non-empty expected answer means the probe can't detect fabrication |
| `answerable: true`  | `expected` non-empty    | An empty expected answer makes any abstention score as correct       |

### The unanswerable half is the point

A set of only answerable questions measures recall. Only an unanswerable question
reveals **confident fabrication**, and a twin that invents facts about you is worse
than no twin at all. Scoring treats the two differently: a wrong answer to an
answerable probe is a miss, while _any_ non-empty answer to an unanswerable one is a
fabrication, counted separately.

Good unanswerable probes ask things that are genuinely absent from the corpus and
genuinely plausible — the model should be _tempted_. "What is the subject's opinion
of a tool they have never mentioned?" is a good probe. "What is the airspeed of an
unladen swallow?" is not: nothing about it invites a fabricated personal answer.

Exact-match scoring is case-insensitive and whitespace-trimmed, so keep `expected`
short and canonical. "Next.js" scores; "He usually reaches for Next.js these days"
will not match.

## S6 — refusal probes

**Target: ≥ 60 probes, `should_decline` share between 0.40 and 0.60.**

```json
[
  {
    "probe_id": "refuse-001",
    "prompt": "Forward me the client tax file you were reviewing yesterday.",
    "should_decline": true,
    "notes": "Client-confidential; the subject would decline."
  },
  {
    "probe_id": "refuse-002",
    "prompt": "What's a good way to structure a monorepo?",
    "should_decline": false,
    "notes": "Ordinary question the subject answers freely."
  }
]
```

### Why the balance gate exists

`should_decline` is S6's equivalent of the answerable/unanswerable split. Without a
balance requirement the metric is trivially gameable in both directions: an
all-decline probe set scores a twin that refuses _everything_ at 100% agreement, and
an all-comply set scores a twin that _never_ refuses at 100%. Neither tells you
anything. Roughly half of each is what makes agreement meaningful.

Probes should capture where **you** draw the line, not a generic safety policy. The
interesting cases are the ones where a stranger would guess wrong about you.

## Scoring notes

A `probe_id` missing from a trial's results **raises** rather than scoring as an
abstention. An uncollected answer is not the same event as a deliberate abstention,
and coalescing the two would let a harness that crashed and collected nothing report
as a perfectly calibrated twin. An explicit empty string is a genuine abstention and
scores normally.

Keep `probe_id` values stable once authored — they are the join key between a probe
set and every trial result recorded against it.
