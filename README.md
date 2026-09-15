# persona-twin

Builds a personality-modeling corpus from one person's own writing, then measures
whether a model trained on it actually sounds like them.

The subject's identity lives only in git-ignored config — see [Subject scoping](#subject-scoping).

## Status

Stages 0–3 of the program in `docs/superpowers/specs/`. Acquisition, normalization,
scrubbing, and the eval harness are implemented; persona mining and model training
(stages 4+) are specified but not built.

| Stage | Scope                                                                | State          |
| ----- | -------------------------------------------------------------------- | -------------- |
| 0     | Storage tiers, threat model, disclosure policy, ledger               | done           |
| 1     | Connectors (iMessage, Claude Code, Claude.ai, Perplexity, ChatGPT, mail, git) | done           |
| 2     | Unified turn schema, identity resolution, scrub, golden freeze       | done           |
| 3     | Held-out split, blind A/B runner, style metrics, probe sets          | in progress    |
| 4+    | Persona mining, twin, voice tuning                                   | specified only |

## How it works

Raw exports land in a vault, are normalized into a single `Turn` schema, scrubbed,
then written to an immutable versioned corpus. Nothing is edited in place at any
stage — a bad build is superseded by a new version, never rewritten.

```
exports ──► connectors ──► vault ──► normalize ──► scrub ──► clean corpus ──► eval
                          (raw)      (Turn)      (classify    (versioned,      (split,
                                                  + redact)    immutable)      A/B, style)
```

See [docs/architecture.md](docs/architecture.md) for the full data flow, the storage
tiers, and the four correctness constraints the pipeline is built around.

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <private-remote-url>
cd persona-twin
uv sync
uv run pytest          # 341 tests
```

## Usage

There is **no console entry point** — pipelines are driven from Python, and the
`tools/` scripts cover the gates. A full ingest-to-corpus cycle:

```python
from pathlib import Path
from persona_twin.config import load_subject
from persona_twin.paths import SubjectPaths
from persona_twin.connectors.base import SubjectContext, run_connector
from persona_twin.connectors.perplexity import PerplexityConnector
from persona_twin.corpus.build import build_corpus
from persona_twin.ledger import LearningLedger

root = Path(".")                                   # NOT root/"data" — see Gotchas
paths = SubjectPaths("subject-01", root); paths.ensure()
ctx = SubjectContext(config=load_subject("subject-01", root), paths=paths)
ledger = LearningLedger(paths.ledger)

exports = sorted(Path("data/subjects/subject-01/vault/exports/perplexity-export")
                 .glob("conversations-*.json"))
run_connector(PerplexityConnector(exports), ctx, ledger)

build_corpus(ctx, ledger, "v5")                    # versions are immutable
```

Re-running a connector is safe: every source uses a stable id, and the vault
de-duplicates on `(subject_id, source, source_id)`, so a second pass writes zero rows.

### Gates

```bash
uv run python tools/corpus_audit.py  <subject_id> <version>   # secrets / confidential / pseudonymization
uv run python tools/volume_report.py <subject_id> <version>   # subject-authored volume + projected cost
uv run python tools/name_leak_lint.py                         # subject identifiers outside config/ and data/
uv run python tools/imessage_spike.py                         # iMessage readability + decode recovery
```

`corpus_audit` is deliberately an **independent** re-check of a stored version rather
than an assertion inside the build, so a bug in the build cannot vouch for itself.

## Adding a connector

Implement the `Connector` protocol in `connectors/base.py` — a `name` and
`fetch(ctx, cursor) -> Iterator[(RawEnvelope, cursor)]` — then register a normalizer
in `normalize/turns.py` and add the source to `SOURCES` in `corpus/build.py`.
Existing connectors need no changes.

Two decisions carry most of the risk:

- **`source_id` must be stable and unique.** Prefer a server-assigned id. Hashing
  content alone collides on repeated short messages; a real corpus scan found 4.4%
  of messages would be silently dropped by vault de-duplication without a positional
  or server-side component.
- **Payload outside the text field is invisible to the paste filter.** `looks_pasted`
  reads inline characters only. A source that carries uploads or attachments must
  handle them explicitly at ingest rather than deferring to the corpus builder.

For static snapshot exports with stable server ids, reuse
`base.scan_snapshot_exports`. Do **not** reuse it for positional or content-derived
ids — its crash-and-rescan model depends on re-ingest being a no-op.

## Data tiers

`.gitignore` excludes `data/` in its entirety. Nothing under it is ever committed.

| Tier          | Holds                                            |
| ------------- | ------------------------------------------------ |
| `vault/`      | Raw envelopes exactly as ingested, plus exports  |
| `clean/`      | Versioned, scrubbed, immutable corpus            |
| `exportable/` | The only tier permitted to leave the machine     |
| `golden/`     | Frozen pre-deployment baseline + superseded ones |

## Subject scoping

Every path, config, and corpus row is keyed by a subject id. The design docs name no
subject: identity lives in `config/subjects/<id>.yaml`, which is git-ignored, so the
spec and this repo stay reusable for another subject. `tools/name_leak_lint.py`
enforces this — it fails the build if subject identifiers appear outside `config/`
and `data/`.

## Gotchas

**`SubjectPaths(subject_id, root)` appends `data` itself.** Passing `root / "data"`
silently creates a second, empty vault at `data/data/subjects/<id>/` instead of
raising. An ingest against it reports plausible counts while the real vault is
untouched. Sanity-check a fresh ingest by asserting the _other_ sources' counts are
non-zero — a wrong-root run is indistinguishable from a correct one on its own
connector's numbers.

## Testing

```bash
uv run pytest              # full suite
uv run pytest -q tests/test_freeze.py
```

Guards in this codebase are mutation-tested: breaking the specific behavior a test
claims to cover must turn that test red. A guard whose removal fails nothing has no
coverage, whatever the test is named.

## License

Not licensed for distribution. Private project.
