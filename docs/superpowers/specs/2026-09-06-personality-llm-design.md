# Personality LLM ("persona-twin") — Design Spec

- **Date:** 2026-09-06
- **Status:** Approved design, pending implementation plan
- **Subject:** see `config/subjects/<id>.yaml`
- **Scope:** Staged research-and-build program producing a system that writes as the subject and converses as the subject

---

## 1. Purpose

Two products from one substrate:

1. **Draft-as-me** — a tool that produces text in the subject's voice (email, chat replies, documents, posts) such that the subject would send it with minimal editing.
2. **The twin** — a conversational endpoint that third parties interact with directly, which answers as the subject would: their voice, their opinions, their factual knowledge, and their shifts in register depending on who is asking.

Target depth is **full simulation**: voice (how things are said), substance (what would be said, including opinions and refusals), facts (people, history, projects), and register (how all of the above changes by audience).

### Breadth of knowledge

The system models **everything the subject knows and has views about** — not a professional slice of it. The subject's occupation determines which confidentiality rules apply (§4.1); it does not bound the subject matter. Personal history, relationships, interests, technical work, hobbies, aesthetic preferences, politics, health, grievances, and half-formed opinions are all in scope, because a twin that can only discuss one domain is not a model of a person.

The distinction that matters in §4.1 is **client data versus domain knowledge**: a client's financial records are excluded, while the subject's accounting expertise — how they reason about a tax question, what they think of a standard, how they explain it to a layperson — is core T1 material and is retained.

### Non-goals

- Not a general-purpose assistant. The system is not required to be capable at tasks the subject could not do themselves; its breadth comes from the subject, not from the base model's abilities.
- Not a deception tool. Third-party interactions disclose that the twin is a model (§4.3).
- Not a memory replacement or an archival product. The corpus serves modeling, not retrieval-for-humans.
- Not real-time voice/audio synthesis. Text only.

---

## 2. Success criteria

The program succeeds when all of the following hold on a held-out test set the system has never trained on or retrieved from:

| # | Criterion | Measure | Target |
|---|---|---|---|
| S1 | Voice indistinguishability | Blind A/B: judges shown the subject's real reply and the system's reply to the same context, asked to pick the real one | ≤60% correct, **n ≥ 300 paired trials**, 95% CI reported |
| S2 | Style fingerprint match | Distributional distance on message length, burstiness, punctuation/emoji rates, lexical signature | Inside the subject's **self-distance band**: bootstrap 200 random half-splits of held-out subject data, measure subject-vs-subject distance, require system distance ≤ 95th percentile of that band |
| S3 | Substance agreement | Subject reviews generated responses to novel prompts, marks "I would/wouldn't say this" | ≥80% "would say", **n ≥ 100**, Wilson CI lower bound ≥ 70% |
| S4 | Register discrimination | Same prompt across 4 audience contexts produces measurably different output | Style distance between registers > distance within register |
| S5 | Factual accuracy | Claims about the subject's life/work in generated text | ≥95% correct and 0 confident fabrications on a curated probe set of **n ≥ 200**, half of which are unanswerable-by-design |
| S6 | Refusal fidelity | System declines what the subject would decline | ≥80% agreement on a refusal probe set of **n ≥ 60** |
| S7 | Learning | Scores improve across updates without regression | **Interim proxy (per update):** edit-distance between generated draft and sent message trends down over rolling 100 drafts. **Long-run:** positive S1/S3 trend quarter-over-quarter; zero uncaught S5/S6 regressions |

All targets are evaluated against the stated n; a gate reported without its sample size is not a passed gate. Failure to reach S1 is acceptable at stage boundaries; failure on S5 is not — a twin that invents facts about the subject is worse than no twin.

---

## 3. Evidence model

The central discipline of this project. Every assertion about the subject's personality carries a **provenance tier**. Tiers are never silently mixed.

| Tier | Definition | Examples | Use |
|---|---|---|---|
| **T1** | Observed behavior — the subject's own words in their original context | iMessage, sent email, Claude Code transcripts, git commits, documents | Training data, retrieval exemplars, eval ground truth |
| **T2** | Model-derived description — an LLM's summary of the subject | ChatGPT/Claude/Perplexity self-description reports | Hypotheses only. Never training data. Never eval data. |
| **T3** | Subject self-report — direct assertions by the subject | Interview answers, psychometric instruments, hand-written style notes | Hypotheses and tie-breakers. May inform the persona spec; never eval ground truth. |

### 3.1 Rules

- **R1** — Training sets and retrieval corpora contain T1 only.
- **R2** — The eval harness (§7) is built exclusively from T1. T2 and T3 are excluded to prevent the system from being scored against a description that helped author it.
- **R3** — Every T2/T3 claim enters the **claims ledger** (§6.3) as `unverified` and must be checked against T1 before informing the persona spec.
- **R4** — Contradictions between tiers are preserved, never resolved by averaging. A claim asserted at T2 and contradicted at T1 is recorded as `contradicted` and is itself a finding.

### 3.2 Rationale

Model-derived self-descriptions are summaries of a curated, partial slice of history, produced by systems with a strong prior toward agreeable output. Instructing such a model to be critical changes the register of its output, not its access to evidence — it substitutes confidently-stated criticism for confidently-stated praise, generated by the same process. Invented flaws are more corrosive than invented virtues, because criticism reads as evidence of honesty and is therefore discounted less. T2 is retained because cross-model convergence carries real signal, but it is never promoted by improved prompting.

---

## 4. Constraints

### 4.1 Confidential data

The subject operates a CPA firm. The corpus will contain client-confidential financial data, and the Claude Code transcripts (497 MB) contain API keys, tokens, credentials, and client identifiers.

- **C1** — Karbon exports, client financial records, and anything tagged confidential are excluded from any corpus that reaches a hosted provider.
- **C2** — Secret scanning runs on every artifact before it leaves the vault, on a deny-by-default basis: unrecognized high-entropy strings are redacted, not passed.
- **C3** — Third-party PII (counterparties in iMessage and email) is pseudonymized at normalization. The identity map lives in the vault and never leaves it.
- **C4** — A scrub failure is a pipeline failure. Partial scrubbing does not produce a partial artifact; it produces an error.

These are **handling rules, not scope limits.** They govern which bytes may leave the vault; they do not narrow what the twin knows or will talk about (§1, Breadth of knowledge).

### 4.2 Storage tiers

- **Vault** (`data/vault/`) — raw, immutable, never leaves the machine, never committed, encrypted at rest.
- **Clean** (`data/clean/`) — normalized, scrubbed, pseudonymized. Eligible for local training.
- **Exportable** (`data/exportable/`) — the subset cleared for hosted inference or hosted fine-tuning. Requires explicit per-artifact sign-off.

### 4.3 Disclosure

Any third-party-facing twin interface states that it is a model of the subject, not the subject, on first contact and on request. This is a design constraint from stage 0, not a later addition.

### 4.4 Cost

**The 611 MB figure is misleading and must not drive the budget.** Only *subject-authored* text needs model processing, and it is one to two orders of magnitude smaller than the raw corpus. Measured on a 60-file sample of the Claude Code transcripts: **1.46% of bytes are user-authored** — 497 MB of transcripts reduces to roughly 7 MB of the subject's own words. Applying the same extraction to mail (quoted replies, signatures, threading) and iMessage yields an estimated **10–30 MB total ≈ 3–8 M tokens**, which is one frontier pass in the tens of dollars, not thousands.

**Measured 2026-09-07 (iMessage, full snapshot):** 77,619 messages total, 37,403 authored by the subject, of which 37,387 carry recoverable text — **2.36 MB ≈ 590 K tokens**. Combined with the transcripts' ~7 MB, the working corpus is tracking toward the *low* end of the estimate: roughly 2.5–3 M tokens, or **under $10 per full frontier pass**.

**Consequence — the binding constraint is data, not cost.** At this volume the §4.4 mitigations are insurance rather than necessity, and the risk shifts to the opposite failure: 590 K tokens of conversational text is a modest corpus for SFT (stage 7). Retrieval (stage 5) extracts more from a small corpus than fine-tuning does, which strengthens the staging already specified. The cold-start handling of §13.2 may apply to the primary subject, not only to future ones. The decisive number is not raw volume but the count of usable **reply pairs** (§6.1 thread reconstruction), which is measured at the stage-2 gate.

The remaining estimate is confirmed or refuted by a measured number at that gate, before any mining is authorized.

Mitigations, in order of leverage:

- **M1 — Extract before you spend.** Strip to subject-authored spans first: user turns only from transcripts, sent-mail bodies with quotes and signatures removed, `is_from_me` messages from iMessage. This is the single largest cost reduction available and it is free.
- **M2 — Compute style, don't infer it.** Every S2 metric — length distributions, burstiness, punctuation and emoji rates, function-word signature — is arithmetic over text. Zero model calls. Only substance, opinions, and register semantics require an LLM.
- **M3 — Deduplicate.** Boilerplate, quoted text, repeated openers, and re-sent content inflate volume without adding signal.
- **M4 — Tiered funnel.** Local model on the 48 GB machine for high-volume passes (candidate extraction, topic and register labeling, filtering); frontier model only for synthesis of the persona spec from pre-filtered candidates.
- **M5 — Embeddings over prompts** for register clustering and retrieval. Roughly two orders of magnitude cheaper per unit of text than generative calls.
- **M6 — Batch API and prompt caching.** All mining is offline with no latency requirement, so batch pricing applies; a stable persona/context prefix is cacheable across calls.
- **M7 — Saturation sampling.** Mine in batches and track new claims per batch. Stop when marginal yield falls below threshold rather than processing everything. Most corpora saturate long before exhaustion.
- **M8 — Budget as a hard gate.** Each stage declares a cost ceiling. A dry-run token count and projected spend is printed before execution, and any run projected above its ceiling stops and asks rather than proceeding. A per-stage cost ledger records actual spend against estimate.

---

## 5. Data sources

### 5.1 Local, confirmed present

| Source | Location | Volume | Tier | Sensitivity | Value |
|---|---|---|---|---|---|
| iMessage | `~/Library/Messages/chat.db` | 114 MB | T1 | High (third-party PII) | Primary voice source — unguarded, conversational, register-rich. **Two blockers, see §5.6** |
| Claude Code transcripts | `~/.claude/projects/**/*.jsonl`, `~/.claude-acct2/projects/` | 1,084 sessions, 497 MB | T1 | Critical (secrets) | Working judgment, technical voice, decision-making verbatim |
| claude-mem observations | `~/.claude-mem/claude-mem.db` | 30 MB | T2 (derived) | Medium | **Deferred.** Model-written summaries of transcripts already ingested in full; carries the only mixed-provenance problem in the table. Revisit only if transcript mining underdelivers |
| Apple Notes | `~/Library/Group Containers/group.com.apple.notes/NoteStore.sqlite` | 3.9 MB | T1 | Medium | Private unedited thought |
| Git history | `~/dev/**/.git` (30 repos) | — | T1 | Low | Terse decision voice; commit messages are high signal per token |
| Authored docs | `~/dev/**/*.md`, `CLAUDE.md` files | — | T1 | Medium | Considered written voice, standards, preferences |

### 5.2 Connected, pullable via MCP

Gmail, Outlook/M365 mail, Teams chat and channel posts, Google Drive, SharePoint/OneDrive, Notion, Calendar.

**Meeting transcripts** (Teams/Zoom, via M365) are called out specifically: they are the only record of the subject's *spoken* register, which differs materially from written register and is otherwise entirely absent from the corpus.

### 5.3 Request-and-wait (initiate at stage 0; latency is days)

ChatGPT data export, Claude.ai data export, Perplexity threads, Google Takeout, X/Reddit/LinkedIn archives if applicable.

Raw conversation exports from these services are T1 and outrank any summary of them: years of questions asked is a better map of what the subject cares about than a paragraph asserting what the subject cares about.

### 5.4 T2 harvest — self-description reports

Collected from ChatGPT, Claude, and Perplexity independently, with an identical prompt. Protocol:

- Request **falsifiable behavioral claims**, not a description. (How does the subject open a message to someone who has annoyed them? What do they do when they don't know something? What are their tells when they have already decided? What do they never say?)
- Require a **verbatim quote** as evidence for each claim. A claim without a citation is unusable, because it cannot be checked.
- Require a **confidence rating** and make abstention explicitly available. A fixed claim quota manufactures claims; "as many as the evidence supports" does not.
- Require the **disconfirming case** for each claim.
- **Run twice per model in separate sessions and diff.** Claims that recur are grounded in retrievable history; claims that drift were generated at request time. This is the primary confabulation detector.
- Criticality/anti-flattery framing is retained — it is a net improvement — but confers no tier promotion.

**Availability risk:** this protocol assumes each service retains enough history to ground a report. Claude.ai has no cross-conversation memory unless explicitly enabled, and Perplexity retains little. Where a service cannot produce a grounded report, the fallback is to supply that service's *exported conversations* as context and request the same claim format over them. A report generated with no retrievable history is discarded, not filed as T2.

### 5.5 T3 — subject-supplied

Structured interview, with questions generated from the claims-ledger contradiction set (§6.3) rather than asked open-endedly — targeted questions about specific disagreements between what models assert and what the corpus shows are worth far more per minute of the subject's time. Long-form psychometric inventories were considered and cut: R2 bars them from eval, they inform the persona spec only weakly, and they cost significant subject time.

### 5.6 iMessage extraction blockers

Verified 2026-09-07: `sqlite3 -readonly ~/Library/Messages/chat.db` returns `authorization denied`. Two prerequisites stand between the plan and its primary source, and both are resolved by a **stage-0 spike** before any sequencing commits to iMessage-first.

1. **Full Disk Access.** The executing process — terminal, and separately any scheduled `launchd` job — must be granted FDA in System Settings → Privacy & Security. The daily ingest job needs its own grant; a grant to the interactive terminal does not cover it.
2. **`attributedBody` decoding.** On current macOS the `message.text` column is frequently `NULL`, with the real content stored in `message.attributedBody` as an `NSKeyedArchiver` blob. A connector reading only `text` silently returns a corpus with large, non-random holes — the worst failure mode available, because it looks like success.

**Spike exit criteria:** read the database, report total messages, the `is_from_me` count, the fraction of rows where `text` is NULL, and the fraction of those recovered by decoding `attributedBody`. Recovery below 95% means the extraction approach changes before stage 1 begins.

**Result 2026-09-07: PASS.** Working from a vault snapshot rather than the live database (which also avoids lock contention with Messages.app), 1,256 of 77,619 rows had empty `text` (1.6% — far below the feared rate), 1,145 of those carried an `attributedBody` blob, and **1,145 of 1,145 decoded successfully (100%)**. The heuristic decoder is adequate; no fallback to a full typedstream parser is required.

---

## 6. Architecture

### 6.1 Components

```
[connectors] → [vault] → [normalize] → [scrub] → [clean corpus] ─┬→ [persona mining] → [persona spec]
                                                                  ├→ [exemplar index]
                                                                  ├→ [preference set builder]
                                                                  └→ [eval harness]  (held-out split)

[T2 harvest] ─┐
[T3 intake]  ─┴→ [claims ledger] ←verify against→ [clean corpus]
                        ↓
                 [persona spec]
                        ↓
        ┌───────────────────────────────┐
        │        serving layer          │
        │  substance model → voice pass │  ← register router
        └───────────────────────────────┘
                        ↓
              [draft-as-me]  [twin endpoint]
```

Each component is independently testable and communicates through files with declared schemas. No component reaches into another's internals.

| Component | Does | Depends on |
|---|---|---|
| Connectors | One per source. Read a source, write raw envelopes to vault. Idempotent, resumable. | Source credentials |
| Normalize | Raw envelopes → unified turn records. Thread reconstruction, identity resolution. | Vault |
| Scrub | Secret detection, PII pseudonymization, confidentiality classification. Fails closed. | Normalized turns |
| Clean corpus | Queryable store of scrubbed turns with metadata. The single source of T1 truth. | Scrub |
| Eval harness | Held-out split, blind A/B runner, style metrics, judge rubric. | Clean corpus |
| Persona mining | Style card, opinion ledger, fact KB, register map. | Clean corpus, claims ledger |
| Claims ledger | T2/T3 claims with provenance, verification status, evidence links. | T2 harvest, T3 intake, clean corpus |
| Exemplar index | Retrieval over real messages, filtered by register and topic. | Clean corpus |
| Serving | Substance generation, voice pass, register routing, disclosure. | Persona spec, exemplar index, models |

### 6.2 Architecture options and where each lands

Eight candidate mechanisms, staged rather than chosen up front. Each is gated on measured improvement over the previous stage; any that fails its gate is cut, not carried.

| Option | Mechanism | Stage | Rationale |
|---|---|---|---|
| **A** Persona-as-context | Persona spec + retrieved real-message exemplars on a frontier model | 5 | Fastest path to a working twin; best reasoning; fully auditable and hand-editable |
| **G** Constitution + self-critique | Mined constitution, generate → critique → revise | 6 | Cheap, model-agnostic, stacks on everything else |
| **B** Local SFT LoRA | Fine-tune open-weights on T1 messages | 7 | Voice fidelity: cadence, length distribution, punctuation habits come free |
| **D** DPO on real choices | Chosen = subject's actual reply; rejected = vanilla model reply to same context | 7 | Directly trains "be the subject, not an assistant." Highest-leverage single technique; the corpus is natively a preference dataset |
| **C** Hybrid | Substance model decides what; voice model rewrites how | 8 | Each part supplied by the mechanism that is good at it |
| **F** Register LoRAs | Per-register adapters, routed or merged by audience | 8 | The concrete implementation of register switching |
| **E** Persona vectors / steering | Contrastive activation direction, applied with an intensity dial | 9 | Intensity control (10% for a client email, 90% for a text to a friend). Research stage |
| **H** Distillation bridge | Frontier + persona generates in-voice data; small local model trains on it | 9 (fallback) | Path to a local model without a second raw-corpus exposure |

### 6.3 Claims ledger

Each claim: `id`, `text`, `tier`, `source`, `evidence_quote`, `source_confidence`, `disconfirming_case`, `recurrence` (did it reappear on the repeat run), `verification_status` ∈ {unverified, corroborated, contradicted, unfalsifiable}, `t1_evidence[]`.

The verification pass queries the clean corpus for behavioral evidence for and against each claim. The **contradicted** set is a primary deliverable: where three models agree on a trait and the subject's actual messages disagree is precisely where a naive persona build would go wrong, and it is also the best source of interview questions for T3.

### 6.4 Persona spec

The human-readable, hand-editable artifact that drives generation. Versioned in git. Sections:

- **Style card** — measured, not asserted: length distributions by register, punctuation and capitalization habits, emoji use, opener/closer inventory, hedging and intensifier rates, characteristic constructions, misspelling and typo patterns.
- **Opinion ledger** — positions held, with T1 evidence and strength. Includes what the subject argues against.
- **Fact KB** — people (pseudonymized externally, resolved internally), projects, history, roles, timeline.
- **Register map** — audience clusters derived from the corpus by a stated method: embed the subject's outgoing turns, cluster by style-feature vector (length, formality markers, punctuation, emoji, profanity, hedging) rather than by topic, then label clusters against known relationship metadata (contact, channel, thread participants). Clusters are validated by checking they separate on held-out data; a cluster that does not survive validation is merged. Each surviving cluster carries its own style card delta and topic boundaries. S4 is measured against exactly these clusters.
- **Refusal profile** — what the subject declines, deflects, or does not engage with.
- **Constitution** — behavioral rules derived from the above, used by the stage-6 critique loop.

---

## 7. Eval harness

Built at stage 3 — after normalization, because held-out splits and style fingerprints are computed over normalized turns, but before persona mining (stage 4) and the first twin (stage 5). What matters is that the scoreboard exists before anything it scores, not that it precedes the pipeline that feeds it.

- **Held-out split** — by time and by thread, not randomly. Random splits leak: adjacent turns in one conversation land on both sides. The most recent N months are reserved and never trained on or retrieved from.
- **Blind A/B (S1)** — for a held-out incoming message, present the subject's real reply and the system's reply in random order to judges (human, and separately an LLM judge with no access to the persona spec). Report accuracy; 50% is perfect.
- **Style fingerprint (S2)** — computed distributional metrics, not model judgments: token/char length distributions, burstiness, punctuation and emoji rates, type-token ratio, function-word signature, sentence-initial patterns. Compared against the subject's own held-out distribution, so the target is a real number rather than an opinion.
- **Substance review (S3)** and **refusal probes (S6)** — subject-scored, on novel prompts.
- **Register discrimination (S4)** — same prompt, four audiences, measure between-register vs within-register style distance.
- **Fact probes (S5)** — curated questions with known answers, scored for correctness and for confident fabrication separately.
- **Baseline** — every metric is first computed for a pinned reference configuration, defined exactly and never changed without renaming: a named model version, temperature and top-p fixed, and the system prompt `"Reply to this message."` with no name, biography, persona spec, or exemplars. A second *informed* baseline — same model, system prompt naming the subject and their occupation only — is also scored, because it separates "the model knows a CPA with the subject's name" from "the system learned the subject." Every later stage reports its delta against both.

**Contamination rules:** no T2 or T3 input reaches the harness; the LLM judge never sees the persona spec; held-out threads are excluded from the exemplar index at index build time, not at query time.

---

## 8. Stages

Each stage produces a usable artifact and has a numeric exit gate. The program can stop at any stage boundary with something that works.

| # | Stage | Deliverables | Exit gate |
|---|---|---|---|
| **0** | Foundations | Repo, storage tiers, encryption, threat model, disclosure policy, secret-scanning config. **Learning scaffolding: CC1 tagging schema, CC3 quarantine policy, learning ledger. Subject-scoped schemas and paths per §13.1.** All external data exports requested. **iMessage extraction spike (§5.6).** | Vault encrypted; iMessage spike passes its exit criteria; every turn carries an assisted/unassisted tag and a subject ID; name-leak lint passes; export requests filed |
| **1** | Acquisition (connectors are independent and may be built in parallel) | Connectors for iMessage, Claude Code transcripts, mail, docs, git, Notes, built against **one shared connector interface (F6)**. T2 harvest executed per §5.4. | ≥3 T1 sources landed in vault; a new source can be added without touching existing connectors; T2 reports collected from 3 models × 2 runs |
| **2** | Normalize & scrub | Unified turn schema, thread reconstruction, identity resolution, secret and PII scrubbing. **Golden corpus frozen (CC2).** Subject-authored volume measured (§4.4). | Zero secrets and zero client-confidential records in clean corpus, verified by audit; golden snapshot immutable; measured token volume and projected stage-4 cost reported |
| **3** | Eval harness | Held-out split with **rolling quarantine (CC3)**, blind A/B runner, style metrics, judge rubric, fact/refusal probe sets | Vanilla baseline scored on S1–S6 at stated sample sizes; quarantine window rolls automatically |
| **4** | Persona mining | Style card, opinion ledger, fact KB, register map, refusal profile, claims ledger with verification pass | Persona spec reviewed and corrected by subject; contradiction set produced |
| **5** | **Twin v1 (A)** | Register-aware RAG twin + draft-as-me CLI, **shipping with edit capture (F1) and ratings (F2) from first use** | Beats vanilla baseline on S1 and S3; preference pairs accumulating from day one |
| **6** | Constitution (G) | Critique-and-revise loop | Measurable lift over stage 5, or the stage is cut |
| **7** | Voice tuning (B, D) | Local SFT LoRA; preference set; DPO run | Beats stage 5/6 on blind A/B (S1) and style fingerprint (S2) |
| **8** | Hybrid (C, F) | Substance model + voice pass, register routing | Best composite score across S1–S6 |
| **9** | Steering (E), distillation (H) | Intensity dial; optional local distilled model | Research stage; optional |
| **10** | Learning loop & serving | Twin endpoint with disclosure, episodic memory, continual ingest (F4), active elicitation (F5), drift-vs-error analysis, change control and learning ledger | Runs unattended 7 days; a correction demonstrably propagates at L1/L2 and is reverted cleanly on a forced regression |

---

## 9. Continual learning

The system must improve after deployment — from new data, from feedback, and from its own observed failures.

### 9.1 Three learning layers

| Layer | Mechanism | Latency | Reversibility | Approval |
|---|---|---|---|---|
| **L1 Retrieval** | New T1 turns enter the exemplar index | Minutes | Trivial (reindex) | Automatic |
| **L2 Persona spec** | Mined changes proposed as reviewable diffs | Days–weeks | Easy (git revert) | Human for substance/opinions; automatic for measured style metrics |
| **L3 Weights** | LoRA/DPO retrain on accumulated preference pairs | Months | Costly (rollback to prior adapter) | Human, gated by the full S1–S6 suite |

**Principle:** absorb every correction at the fastest, cheapest, most reversible layer that can hold it. Weight changes are the last resort, never the first response to a bad output.

### 9.2 Feedback channels

- **F1 — Edit capture (primary).** Draft-as-me records `(context, generated draft, what was actually sent)`. The diff is a natural preference pair: rejected = the draft, chosen = the sent message. It costs the subject nothing, and unlike the original corpus it shows *where the system fails* rather than merely how the subject writes. This is the flywheel; every other channel is supplementary.
- **F2 — Explicit ratings.** Accept / edit / reject, with an optional one-line reason.
- **F3 — Twin transcript review.** The subject reviews third-party conversations and flags divergence with annotations.
- **F4 — Ongoing ingest.** New iMessage, email, Claude Code sessions, and documents land continuously through the same connectors. Free, permanent T1 growth.
- **F5 — Active elicitation.** The system identifies its own weak spots — thin registers, low retrieval coverage, high substance/voice disagreement, `contradicted` claims — and asks targeted questions. Far higher yield per minute of the subject's time than open-ended interviews.
- **F6 — New source onboarding.** Connectors implement a single interface, so adding a source later is a plug-in rather than a redesign.

### 9.3 Contamination controls

**These cannot be retrofitted.** If they are not built before the system is first used, the data needed to untangle the results is permanently unrecoverable.

- **CC1 — Assisted/unassisted tagging.** Every ingested turn records whether the system helped write it. Once draft-as-me is in daily use, the subject's outgoing messages stop being independent evidence of the subject's voice.
- **CC2 — Golden corpus freeze.** An immutable snapshot of the pre-deployment corpus is the permanent reference for the subject's voice *as it was before the system existed*. All drift measurement is against it.
- **CC3 — Rolling held-out quarantine.** The newest N weeks are always reserved — never trained on, never indexed for retrieval. The window rolls forward, so a fresh uncontaminated eval set always exists.
- **CC4 — Model-collapse guard.** Training sets weight unassisted turns above assisted ones, and assisted-origin data may never exceed a fixed share of any training set. Without this the system progressively trains on its own output wearing the subject's name.

### 9.4 Drift versus error

A shift in style metrics has two possible causes: the model is wrong, or the subject changed. People's voices genuinely evolve.

These are distinguished by measuring recent **unassisted** subject data against the golden corpus. If the subject's own metrics moved, the persona spec follows and the change is recorded as drift. If only the system's metrics moved, it is error and is corrected. Conflating the two silently converges the twin on a portrait of who the subject used to be.

### 9.5 Change control

- The persona spec is versioned in git. Every change is a diff carrying its trigger evidence, provenance tier, observation count, and measured eval delta.
- **Threshold rule** — no rule enters the persona spec on fewer than N independent observations. A single emphatic correction does not become law.
- Every L2 and L3 change runs the full S1–S6 suite. **A regression on S5 (facts) or S6 (refusals) blocks the change automatically, regardless of any S1 gain.**
- **Learning ledger** — append-only record of every change, its trigger, and its measured delta, so any regression can be traced to the change that caused it and reverted.
- Rollback: persona spec via git revert; adapters versioned and hot-swappable.

### 9.6 Twin episodic memory

Distinct from personality learning: the twin retains per-interlocutor conversation history so it stays coherent across sessions and does not reintroduce itself to people it has already met. Stored per person, subject to the same PII rules as §4.1, and never used as training data.

### 9.7 Cadence

Ingest continuous. Metrics recomputed weekly. Persona-spec proposals monthly or on threshold breach. Retrain quarterly or when accumulated preference pairs exceed K, whichever comes first.

### 9.8 Ingest pipeline

New text messages, emails, documents, and Claude Code sessions must flow into the persona continuously, without a human step and without a second code path.

- **Cursors, not rescans.** Every connector persists a durable watermark and resumes from it: iMessage by `ROWID`/date, mail by history ID, Claude Code transcripts by file mtime plus byte offset within each `.jsonl`, git by last commit SHA, Notes by modification date. Incremental runs are cheap enough to schedule daily.
- **One path for backfill and incremental.** The initial load and the nightly delta run the *same* connector code, differing only in where the cursor starts. A separate bulk-import path would drift from the incremental path and silently produce two differently-shaped corpora.
- **Idempotent upsert.** Records key on a stable source identifier, so a re-run, an overlapping window, or a crashed job re-executed produces no duplicates. Duplicates would quietly reweight the training distribution.
- **Same normalize → scrub path.** New turns are never fast-tracked around scrubbing. A secret that arrives tomorrow is as disqualifying as one that arrived last year.
- **Arrival routing.** New data lands first inside the CC3 quarantine window, where it serves as fresh eval material; as the window rolls forward it ages out into the trainable and retrievable corpus. New data therefore *tests* the current system before it *trains* the next one — which is the correct order, and free.
- **Scheduling.** A `launchd` job runs the delta daily, with a manual trigger available. Failures alert rather than silently skipping, since a connector that quietly stops is indistinguishable from a subject who stopped writing.
- **Propagation.** On arrival: L1 index updates immediately. Weekly metric recompute detects distribution shift. Monthly persona proposals and quarterly retrains consume what has aged out of quarantine (§9.7).

---

## 10. Error handling and failure modes

| Failure | Handling |
|---|---|
| Scrub detects an unclassifiable high-entropy string | Redact and flag. Never pass through. |
| Connector partially fails | Resume from checkpoint. Partial ingest is marked incomplete and excluded from eval splits until complete. |
| A stage misses its exit gate | The stage's mechanism is cut, not carried forward. Recorded with its measurements. |
| Claims ledger contradiction | Preserved as a finding; escalated to a T3 interview question. Never averaged away. |
| Twin asked something outside the fact KB | Abstains in the subject's own voice. Confident fabrication is an S5 failure and takes priority over voice fidelity. |
| Normalization produces a defective corpus already consumed downstream | Clean corpus builds are versioned and immutable; each persona version, index, and adapter records the corpus version it consumed. Rebuild and re-point rather than mutate in place. |
| Identity map or vault would be committed | Blocked by gitignore and a pre-commit hook; both are stage-0 deliverables. |

---

## 11. Decisions taken (defaults, revisable in the plan)

| Decision | Default | Reasoning |
|---|---|---|
| Project location | `~/dev/persona-twin` | Consistent with existing project layout |
| Substance model | Frontier hosted (Claude) on exportable-tier data only | Best reasoning; raw corpus never leaves the machine |
| Local base model | Qwen or Llama class, LoRA via MLX on Apple Silicon | **Verified 2026-09-07: 48 GB M5 Pro.** Comfortable for 7B–14B LoRA, 32B quantized feasible. No new hardware required |
| Corpus store | SQLite + FTS, plus a vector index for exemplars | Matches existing tooling; no server to operate |
| Pseudonymization | Deterministic, reversible only inside the vault | Enables thread coherence without exposing counterparties |
| Held-out split | Most recent 3 months, whole threads | Time-based split prevents adjacent-turn leakage |
| Judge | Human primary for S1; LLM judge secondary and persona-blind | Human judgment is the criterion; LLM judge provides throughput |

---

## 12. Sequencing decisions

Each has a default so that nothing here blocks implementation; each is revisable when the stage is reached.

1. **Connector build order in stage 1.** Default: iMessage first — largest T1 signal per unit of effort — then Claude Code transcripts, mail, docs, git, Notes.
2. **Stage 7 training sequence.** Default: SFT LoRA first, then DPO on top of it. SFT establishes voice; DPO removes assistant-register bleed. Running DPO alone against the stage-5 system is the fallback if SFT underperforms its gate.
3. **Twin exposure at stage 10.** Default: local-only, single user, no network exposure. Any third-party access is a separate decision requiring the disclosure mechanism (§4.3) to be verified first.

---

## 13. Generalization to other subjects

The end-state goal is a system that builds a personality for **any** subject, not only the initial one. Full multi-subject support is **post-MVP**. What is *not* deferred is the set of seams that make it possible, because those cost almost nothing now and are expensive to retrofit — the same argument as the contamination controls in §9.3.

### 13.1 Built now (cheap seams)

- **Subject ID is a first-class key** in every schema, path, and index. Vault, clean corpus, persona spec, exemplar index, claims ledger, eval sets, and learning ledger are all subject-scoped from the first commit.
- **No subject-specific logic in code.** Everything particular to a person lives in data — the persona spec, the config, the identity map. A hardcoded name, habit, or relationship anywhere in the codebase is a defect.
- **Connectors take a subject context** (credentials, paths, cursors) as a parameter rather than reading a global.
- **The eval harness is parameterized by subject**, so S1–S7 run unchanged against anyone with a corpus.
- **Prompts and templates reference the persona spec**, never a literal name or trait.

Enforcement: a lint check that fails the build if the initial subject's name or identifiers appear outside `config/` and `data/`.

### 13.2 Deferred (post-MVP)

- Multi-tenant isolation, authentication, and per-subject access control.
- A guided onboarding flow for a new subject — source connection, consent capture, extraction walkthrough.
- Adapter registry and shared base-model hosting across subjects.
- **Cold-start handling.** A new subject may have a fraction of the initial corpus. The system needs graceful degradation: persona from interview plus a thin corpus, with the eval harness reporting reduced confidence rather than silently producing a confident and wrong portrait.

### 13.3 Consent model

Modeling oneself and modeling another person are ethically different acts. Any subject other than the operator requires recorded informed consent before ingest begins. This is a hard gate on multi-subject support, not a policy to be written later.

**Parity of corpus.** Consent covers the *full* source set required to reach the same fidelity the operator's own twin achieves — messages, mail, documents, LLM conversation histories, transcripts. A subject is not silently half-modeled: the default authorization is the complete corpus, because a personality built from a polite subset is a portrait of the subject's public face, not of the subject.

**Source-enumerated, not blanket.** Parity is the default, not a trick. The consent record lists each source individually, with its volume and date range, because "my text messages" abstractly and *every message I have sent since 2014* concretely are not the same thing to the person agreeing. Any source may be excluded.

**Exclusions degrade, they do not silently shrink.** An excluded source reduces corpus coverage, and the eval harness reports the resulting confidence reduction rather than producing an equally confident thinner portrait. This is the same mechanism as cold start (§13.2) — partial consent and a small corpus are the same problem and share one code path.

**Counterparties are context, never targets.** Every corpus is two-sided: it contains people who never consented to anything. Their turns are retained only as the context that makes the subject's replies interpretable, are pseudonymized per C3, and are never modeled as personalities in their own right or used to answer questions about them.

**Ingest consent and deployment consent are separate.** Agreeing to be modeled is not agreeing that a twin will converse with one's colleagues, clients, or family. Third-party-facing deployment requires its own authorization, naming the audience.

**Revocation must be honorable.** Consent is withdrawable, and withdrawal has to be executable rather than aspirational: corpus and derived artifacts are deleted, and any adapter or persona version trained on the withdrawn data is retired. This requires end-to-end provenance from source record → training set → adapter version, which the learning ledger (§9.5) already records. Without that chain, revocation is a promise the system cannot keep.

### 13.4 Why this helps the MVP

Subject-scoping is not overhead paid for a future feature. It prevents the initial subject's particulars from leaking into code and prompts — the most common way a personality system becomes unmaintainable — and it makes the eval harness reusable, which is what allows the same measurements to be trusted across stages.

---

## 14. Planning scope

This spec covers the whole program. Note that the learning scaffolding of §9.3 and the generalization seams of §13.1 land in stages 0–2 rather than at stage 10, because CC1, CC2, and CC3 are unrecoverable if deferred. The **first implementation plan covers stages 0–2 only** (foundations, acquisition, eval harness). Stages 3+ are specified here at design resolution but are deliberately not planned in detail yet: their tasks depend on what the corpus and the baseline measurements actually show. Each subsequent stage gets its own plan, written when its predecessor clears its gate.
