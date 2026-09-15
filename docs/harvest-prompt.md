# T2 harvest prompt (spec §5.4)

Run this once per model (ChatGPT, Claude, Perplexity), then AGAIN in a fresh
session for the same model. Save each as JSON. Do not edit between runs.

Record which prompt version produced each report — claims are only comparable
across runs of the same version.

- **v1** — produced `chatgpt-run1.json`. Superseded; see "Why v1 failed" below.
- **v2** — current.

## v2 (current)

---

Use only what you can actually retrieve from our conversation history. Do not
reconstruct, infer from how people like me usually write, or fill gaps with
what is probably true. If it isn't in something I actually said, it doesn't go
in the list.

First, before any claims, tell me in two or three sentences what you can
actually retrieve: roughly how many of my messages, over what period, and on
what topics. If that pool is thin, say so plainly. Three grounded claims are
worth more to me than fifteen plausible ones, and zero is a legitimate answer.

Then give me falsifiable behavioural claims about how I communicate — the kind
a reader holding my full message archive could check and find WRONG.

Each claim must take the form: **in situation X, I do Y.** Both the situation
and the behaviour have to be things you could point at in a message. "I am
direct" is not a claim. "When I disagree with a recommendation, I restate it in
my own words before rejecting it" is.

Probes — answer only the ones you have real evidence for, and skip the rest.
Skipping is the expected outcome for most of them:

- How do I open a message to someone who has annoyed me?
- What do I do when I don't know something?
- What are my tells that I had already decided before I asked?
- What do I never say?
- How do I disagree with someone I need something from?
- What do I do when I turn out to be wrong?
- How do I end a conversation I want out of?
- What makes me go from short messages to long ones?

Rules:

1. **No hedges.** "sometimes", "often", "tends to", "can be", "may", "in some
   contexts" — any claim carrying one of these is unfalsifiable. Drop it rather
   than soften it. If the behaviour is genuinely conditional, put the condition
   in situation X where it can be checked.
2. **Verbatim quotes only.** Copy my words exactly: no ellipsis, no trimming,
   no tidied typos, no paraphrase, no stitching two messages together. If you
   cannot reproduce the words exactly as I typed them, you do not have the
   evidence — drop the claim.
3. **State the falsifier**: a specific thing that, if found in my archive,
   proves the claim wrong.
4. **State the disconfirming case**: something in my real retrieved history
   that argues against the claim. It must be a *different* message than the
   evidence quote, and it must not be the evidence quote for another claim in
   this same list. If there genuinely isn't one, write "none found in
   retrieved history" — do not invent one, and do not recycle another claim.
5. **Show the sample.** For each claim, how many distinct conversations it
   draws on, and roughly when they are from.
6. **Confidence rates retrieval, not plausibility.** high = several independent
   conversations. medium = a couple. low = a single message. A claim that feels
   obviously true but rests on one message is low.
7. **No quota.** As many claims as the evidence supports and no more. There is
   no target number, and a short list is not a worse answer. If your retrievable
   history is too thin to ground anything, say so and return `[]`.
8. **Critical, not flattering** — but do not manufacture flaws to look
   balanced. An invented weakness is as useless as an invented strength.

Return the retrieval summary as plain prose, then the claims as a JSON array of
objects with keys:

`text`, `evidence_quote`, `falsifier`, `disconfirming_case`,
`evidence_conversations`, `source_confidence`

---

## Why v1 failed

Kept for provenance — this is the prompt that produced `chatgpt-run1.json`.

Observed failure modes, all traceable to the prompt rather than the model:

- Asking about "how I communicate" in the abstract invites style adjectives,
  and an adjective can only be hedged. All five returned claims were
  "sometimes"/"can"/"may" and none could fail.
- "Verbatim quote" without "no ellipsis" let an elided quote through, which
  cannot be anchored against the vault.
- Requiring a disconfirming case without requiring it be a *distinct* message
  produced a set where claims 1, 3 and 4 disconfirmed each other — one finding
  split four ways.
- The abstention clause was present but unattractive; five thin claims were
  returned where the honest answer was one or none.

### v1 text

Based only on what you can actually retrieve from our conversation history,
give me falsifiable behavioural claims about how I communicate.

Rules:

- Give as many claims as the evidence supports, and no more. If you have
  little retrievable history, say so and return an empty list. Do not fill a quota.
- Every claim needs a verbatim quote from me as evidence. No quote, no claim.
- Rate your confidence high / medium / low.
- For each claim, state what in my history argues AGAINST it.
- Be objective and critical rather than flattering. Do not invent flaws to
  appear balanced; an invented weakness is as useless as an invented strength.

Return JSON: a list of objects with keys
text, evidence_quote, source_confidence, disconfirming_case
