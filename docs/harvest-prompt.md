# T2 harvest prompt (spec §5.4)

Run this once per model (ChatGPT, Claude, Perplexity), then AGAIN in a fresh
session for the same model. Save each as JSON. Do not edit between runs.

---

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
