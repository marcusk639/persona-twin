# Threat model

## Assets

- **Vault** — raw personal corpus: iMessage, mail, transcripts, notes. Highest value.
- **Identity map** — reverses pseudonymization. Compromise re-identifies every counterparty.
- **Client-confidential material** — CPA client financial data. Regulated; must never reach a hosted provider.
- **Credentials in transcripts** — API keys and tokens pasted into Claude Code sessions.

## Adversaries and exposures

| Exposure                                               | Mitigation                                                                                                                                                                                            |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Repo pushed with data included                         | `data/` gitignored; verified by test in Task 1                                                                                                                                                        |
| Corpus sent to hosted provider                         | Only `exportable/` tier leaves the machine, per-artifact sign-off (§4.2)                                                                                                                              |
| Client-confidential material reaches a hosted provider | Classified at normalization and excluded from the corpus entirely; never eligible for the `exportable/` tier regardless of scrubbing (spec C1, implemented in the Task 16 confidentiality classifier) |
| Secret leaked in a training set                        | Deny-by-default scanning, fail-closed (C2/C4, Task 15)                                                                                                                                                |
| Counterparty re-identified                             | Pseudonymization; identity map never leaves the vault (C3, Task 14)                                                                                                                                   |
| Laptop theft                                           | FileVault full-disk encryption (verified in Task 1)                                                                                                                                                   |
| Twin impersonates the subject                          | Disclosure on first contact (§4.3, docs/disclosure-policy.md)                                                                                                                                         |

## Out of scope

Nation-state adversaries, physical coercion, and macOS platform compromise.
