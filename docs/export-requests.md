# Export requests (spec §5.3)

File these on day one; they take hours to days to arrive and gate nothing else.

- [x] **ChatGPT** — Settings → Data Controls → Export data. Arrives by email as a zip containing `conversations.json`.
- [x] **Claude.ai** — arrived 2026-09-12. Filed in `data/subjects/subject-01/vault/exports/claude-ai/`.
      The emailed manifest JSON (now `manifest.json` alongside the zips) is a **manifest**,
      not the data: 4 single-use
      `claude.ai` URLs (light_metadata, projects, memories, conversations). They are NOT presigned —
      no query signature — so `curl` gets a Cloudflare "Just a moment..." 403 at the edge and the
      token is not spent. Open them in a logged-in browser (`open <url>`) instead.
- [x] **Perplexity** — Settings → Account → download threads if offered; otherwise skip and rely on the T2 harvest.
- [x] **Google Takeout** — select Mail (mbox) and Drive. Large; request early.
- [x] **Microsoft 365** — mail export via Outlook, plus Teams meeting transcripts if retained.
- [x] **X / Reddit / LinkedIn** — archive requests, if those accounts are used.

Record arrival dates here. Exports should be placed in `data/subjects/<id>/vault/exports/`.

## Request log

- **2026-09-12** — all export requests filed by the operator; awaiting delivery. Exports land in `data/subjects/<id>/vault/exports/` (created on first arrival).
