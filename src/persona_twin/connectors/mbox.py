"""Mail connector (spec §5.4/§4.4 M1): the subject's considered written voice.

Reads mbox exports (Gmail Takeout / Outlook). Sent mail is more deliberate
than text messages and less formal than documents, which makes it a distinct
register worth its own source. Only mail the subject SENT is emitted --
received mail is someone else's voice, not the subject's (contrast with the
iMessage connector, which deliberately keeps both sides because the
counterparty's messages supply reply-pair context; that context isn't needed
here). Quoted reply chains and trailing signatures are stripped via
`strip_quoted` so bodies read as considered writing, not quoted noise.
"""
from __future__ import annotations
import json, mailbox, re, sys
from datetime import datetime, timezone
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path
from typing import Iterator
from persona_twin.connectors.base import SubjectContext
from persona_twin.schema import RawEnvelope

_QUOTE_HEAD = re.compile(r"^On .*wrote:\s*$", re.M)
_SIG = re.compile(r"^--\s*$", re.M)


def strip_quoted(body: str) -> str:
    """Drop the quoted reply chain and trailing signature (spec §4.4 M1)."""
    for pattern in (_QUOTE_HEAD, _SIG):
        m = pattern.search(body)
        if m:
            body = body[: m.start()]
    lines = [ln for ln in body.splitlines() if not ln.lstrip().startswith(">")]
    return "\n".join(lines).strip()


def _plain_body(msg) -> str:
    """Extract the text/plain part, decoding defensively.

    Real exports carry arbitrary charsets, and the declared charset
    sometimes doesn't match the actual bytes. `errors="ignore"` and a
    fallback to utf-8 on an unknown/unsupported codec name keep one
    mis-labeled message from raising and aborting the whole run.
    """
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True) or b""
                return _decode(payload, part.get_content_charset())
        return ""
    payload = msg.get_payload(decode=True)
    if payload is None:
        # get_payload(decode=True) returns None when the message has no
        # decodable body at all (e.g. a bare Content-Type with no
        # transfer-encoded payload); fall back to whatever get_payload()
        # gives us rather than crashing on a NoneType.
        return msg.get_payload() or ""
    return _decode(payload, msg.get_content_charset())


def _decode(payload: bytes, charset: str | None) -> str:
    try:
        return payload.decode(charset or "utf-8", "ignore")
    except (LookupError, TypeError):
        # Unknown/bogus charset name, or a charset value that isn't a str
        # at all -- both happen on real-world exports.
        return payload.decode("utf-8", "ignore")


class MboxConnector:
    name = "mail"

    def __init__(self, mbox_paths: list[Path], subject_addresses: list[str]) -> None:
        self.paths = [Path(p) for p in mbox_paths]
        self.addresses = {a.lower() for a in subject_addresses}

    def _is_from_subject(self, msg) -> bool:
        # parseaddr extracts the bare address, so a display name or a
        # different mailbox that merely contains the subject's address as
        # a substring (e.g. "not-alice@example.com") doesn't false-match.
        _, addr = parseaddr(msg.get("From") or "")
        return addr.lower() in self.addresses

    def fetch(self, ctx: SubjectContext,
              cursor: str | None) -> Iterator[tuple[RawEnvelope, str]]:
        """Cursor is a JSON map {mbox_path: last_index_seen}.

        Fragility (documented, not handled): mbox message *order* is
        stable only for a given export file. If Gmail Takeout regenerates
        the export, the same email can land at a different index -- so a
        fresh Takeout can cause messages to be re-ingested or, worse,
        skipped if the new file happens to be shorter at the cursor's
        recorded index. No detection or recovery for this is implemented.
        """
        seen: dict[str, int] = json.loads(cursor) if cursor else {}
        # Message-ID collisions happen in practice, and specifically
        # *across files* -- a thread exported into both "All Mail" and a
        # label export lands as two on-disk copies sharing one
        # Message-ID header. If two distinct messages mapped to the same
        # source_id, VaultWriter's insert-or-ignore would silently drop
        # the second one's content. seen_message_ids tracks which
        # Message-IDs have already claimed a source_id *anywhere in this
        # fetch() call* -- deliberately scoped to the whole call, not
        # per-path, since a per-file set would miss exactly the
        # cross-file case this exists for. A repeat falls back to the
        # path-qualified index, which is always unique because idx is
        # monotonic per file.
        #
        # Stability across separate runs/calls: this set is local to one
        # fetch() invocation and starts empty every time, so it does NOT
        # persist which Message-IDs were already claimed in a prior run.
        # Two consequences follow. (1) A message that is a byte-for-byte
        # re-export of one already ingested in an earlier run keeps the
        # same Message-ID and so still lands on the same source_id --
        # the vault dedups it correctly, no regression there. (2) A
        # message whose Message-ID collided with another *within* one
        # run and so fell back to `{path}:{idx}` will, if the exact same
        # duplicate-ID pair is re-exported into new files in a later run,
        # NOT necessarily reproduce the same fallback assignment -- this
        # is a narrower instance of the cursor-index fragility already
        # documented above and is not handled. Separately: the fallback
        # id itself (`{path}:{idx}`) is unaffected by the *order* of
        # `mbox_paths` -- it's keyed by the literal path string and the
        # message's own index within that file, not by which file was
        # processed first, so reordering `mbox_paths` between runs does
        # not shift any previously-assigned fallback id. Reordering can
        # only change which of two colliding files "wins" the bare
        # Message-ID when both are new in the *same* fetch() call, which
        # cannot happen for a message already past its file's cursor.
        seen_message_ids: set[str] = set()
        for path in self.paths:
            if not path.exists():
                # A missing mbox must not look like "zero new messages" --
                # that's indistinguishable from a fully-caught-up cursor
                # and would silently drop an entire export.
                raise FileNotFoundError(f"mbox connector: mbox file not found: {path}")
            try:
                box = mailbox.mbox(path)
            except Exception as exc:
                raise OSError(f"mbox connector: failed to open {path}: {exc}") from exc

            key = str(path)
            start = seen.get(key, -1)
            for idx, msg in enumerate(box):
                if idx <= start:
                    continue
                seen[key] = idx
                try:
                    if not self._is_from_subject(msg):
                        continue
                    text = strip_quoted(_plain_body(msg))
                    if not text:
                        continue
                    mid = msg.get("Message-ID")
                    if mid and mid not in seen_message_ids:
                        source_id = mid
                        seen_message_ids.add(mid)
                    else:
                        source_id = f"{key}:{idx}"
                    try:
                        ts = parsedate_to_datetime(msg.get("Date")).astimezone(timezone.utc)
                    except (TypeError, ValueError):
                        ts = datetime.now(timezone.utc)
                    yield RawEnvelope(
                        subject_id=ctx.config.subject_id, source=self.name,
                        source_id=source_id, ts=ts,
                        payload={"text": text, "subject": msg.get("Subject", ""),
                                 "to": msg.get("To", "")},
                        ingested_at=datetime.now(timezone.utc)), json.dumps(seen)
                except Exception as exc:
                    # A single malformed message (bad charset, missing/
                    # corrupt headers, broken MIME) must not abort the
                    # whole mbox -- warn loudly, on to the next message.
                    print(f"mbox connector: skipping unreadable message "
                          f"{key}:{idx}: {exc}", file=sys.stderr)
                    continue
