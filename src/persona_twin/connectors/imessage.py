"""iMessage connector (spec §5.6): the subject's unguarded conversational voice.

Reads a read-only snapshot of `chat.db` -- never the live Messages.app database.
`message.text` is NULL for a meaningful fraction of rows on modern macOS; those
rows are recovered via `decode_attributed_body` and only skipped when both are
empty. Both sides of every conversation are emitted (see `is_from_me` in the
payload) -- the subject's replies are meaningless without what they're replying
to. Counterparty identities are pseudonymized later, in a downstream stage, not
here.
"""
from __future__ import annotations
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator
from persona_twin.connectors.base import SubjectContext
from persona_twin.imessage.typedstream import decode_attributed_body
from persona_twin.schema import RawEnvelope

APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)

_SQL = """
select m.ROWID, m.text, m.attributedBody, m.is_from_me, m.date,
       coalesce(h.id, 'unknown') as handle,
       coalesce(c.guid, 'unknown') as chat_guid
from message m
left join handle h on h.ROWID = m.handle_id
left join chat_message_join cmj on cmj.message_id = m.ROWID
left join chat c on c.ROWID = cmj.chat_id
where m.ROWID > ?
group by m.ROWID
order by m.ROWID
"""
# `group by m.ROWID` guards against fan-out: a message can have more than one
# chat_message_join row (macOS chat merge/split), which would otherwise
# duplicate the LEFT JOIN output and emit two envelopes sharing one
# source_id, violating the uniqueness-within-a-run contract. This did not
# occur on the real snapshot (77,508 emitted, 77,508 distinct ids) but
# nothing in the schema prevents it, so it is guarded structurally rather
# than left to be "empirically absent." The chat_guid picked for a fanned-out
# row is deterministic (SQLite's default aggregate-free GROUP BY resolves
# ties to the first row in scan order) but arbitrary among the candidates.

def _to_utc(apple_ts: int | None) -> datetime:
    """Apple epoch (2001-01-01 UTC), nanoseconds on modern macOS, seconds on
    older databases. `message.date` values switch magnitude between the two
    -- a wrong guess here silently places messages decades away, corrupting
    thread reconstruction and the held-out time split without raising an
    error, so the threshold is deliberately conservative (10**12 is ~31,700
    years in seconds, far past any real nanosecond value's seconds-only
    misinterpretation).
    """
    if not apple_ts:
        return APPLE_EPOCH
    seconds = apple_ts / 1_000_000_000 if apple_ts > 10**12 else float(apple_ts)
    return APPLE_EPOCH + timedelta(seconds=seconds)

class IMessageConnector:
    name = "imessage"

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    def fetch(self, ctx: SubjectContext,
              cursor: str | None) -> Iterator[tuple[RawEnvelope, str]]:
        # source_id is the raw ROWID (spec lesson: hashing text loses ~4.4%
        # of rows to collisions in the claude_code connector's history).
        # ROWID is stable across runs on an append-only table and unique
        # within a run, which is exactly what run_connector's dedup key
        # needs.
        last = int(cursor) if cursor else 0
        uri = f"file:{self.db_path}?mode=ro"
        con = sqlite3.connect(uri, uri=True)
        try:
            for rid, text, blob, from_me, date, handle, chat_guid in con.execute(_SQL, (last,)):
                body = text if (text and text.strip()) else decode_attributed_body(blob)
                if not body or not body.strip():
                    continue
                yield RawEnvelope(
                    subject_id=ctx.config.subject_id, source=self.name,
                    source_id=str(rid), ts=_to_utc(date),
                    payload={"text": body, "is_from_me": bool(from_me),
                             "handle": handle, "chat_guid": chat_guid},
                    ingested_at=datetime.now(timezone.utc)), str(rid)
        finally:
            con.close()
