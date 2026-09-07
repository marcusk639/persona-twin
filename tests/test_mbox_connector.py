import email.message
import mailbox
import pytest
from persona_twin.config import SubjectConfig
from persona_twin.paths import SubjectPaths
from persona_twin.connectors.base import SubjectContext
from persona_twin.connectors.mbox import MboxConnector, strip_quoted

MBOX = """From alice@example.com Mon Jan  1 00:00:00 2024
From: alice@example.com
To: bob@example.com
Subject: Re: lunch
Date: Mon, 1 Jan 2024 10:00:00 +0000

Sounds good, see you at noon.

On Mon, Jan 1, 2024 at 9:00 AM Bob wrote:
> where should we meet?

--
Alice

From bob@example.com Mon Jan  1 00:00:00 2024
From: bob@example.com
To: alice@example.com
Subject: hi
Date: Mon, 1 Jan 2024 09:00:00 +0000

where should we meet?
"""


def _ctx(tmp_path):
    p = SubjectPaths("alice", tmp_path)
    p.ensure()
    return SubjectContext(config=SubjectConfig(subject_id="alice", display_name="A"), paths=p)


def _write_mbox(path, messages):
    """Build a real mbox file via the stdlib so envelope framing (including
    "From "-line escaping inside bodies) matches what a real export does,
    rather than relying on hand-typed separators for every fixture."""
    box = mailbox.mbox(str(path))
    box.lock()
    try:
        for msg in messages:
            box.add(msg)
        box.flush()
    finally:
        box.unlock()
        box.close()


def _msg(from_addr, subject, body, *, to="bob@example.com",
         date="Mon, 1 Jan 2024 10:00:00 +0000", message_id=None):
    msg = email.message.Message()
    msg["From"] = from_addr
    msg["To"] = to
    msg["Subject"] = subject
    msg["Date"] = date
    if message_id is not None:
        msg["Message-ID"] = message_id
    msg.set_payload(body)
    return msg


def test_strip_quoted_removes_reply_chain_and_signature():
    body = "Sounds good.\n\nOn Mon Bob wrote:\n> hi\n\n--\nAlice\n"
    assert strip_quoted(body) == "Sounds good."


def test_strip_quoted_handles_dash_dash_space_signature_delimiter():
    # "-- " (trailing space) is the RFC 3676 / common-client signature
    # delimiter convention, distinct from a bare "--".
    body = "Thanks!\n\n-- \nAlice\nSent from my phone\n"
    assert strip_quoted(body) == "Thanks!"


def test_strip_quoted_with_no_reply_chain_or_signature_is_unchanged():
    assert strip_quoted("Just a plain note, nothing to strip.") == \
        "Just a plain note, nothing to strip."


def test_only_subject_sent_mail_is_emitted(tmp_path):
    p = tmp_path / "m.mbox"
    p.write_text(MBOX)
    c = MboxConnector([p], ["alice@example.com"])
    texts = [e.payload["text"] for e, _ in c.fetch(_ctx(tmp_path), None)]
    assert texts == ["Sounds good, see you at noon."]


def test_resume_yields_nothing(tmp_path):
    p = tmp_path / "m.mbox"
    p.write_text(MBOX)
    c = MboxConnector([p], ["alice@example.com"])
    ctx = _ctx(tmp_path)
    cur = [x for _, x in c.fetch(ctx, None)][-1]
    assert list(c.fetch(ctx, cur)) == []


def test_from_header_substring_match_is_not_a_false_positive(tmp_path):
    # A naive `subject_address in from_header` check would wrongly treat
    # "not-alice@example.com" as sent by "alice@example.com". parseaddr
    # must be used to compare the exact mailbox.
    msgs = [
        _msg("not-alice@example.com", "decoy", "should not be emitted"),
        _msg("Alice <alice@example.com>", "real", "should be emitted"),
    ]
    p = tmp_path / "m.mbox"
    _write_mbox(p, msgs)
    texts = [e.payload["text"] for e, _ in MboxConnector([p], ["alice@example.com"]).fetch(_ctx(tmp_path), None)]
    assert texts == ["should be emitted"]


def test_distinct_source_ids_when_message_id_missing_or_duplicated(tmp_path):
    # Two risks call out explicitly: a message with no Message-ID at all,
    # and two distinct messages sharing the same Message-ID (happens when
    # a thread is exported into both "All Mail" and a label export). In
    # neither case may two distinct messages collapse onto one source_id --
    # the vault's insert-or-ignore would silently drop the second one.
    msgs = [
        _msg("alice@example.com", "no id", "first message body"),
        _msg("alice@example.com", "dup 1", "second message body", message_id="<dup@example.com>"),
        _msg("alice@example.com", "dup 2", "third message body", message_id="<dup@example.com>"),
    ]
    p = tmp_path / "m.mbox"
    _write_mbox(p, msgs)
    envs = [e for e, _ in MboxConnector([p], ["alice@example.com"]).fetch(_ctx(tmp_path), None)]
    ids = [e.source_id for e in envs]
    assert len(ids) == 3
    assert len(set(ids)) == 3
    assert [e.payload["text"] for e in envs] == [
        "first message body", "second message body", "third message body",
    ]


def test_vault_does_not_swallow_distinct_messages_sharing_message_id(tmp_path):
    # End-to-end version of the above: prove VaultWriter actually persists
    # all three envelopes rather than "insert or ignore"-ing the collision
    # away, which is the concrete failure mode the source_id design guards
    # against.
    from persona_twin.vault import VaultWriter

    msgs = [
        _msg("alice@example.com", "dup 1", "first body", message_id="<dup@example.com>"),
        _msg("alice@example.com", "dup 2", "second body", message_id="<dup@example.com>"),
    ]
    p = tmp_path / "m.mbox"
    _write_mbox(p, msgs)
    ctx = _ctx(tmp_path)
    writer = VaultWriter(ctx.paths)
    results = [writer.write(env) for env, _ in MboxConnector([p], ["alice@example.com"]).fetch(ctx, None)]
    assert results == [True, True]
    assert writer.count("mail") == 2


def test_missing_mbox_file_raises_instead_of_yielding_silently(tmp_path):
    missing = tmp_path / "does-not-exist.mbox"
    c = MboxConnector([missing], ["alice@example.com"])
    with pytest.raises(FileNotFoundError):
        list(c.fetch(_ctx(tmp_path), None))


def test_single_unreadable_message_is_skipped_not_fatal(tmp_path, monkeypatch, capsys):
    # One malformed message must not abort ingestion of the rest of the
    # mbox. Force strip_quoted to blow up for exactly one message body and
    # verify (a) the other message still comes through and (b) the failure
    # is surfaced loudly (stderr), not swallowed.
    import persona_twin.connectors.mbox as mbox_mod

    real_strip_quoted = mbox_mod.strip_quoted

    def flaky(body):
        if body.strip() == "BOOM":
            raise ValueError("simulated corrupt message")
        return real_strip_quoted(body)

    monkeypatch.setattr(mbox_mod, "strip_quoted", flaky)

    msgs = [
        _msg("alice@example.com", "bad", "BOOM"),
        _msg("alice@example.com", "good", "this one is fine"),
    ]
    p = tmp_path / "m.mbox"
    _write_mbox(p, msgs)
    texts = [e.payload["text"] for e, _ in MboxConnector([p], ["alice@example.com"]).fetch(_ctx(tmp_path), None)]
    assert texts == ["this one is fine"]
    err = capsys.readouterr().err
    assert "skipping unreadable message" in err
    assert "simulated corrupt message" in err


def test_multipart_message_uses_text_plain_part(tmp_path):
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    outer = MIMEMultipart("alternative")
    outer["From"] = "alice@example.com"
    outer["To"] = "bob@example.com"
    outer["Subject"] = "multipart"
    outer["Date"] = "Mon, 1 Jan 2024 10:00:00 +0000"
    outer.attach(MIMEText("<b>html body</b>", "html"))
    outer.attach(MIMEText("plain text body", "plain"))

    p = tmp_path / "m.mbox"
    _write_mbox(p, [outer])
    texts = [e.payload["text"] for e, _ in MboxConnector([p], ["alice@example.com"]).fetch(_ctx(tmp_path), None)]
    assert texts == ["plain text body"]


def test_multipart_message_with_no_text_plain_part_is_skipped(tmp_path):
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    outer = MIMEMultipart("alternative")
    outer["From"] = "alice@example.com"
    outer["To"] = "bob@example.com"
    outer["Subject"] = "html only"
    outer["Date"] = "Mon, 1 Jan 2024 10:00:00 +0000"
    outer.attach(MIMEText("<b>html only body</b>", "html"))

    p = tmp_path / "m.mbox"
    _write_mbox(p, [outer])
    assert list(MboxConnector([p], ["alice@example.com"]).fetch(_ctx(tmp_path), None)) == []


def test_unknown_charset_label_does_not_raise(tmp_path):
    # A declared charset that isn't a real codec name (garbled export
    # metadata) must fall back to utf-8 decoding rather than raising
    # LookupError and killing the whole run.
    msg = email.message.Message()
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Subject"] = "bad charset"
    msg["Date"] = "Mon, 1 Jan 2024 10:00:00 +0000"
    msg["Content-Type"] = 'text/plain; charset="x-not-a-real-charset"'
    msg["Content-Transfer-Encoding"] = "8bit"
    msg.set_payload(b"still readable text", charset=None)

    p = tmp_path / "m.mbox"
    _write_mbox(p, [msg])
    texts = [e.payload["text"] for e, _ in MboxConnector([p], ["alice@example.com"]).fetch(_ctx(tmp_path), None)]
    assert texts == ["still readable text"]


def test_missing_date_header_still_yields_with_fallback_timestamp(tmp_path):
    msg = email.message.Message()
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Subject"] = "no date"
    msg.set_payload("body with no date header")

    p = tmp_path / "m.mbox"
    _write_mbox(p, [msg])
    envs = [e for e, _ in MboxConnector([p], ["alice@example.com"]).fetch(_ctx(tmp_path), None)]
    assert len(envs) == 1
    # `envs[0].ts is not None` alone would still pass for a regression to
    # a naive `datetime.now()` (no tzinfo) -- exactly the branch a bare
    # `now()` is tempting on, and exactly what schema.py's frozen
    # RawEnvelope forbids. Assert timezone-awareness and UTC explicitly.
    assert envs[0].ts.tzinfo is not None
    assert envs[0].ts.utcoffset().total_seconds() == 0


def test_multiple_mbox_files_tracked_independently_in_cursor(tmp_path):
    p1 = tmp_path / "a.mbox"
    p2 = tmp_path / "b.mbox"
    _write_mbox(p1, [_msg("alice@example.com", "one", "from file one")])
    _write_mbox(p2, [_msg("alice@example.com", "two", "from file two")])
    c = MboxConnector([p1, p2], ["alice@example.com"])
    ctx = _ctx(tmp_path)
    texts = [e.payload["text"] for e, _ in c.fetch(ctx, None)]
    assert set(texts) == {"from file one", "from file two"}
    assert list(c.fetch(ctx, [x for _, x in c.fetch(ctx, None)][-1])) == []


def test_distinct_source_ids_when_message_id_duplicated_across_files(tmp_path):
    # Fix round 1, item 1: the collision guard must span the whole
    # fetch() call, not reset per-file. This is the actual motivating
    # scenario -- a thread exported into both "All Mail" and a label
    # export -- and it is cross-file by construction. A per-file guard
    # (the round-1 bug) lets each file's first occurrence of the shared
    # Message-ID claim it independently, so two DIFFERENT bodies collide
    # on one source_id and the second is silently dropped by the vault's
    # insert-or-ignore. Fails if seen_message_ids is re-initialized
    # inside the `for path in self.paths` loop.
    p1 = tmp_path / "all_mail.mbox"
    p2 = tmp_path / "label_export.mbox"
    _write_mbox(p1, [_msg("alice@example.com", "dup", "body from All Mail export",
                          message_id="<shared@x>")])
    _write_mbox(p2, [_msg("alice@example.com", "dup", "DIFFERENT body from label export",
                          message_id="<shared@x>")])
    c = MboxConnector([p1, p2], ["alice@example.com"])
    envs = [e for e, _ in c.fetch(_ctx(tmp_path), None)]
    ids = [e.source_id for e in envs]
    assert len(ids) == 2
    assert len(set(ids)) == 2
    assert [e.payload["text"] for e in envs] == [
        "body from All Mail export", "DIFFERENT body from label export",
    ]


def test_vault_does_not_swallow_cross_file_message_id_collision(tmp_path):
    # End-to-end version of the above, proving VaultWriter actually
    # persists both envelopes.
    from persona_twin.vault import VaultWriter

    p1 = tmp_path / "all_mail.mbox"
    p2 = tmp_path / "label_export.mbox"
    _write_mbox(p1, [_msg("alice@example.com", "dup", "first body", message_id="<shared@x>")])
    _write_mbox(p2, [_msg("alice@example.com", "dup", "second body", message_id="<shared@x>")])
    ctx = _ctx(tmp_path)
    writer = VaultWriter(ctx.paths)
    c = MboxConnector([p1, p2], ["alice@example.com"])
    results = [writer.write(env) for env, _ in c.fetch(ctx, None)]
    assert results == [True, True]
    assert writer.count("mail") == 2


def test_mbox_path_that_is_a_directory_raises_instead_of_yielding_silently(tmp_path):
    # Fix round 1, item 3: a path that exists but can't be opened as an
    # mbox (e.g. it's a directory) must raise, not look like an empty
    # inbox.
    not_a_file = tmp_path / "looks_like_mbox_but_is_a_dir"
    not_a_file.mkdir()
    c = MboxConnector([not_a_file], ["alice@example.com"])
    with pytest.raises(OSError):
        list(c.fetch(_ctx(tmp_path), None))


def test_multiple_subject_addresses_all_emit_and_others_do_not(tmp_path):
    # Fix round 1, item 4: subject_addresses is a list of aliases (e.g.
    # a work address and a personal address); mail from any of them
    # counts as sent-by-subject, mail from an address not in the list
    # does not.
    msgs = [
        _msg("alice@work.example.com", "work", "sent from work alias"),
        _msg("alice@home.example.com", "home", "sent from home alias"),
        _msg("someone.else@example.com", "other", "not the subject"),
    ]
    p = tmp_path / "m.mbox"
    _write_mbox(p, msgs)
    c = MboxConnector([p], ["alice@work.example.com", "alice@home.example.com"])
    texts = [e.payload["text"] for e, _ in c.fetch(_ctx(tmp_path), None)]
    assert texts == ["sent from work alias", "sent from home alias"]


def test_message_id_collision_fallback_is_independent_of_mbox_paths_order(tmp_path):
    # Fix round 2: when two files new in the same fetch() call share a
    # Message-ID, whichever is scanned first keeps the bare Message-ID
    # and the other falls back to `{path}:{idx}`. If that scan order
    # depended on the order the caller happened to pass `mbox_paths` in,
    # flipping the order would swap which message gets which id between
    # runs -- and since `source_id` is the vault's dedup key, a swap
    # duplicates content rather than merely renaming it: whichever
    # message's id changes gets inserted as "new" under an id the vault
    # has never seen, while its original copy (under the id that didn't
    # change) is left in place too. Asserts the *same message* gets the
    # *same id* regardless of the order `mbox_paths` is given in --
    # fails without sorting `self.paths` in fetch().
    p1 = tmp_path / "all_mail.mbox"
    p2 = tmp_path / "label_export.mbox"
    _write_mbox(p1, [_msg("alice@example.com", "dup", "body one", message_id="<shared@x>")])
    _write_mbox(p2, [_msg("alice@example.com", "dup", "body two", message_id="<shared@x>")])

    forward = {e.payload["text"]: e.source_id
               for e, _ in MboxConnector([p1, p2], ["alice@example.com"]).fetch(_ctx(tmp_path), None)}
    reversed_ = {e.payload["text"]: e.source_id
                 for e, _ in MboxConnector([p2, p1], ["alice@example.com"]).fetch(_ctx(tmp_path), None)}

    assert forward == reversed_
    assert len(set(forward.values())) == 2
