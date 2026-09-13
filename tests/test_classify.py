from datetime import datetime, timezone
from types import SimpleNamespace
import pytest
from persona_twin.schema import Turn
from persona_twin.scrub.classify import (
    classify, is_exportable, CONFIDENTIAL_SOURCES, ClassifyError,
)


def _t(text, source="imessage"):
    return Turn(
        subject_id="alice",
        source=source,
        source_id="1",
        thread_id="c1",
        ts=datetime.now(timezone.utc),
        author_id="alice",
        is_subject=True,
        text=text,
        assisted=False,
    )


# --- Basic cases from the brief ---


def test_client_financial_detail_is_confidential():
    assert classify(_t("Client SSN 123-45-6789 owes 40k on the 1120S")) == "confidential"


def test_ein_is_confidential():
    assert classify(_t("their EIN is 12-3456789")) == "confidential"


def test_domain_expertise_is_open():
    t = _t("Section 179 lets you expense it up front instead of depreciating over five years")
    assert classify(t) == "open" and is_exportable(t)


def test_karbon_export_source_is_confidential():
    assert classify(_t("anything", source="karbon")) == "confidential"


def test_ordinary_chat_is_open():
    assert is_exportable(_t("running late, be there in 10"))


# --- Domain expertise that plausibly could trip a marker, and doesn't ---
# These deliberately include figures, dollar amounts, and form numbers so
# the "open" result actually exercises the boundary rather than trivially
# avoiding it.


def test_expertise_with_dollar_figures_stays_open():
    t = _t(
        "For a $50,000 Section 179 deduction on Form 4562, you'd typically "
        "see the write-off hit in year one instead of spread across 2026-2031."
    )
    assert classify(t) == "open" and is_exportable(t)


def test_expertise_explaining_depreciation_math_stays_open():
    t = _t(
        "If the asset cost $12,500 and has a 5 year life, straight-line "
        "depreciation is roughly $2,500 a year under MACRS."
    )
    assert classify(t) == "open" and is_exportable(t)


def test_expertise_mentioning_form_1120s_stays_open():
    t = _t("An 1120S is due March 15th; late filing runs about $220 per shareholder per month.")
    assert classify(t) == "open" and is_exportable(t)


# --- Source rule and content rule tested independently ---


def test_karbon_source_alone_is_confidential_even_with_no_content_markers():
    # No SSN/EIN/account-number text at all — the source rule alone must fire.
    t = _t("client called about scheduling", source="karbon")
    assert "karbon" in CONFIDENTIAL_SOURCES
    assert classify(t) == "confidential"


def test_non_karbon_source_with_marker_is_still_confidential():
    # Not a confidential source — the content rule alone must fire.
    t = _t("routing number for the escrow account is 021000021", source="imessage")
    assert classify(t) == "confidential"


def test_non_karbon_source_without_marker_is_open():
    # Neither rule should fire.
    t = _t("let's grab lunch tomorrow", source="imessage")
    assert classify(t) == "open"


# --- What the \b\d{9,17}\b account-number marker does to ordinary numbers ---
# Reported honestly per the task brief, not just asserted as "correct":
# a bare 9-17 digit contiguous run has no way to distinguish a real account
# number from a phone number written without separators or a unix
# timestamp. Both of the following are FALSE POSITIVES the marker produces.


def test_year_alone_does_not_trigger_marker():
    # 4 digits: well under the 9-digit floor.
    assert classify(_t("the 2026 return is due in April")) == "open"


def test_hyphenated_phone_number_does_not_trigger_marker():
    # Hyphens break the digit run into segments of 3/3/4, none >= 9.
    assert classify(_t("call me at 555-123-4567")) == "open"


def test_phone_number_without_separators_is_a_false_positive():
    # 10 contiguous digits, no separators: falls inside 9-17 and IS flagged,
    # even though it's an ordinary phone number, not an account number.
    assert classify(_t("call me at 5551234567")) == "confidential"


def test_unix_timestamp_is_a_false_positive():
    # 10-digit unix timestamp, contiguous: also falls inside 9-17.
    assert classify(_t("event logged at 1735689600 in the audit trail")) == "confidential"


def test_comma_formatted_dollar_figure_does_not_trigger_marker():
    # Commas break the digit run; ordinary large dollar figures survive.
    assert classify(_t("the portfolio is worth $1,234,567 today")) == "open"


def test_large_bare_dollar_figure_is_a_false_positive():
    # Written without separators, a >= 9-digit dollar figure also trips the
    # marker even though it names no one's account.
    assert classify(_t("total assessed damages were $123456789 this year")) == "confidential"


def test_is_exportable_is_false_for_confidential_turns():
    """`is_exportable` must track `classify` on BOTH branches.

    Without this, replacing its body with `return True` passes the whole
    suite — every other call site pairs it with an "open" expectation.
    """
    for turn in (
        _t("Client SSN 123-45-6789 owes 40k on the 1120S"),
        _t("their EIN is 12-3456789"),
        _t("anything at all", source="karbon"),
    ):
        assert classify(turn) == "confidential"
        assert not is_exportable(turn)


def test_separated_identifier_formats_are_confidential():
    """The same identifier written with spaces or dots, not just hyphens."""
    for text in (
        "ssn 123 45 6789",
        "ssn 123.45.6789",
        "ein 12 3456789",
        "acct 4093 8172 6354",
        "acct 4093-8172-6354",
    ):
        assert classify(_t(text)) == "confidential", text


# --- Fix round: six confirmed gaps from independent review, each isolated ---


def test_acct_abbreviation_with_hash_is_confidential():
    # The demonstrated gap: "acct" alone did not match the literal "account"
    # phrase marker, and 8 digits falls under the bare-digit-run floor, so
    # neither mechanism fired.
    assert classify(_t("acct #12345678")) == "confidential"


def test_a_slash_c_abbreviation_with_number_is_confidential():
    assert classify(_t("a/c number 12345678")) == "confidential"


def test_ordinary_act_reference_stays_open():
    # Bare "act" is deliberately excluded from the abbreviation set: this is
    # a tax practice's data, where "the Act" (Tax Cuts and Jobs Act, CARES
    # Act, "Act #115") is ordinary domain vocabulary, not an account
    # reference. Including it would trade the content false-negative for a
    # false-positive on real discussion of legislation.
    assert classify(_t("under the Tax Cuts and Jobs Act, bonus depreciation phases out")) == "open"


def test_nonbreaking_space_separated_ssn_is_confidential():
    # A non-breaking space (U+00A0), as pasted from a PDF or Word export,
    # renders identically to an ASCII space but is a different codepoint —
    # the matcher must not be fooled by it.
    assert classify(_t("SSN 123 45 6789")) == "confidential"


def test_nonbreaking_hyphen_separated_ssn_is_confidential():
    assert classify(_t("SSN 123‑45‑6789")) == "confidential"


def test_classify_does_not_mutate_stored_text():
    # The normalization used for matching must be a view, never applied to
    # the Turn's own text (which also can't happen in place -- Turn is
    # frozen -- but the contract is worth pinning explicitly).
    t = _t("SSN 123 45 6789")
    original = t.text
    classify(t)
    assert t.text == original


def test_ssn_wrapped_across_newline_is_confidential():
    # A hard line-wrap landing right after the separator must not hide the
    # identifier: the separator itself is still present, just followed by
    # incidental whitespace.
    assert classify(_t("SSN 123-45-\n6789")) == "confidential"


def test_eighteen_digit_run_is_confidential():
    # \b\d{9,17}\b could never match inside an 18+ digit run at all: a \b
    # requires a word/non-word transition, and no interior position of a
    # longer contiguous run qualifies. The floor stays at 9; only the
    # (previously unintended) ceiling is removed.
    assert classify(_t("wire to account 123456789012345678 today")) == "confidential"


def test_non_string_text_raises_classify_error():
    # Mirrors secrets.py's redact(): fail loudly rather than let an
    # unexpected type make every marker's .search() silently a no-op.
    t = SimpleNamespace(text=None, source="imessage")
    with pytest.raises(ClassifyError):
        classify(t)


def test_routing_number_phrase_alone_is_confidential():
    # No digit-shape marker qualifies here at all -- this isolates the
    # routing-number phrase marker, which the existing suite never did
    # (its one "marker" fixture also contained a qualifying bare digit run).
    assert classify(_t("please confirm the routing number before we wire")) == "confidential"


def test_account_number_phrase_alone_is_confidential():
    # Same isolation for the account-number phrase marker: no digits at all.
    assert classify(_t("what's the account number for this client")) == "confidential"
