from datetime import datetime, timezone
from persona_twin.schema import Turn
from persona_twin.scrub.classify import classify, is_exportable, CONFIDENTIAL_SOURCES


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
