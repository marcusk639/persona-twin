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


# --- Round 2: independent review found three of these are siblings of gaps
# round 1 already closed on a different axis. These tests target the FAMILY
# each fix closes, not just the reviewer's demonstrated instance. ---


# Gap 7 (HIGH): \b\d{9,}\b cannot match a digit run adjacent to ANY word
# character (a letter or underscore is also \w, so a \b never forms there).


def test_bare_digit_run_confidential_when_prefixed_by_a_letter():
    assert classify(_t("ID123456789 needs review")) == "confidential"


def test_bare_digit_run_confidential_when_suffixed_by_a_letter():
    assert classify(_t("reference 123456789Z on file")) == "confidential"


def test_bare_digit_run_confidential_when_joined_by_underscore():
    # Underscore is a word character too -- the same escape, a different
    # adjoining character.
    assert classify(_t("field acct_123456789 in the export")) == "confidential"


def test_bare_digit_run_confidential_when_embedded_in_a_longer_alnum_token():
    assert classify(_t("token x1234567890123456 expired")) == "confidential"


def test_bare_digit_run_confidential_when_glued_to_lowercase_label():
    assert classify(_t("ssn123456789 typo'd with no space")) == "confidential"


# Gap 8 (MEDIUM, latent): every Unicode Pd (Dash Punctuation) character folds
# to ASCII '-', every Cf (Format) character is dropped, and the one dash
# lookalike outside Pd (MINUS SIGN, category Sm) is named explicitly. Tested
# with a mix of the reviewer's examples and DIFFERENT category members the
# reviewer did not name, to demonstrate this is a category sweep and not a
# longer copy of the same list.


def test_en_dash_separated_ssn_is_confidential():
    assert classify(_t("SSN 123–45–6789")) == "confidential"


def test_figure_dash_separated_ssn_is_confidential():
    assert classify(_t("SSN 123‒45‒6789")) == "confidential"


def test_horizontal_bar_separated_ssn_is_confidential():
    # U+2015 HORIZONTAL BAR: Pd, like the dashes above, but not one the
    # review named -- this is what pins the fix to the category rather than
    # to a longer enumeration of the same handful of characters.
    assert classify(_t("SSN 123―45―6789")) == "confidential"


def test_minus_sign_separated_ssn_is_confidential():
    # U+2212 MINUS SIGN: category Sm, not Pd -- the one deliberate named
    # exception, since Sm also contains '+', '=', '<', '>' and sweeping the
    # whole category would be wrong.
    assert classify(_t("SSN 123−45−6789")) == "confidential"


def test_soft_hyphen_in_ssn_is_confidential():
    # U+00AD SOFT HYPHEN: category Cf, dropped rather than folded to '-'
    # because it renders as invisible, not as a visible separator.
    assert classify(_t("SSN 123\xad45\xad6789")) == "confidential"


def test_zero_width_non_joiner_in_ssn_is_confidential():
    # U+200C ZERO WIDTH NON-JOINER: Cf, like ZWSP/BOM/word joiner, but not
    # one the review named -- the Cf-category counterpart to the Pd proof
    # above.
    assert classify(_t("SSN 123‌45‌6789")) == "confidential"


def test_plus_and_equals_are_not_folded_to_hyphen():
    # The Sm exception for MINUS SIGN must stay narrow: '+' and '=' are also
    # Sm and must never be treated as separators, or ordinary arithmetic in
    # domain-expertise text would start tripping the SSN/EIN shape markers.
    assert classify(_t("5+3=8 is not an identifier")) == "open"


# Gap 9 (MEDIUM-LOW): the qualifier after routing/account/acct/a-c is a
# family -- number, #, no[.], num[.], nbr[.] -- not just the two spellings
# the original phrase markers recognised.


def test_acct_no_dot_qualifier_is_confidential():
    assert classify(_t("acct no. 12345678")) == "confidential"


def test_account_bare_no_qualifier_is_confidential():
    assert classify(_t("account no 12345678")) == "confidential"


def test_routing_no_dot_qualifier_is_confidential():
    assert classify(_t("routing no. 12345678")) == "confidential"


def test_a_slash_c_bare_no_qualifier_is_confidential():
    assert classify(_t("a/c no 12345678")) == "confidential"


def test_account_nbr_qualifier_is_confidential():
    assert classify(_t("account nbr 12345678")) == "confidential"


def test_acct_num_qualifier_is_confidential():
    assert classify(_t("acct num 12345678")) == "confidential"


def test_account_no_longer_is_an_accepted_false_positive():
    """Reported honestly, like the bare-digit-run false positives above: a
    standalone "no" qualifier (per the review's own "account no 12345678"
    example, required to catch it) is followed by a SPACE here, then an
    unrelated word -- "no" as its own complete word cannot be distinguished
    from ordinary English "no longer" / "no more" without a digit nearby,
    and no trailing-letter guard can fix that (the guard only rejects "no"
    glued directly onto a following word with no space, see the
    prefix-swallowing tests below).

    Corrected attribution, on re-review: the original 2-of-85,330 corpus
    turns cited as this trade-off's cost were NOT this case -- they were
    "account notifications" and "account \\nnow", both prefix-swallowing
    (a letter glued directly onto "no" with no space), which the guard
    below now closes for free. This exact standalone-word case, the one
    this test actually documents, costs 0 additional turns in the real
    corpus today; the trade-off is accepted here as a matter of principle
    (a bare "no" this specific and this word-bounded could still appear in
    a larger or different corpus), not because it is currently paid."""
    assert classify(_t("the account no longer has any activity")) == "confidential"


# Gap 10 (LOW): the old explicit alias table had one dead entry (U+00A0,
# already handled by NFKC) folded into a general category sweep that
# replaces the table entirely. This test pins NFKC's remaining
# responsibility -- non-breaking space and other Unicode space variants --
# so a future refactor that drops it is caught. (The Pd/Cf sweep above does
# NOT reach U+00A0: its category is Zs, not Pd or Cf.)


def test_nfkc_is_still_load_bearing_for_nonbreaking_space():
    # This is the same case as test_nonbreaking_space_separated_ssn_is_
    # confidential above; named and commented here specifically as the
    # regression pin for NFKC, since nothing else in _for_matching reaches
    # a Zs-category space.
    assert classify(_t("SSN 123\xa045\xa06789")) == "confidential"


# Gap 11 (LOW-MEDIUM, latent): CONFIDENTIAL_SOURCES matching is now
# case-insensitive and tolerates a "<name>_<subtype>" compound, matching
# this codebase's own connector-naming convention -- not a substring match,
# so an unrelated source that happens to contain "karbon" does not match.


def test_karbon_source_matching_is_case_insensitive():
    for source in ("Karbon", "KARBON", "kArBoN"):
        assert classify(_t("client called about scheduling", source=source)) == "confidential", source


def test_karbon_compound_source_name_is_confidential():
    # No Karbon connector exists yet; this is the plausible name one would
    # use, following this codebase's own "<source>_<subtype>" convention
    # (git_repos, claude_ai, claude_code).
    assert classify(_t("client called about scheduling", source="karbon_export")) == "confidential"


def test_source_merely_containing_karbon_is_not_matched():
    # The fix is a prefix match on "karbon_", not a substring match anywhere
    # in the source name -- an unrelated source name should not collide.
    assert classify(_t("anything", source="unkarbon")) == "open"
    assert classify(_t("anything", source="prekarbon_thing")) == "open"


# --- Round 3: gap 7's boundary fix extended to SSN and EIN, on review ---
# (grouped-account/card deliberately left alone -- see the module comment).


def test_ssn_confidential_when_prefixed_by_a_letter():
    assert classify(_t("ref123-45-6789 on the form")) == "confidential"


def test_ssn_confidential_when_suffixed_by_a_letter():
    assert classify(_t("filed under 123-45-6789x")) == "confidential"


def test_ssn_confidential_when_joined_by_underscore():
    assert classify(_t("field id_123-45-6789 in the export")) == "confidential"


def test_ein_confidential_when_prefixed_by_a_letter():
    assert classify(_t("see x12-3456789 on the filing")) == "confidential"


def test_ein_confidential_when_suffixed_by_a_letter():
    assert classify(_t("reference 12-3456789z on the 1120S")) == "confidential"


# --- Round 4: boundary fix extended to the grouped-account/card pattern ---
# (the earlier "already safe" check used a space-separated prefix, which
# has a boundary regardless of \b; a directly-glued prefix does not).


def test_grouped_account_confidential_when_glued_to_a_label_no_space():
    assert classify(_t("acct4093-8172-6354 needs updating")) == "confidential"


def test_grouped_account_confidential_when_prefixed_by_a_bare_letter():
    assert classify(_t("see x4093-8172-6354 on the statement")) == "confidential"


def test_grouped_account_confidential_when_joined_by_underscore():
    assert classify(_t("field id_4093-8172-6354 in the export")) == "confidential"


def test_grouped_account_still_confidential_with_a_space_before_it():
    # The case the earlier check actually verified -- pinned so this
    # already-working path can't regress while the glued-prefix cases above
    # are being fixed.
    assert classify(_t("acct 4093-8172-6354 needs updating")) == "confidential"


# --- Round 5: qualifier-family false positives, phrase-\b pinning, and the
# noun-direct-digits connector axis, found by a second independent pass ---


# Finding 4 (MEDIUM, live): a bare "no"/"num"/"nbr" with nothing after it
# to say the match should stop there is a prefix of ordinary English words.
# Each of these was measured confidential before the trailing-letter guard.


def test_account_notes_stays_open():
    assert classify(_t("account notes are attached")) == "open"


def test_account_now_stays_open():
    assert classify(_t("account now please")) == "open"


def test_acct_nothing_stays_open():
    assert classify(_t("acct nothing to report")) == "open"


def test_routing_normally_stays_open():
    assert classify(_t("routing normally takes a week")) == "open"


def test_a_slash_c_not_stays_open():
    assert classify(_t("a/c not needed")) == "open"


def test_account_november_stays_open():
    assert classify(_t("account november filing")) == "open"


def test_guard_does_not_break_any_intended_qualifier_catch():
    """The trailing-letter guard on number/no/num/nbr must not narrow the
    family it was built to widen -- every phrasing from the gap-9 and
    gap-1 fixes, plus the bare "number" word, must still fire."""
    for text in (
        "acct no. 12345678", "account no 12345678", "acct num 12345678",
        "account nbr 12345678", "routing no. 12345678", "a/c no 12345678",
        "account number is on file", "acct #12345678",
    ):
        assert classify(_t(text)) == "confidential", text


# Finding 3 (LOW): "number" was dead code before the guard -- "num\.?" with
# no trailing check silently swallowed it. The guard above makes it
# unreachable via "num" alone again (the guard rejects "num" followed by
# "ber"), so "number" must stay in the alternation, not be removed.


def test_bare_number_word_alone_is_confidential():
    # Isolates "number" specifically: no period, no other qualifier present,
    # nothing this could be confused with except the guarded "num" prefix
    # the test above already proves does NOT reach it.
    assert classify(_t("please read me back the routing number")) == "confidential"


# "number" carries the identical prefix-swallowing risk as the short forms
# it was found alongside: without its own trailing guard, it is a prefix of
# "numbering" / "numbered" / "numberless" the same way "no" is a prefix of
# "notes" / "now" / "nothing". Guarding only the short alternatives and
# treating "number" as a safe, complete word was the same unswept-sibling
# mistake this module has made before -- caught this time before it shipped
# rather than by a later review round.


def test_account_numbering_stays_open():
    assert classify(_t("account numbering scheme needs review")) == "open"


def test_account_numbered_stays_open():
    assert classify(_t("the account numbered accounts twice")) == "open"


def test_account_numberless_stays_open():
    assert classify(_t("account numberless entries exist")) == "open"


# Finding 1 (LOW): the phrase markers' \b bookends are correct (unlike the
# digit-shape patterns, the ambiguous side here is the NOUN) but nothing
# asserted it -- deleting either \b left the whole suite green before these.


def test_rerouting_number_stays_open():
    assert classify(_t("rerouting number for this transfer")) == "open"


def test_subaccount_number_stays_open():
    assert classify(_t("subaccount number listed")) == "open"


def test_myaccount_hash_stays_open():
    assert classify(_t("myaccount #123 is active")) == "open"


# Gap 12, partial (deliberately narrow): a noun connected directly to a bare
# number via a colon or plain whitespace, with no qualifier word at all.
# Slash-as-connector and last-4-digit "ending in" disclosure were both
# considered and rejected -- zero corpus exposure and unmeasured new risk
# for the former, an explicit prior decline for the latter.


def test_acct_colon_bare_digits_is_confidential():
    assert classify(_t("acct: 12345678")) == "confidential"


def test_account_colon_bare_digits_is_confidential():
    assert classify(_t("account: 12345678")) == "confidential"


def test_bare_acct_directly_followed_by_digits_is_confidential():
    assert classify(_t("bank acct 12345678 is the one to use")) == "confidential"


def test_aba_number_phrase_is_confidential():
    # "aba" joins "routing" as a noun for the SAME qualifier family --
    # "ABA number" is the standard name for a routing number.
    assert classify(_t("the aba number is on the check")) == "confidential"


def test_aba_direct_digits_is_confidential():
    # Deliberately fewer than 9 digits: a 9+ digit number here would also
    # satisfy the unrelated bare-digit-run marker on its own, making this
    # test pass regardless of whether the noun-direct-digits marker fires
    # at all. This isolates the marker actually under test.
    assert classify(_t("aba: 2100")) == "confidential"


def test_bare_acct_without_any_digits_stays_open():
    # The noun-direct-digits marker must not fire on the noun alone --
    # digits are what distinguish this from ordinary mentions of "acct".
    assert classify(_t("let's acct for the discrepancy next quarter")) == "open"
