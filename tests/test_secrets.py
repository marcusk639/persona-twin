import persona_twin.scrub.secrets as secrets_module
import pytest
from persona_twin.scrub.secrets import (
    _ENTROPY_THRESHOLD,
    ScrubError,
    redact,
    scan,
    shannon_entropy,
)


def test_detects_known_key_shapes():
    text = "use sk-ant-api03-AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHHIIIIJJJJKKKKLLLL now"
    assert scan(text), "vendor key prefix must be detected"
    assert "sk-ant" not in redact(text)


def test_detects_aws_access_key():
    assert "AKIAIOSFODNN7EXAMPLE" not in redact("key AKIAIOSFODNN7EXAMPLE here")


def test_high_entropy_unknown_string_is_redacted_deny_by_default():
    blob = "b7Xq2LmZ9pR4vNc1TgHs6JdKfW0yEuAo3iQz8Vbn5"
    assert blob not in redact(f"token {blob} end")


def test_ordinary_prose_survives():
    prose = "I think we should ship the auth fix before Friday, it is simple enough."
    assert redact(prose) == prose


def test_entropy_ranks_random_above_prose():
    assert shannon_entropy("b7Xq2LmZ9pR4vNc1TgHs") > shannon_entropy("hello there friend")


def test_redaction_is_total_not_partial():
    text = "a AKIAIOSFODNN7EXAMPLE b AKIAIOSFODNN7EXAMPLE c"
    assert "AKIA" not in redact(text)


def test_scrub_error_is_raised_for_unencodable_input():
    with pytest.raises(ScrubError):
        redact(None)  # type: ignore[arg-type]


# --- Additional coverage: the two detection routes must work independently ---


def test_vendor_pattern_route_catches_low_entropy_key_alone():
    """An AKIA-shaped key with deliberately low entropy must still be caught
    by the pattern route. This proves pattern matching does not silently
    depend on the entropy check also firing."""
    key = "AKIA" + "A" * 16
    assert shannon_entropy(key) < _ENTROPY_THRESHOLD, (
        "fixture must be low-entropy to isolate the pattern route from the entropy route"
    )
    text = f"key {key} here"
    spans = scan(text)
    assert spans and all(label == "aws_access_key" for _, _, label in spans)
    assert key not in redact(text)


def test_entropy_route_catches_token_matching_no_known_pattern():
    """A high-entropy blob that matches none of the vendor regexes must
    still be caught by the entropy route alone (deny-by-default, spec C2)."""
    blob = "b7Xq2LmZ9pR4vNc1TgHs6JdKfW0yEuAo3iQz8Vbn5"
    text = f"token {blob} end"
    spans = scan(text)
    assert spans, "high-entropy blob must be flagged even though it matches no vendor pattern"
    assert all(label == "high_entropy" for _, _, label in spans)
    assert shannon_entropy(blob) >= _ENTROPY_THRESHOLD
    assert blob not in redact(text)


@pytest.mark.parametrize(
    ("token", "expected_label"),
    [
        ("sk_live_EXAMPLE1234567890", "stripe_secret_key"),
        ("pk_test_EXAMPLE1234567890", "stripe_publishable_key"),
        ("rk_live_EXAMPLE1234567890", "stripe_restricted_key"),
        ("npm_EXAMPLE12345678901234", "npm_token"),
        ("pypi-EXAMPLE1234567890123456", "pypi_token"),
        ("ntn_EXAMPLE12345678901234", "notion_token"),
        ("secret_EXAMPLE12345678901234", "notion_token_legacy"),
    ],
)
def test_detects_delimiter_prefixed_vendor_key(token, expected_label):
    """Round 2: vendor-prefixed keys whose prefix is separated from the body
    by '_' or '-' fragment on that delimiter under the narrowed _CANDIDATE
    pattern, so they're invisible to the entropy route regardless of
    threshold. Coverage for them comes only from a dedicated pattern per
    vendor — assert the specific label, not just that some span exists, so
    a future pattern collision or removal is caught precisely."""
    text = f"key {token} here"
    spans = scan(text)
    assert spans and any(label == expected_label for _, _, label in spans)
    assert token not in redact(text)


def test_openai_project_scoped_key_is_detected():
    """The original openai_key pattern (`sk-[A-Za-z0-9]{32,}`) doesn't match
    project-scoped keys shaped like sk-proj-... because their body contains
    hyphens. Widening the body class to [A-Za-z0-9_-] fixes this."""
    token = "sk-proj-" + "EXAMPLE1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    text = f"key {token} here"
    spans = scan(text)
    assert spans and any(label == "openai_key" for _, _, label in spans)
    assert token not in redact(text)


def test_short_delimiter_prefixed_token_below_entropy_floor_is_still_caught():
    """The regression this round exists to prevent: a vendor-prefixed key
    can be well under the entropy route's 32-character floor (and would
    fragment on its own delimiter even if it weren't), making it invisible
    to both detection routes unless a dedicated pattern exists. This test
    fails without the npm_token pattern."""
    token = "npm_" + "aB3dE6gH9jK1mN4pQ7rS"
    assert len(token) < 32, "fixture must be short enough that the entropy floor can't save it"
    text = f"token {token} end"
    spans = scan(text)
    assert spans and any(label == "npm_token" for _, _, label in spans)
    assert token not in redact(text)


def test_ordinary_secret_prefixed_identifiers_survive_redaction():
    """Round 3: `notion_token_legacy` matches `secret_[A-Za-z0-9]{20,}`, and
    "secret_" alone is a common prefix in ordinary code/prose. This pins the
    boundary: real identifiers carry internal underscores that break the
    required contiguous alphanumeric run, so they survive; an actual token
    with no internal separators does not. If the pattern is ever relaxed to
    permit '_' or '-' in the body, this test should start failing on the
    ordinary-identifier side."""
    ordinary = [
        "secret_key",
        "secret_manager_client",
        "secret_token_value",
        "AWS_SECRET_ACCESS_KEY",
        "secret_key_base_configuration",
        'the secret_sauce of good design',
        "secret_abcdefghij",
    ]
    for identifier in ordinary:
        assert scan(identifier) == [], f"{identifier!r} must not be flagged"

    prose = "I stored the secret_key in secret_manager_client and it worked."
    assert redact(prose) == prose

    # Positive case, shown alongside the negative ones so both sides of the
    # boundary are visible together: no internal separator, still flagged.
    real_token = "secret_4827JkLmNoPqRsTuVwXyZ0123456789"
    spans = scan(real_token)
    assert spans and any(label == "notion_token_legacy" for _, _, label in spans)
    assert real_token not in redact(real_token)


def test_realistic_prose_with_hyphenation_survives():
    """Prose with long-ish words, hyphenation, and punctuation must pass
    through unredacted. A prose fixture made only of short plain words
    would prove nothing about the entropy threshold."""
    prose = (
        "I think we should ship the client-facing auth-flow fix before "
        "Friday's end-of-day cutoff, since nobody's re-tested the onboarding "
        "process yet, and the self-service dashboard still needs its "
        "rate-limiting logic double-checked. It's not urgent, but let's "
        "avoid a last-minute scramble."
    )
    assert redact(prose) == prose


def test_path_shaped_token_is_not_flagged_despite_high_entropy():
    """A lowercase, multi-segment filesystem path can be genuinely
    high-entropy (order-0 character entropy doesn't know it's a path), so
    raising the threshold alone can't exclude it — the path-shape exclusion
    must. This fixture's entropy is deliberately above _ENTROPY_THRESHOLD to
    prove the exclusion, not the threshold, is what's keeping it out."""
    path = "home/mk7391/dropbox/reportsjan2026/budgetreviewfinal"
    assert shannon_entropy(path) >= _ENTROPY_THRESHOLD, (
        "fixture must be high-entropy on its own so a bare threshold check would "
        "wrongly flag it; only the path-shape exclusion should save it"
    )
    assert scan(f"see {path} for details") == []
    assert redact(f"see {path} for details") == f"see {path} for details"


def test_base64_blob_with_slash_is_still_flagged():
    """Pins the decision to keep '/' in the candidate character class: a
    base64-shaped secret containing '/' must still be redacted. If a future
    change drops '/' to "simplify" the pattern, this test breaks loudly."""
    blob = "R3JlYXRlckVudHJvcHk/TWl4ZWRDYXNlMTIzNDU2Nzg5MA=="
    assert "/" in blob
    spans = scan(f"payload {blob} end")
    assert spans and all(label == "high_entropy" for _, _, label in spans)
    assert blob not in redact(f"payload {blob} end")


def test_redact_raises_when_rescan_finds_residual_match(monkeypatch):
    """Prove the fail-closed re-scan (spec C4) is actually load-bearing.

    This patches `scan` with a fake detector that flags the literal
    substring "REDACTED" as a secret. Redacting a span with that marker
    inevitably leaves a residual match inside the replacement token itself
    ("[REDACTED]" contains "REDACTED"), which the post-substitution re-scan
    must catch. If the "if scan(result): raise" line in redact() were
    deleted, this test would fail: redact() would return the
    still-matching text instead of raising ScrubError.
    """

    def fake_scan(text: str) -> list[tuple[int, int, str]]:
        marker = "REDACTED"
        idx = text.find(marker)
        return [(idx, idx + len(marker), "fake")] if idx != -1 else []

    monkeypatch.setattr(secrets_module, "scan", fake_scan)
    with pytest.raises(ScrubError):
        redact("prefix REDACTEDxyz suffix")
