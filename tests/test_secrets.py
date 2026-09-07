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
