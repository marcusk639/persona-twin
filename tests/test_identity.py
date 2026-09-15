import hmac
import stat
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from persona_twin.paths import SubjectPaths
from persona_twin.schema import Turn
from persona_twin.normalize.identity import Pseudonymizer

def _paths(tmp_path):
    p = SubjectPaths("alice", tmp_path); p.ensure(); return p

def _turn(author="+15551234567", is_subject=False):
    return Turn(subject_id="alice", source="imessage", source_id="1", thread_id="c1",
                ts=datetime.now(timezone.utc), author_id=author,
                is_subject=is_subject, text="hi", assisted=False)

def test_pseudonym_is_deterministic(tmp_path):
    p = Pseudonymizer(_paths(tmp_path))
    assert p.pseudonym("+15551234567") == p.pseudonym("+15551234567")

def test_distinct_identifiers_get_distinct_pseudonyms(tmp_path):
    p = Pseudonymizer(_paths(tmp_path))
    assert p.pseudonym("a@x.com") != p.pseudonym("b@x.com")

def test_subject_author_is_not_pseudonymized(tmp_path):
    p = Pseudonymizer(_paths(tmp_path))
    assert p.apply(_turn(author="alice", is_subject=True)).author_id == "alice"

def test_counterparty_is_replaced(tmp_path):
    out = Pseudonymizer(_paths(tmp_path)).apply(_turn())
    assert out.author_id.startswith("P-") and "5551234567" not in out.author_id

def test_map_lives_in_vault_and_resolves(tmp_path):
    paths = _paths(tmp_path)
    p = Pseudonymizer(paths)
    tok = p.pseudonym("+15551234567")
    assert (paths.vault / "identity_map.json").exists()
    assert p.resolve(tok) == "+15551234567"

def test_key_is_stable_across_instances(tmp_path):
    paths = _paths(tmp_path)
    first = Pseudonymizer(paths).pseudonym("x@y.com")
    assert Pseudonymizer(paths).pseudonym("x@y.com") == first

def test_existing_key_is_never_overwritten(tmp_path):
    paths = _paths(tmp_path)
    p1 = Pseudonymizer(paths)
    tok_before = p1.pseudonym("carol@x.com")
    key_bytes_before = p1.key_path.read_bytes()

    p2 = Pseudonymizer(paths)  # constructed against a vault that already has a key

    assert p2.key_path.read_bytes() == key_bytes_before
    assert p2.key == key_bytes_before
    assert p2.pseudonym("carol@x.com") == tok_before

def test_key_and_map_files_are_owner_only(tmp_path):
    paths = _paths(tmp_path)
    p = Pseudonymizer(paths)
    p.pseudonym("+15551234567")
    assert stat.S_IMODE(p.key_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(p.map_path.stat().st_mode) == 0o600

def test_key_creation_ignores_stale_exists_check(tmp_path, monkeypatch):
    # Deterministic TOCTOU test: simulate the exact stale-check window a
    # check-then-write implementation is vulnerable to ("the check said the
    # key was absent, but it wasn't by write time") without racing real
    # processes, which would only sometimes catch a broken implementation.
    paths = _paths(tmp_path)
    known_key = b"\x01" * 32
    key_path = paths.vault / "identity_key.bin"
    key_path.write_bytes(known_key)
    key_path.chmod(0o600)
    expected_token = "P-" + hmac.new(
        known_key, b"dave@x.com", sha256).hexdigest()[:12]

    # Every Path.exists() call in the constructor now lies and reports
    # absent, no matter what's actually on disk.
    monkeypatch.setattr(Path, "exists", lambda self: False)

    p = Pseudonymizer(paths)

    assert key_path.read_bytes() == known_key
    assert p.pseudonym("dave@x.com") == expected_token


# --- text-level identity scrub (spec C3 / §13.1) ---

NEEDLES = ["Marcus Klein", "Marcus", "Klein"]


def test_replaces_the_subject_name_in_turn_text():
    from persona_twin.normalize.identity import scrub_identity
    assert scrub_identity("Hey Marcus, you around?", NEEDLES, "Dylan") == \
        "Hey Dylan, you around?"


def test_matches_case_insensitively_because_inbound_sms_shouts():
    """Appointment reminders arrive as 'MARCUS KLEIN'."""
    from persona_twin.normalize.identity import scrub_identity
    assert scrub_identity("MARCUS KLEIN is scheduled", NEEDLES, "Dylan") == \
        "Dylan is scheduled"


def test_full_name_is_replaced_as_one_unit_not_twice():
    """Shortest-first would turn 'Marcus Klein' into 'Dylan Dylan'."""
    from persona_twin.normalize.identity import scrub_identity
    assert scrub_identity("from Marcus Klein today", NEEDLES, "Dylan") == \
        "from Dylan today"


def test_does_not_match_inside_a_longer_word():
    """Aliases here are 5-6 chars; unanchored matching corrupts ordinary prose."""
    from persona_twin.normalize.identity import scrub_identity
    for safe in ("Kleiner Perkins", "marcuscolumn", "Kleines"):
        assert scrub_identity(safe, NEEDLES, "Dylan") == safe


def test_text_without_the_name_is_returned_unchanged():
    from persona_twin.normalize.identity import scrub_identity
    s = "the billing flow is finicky on complex views"
    assert scrub_identity(s, NEEDLES, "Dylan") is s or scrub_identity(s, NEEDLES, "Dylan") == s


def test_empty_needles_are_ignored_rather_than_matching_everything():
    """An empty alias compiles to a pattern matching the empty string at every
    non-word boundary. The text MUST contain punctuation: without the guard,
    "hi -- there" becomes "hi Dylan-Dylan-Dylan there", but a string of plain
    words has no such position and the test would pass either way."""
    from persona_twin.normalize.identity import scrub_identity
    for s in ("hi -- there", "a, b", "(note)"):
        assert scrub_identity(s, ["", "  "], "Dylan") == s


def test_needles_shorter_than_three_characters_are_ignored():
    """'A' or 'Jo' match common words; case-insensitively, a one-character
    needle rewrites every standalone 'a' in the corpus. Substitution cannot
    safely scrub a name that short -- excluding it is the lesser harm, and the
    existing test fixture (display_name='A') proved the hazard is real."""
    from persona_twin.normalize.identity import scrub_identity
    s = "a quick note about the build"
    assert scrub_identity(s, ["A"], "Dylan") == s
    assert scrub_identity(s, ["Jo"], "Dylan") == s


def test_a_long_enough_needle_still_applies_alongside_short_ones():
    from persona_twin.normalize.identity import scrub_identity
    assert scrub_identity("A note from Marcus", ["A", "Marcus"], "Dylan") == \
        "A note from Dylan"
