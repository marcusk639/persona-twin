import stat
from datetime import datetime, timezone
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
