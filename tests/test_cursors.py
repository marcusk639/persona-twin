from persona_twin.cursors import CursorStore

def test_roundtrip(tmp_path):
    cs = CursorStore(tmp_path / "c.json")
    assert cs.get("imessage") is None
    cs.set("imessage", "1234")
    assert cs.get("imessage") == "1234"

def test_persists_across_instances(tmp_path):
    p = tmp_path / "c.json"
    CursorStore(p).set("git", "abc")
    assert CursorStore(p).get("git") == "abc"

def test_sources_are_independent(tmp_path):
    cs = CursorStore(tmp_path / "c.json")
    cs.set("a", "1"); cs.set("b", "2")
    assert (cs.get("a"), cs.get("b")) == ("1", "2")
