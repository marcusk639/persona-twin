from persona_twin.imessage.typedstream import decode_attributed_body

def _blob(text: str) -> bytes:
    """Minimal typedstream fixture: NSString marker, 0x2b, 1-byte length, UTF-8 payload."""
    body = text.encode("utf-8")
    return b"\x04\x0bstreamtyped\x81\xe8\x03\x84\x01@\x84\x84\x84\x08NSString\x01\x94\x84\x01+" \
           + bytes([len(body)]) + body + b"\x86"

def test_decodes_short_string():
    assert decode_attributed_body(_blob("hello there")) == "hello there"

def test_decodes_long_string_two_byte_length():
    text = "x" * 300
    body = text.encode("utf-8")
    blob = (b"\x04\x0bstreamtyped\x84\x84\x08NSString\x01\x94\x84\x01+"
            + b"\x81" + len(body).to_bytes(2, "little") + body)
    assert decode_attributed_body(blob) == text

def test_returns_none_when_no_marker():
    assert decode_attributed_body(b"\x00\x01\x02nothing here") is None

def test_returns_none_on_empty():
    assert decode_attributed_body(b"") is None
    assert decode_attributed_body(None) is None
