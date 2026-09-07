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

def test_returns_none_on_truncated_payload():
    """Payload declared longer than bytes present should return None."""
    # Blob claiming 200 bytes of payload but only 10 bytes provided
    blob = b"\x04\x0bstreamtyped\x81\xe8\x03\x84\x01@\x84\x84\x84\x08NSString\x01\x94\x84\x01+" + bytes([200]) + b"short"
    assert decode_attributed_body(blob) is None

def test_returns_none_on_invalid_utf8():
    """Invalid UTF-8 bytes should return None, never raise."""
    # Build blob with invalid UTF-8 payload (0xff 0xfe 0xff)
    payload = b"\xff\xfe\xff"
    blob = b"\x04\x0bstreamtyped\x81\xe8\x03\x84\x01@\x84\x84\x84\x08NSString\x01\x94\x84\x01+" \
           + bytes([len(payload)]) + payload + b"\x86"
    assert decode_attributed_body(blob) is None

def test_decodes_long_string_four_byte_length():
    """Test 0x82 form: 4-byte little-endian length."""
    text = "y" * 70000  # Long enough to show 4-byte length is needed
    body = text.encode("utf-8")
    blob = (b"\x04\x0bstreamtyped\x84\x84\x08NSString\x01\x94\x84\x01+"
            + b"\x82" + len(body).to_bytes(4, "little") + body)
    assert decode_attributed_body(blob) == text
