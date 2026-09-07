"""Decode Apple's legacy typedstream `attributedBody` blobs (spec §5.6).

`message.text` is frequently NULL on current macOS; the message body lives in
`attributedBody` as an NSArchiver typedstream. This is a heuristic reader for the
one shape that matters -- the NSString payload -- not a general typedstream parser.
Its adequacy is decided by the measured recovery rate, not by its completeness.
"""
from __future__ import annotations

_MARKER = b"NSString"
_PLUS = 0x2B

def decode_attributed_body(blob: bytes | None) -> str | None:
    if not blob:
        return None
    idx = blob.find(_MARKER)
    if idx == -1:
        return None
    # Assumption: 0x2b length marker is the first occurrence after NSString marker.
    # This works for all observed blob shapes in the current corpus (100% recovery),
    # but could mis-parse richer attribute dictionaries with extra bytes between
    # the class descriptor and length byte in other subjects' data.
    i = blob.find(bytes([_PLUS]), idx)
    if i == -1 or i + 1 >= len(blob):
        return None
    i += 1
    n = blob[i]
    if n == 0x81:                      # 2-byte little-endian length follows
        if i + 3 > len(blob):
            return None
        length = int.from_bytes(blob[i + 1:i + 3], "little")
        start = i + 3
    elif n == 0x82:                    # 4-byte little-endian length
        if i + 5 > len(blob):
            return None
        length = int.from_bytes(blob[i + 1:i + 5], "little")
        start = i + 5
    else:
        length = n
        start = i + 1
    payload = blob[start:start + length]
    if len(payload) < length:
        return None
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return None
