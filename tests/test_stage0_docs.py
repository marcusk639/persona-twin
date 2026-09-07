from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "docs/threat-model.md": ["Vault", "Identity map", "Mitigation"],
    "docs/disclosure-policy.md": ["model of a person", "first contact"],
    "docs/export-requests.md": ["ChatGPT", "Google Takeout"],
}

@pytest.mark.parametrize("rel,needles", REQUIRED.items())
def test_stage0_document_exists_and_is_substantive(rel, needles):
    path = ROOT / rel
    assert path.exists(), f"{rel} is a stage-0 deliverable (spec §4.1/§4.3/§5.3)"
    text = path.read_text(encoding="utf-8")
    assert len(text) > 400, f"{rel} is present but empty of content"
    for needle in needles:
        assert needle in text, f"{rel} is missing required content: {needle!r}"
