from __future__ import annotations
import hmac, json, os
from hashlib import sha256
from pathlib import Path
from persona_twin.paths import SubjectPaths
from persona_twin.schema import Turn

class Pseudonymizer:
    """Deterministic, reversible only inside the vault (spec C3)."""

    def __init__(self, paths: SubjectPaths) -> None:
        paths.ensure()
        self.vault = Path(paths.vault)
        self.key_path = self.vault / "identity_key.bin"
        self.map_path = self.vault / "identity_map.json"
        if not self.key_path.exists():
            self.key_path.write_bytes(os.urandom(32))
            self.key_path.chmod(0o600)
        self.key = self.key_path.read_bytes()

    def _load_map(self) -> dict[str, str]:
        if not self.map_path.exists():
            return {}
        return json.loads(self.map_path.read_text(encoding="utf-8"))

    def pseudonym(self, raw_identifier: str) -> str:
        token = "P-" + hmac.new(
            self.key, raw_identifier.encode("utf-8"), sha256).hexdigest()[:12]
        mapping = self._load_map()
        if token not in mapping:
            mapping[token] = raw_identifier
            self.map_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
            self.map_path.chmod(0o600)
        return token

    def resolve(self, pseudonym: str) -> str | None:
        return self._load_map().get(pseudonym)

    def apply(self, turn: Turn) -> Turn:
        if turn.is_subject:
            return turn
        return turn.model_copy(update={"author_id": self.pseudonym(turn.author_id)})
