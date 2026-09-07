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
        self.key = self._load_or_create_key()

    def _load_or_create_key(self) -> bytes:
        # Atomic exclusive create: two Pseudonymizer instances constructed
        # concurrently against an empty vault must not let the second one
        # overwrite the first's key (that would silently re-map every
        # counterparty pseudonym generated between the two writes).
        try:
            fd = os.open(self.key_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return self.key_path.read_bytes()
        try:
            os.write(fd, os.urandom(32))
        finally:
            os.close(fd)
        return self.key_path.read_bytes()

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
            # Write-then-rename: a process killed mid-write leaves the .tmp
            # file corrupt but never touches identity_map.json itself, so a
            # crash can't truncate the one artifact that can re-identify a
            # third party. chmod the .tmp before the replace so 0o600 holds
            # continuously at the final path — there's no window where a
            # default-permission file sits there, even briefly.
            tmp = self.map_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
            tmp.chmod(0o600)
            tmp.replace(self.map_path)
        return token

    def resolve(self, pseudonym: str) -> str | None:
        return self._load_map().get(pseudonym)

    def apply(self, turn: Turn) -> Turn:
        if turn.is_subject:
            return turn
        return turn.model_copy(update={"author_id": self.pseudonym(turn.author_id)})
