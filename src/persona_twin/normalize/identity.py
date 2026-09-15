from __future__ import annotations
import hmac, json, os, re
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


def scrub_identity(text: str, needles: list[str], replacement: str) -> str:
    """Replace the subject's own name in turn TEXT (spec C3 / §13.1).

    Pseudonymizer.apply() rewrites `author_id` only, which leaves the subject's
    name wherever it appears in a message body -- inbound SMS that address him
    by name, appointment reminders, signature lines. Measured on v5 that was
    1,828 turns across train, held-out and quarantined, including the frozen
    CC2 baseline. Neither existing check catches it: corpus_audit asks about
    ids, and name_leak_lint skips data/ by design.

    A real first name rather than a token, deliberately: the corpus is training
    data for a voice model, and 1,828 instances of "[SUBJECT]" would teach the
    twin to emit "[SUBJECT]" and would shift the type-token ratio S2 measures.

    Longest needle first, so a full name is replaced as one unit instead of
    each part separately ("Marcus Klein" -> "Dylan", never "Dylan Dylan").
    Word-anchored, because the aliases here are 5-6 characters and unanchored
    matching corrupts ordinary prose ("Kleiner Perkins", "Kleines").
    """
    # A 1-2 character needle matches common words -- case-insensitively "A"
    # rewrites every standalone "a" in the corpus, and "Jo" hits "Jo" in any
    # list of names. Substitution cannot safely scrub a token that short, so it
    # is excluded rather than applied; the tradeoff is documented because the
    # alternative silently destroys the corpus.
    cleaned = [n.strip() for n in needles if n and len(n.strip()) >= 3]
    if not cleaned:
        # An empty needle compiles to a pattern matching at every position,
        # which would rewrite the entire corpus into the replacement name.
        return text
    pattern = "|".join(re.escape(n) for n in sorted(cleaned, key=len, reverse=True))
    return re.sub(rf"(?<!\w)(?:{pattern})(?!\w)", replacement, text, flags=re.I)
