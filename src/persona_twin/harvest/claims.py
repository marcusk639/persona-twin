from __future__ import annotations
import hashlib, json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal
from pydantic import BaseModel


class Claim(BaseModel):
    claim_id: str
    text: str
    tier: Literal["T2", "T3"] = "T2"
    source_model: str
    run: int
    evidence_quote: str
    source_confidence: str
    disconfirming_case: str
    recurrence: bool = False
    verification_status: Literal[
        "unverified", "corroborated", "contradicted", "unfalsifiable"
    ] = "unverified"


def load_report(path: Path, source_model: str, run: int) -> list[Claim]:
    """Intake a T2 report. Claims always enter unverified (spec R3).

    An empty list is a legitimate, meaningful result (a model with no
    retrievable history about the subject) and is returned as-is, not
    treated as an error. A malformed file (missing, not JSON, not a JSON
    list) raises instead of silently returning an empty list, so "no
    claims" and "could not read the file" stay distinguishable.
    """
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON list of claim objects, got {type(rows).__name__}")
    out: list[Claim] = []
    for row in rows:
        cid = hashlib.sha256(
            f"{source_model}:{run}:{row['text']}".encode("utf-8")
        ).hexdigest()[:16]
        out.append(
            Claim(
                claim_id=cid,
                text=row["text"],
                source_model=source_model,
                run=run,
                evidence_quote=row.get("evidence_quote", ""),
                source_confidence=row.get("source_confidence", "unknown"),
                disconfirming_case=row.get("disconfirming_case", ""),
            )
        )
    return out


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def mark_recurrence(
    run_a: list[Claim], run_b: list[Claim], threshold: float = 0.6
) -> list[Claim]:
    """Claims recurring across independent runs are grounded; drifting ones are not (§5.4).

    Recurrence is the confabulation detector: a claim reappearing across two
    separately-run sessions is grounded in retrievable history, one that
    only shows up once was generated on the spot. Returns run_a's claims
    with `recurrence` set; run_a and run_b are not mutated (Claim is
    frozen-by-convention pydantic, so model_copy is used instead of item
    assignment).
    """
    merged: list[Claim] = []
    for claim in run_a:
        hit = any(_similar(claim.text, other.text) >= threshold for other in run_b)
        merged.append(claim.model_copy(update={"recurrence": hit}))
    return merged
