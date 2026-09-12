from __future__ import annotations
from datetime import datetime, timedelta
from persona_twin.schema import Turn

DEFAULT_WEEKS = 12

def is_quarantined(turn: Turn, now: datetime, weeks: int = DEFAULT_WEEKS) -> bool:
    """CC3: the newest `weeks` are never trained on and never indexed."""
    return turn.ts > now - timedelta(weeks=weeks)

def split(turns: list[Turn], now: datetime,
          weeks: int = DEFAULT_WEEKS) -> tuple[list[Turn], list[Turn]]:
    trainable = [t for t in turns if not is_quarantined(t, now, weeks)]
    quarantined = [t for t in turns if is_quarantined(t, now, weeks)]
    return trainable, quarantined
