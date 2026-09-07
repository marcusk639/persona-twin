from __future__ import annotations
from typing import Callable
from persona_twin.schema import RawEnvelope, Turn

_REGISTRY: dict[str, Callable[[RawEnvelope], Turn | None]] = {}


def register(source: str, fn: Callable[[RawEnvelope], Turn | None]) -> None:
    _REGISTRY[source] = fn


def normalize(env: RawEnvelope) -> Turn | None:
    if env.source not in _REGISTRY:
        raise KeyError(f"no normalizer registered for source {env.source!r}")
    return _REGISTRY[env.source](env)


def _turn(env: RawEnvelope, *, thread_id: str, author_id: str,
          is_subject: bool, text: str) -> Turn | None:
    if not text.strip():
        return None
    return Turn(subject_id=env.subject_id, source=env.source, source_id=env.source_id,
                thread_id=thread_id, ts=env.ts, author_id=author_id,
                is_subject=is_subject, text=text,
                assisted=False)   # CC1: historical data predates the tool


def _imessage(env: RawEnvelope) -> Turn | None:
    p = env.payload
    from_me = bool(p.get("is_from_me"))
    return _turn(env, thread_id=str(p.get("chat_guid", "unknown")),
                 author_id=env.subject_id if from_me else str(p.get("handle", "unknown")),
                 is_subject=from_me, text=str(p.get("text", "")))


def _subject_authored(thread_key: str):
    def fn(env: RawEnvelope) -> Turn | None:
        return _turn(env, thread_id=str(env.payload.get(thread_key, "unknown")),
                     author_id=env.subject_id, is_subject=True,
                     text=str(env.payload.get("text", "")))
    return fn


register("imessage", _imessage)
register("claude_code", _subject_authored("file"))
register("git_repos", _subject_authored("repo"))
register("mail", _subject_authored("to"))
