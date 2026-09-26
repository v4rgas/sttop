"""`sttop follow`: the live transcript as a stream, for an agent to watch.

Reads the session file rather than hooking the recorder, so it can attach to
a meeting already in progress and a slow or dead reader never touches the
recording. New utterances come out in batches - one burst every few seconds,
not one line per sentence - which is the cadence an agent can think at.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from pathlib import Path

from .journal import STAMP, list_sessions

#: `- `12:41` **speaker** — text`: the header and footer bullets never start
#: with a backticked clock.
UTTERANCE = re.compile(r"^- `\d+:\d{2}(?::\d{2})?` ")
ENDED = "\n- ended: "


def utterances(text: str) -> list[str]:
    """The complete utterance lines. A line still being written (no newline
    yet) waits for the next read."""
    complete = text[: text.rfind("\n") + 1]
    return [line for line in complete.splitlines() if UTTERANCE.match(line)]


def is_finished(text: str) -> bool:
    return ENDED in text


def relocate(directory: Path, path: Path) -> Path | None:
    """Where a session went when naming it on quit moved it into a folder:
    the new name keeps the stamp it was created with."""
    match = STAMP.search(path.name)
    if not match:
        return None
    for candidate in list_sessions(directory, limit=1000):
        if match.group() in candidate.name:
            return candidate
    return None


def live_session(
    directory: Path,
    read: Callable[[Path], str | None],
    *,
    waiting: Callable[[], None] = lambda: None,
    poll: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """The newest session that is still recording - waiting for one to start
    when the newest is finished, so the watcher can be up before the meeting.
    `waiting` is called once, the first time there is nothing live yet."""
    announced = False
    while True:
        sessions = list_sessions(directory, limit=1)
        if sessions:
            text = read(sessions[0])
            if text is not None and not is_finished(text):
                return sessions[0]
        if not announced:
            waiting()
            announced = True
        sleep(poll)


def follow(
    path: Path,
    directory: Path,
    read: Callable[[Path], str | None],
    emit: Callable[[str], None],
    *,
    every: float = 20.0,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Emit each batch of new utterances until the session ends.

    `read` returns the plaintext, or None when the file is unreadable right
    now (gone, or caught mid-write while sealed) - a transient, retried on
    the next tick.
    """
    seen = 0
    emit(f"[sttop] following {path.name}")
    while True:
        text = read(path)
        if text is None and not path.exists():
            moved = relocate(directory, path)
            if moved is None:
                emit(f"[sttop] session file is gone: {path}")
                return
            if moved != path:
                path = moved
                continue
        if text is not None:
            lines = utterances(text)
            if len(lines) > seen:
                emit("\n".join(lines[seen:]))
                seen = len(lines)
            if is_finished(text):
                emit(f"[sttop] session ended: {path}")
                return
        sleep(every)
