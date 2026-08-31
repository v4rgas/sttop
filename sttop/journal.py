"""Session storage: one Markdown file per session, appended as you speak.

Every utterance is written and flushed the moment it is transcribed, so the
file on disk is always current - if sttop is killed mid-meeting, the transcript
up to that point is already saved and readable.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # the journal never derives keys; it is handed a Cipher
    from .crypto import Cipher


@dataclass
class Utterance:
    """One line of transcript."""

    source: str
    speaker: str
    start: float  # seconds since session start
    end: float
    text: str
    language: str | None = None
    confidence: float | None = None


def clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:48] or "session"


class Journal:
    """An open transcript being written to - plaintext Markdown, or the same
    Markdown sealed chunk by chunk when a Cipher is supplied.

    Everything written also lives in `_parts`, an in-memory mirror of the
    plaintext. That is what `snapshot()` hands the UI mid-meeting, and what
    renames rewrite - the file on disk may be ciphertext, so it stopped being
    a place to read the transcript back from.
    """

    def __init__(self, path: Path, title: str, cipher: Cipher | None = None) -> None:
        self.path = path
        self.title = title
        self.started = datetime.now().astimezone()
        self.count = 0
        self._cipher = cipher
        self._parts: list[str] = []
        self._last_speaker: str | None = None
        self._lock = threading.Lock()
        self._file = None

    @classmethod
    def create(
        cls,
        directory: Path,
        title: str | None = None,
        *,
        mic_source: str = "",
        sys_source: str = "",
        backend: str = "",
        cipher: Cipher | None = None,
    ) -> Journal:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone()
        title = title or f"Session {stamp:%Y-%m-%d %H:%M}"

        extension = ".md.enc" if cipher else ".md"
        base = f"{stamp:%Y-%m-%d-%H%M}-{slugify(title)}"
        path = directory / f"{base}{extension}"
        suffix = 2
        while path.exists():  # two sessions in the same minute
            path = directory / f"{base}-{suffix}{extension}"
            suffix += 1

        journal = cls(path, title, cipher)
        journal._open(mic_source, sys_source, backend)
        return journal

    def _open(self, mic_source: str, sys_source: str, backend: str) -> None:
        self._file = self.path.open("w", encoding="utf-8")
        if self._cipher is not None:
            self._file.write(self._cipher.header)
        self._write(
            f"# {self.title}\n\n"
            f"- started: {self.started:%Y-%m-%d %H:%M:%S %Z}\n"
            f"- mic: `{mic_source}`\n"
            f"- system: `{sys_source}`\n"
            f"- backend: `{backend}`\n\n"
            "## Transcript\n\n"
        )

    def _write(self, text: str) -> None:
        """One flushed chunk: a plaintext piece, or one sealed base64 line."""
        self._parts.append(text)
        self._file.write(self._cipher.seal(text) if self._cipher else text)
        self._file.flush()

    def snapshot(self) -> str:
        """The transcript so far, as plaintext Markdown. Current the moment an
        utterance lands, whatever the on-disk format."""
        with self._lock:
            return "".join(self._parts)

    def append(self, utterance: Utterance) -> None:
        """Write one utterance and flush, so the file survives a hard kill."""
        with self._lock:
            if self._file is None:
                return
            # A blank line between speaker turns keeps the raw file readable.
            prefix = ""
            if utterance.speaker != self._last_speaker:
                if self._last_speaker is not None:
                    prefix = "\n"
                self._last_speaker = utterance.speaker
            stamp, speaker = clock(utterance.start), utterance.speaker
            self._write(f"{prefix}- `{stamp}` **{speaker}** — {utterance.text}\n")
            self.count += 1

    def close(self, elapsed: float) -> Path:
        with self._lock:
            if self._file is not None:
                ended = datetime.now().astimezone()
                self._write(
                    f"\n---\n\n"
                    f"- ended: {ended:%Y-%m-%d %H:%M:%S %Z}\n"
                    f"- duration: {clock(elapsed)}\n"
                    f"- utterances: {self.count}\n"
                )
                self._file.close()
                self._file = None
        return self.path

    def rename_speaker(self, old: str, new: str) -> int:
        """Rewrite past lines after you identify a voice. Safe while recording."""
        with self._lock:
            text = "".join(self._parts)
            replaced = text.count(f"**{old}**")
            if replaced:
                text = text.replace(f"**{old}**", f"**{new}**")
                self._parts = [text]
                # Write-then-rename: rewriting in place would leave the whole
                # session truncated if we were killed mid-write, and renaming
                # is something you do *during* a meeting.
                temporary = self.path.with_name(self.path.name + ".tmp")
                if self._cipher is not None:
                    temporary.write_text(
                        self._cipher.header + self._cipher.seal(text),
                        encoding="utf-8",
                    )
                else:
                    temporary.write_text(text, encoding="utf-8")
                temporary.replace(self.path)
                # The handle still points at the old file; reopen at the end.
                if self._file is not None:
                    self._file.close()
                    self._file = self.path.open("a", encoding="utf-8")
            if self._last_speaker == old:
                self._last_speaker = new
            return replaced


def list_sessions(directory: Path, limit: int = 50) -> list[Path]:
    """Newest first. A session with both a plaintext file and its sealed twin
    (sync mode) is one session, and the plaintext - the readable one - wins."""
    if not directory.is_dir():
        return []
    sessions: dict[str, Path] = {}
    for path in directory.glob("*.md.enc"):
        sessions[path.name.removesuffix(".enc")] = path
    for path in directory.glob("*.md"):
        sessions[path.name] = path
    return sorted(sessions.values(), key=lambda p: p.name, reverse=True)[:limit]
