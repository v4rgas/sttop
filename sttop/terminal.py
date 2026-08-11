"""Detect the terminal's colour scheme so the UI can match it.

The `ansi-dark` / `ansi-light` Textual themes paint using only the terminal's
own 16 ANSI colours, so sttop inherits whatever palette the user has already
configured. All we have to choose is which of the two - i.e. whether the
background is dark or light.

Three sources, cheapest and most reliable first:

1. COLORFGBG, set by urxvt/konsole and friends.
2. An OSC 11 query, which asks the terminal for its background colour outright.
   Supported by ghostty, kitty, alacritty, wezterm, foot, xterm, iTerm2.
3. Assume dark, which is what most terminals are.
"""

from __future__ import annotations

import os
import re
import select
import sys
import time
from collections.abc import Iterator

DARK = "ansi-dark"
LIGHT = "ansi-light"

#: OSC 11 replies as rgb:RRRR/GGGG/BBBB, but component width varies by terminal.
_OSC11_REPLY = re.compile(rb"rgb:([0-9a-f]+)/([0-9a-f]+)/([0-9a-f]+)", re.I)


def _scale(component: bytes) -> float:
    """Normalise a hex component of any width to 0..1."""
    return int(component, 16) / (16 ** len(component) - 1)


def luminance(red: float, green: float, blue: float) -> float:
    """Perceived brightness, 0 (black) .. 1 (white). Rec. 601 weights."""
    return 0.299 * red + 0.587 * green + 0.114 * blue


def parse_osc11(reply: bytes) -> float | None:
    """Pull a luminance out of an OSC 11 reply, or None if it isn't one."""
    match = _OSC11_REPLY.search(reply)
    if match is None:
        return None
    return luminance(*(_scale(part) for part in match.groups()))


def from_colorfgbg(value: str | None) -> str | None:
    """Read the `foreground;background` env var some terminals export."""
    if not value:
        return None
    parts = value.split(";")
    if len(parts) < 2 or not parts[-1].isdigit():
        return None
    # ANSI colour 0-6 and 8 are dark; 7 and 9-15 are light.
    background = int(parts[-1])
    return LIGHT if background == 7 or background >= 9 else DARK


def query_osc11(timeout: float = 0.15) -> str | None:
    """Ask the terminal for its background colour. None if it stays silent.

    Must run before Textual takes over the terminal, since it needs raw mode
    and reads the reply straight off the tty.
    """
    try:
        import termios
        import tty
    except ImportError:  # pragma: no cover - not POSIX
        return None

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return None

    try:
        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
    except (termios.error, ValueError, OSError):
        return None

    try:
        tty.setraw(fd)
        sys.stdout.write("\033]11;?\033\\")
        sys.stdout.flush()

        reply = b""
        # Terminals that do not implement OSC 11 simply never answer, so the
        # whole thing is bounded by a deadline rather than by a sentinel. The
        # deadline is for the query as a whole: re-arming select with the full
        # timeout each pass would let unrelated input (a paste, buffered
        # keystrokes) hold the terminal in raw mode for far longer.
        deadline = time.monotonic() + timeout
        while (left := deadline - time.monotonic()) > 0 and select.select(
            [fd], [], [], left
        )[0]:
            chunk = os.read(fd, 64)
            if not chunk:
                break
            reply += chunk
            if b"\033\\" in reply or reply.endswith(b"\a"):
                break
    except (OSError, ValueError):
        return None
    finally:
        _restore_tty(fd, saved)

    value = parse_osc11(reply)
    if value is None:
        return None
    return LIGHT if value > 0.5 else DARK


def _restore_tty(fd: int, saved) -> None:
    """Put the terminal back the way we found it, never raising over a dead tty."""
    try:
        import termios

        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
    except Exception:  # pragma: no cover - the tty went away mid-query
        pass


def theme_sources() -> Iterator[tuple[str, str | None]]:
    """Each source of evidence and its verdict, in the order they are consulted.

    Detection policy lives here rather than in the caller, so `sttop theme`
    reports what `detect_theme` actually does instead of a copy that can drift.
    Lazy on purpose: a caller that stops at the first answer never pays for the
    OSC 11 query, which costs a timeout and takes over the tty.
    """
    yield "COLORFGBG", from_colorfgbg(os.environ.get("COLORFGBG"))
    yield "OSC 11", query_osc11()


def detect_theme(preference: str = "auto") -> str:
    """Resolve a configured theme name, following the terminal when asked to."""
    if preference and preference != "auto":
        return preference
    for _, verdict in theme_sources():
        if verdict:
            return verdict
    return DARK
