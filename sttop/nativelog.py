"""Keep native libraries from writing over the UI.

Textual owns the terminal for the length of a session and redraws it by
absolute cursor position. Anything else that writes to the terminal lands in
the middle of that and corrupts the screen - and Python's `sys.stderr` is no
defence, because the offenders are C++ libraries writing to file descriptor 2
directly. onnxruntime is the one that bites: on macOS it registers a CoreML
execution provider and warns about every operator that falls back to CPU,
straight down fd 2, in the seconds while the model loads and the UI is up.

So fd 2 is pointed at a file for the length of the run. The output is kept
rather than discarded - a native crash message is often the only evidence of
what went wrong - and the tail is printed if the session ends badly.
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator
from pathlib import Path


def quiet_onnxruntime() -> None:
    """Ask onnxruntime for errors only, before any session is built.

    Fixing it at the source as well as at fd 2: the warnings are noise in the
    log file too, and a log that is mostly noise does not get read.
    """
    with contextlib.suppress(Exception):
        import onnxruntime

        onnxruntime.set_default_logger_severity(3)  # 3 = error


@contextlib.contextmanager
def stderr_to(path: Path) -> Iterator[Path]:
    """Redirect fd 2 to `path` for the duration of the block.

    Only fd 2: Textual draws on stdout, and redirecting that would hide the
    UI rather than protect it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stderr.flush()

    saved = os.dup(2)
    handle = open(path, "wb")
    try:
        os.dup2(handle.fileno(), 2)
        yield path
    finally:
        # Restore first, so that anything raised while restoring still has a
        # working stderr to be reported on.
        with contextlib.suppress(Exception):
            sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)
        handle.close()


def tail(path: Path, lines: int = 10) -> str:
    """The end of a captured log, for when a run fails and it is all we have."""
    try:
        captured = path.read_text(errors="replace").strip().splitlines()
    except OSError:
        return ""
    return "\n".join(captured[-lines:])
