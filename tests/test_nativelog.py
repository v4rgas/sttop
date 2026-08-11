"""Which writer goes to the log and which one still reaches the terminal.

Getting this backwards has no error and no traceback: the session records
perfectly and the user watches a blank screen for the whole of it.
"""

import os
import sys

from sttop.nativelog import stderr_to, tail


def redirect_fd2_to(path):
    """Stand in for the terminal, which a test does not have."""
    handle = open(path, "wb")
    saved = os.dup(2)
    os.dup2(handle.fileno(), 2)
    return handle, saved


def restore_fd2(handle, saved):
    os.dup2(saved, 2)
    os.close(saved)
    handle.close()


def test_the_ui_still_reaches_the_terminal(tmp_path):
    """Textual's Linux and macOS driver draws through `sys.__stderr__`, so a
    redirect that moves it along with fd 2 files the entire UI into the log."""
    terminal_path, log_path = tmp_path / "terminal", tmp_path / "session.log"
    handle, saved = redirect_fd2_to(terminal_path)
    try:
        with stderr_to(log_path):
            sys.__stderr__.write("the user interface")
            sys.__stderr__.flush()
    finally:
        restore_fd2(handle, saved)

    assert "the user interface" in terminal_path.read_text()
    assert "the user interface" not in log_path.read_text()


def test_a_native_library_writing_to_the_descriptor_is_caught(tmp_path):
    """onnxruntime's CoreML warnings go straight down fd 2, in the seconds
    while the model loads and the UI is already up."""
    terminal_path, log_path = tmp_path / "terminal", tmp_path / "session.log"
    handle, saved = redirect_fd2_to(terminal_path)
    try:
        with stderr_to(log_path):
            os.write(2, b"[W:onnxruntime] falling back to CPU\n")
    finally:
        restore_fd2(handle, saved)

    assert "onnxruntime" in log_path.read_text()
    assert "onnxruntime" not in terminal_path.read_text()
    # And it is the log the failure path reads back for the user.
    assert "onnxruntime" in tail(log_path)


def test_a_python_traceback_is_caught_too(tmp_path):
    """It would corrupt the screen exactly as a native one does, so only
    `sys.__stderr__` is moved aside - `sys.stderr` stays on the descriptor and
    follows it into the log.

    Written through a stream of our own rather than `sys.stderr`, which pytest
    has already replaced with a capture buffer that fd 2 knows nothing about.
    """
    terminal_path, log_path = tmp_path / "terminal", tmp_path / "session.log"
    handle, saved = redirect_fd2_to(terminal_path)
    outer = sys.stderr
    try:
        with stderr_to(log_path):
            assert sys.stderr is outer  # not repointed at the terminal
            stream = open(os.dup(2), "w", errors="replace")
            stream.write("Traceback (most recent call last):\n")
            stream.close()
    finally:
        restore_fd2(handle, saved)

    assert "Traceback" in log_path.read_text()
    assert "Traceback" not in terminal_path.read_text()


def test_everything_is_put_back_even_when_the_block_raises(tmp_path):
    """The next thing to run is the `transcript:` line, or a report of
    whatever went wrong - both of which need a working stderr."""
    terminal_path, log_path = tmp_path / "terminal", tmp_path / "session.log"
    handle, saved = redirect_fd2_to(terminal_path)
    before = sys.__stderr__
    try:
        try:
            with stderr_to(log_path):
                raise RuntimeError("the session died")
        except RuntimeError:
            pass

        assert sys.__stderr__ is before
        os.write(2, b"back on the terminal\n")
    finally:
        restore_fd2(handle, saved)

    assert "back on the terminal" in terminal_path.read_text()
