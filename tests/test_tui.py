"""Layout checks: the UI has to survive a narrow terminal without a scrollbar
and without wrapping a turn back to the left margin, where the continuation
reads as a new speaker.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from rich.console import Console
from rich.text import Text

from sttop.config import Config
from sttop.engine import EngineStatus
from sttop.journal import Utterance
from sttop.tui import NARROW, SttopApp, shorten, status_line

LONG = (
    "Loading model. Entonces usa dos modelos y eso basicamente dice como que "
    "el, o sea eso es el, la weá de escritura, pues está usando lo mismo"
)


def screen_lines(width: int, height: int, utterances) -> list[str]:
    return asyncio.run(_screen_lines(width, height, utterances))


async def _screen_lines(width: int, height: int, utterances) -> list[str]:
    app = SttopApp(Config())
    async with app.run_test(size=(width, height)) as pilot:
        log = app.query_one("TranscriptLog")
        for speaker, text in utterances:
            log.add(Utterance("system", speaker, 15.0, 16.0, text))
        await pilot.pause()
        # color_system=None so the capture is plain text: with escapes left in,
        # a length assertion measures the styling rather than the layout.
        console = Console(width=width, height=height, color_system=None)
        with console.capture() as capture:
            console.print(app.screen._compositor)
    return capture.get().splitlines()


@pytest.mark.parametrize("width", [120, 80, 60, NARROW, 40, 30])
def test_nothing_overflows_the_terminal(width):
    lines = screen_lines(width, 20, [("spk1", LONG), ("you", LONG)])
    assert all(len(line) <= width for line in lines)


def test_no_scrollbar_is_drawn():
    """Textual draws scrollbars with these block glyphs; none should appear."""
    lines = screen_lines(60, 12, [("spk1", LONG)] * 8)
    assert not any(glyph in line for line in lines for glyph in ("▊", "▎", "█"))


def test_a_wrapped_turn_continues_under_its_text():
    """The bug this guards: a continuation starting at column 0 looks like a
    new speaker's line."""
    lines = screen_lines(80, 20, [("spk1", LONG)])
    body = [line for line in lines if "Loading model" in line or "escritura" in line]
    assert len(body) == 2, "the long turn should have wrapped"
    first, second = body
    assert second.index(second.strip()[0]) == first.index("Loading model")


def test_a_narrow_terminal_stacks_instead_of_wrapping_to_a_sliver():
    """Below NARROW the gutter is dropped, so text gets the whole width."""
    lines = screen_lines(38, 20, [("spk1", LONG)])
    text = [line for line in lines if "Loading model" in line]
    assert text and text[0].strip().startswith("Loading model")


def test_the_status_line_sheds_detail_before_it_overflows():
    status = EngineStatus(elapsed=61.0, backend="parakeet-tdt/cpu", diarizer="ecapa")
    for width in (120, 96, 80, 60, 46, 34, 20):
        assert len(Text.from_markup(status_line(status, width))) <= width


def test_the_status_line_keeps_the_clock_at_any_width():
    status = EngineStatus(elapsed=61.0)
    assert all("01:01" in status_line(status, width) for width in (20, 46, 120))


def test_a_long_session_path_gives_up_directories_not_the_filename(tmp_path):
    path = Path.home() / ".local/share/sttop/sessions/2026-08-11-1122-standup.md"
    assert shorten(path, 200).startswith("~/")
    assert shorten(path, 30) == "…/2026-08-11-1122-standup.md"
    assert shorten(path, 10) == "2026-08-11-1122-standup.md"
    assert shorten(None, 30) == ""
