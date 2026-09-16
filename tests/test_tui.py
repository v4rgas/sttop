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


@pytest.fixture(autouse=True)
def no_real_engine_boot(monkeypatch):
    """UI tests exercise widgets, not audio hardware or downloaded models."""
    class FakeEngine:
        def __init__(self, *args, **kwargs):
            self._status = EngineStatus()

        def status(self):
            return self._status

        def transcript(self):
            return ""

        def toggle_pause(self):
            return False

        def rename_speaker(self, old, new):
            return 0

        async def start(self, title=None):
            return Path("/tmp/sttop-test-session.md")

        async def stop(self):
            return None

    monkeypatch.setattr("sttop.tui.Engine", FakeEngine)


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


def test_yank_copies_the_transcript_so_far():
    """`y` mid-meeting: the transcript so far lands on the clipboard, whole."""

    async def scenario():
        app = SttopApp(Config())
        copied: list[str] = []
        app.copy_to_clipboard = copied.append
        app.engine.transcript = lambda: "# demo\n\n- `00:01` **you** — hola\n"
        async with app.run_test() as pilot:
            await pilot.press("y")
        return copied

    assert asyncio.run(scenario()) == ["# demo\n\n- `00:01` **you** — hola\n"]


def test_yank_before_any_transcript_copies_nothing():
    async def scenario():
        app = SttopApp(Config())
        copied: list[str] = []
        app.copy_to_clipboard = copied.append
        async with app.run_test() as pilot:  # no journal yet: transcript is ""
            await pilot.press("y")
        return copied

    assert asyncio.run(scenario()) == []


def test_quit_waits_for_a_background_pull():
    """A pull racing shutdown must finish before the app exits - quitting
    mid-rebase would leave the repo half-done for the close-time sync."""
    events: list[str] = []

    async def scenario():
        config = Config()
        config.storage.git_remote = "git@example.com:x.git"
        app = SttopApp(config)
        app._banner = lambda message: None
        app.exit = lambda result=None: None

        async def stop():
            events.append("capture stopped")

        app.engine.stop = stop
        app._pull_worker = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_later(
            0.01,
            lambda: (events.append("pull finished"), app._pull_worker.set_result(None)),
        )
        await app.action_quit()
        return events

    assert asyncio.run(scenario()) == ["capture stopped", "pull finished"]


def test_no_remote_means_no_pull_worker():
    async def scenario():
        app = SttopApp(Config())
        async with app.run_test():
            return app._pull_worker

    assert asyncio.run(scenario()) is None


def test_a_long_session_path_gives_up_directories_not_the_filename(tmp_path):
    path = Path.home() / ".local/share/sttop/sessions/2026-08-11-1122-standup.md"
    assert shorten(path, 200).startswith("~/")
    assert shorten(path, 30) == "…/2026-08-11-1122-standup.md"
    assert shorten(path, 10) == "2026-08-11-1122-standup.md"
    assert shorten(None, 30) == ""


def test_quit_asks_for_a_name_and_tab_completes(tmp_path):
    (tmp_path / "micelio").mkdir()
    (tmp_path / "micelio" / "daily.2026-09-13-1030.md").write_text("x")
    session = tmp_path / "2026-09-14-1030-session.md"
    session.write_text("x")

    async def scenario():
        config = Config()
        config.sessions_dir = str(tmp_path)
        app = SttopApp(config)

        async def stop():
            return session

        app.engine.stop = stop
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("q", "m", "tab")
            field = app.screen.query_one("Input")
            assert field.value == "micelio"
            await pilot.press("slash", "d", "tab")
            assert field.value == "micelio/daily"
            await pilot.press("enter")
            await pilot.pause()
        return app.return_value

    assert asyncio.run(scenario()) == tmp_path / "micelio" / "daily.2026-09-14-1030.md"


def test_loading_shows_stage_and_progress_without_claiming_to_record(monkeypatch):
    async def run():
        app = SttopApp(Config())
        entered = asyncio.Event()
        release = asyncio.Event()

        async def slow_start(title=None):
            app.engine._status.loading = "Loading speech model"
            app.engine._status.loading_elapsed = 3
            entered.set()
            await release.wait()
            app.engine._status.loading = ""
            return Path("/tmp/sttop-test-session.md")

        monkeypatch.setattr(app.engine, "start", slow_start)
        async with app.run_test() as pilot:
            await entered.wait()
            app._refresh_status()
            assert app.query_one("#loading-progress").display
            assert "Loading speech model" in str(app.query_one("#banner").render())
            assert "rec" not in status_line(app.engine.status(), 80)
            release.set()
            await pilot.pause()
            assert not app.query_one("#loading-progress").display

    asyncio.run(run())


def test_tui_explains_recording_loading_catchup_and_live_states():
    async def run():
        app = SttopApp(Config())
        async with app.run_test():
            status = app.engine._status
            status.running = True
            status.loading = "Loading speech model"
            status.backlog = 4
            app._refresh_status()
            banner = app.query_one("#banner")
            assert "Recording — buffering speech" in str(banner.render())
            assert "Loading speech model" in str(banner.render())
            assert "rec" in status_line(status, 80)
            assert app.query_one("#loading-progress").display

            status.loading = ""
            status.catching_up = True
            app._refresh_status()
            assert "catching up: 4" in str(banner.render())
            assert not app.query_one("#loading-progress").display

            status.backlog = 0
            status.catching_up = False
            app._refresh_status()
            assert "live transcription" in str(banner.render())

            status.loading = "Loading speaker model"
            status.paused = True
            app._refresh_status()
            assert "Paused" in str(banner.render())
            assert "Recording — buffering" not in str(banner.render())

            app._finishing = True
            status.running = False
            app._refresh_status()
            assert "Recording stopped" in str(banner.render())

    asyncio.run(run())
