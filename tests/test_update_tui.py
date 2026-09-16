"""Update notices stay in the UI and never become transcript content."""

import asyncio
import threading
from pathlib import Path

from sttop.config import Config
from sttop.tui import SttopApp, TranscriptLog
from sttop.updates import UpdateNotice


def test_background_update_notice_reaches_ui_without_waiting_for_boot(monkeypatch):
    started = threading.Event()
    notice = UpdateNotice("0.9.0", "0.10.0")
    notes = []

    def start_check(callback):
        def check():
            started.set()
            callback(notice)

        threading.Thread(target=check, daemon=True).start()

    monkeypatch.setattr("sttop.tui.start_update_check", start_check)
    monkeypatch.setattr(
        TranscriptLog, "note", lambda self, message: notes.append(message)
    )

    async def scenario():
        app = SttopApp(Config())
        boot_release = asyncio.Event()

        async def boot():
            await boot_release.wait()
            app.journal_path = Path("/tmp/test.md")

        app._boot = boot
        async with app.run_test() as pilot:
            for _ in range(20):
                if notes:
                    break
                await pilot.pause(0.01)
            assert started.is_set()
            assert notes == [notice.message]
            assert app.journal_path is None
            assert app.engine.transcript() == ""
            boot_release.set()

    asyncio.run(scenario())


def test_update_arriving_after_app_exit_is_ignored():
    app = SttopApp(Config())
    # A daemon network request can complete after Textual has closed.
    app._on_update_available(UpdateNotice("0.9.0", "0.10.0"))
