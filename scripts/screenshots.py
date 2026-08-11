"""Render the README screenshots.

Drives the real widgets from `sttop.tui` with a scripted transcript instead of a
live engine, so the images in `docs/` can never drift from the actual UI.

    uv run --extra dev python scripts/screenshots.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Footer, Input, Static

from sttop.engine import EngineStatus
from sttop.journal import Utterance
from sttop.tui import StatusBar, SttopApp, TranscriptLog

DOCS = Path(__file__).resolve().parent.parent / "docs"
SIZE = (128, 17)

SCRIPT = [
    ("you", 3.4, "okay, migration — are we still saying friday?"),
    ("spk1", 8.1, "Friday is tight. The backfill alone is six hours."),
    ("spk1", 14.0, "If anything slips we'd be debugging it over the weekend."),
    ("spk2", 21.7, "Monday morning, with the whole team awake. Plus one."),
    ("you", 27.2, "fine — monday. who owns the runbook?"),
    ("spk2", 33.9, "I'll write it today and put it in the deploy channel."),
    ("spk1", 41.5, "I'll do a dry run against staging tomorrow afternoon."),
    ("you", 48.0, "great. anything else before we drop?"),
    ("spk?", 52.6, "…"),
    ("spk1", 55.3, "Nothing from me."),
]

STATUS = EngineStatus(
    elapsed=252.0,
    backlog=1,
    levels={"mic": 0.35, "system": 0.72},
    running=True,
    utterances=len(SCRIPT),
    speakers=3,
    backend="parakeet-tdt/cpu onnx",
    diarizer="ecapa @0.50",
)


class DemoApp(App):
    """The real widgets and stylesheet, with the engine stubbed out.

    Deliberately not a subclass of SttopApp: Textual dispatches `on_mount` to
    every class in the MRO, so inheriting would also boot the real engine.
    """

    CSS = SttopApp.CSS
    BINDINGS = SttopApp.BINDINGS

    def __init__(self, theme: str, banner: str, status: EngineStatus) -> None:
        super().__init__()
        self._theme = theme
        self._banner_text = banner
        self._status = status

    def _banner(self, message: str) -> None:
        self.query_one("#banner", Static).update(f"[dim]{message}[/]")

    def compose(self) -> ComposeResult:
        yield StatusBar()
        yield Static("", id="banner")
        with Horizontal():
            yield TranscriptLog()
        yield Input(placeholder="rename: old=new  (e.g. spk1=Ana)", id="rename")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "sttop"
        self.theme = self._theme
        self._banner(self._banner_text)
        self.query_one(StatusBar).render_status(self._status)
        log = self.query_one(TranscriptLog)
        for speaker, start, text in SCRIPT:
            log.add(Utterance("mic", speaker, start, start + 2, text))
        log.focus()


async def shoot(name: str, **kwargs) -> None:
    app = DemoApp(**kwargs)
    async with app.run_test(size=SIZE) as pilot:
        if name == "rename":
            field = app.query_one("#rename", Input)
            field.add_class("visible")  # what action_rename does on `r`
            field.focus()
            await pilot.press(*"spk1=Ana")
        await pilot.pause()
        app.save_screenshot(str(DOCS / f"{name}.svg"))
    print(f"wrote docs/{name}.svg")


async def main() -> None:
    DOCS.mkdir(exist_ok=True)
    await shoot(
        "sttop",
        theme="ansi-dark",
        banner="writing → ~/.local/share/sttop/sessions/2026-08-10-1432-standup.md",
        status=STATUS,
    )
    # The default ansi-dark/ansi-light themes paint with the terminal's own 16
    # colours, which an SVG has no access to - so the theming shots use named
    # Textual themes, which carry their own palette.
    for theme in ("gruvbox", "solarized-light"):
        await shoot(
            f"theme-{theme}",
            theme=theme,
            banner=f"ui.theme = \"{theme}\"",
            status=STATUS,
        )
    await shoot(
        "rename",
        theme="ansi-dark",
        banner="rename a speaker — the transcript on disk is rewritten too",
        status=STATUS,
    )

if __name__ == "__main__":
    asyncio.run(main())
