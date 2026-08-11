"""The terminal UI - a live transcript with level meters, in the `top` idiom."""

from __future__ import annotations

import contextlib
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Footer, Input, RichLog, Static

from .config import Config
from .diarize import SELF_LABEL
from .engine import Engine, EngineStatus
from .journal import Utterance, clock
from .terminal import detect_theme

SPEAKER_COLORS = ["cyan", "magenta", "yellow", "green", "blue", "red"]


def meter(level: float, width: int = 10) -> str:
    filled = int(max(0.0, min(1.0, level)) * width)
    return "█" * filled + "·" * (width - filled)


def _parse_rename(raw: str) -> tuple[str, str] | None:
    """Split an `old=new` rename entry, or None if it isn't one."""
    old, sep, new = raw.partition("=")
    if not sep:
        return None
    old, new = old.strip(), new.strip()
    return (old, new) if old and new else None


class StatusBar(Static):
    """Top line: clock, sources, backend, backlog."""

    def render_status(self, status: EngineStatus) -> None:
        state = "[yellow]paused[/]" if status.paused else "[green]●[/] rec"
        mic = meter(status.levels.get("mic", 0.0))
        system = meter(status.levels.get("system", 0.0))
        backlog = (
            f"[red]queue {status.backlog}[/]"
            if status.backlog > 3
            else f"queue {status.backlog}"
        )
        self.update(
            f"[b]sttop[/]  {state} [b]{clock(status.elapsed)}[/]   "
            f"mic [cyan]{mic}[/]  sys [magenta]{system}[/]   "
            f"[dim]{status.backend} · {status.diarizer} · {backlog} · "
            f"{status.utterances} lines · {status.speakers} voices[/]"
        )


class TranscriptLog(RichLog):
    """The scrolling transcript."""

    def __init__(self) -> None:
        super().__init__(wrap=True, markup=True, auto_scroll=True)
        self._colors: dict[str, str] = {}

    def color_for(self, speaker: str) -> str:
        if speaker == SELF_LABEL:
            return "bold cyan"
        if speaker not in self._colors:
            index = len(self._colors) % len(SPEAKER_COLORS)
            self._colors[speaker] = SPEAKER_COLORS[index]
        return self._colors[speaker]

    def add(self, utterance: Utterance) -> None:
        color = self.color_for(utterance.speaker)
        self.write(
            f"[dim]{clock(utterance.start)}[/] "
            f"[{color}]{utterance.speaker:<6}[/] {utterance.text}"
        )


class SttopApp(App):
    CSS = """
    Screen { layout: vertical; }
    StatusBar { height: 1; padding: 0 1; background: $panel; color: $text; }
    #banner { height: auto; padding: 0 1; color: $text-muted; }
    TranscriptLog { height: 1fr; padding: 0 1; }
    Input { display: none; dock: bottom; }
    Input.visible { display: block; }
    """

    BINDINGS = [
        Binding("q", "quit", "quit"),
        Binding("space", "pause", "pause"),
        Binding("r", "rename", "rename speaker"),
        Binding("ctrl+c", "quit", show=False),
    ]

    def __init__(self, config: Config, title: str | None = None) -> None:
        super().__init__()
        self.config = config
        self.session_title = title
        self.engine = Engine(
            config, self._on_utterance, self._on_error, self._on_rename
        )
        self.journal_path: Path | None = None
        # Resolved before Textual grabs the terminal - the OSC 11 query needs
        # raw mode and a tty that nobody else is reading.
        self._theme = detect_theme(config.ui.theme)

    # -- composition -------------------------------------------------------

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
        self._banner("loading model…")
        # prepare() offloads the slow model load itself, so this stays on the loop.
        self.run_worker(self._boot(), exclusive=True)

    async def _boot(self) -> None:
        try:
            self.journal_path = await self.engine.start(self.session_title)
        except Exception as exc:
            # A start that failed halfway still left the models loaded, the
            # executor thread running, the journal open and - if the mic came
            # up before the failure - an ffmpeg reading from it.
            with contextlib.suppress(Exception):
                await self.engine.stop()
            self._fatal(f"{type(exc).__name__}: {exc}")
            return
        self._banner(self._recording_message())
        self.set_interval(0.2, self._refresh_status)

    def _recording_message(self) -> str:
        return f"writing → {self.journal_path}"

    def _banner(self, message: str) -> None:
        self.query_one("#banner", Static).update(f"[dim]{message}[/]")

    def _fatal(self, message: str) -> None:
        self.query_one(TranscriptLog).write(f"[bold red]error[/] {message}")
        self._banner("startup failed — press q to quit")

    # -- engine callbacks --------------------------------------------------
    # The engine calls these on the event loop, so they can touch widgets
    # directly - no call_from_thread marshalling.

    def _on_utterance(self, utterance: Utterance) -> None:
        self.query_one(TranscriptLog).add(utterance)

    def _on_error(self, message: str) -> None:
        self.query_one(TranscriptLog).write(f"[yellow]warn[/] {message}")

    def _on_rename(self, old: str, new: str, lines: int) -> None:
        """The diarizer decided two voices were one person.

        The transcript file is rewritten, but lines already scrolled past in
        the log are not - RichLog has no way to reach back into them - so this
        says what happened instead of silently disagreeing with the file.
        """
        self.query_one(TranscriptLog).write(
            f"[dim]— {old} was {new}; {lines} earlier line(s) relabelled[/]"
        )

    def _refresh_status(self) -> None:
        self.query_one(StatusBar).render_status(self.engine.status())

    # -- actions -----------------------------------------------------------

    def action_pause(self) -> None:
        paused = self.engine.toggle_pause()
        self._banner(
            "paused — audio is discarded while paused"
            if paused
            else self._recording_message()
        )

    def action_rename(self) -> None:
        field = self.query_one("#rename", Input)
        field.add_class("visible")
        field.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        field = event.input
        field.remove_class("visible")
        field.value = ""
        self.query_one(TranscriptLog).focus()

        pair = _parse_rename(event.value)
        if pair is None:
            self._banner("rename needs the form old=new, e.g. spk1=Ana")
            return
        old, new = pair
        changed = self.engine.rename_speaker(old, new)
        self._banner(f"renamed {old} → {new} in {changed} line(s)")

    async def action_quit(self) -> None:
        # Awaiting the drain keeps the UI painting while the last few segments
        # finish transcribing, instead of freezing on the way out.
        self._banner("finishing transcription…")
        self.exit(await self.engine.stop())
