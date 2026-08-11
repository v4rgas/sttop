"""The terminal UI - a live transcript with level meters, in the `top` idiom."""

from __future__ import annotations

import contextlib
from pathlib import Path

from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Input, RichLog, Static

from .config import Config
from .diarize import SELF_LABEL
from .engine import Engine, EngineStatus
from .journal import Utterance, clock
from .terminal import detect_theme

SPEAKER_COLORS = ["cyan", "magenta", "yellow", "green", "blue", "red"]

#: Below this width the timestamp/speaker gutter costs more than it earns, and
#: an utterance is given its own header line instead.
NARROW = 46
#: Room a wrapped line needs to the right of the gutter before stacking beats
#: aligning - narrower than this and text arrives a word at a time.
MIN_TEXT = 24


def meter(level: float, width: int = 10) -> str:
    filled = int(max(0.0, min(1.0, level)) * width)
    return "█" * filled + "·" * (width - filled)


def shorten(path: Path | None, width: int) -> str:
    """Fit a session path into `width`, giving up directories before the name.

    The filename is how you find the session again; the directory is the same
    for every session you will ever record.
    """
    if path is None:
        return ""
    text = str(path)
    home = str(Path.home())
    if text.startswith(home + "/"):
        text = "~" + text[len(home) :]
    if len(text) <= width:
        return text
    name = path.name
    return name if len(name) >= width else f"…/{name}"


def _parse_rename(raw: str) -> tuple[str, str] | None:
    """Split an `old=new` rename entry, or None if it isn't one."""
    old, sep, new = raw.partition("=")
    if not sep:
        return None
    old, new = old.strip(), new.strip()
    return (old, new) if old and new else None


class StatusBar(Static):
    """Top line: clock, sources, backend, backlog.

    One line, always - so it sheds detail as the terminal narrows rather than
    letting the useful half scroll off the right edge. What survives longest is
    what you cannot get anywhere else on screen: that it is still recording,
    and for how long.
    """

    def render_status(self, status: EngineStatus) -> None:
        self.update(status_line(status, self.size.width or 80))


def status_line(status: EngineStatus, width: int) -> str:
    """The richest variant that fits, rather than a guess keyed on width.

    Guessing where the cutoffs fall is how a status bar ends up one column too
    long on somebody else's font and backend name; this measures instead.
    """
    state = "[yellow]paused[/]" if status.paused else "[green]●[/] rec"
    head = f"[b]sttop[/]  {state} [b]{clock(status.elapsed)}[/]"

    def meters(bars: int) -> str:
        mic = meter(status.levels.get("mic", 0.0), bars)
        system = meter(status.levels.get("system", 0.0), bars)
        return f"mic [cyan]{mic}[/]  sys [magenta]{system}[/]"

    backlog = (
        f"[red]queue {status.backlog}[/]"
        if status.backlog > 3
        else f"queue {status.backlog}"
    )
    counts = [backlog, f"{status.utterances} lines", f"{status.speakers} voices"]

    def joined(bars: int, tail: list[str]) -> str:
        parts = [head, meters(bars)]
        if tail:
            parts.append(f"[dim]{' · '.join(tail)}[/]")
        return "   ".join(parts)

    # Richest first. Detail goes before the meters, and the meters before the
    # clock: what you cannot learn anywhere else on screen survives longest.
    for candidate in (
        joined(10, [status.backend, status.diarizer, *counts]),
        joined(10, [status.diarizer, *counts]),
        joined(10, counts),
        joined(10, counts[:1]),
        joined(10, []),
        joined(4, []),
        head,
    ):
        if _visible(candidate) <= width:
            return candidate
    return head


def _visible(markup: str) -> int:
    """Width of `markup` once the tags are gone."""
    return len(Text.from_markup(markup))


class TranscriptLog(RichLog):
    """The scrolling transcript.

    Each utterance is a two-column grid rather than one padded string, so a
    line that wraps continues *under the text*, not back at the left margin
    where it reads as a new speaker's turn.
    """

    def __init__(self) -> None:
        super().__init__(wrap=True, markup=True, auto_scroll=True)
        self._colors: dict[str, str] = {}
        #: Widest label seen, so the gutter fits renamed speakers ("Ana",
        #: "Interviewer") instead of clipping them to spk-sized boxes.
        self._label_width = 4

    def color_for(self, speaker: str) -> str:
        if speaker == SELF_LABEL:
            return "bold cyan"
        if speaker not in self._colors:
            index = len(self._colors) % len(SPEAKER_COLORS)
            self._colors[speaker] = SPEAKER_COLORS[index]
        return self._colors[speaker]

    @property
    def _width(self) -> int:
        return self.size.width or 80

    def add(self, utterance: Utterance) -> None:
        speaker = Text(utterance.speaker, style=self.color_for(utterance.speaker))
        stamp = Text(clock(utterance.start), style="dim")
        self._label_width = max(self._label_width, len(utterance.speaker))

        gutter = len(stamp) + self._label_width + 2
        if self._width < NARROW or self._width - gutter < MIN_TEXT:
            # No room for a gutter and a readable column of text both. Give the
            # utterance its own header line and let the text have the width.
            header = stamp.copy()
            header.append(" ")
            header.append(speaker)
            self.write(header)
            self.write(Text(utterance.text))
            return

        grid = Table.grid(expand=True, padding=(0, 1))
        grid.add_column(width=len(stamp), no_wrap=True)
        grid.add_column(width=self._label_width, no_wrap=True, overflow="ellipsis")
        grid.add_column(ratio=1, overflow="fold")
        grid.add_row(stamp, speaker, Text(utterance.text))
        self.write(grid)

    def note(self, message: str) -> None:
        """An aside from the app itself - warnings, merges, errors."""
        self.write(Text(message, style="dim"), shrink=True)


class SttopApp(App):
    CSS = """
    /* No scrollbars anywhere: everything wraps, so a horizontal bar would
       only ever be an artifact, and the vertical one costs a column that
       matters at 40 columns and buys nothing the wheel does not. */
    Screen { layout: vertical; overflow: hidden; }
    StatusBar { height: 1; padding: 0 1; background: $panel; color: $text; }
    #banner { height: auto; max-height: 2; padding: 0 1; color: $text-muted; }
    TranscriptLog {
        height: 1fr;
        padding: 0 1;
        overflow-x: hidden;
        scrollbar-size-vertical: 0;
        scrollbar-size-horizontal: 0;
    }
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
        return f"writing → {shorten(self.journal_path, self.size.width - 11)}"

    def _banner(self, message: str) -> None:
        self.query_one("#banner", Static).update(f"[dim]{message}[/]")

    def _fatal(self, message: str) -> None:
        self.query_one(TranscriptLog).write(Text(f"error {message}", style="bold red"))
        self._banner("startup failed — press q to quit")

    # -- engine callbacks --------------------------------------------------
    # The engine calls these on the event loop, so they can touch widgets
    # directly - no call_from_thread marshalling.

    def _on_utterance(self, utterance: Utterance) -> None:
        self.query_one(TranscriptLog).add(utterance)

    def _on_error(self, message: str) -> None:
        self.query_one(TranscriptLog).write(Text(f"warn {message}", style="yellow"))

    def _on_rename(self, old: str, new: str, lines: int) -> None:
        """The diarizer decided two voices were one person.

        The transcript file is rewritten, but lines already scrolled past in
        the log are not - RichLog has no way to reach back into them - so this
        says what happened instead of silently disagreeing with the file.
        """
        self.query_one(TranscriptLog).note(
            f"— {old} was {new}; {lines} earlier line(s) relabelled"
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
