"""Capture a PulseAudio/PipeWire source as 16 kHz mono PCM via ffmpeg.

ffmpeg is used instead of a PortAudio binding because monitor sources (the
"what you hear" side of the capture) are exposed cleanly by the pulse backend,
whereas PortAudio device indices for monitors are inconsistent under PipeWire.

Reading is an asyncio task, not a thread: it is blocking I/O on a pipe, which
is exactly what an event loop handles well, and it makes cancellation and
shutdown structured rather than hand-rolled from events and joins.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import subprocess
import wave
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .. import FRAME_BYTES, SAMPLE_RATE
from .devices import AudioError

FrameCallback = Callable[[bytes], None]
ErrorCallback = Callable[[str], None]

#: Keep only the tail of ffmpeg's stderr - all we ever report is the last line,
#: and an unbounded buffer would grow for the whole session.
_STDERR_TAIL_LINES = 5


def _ffmpeg_command(pulse_source: str, *extra: str) -> list[str]:
    """The one true ffmpeg invocation: one pulse source in, 16 kHz mono s16le
    out on stdout. `extra` is inserted between the input and output options."""
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-f", "pulse",
        "-i", pulse_source,
        *extra,
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-f", "s16le",
        "-",
    ]


class SourceCapture:
    """Streams one audio source, handing fixed-size PCM frames to a callback.

    `on_frame` runs on the event loop, so it must stay cheap - VAD only. Any
    real work belongs behind the queue the segmenter feeds.
    """

    def __init__(
        self,
        label: str,
        pulse_source: str,
        on_frame: FrameCallback,
        on_error: ErrorCallback | None = None,
        wav_path: Path | None = None,
    ) -> None:
        self.label = label
        self.pulse_source = pulse_source
        self._on_frame = on_frame
        self._on_error = on_error
        self._wav_path = wav_path

        self._proc: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._stderr_tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        self._wav: wave.Wave_write | None = None
        self._stopping = False

        #: Smoothed 0..1 loudness, for the level meters in the TUI.
        self.level: float = 0.0
        self.frames_seen: int = 0

    async def start(self) -> None:
        if not shutil.which("ffmpeg"):
            raise AudioError("ffmpeg not found on PATH")

        if self._wav_path is not None:
            self._wav_path.parent.mkdir(parents=True, exist_ok=True)
            self._wav = wave.open(str(self._wav_path), "wb")
            self._wav.setnchannels(1)
            self._wav.setsampwidth(2)
            self._wav.setframerate(SAMPLE_RATE)

        self._proc = await asyncio.create_subprocess_exec(
            *_ffmpeg_command(self.pulse_source),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._task = asyncio.create_task(self._pump(), name=f"capture-{self.label}")
        self._stderr_task = asyncio.create_task(
            self._drain_stderr(), name=f"capture-{self.label}-stderr"
        )

    async def _pump(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            while True:
                frame = await self._proc.stdout.readexactly(FRAME_BYTES)
                self.frames_seen += 1
                self._update_level(frame)
                if self._wav is not None:
                    self._wav.writeframes(frame)
                self._on_frame(frame)
        except asyncio.IncompleteReadError:
            # The pipe closed: either we are stopping, or the source went away.
            if not self._stopping:
                self._report_failure()

    async def _drain_stderr(self) -> None:
        """Read stderr continuously. Not only for the message - an unread pipe
        fills at 64 KiB and would block ffmpeg, stalling capture entirely."""
        pipe = self._proc.stderr if self._proc is not None else None
        if pipe is None:
            return
        async for raw in pipe:
            line = raw.decode(errors="replace").strip()
            if line:
                self._stderr_tail.append(line)

    def _update_level(self, frame: bytes) -> None:
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples * samples)))
        # Attack fast, release slow, so brief peaks stay visible on the meter.
        weight = 0.5 if rms > self.level else 0.15
        self.level = (1 - weight) * self.level + weight * min(rms * 4.0, 1.0)

    def _report_failure(self) -> None:
        if self._on_error is None:
            return
        code = self._proc.returncode if self._proc is not None else None
        detail = self._stderr_tail[-1] if self._stderr_tail else f"ffmpeg exited {code}"
        self._on_error(f"[{self.label}] {detail}")

    async def stop(self) -> None:
        """Terminate ffmpeg and await the reader tasks. Idempotent."""
        self._stopping = True

        if self._proc is not None and self._proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self._proc.terminate()

        for attr in ("_task", "_stderr_task"):
            task = getattr(self, attr)
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                setattr(self, attr, None)

        if self._proc is not None:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._proc.wait(), timeout=3)
            if self._proc.returncode is None:
                self._proc.kill()
                await self._proc.wait()
            self._proc = None

        if self._wav is not None:
            self._wav.close()
            self._wav = None
        self.level = 0.0


def check_source(pulse_source: str, seconds: float = 1.0) -> float:
    """Record briefly and report peak level - used by `sttop devices --test`."""
    proc = subprocess.run(
        _ffmpeg_command(pulse_source, "-t", str(seconds)),
        capture_output=True,
        timeout=seconds + 10,
    )
    if proc.returncode != 0:
        raise AudioError(
            proc.stderr.decode(errors="replace").strip() or "capture failed"
        )
    if not proc.stdout:
        return 0.0
    samples = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.max(np.abs(samples))) if samples.size else 0.0
