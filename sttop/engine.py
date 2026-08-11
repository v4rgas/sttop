"""Wires capture -> VAD -> transcription -> diarization -> Markdown journal.

Concurrency model - one thread boundary, and it is the model:

  capture tasks (2)   asyncio: read ffmpeg stdout, run VAD inline (microseconds)
        |
   asyncio.Queue
        |
  consume task        asyncio: awaits the executor, appends, notifies the UI
        |
  ThreadPoolExecutor(max_workers=1)   the only thread: transcribe + embed

Everything except the model runs on the event loop, so the UI needs no
cross-thread marshalling and shutdown is ordinary task cancellation. The
executor is deliberately single-threaded: transcription is CPU-bound and
already internally parallel, so a second worker would only thrash the cache -
and serialising it keeps utterances in the order they were spoken.

When transcription falls behind, the queue absorbs the lag and `backlog`
reports it rather than dropping audio.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import diarize as diarize_mod
from . import stt
from .audio import devices
from .audio.capture import SourceCapture
from .audio.segmenter import Segment, Segmenter
from .config import Config
from .journal import Journal, Utterance

MIC = "mic"
SYSTEM = "system"


UtteranceCallback = Callable[[Utterance], None]
ErrorCallback = Callable[[str], None]


def _capture_for(spec: devices.CaptureSpec):
    """Which reader a source needs.

    Everything is an ffmpeg subprocess except macOS system audio, which comes
    from an in-process ScreenCaptureKit stream - imported lazily so a Linux
    run never touches the Apple bindings.
    """
    if spec.backend == "screencapture":
        from .audio.screencapture import ScreenAudioCapture

        return ScreenAudioCapture
    return SourceCapture


@dataclass
class EngineStatus:
    """Everything the UI needs to draw one frame of the status bar."""

    elapsed: float = 0.0
    backlog: int = 0
    levels: dict[str, float] = field(default_factory=dict)
    paused: bool = False
    running: bool = False
    utterances: int = 0
    speakers: int = 0
    backend: str = "no backend"
    diarizer: str = "diarize off"


class Engine:
    def __init__(
        self,
        config: Config,
        on_utterance: UtteranceCallback,
        on_error: ErrorCallback | None = None,
    ) -> None:
        self.config = config
        self._on_utterance = on_utterance
        self._on_error = on_error or (lambda message: None)

        self.transcriber: stt.Transcriber | None = None
        self.diarizer: diarize_mod.SpeakerLabeler | None = None
        self.journal: Journal | None = None
        #: Resolved by prepare(); None until then. `sys_source` stays None when
        #: the platform has no way to hear itself - see _resolve_sources.
        self.mic_source: devices.CaptureSpec | None = None
        self.sys_source: devices.CaptureSpec | None = None
        #: Raised during prepare(), reported on the loop once start() runs.
        self._warnings: list[str] = []

        self._captures: list[SourceCapture] = []
        self._segmenters: dict[str, Segmenter] = {}
        self._queue: asyncio.Queue[Segment | None] = asyncio.Queue()
        self._consumer: asyncio.Task | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
        self._t0: float = 0.0
        self._paused = False
        self._running = False

    # -- lifecycle ---------------------------------------------------------

    async def prepare(self) -> None:
        """Resolve devices and load models - seconds of blocking work, so all
        of it goes to the executor and the caller stays responsive."""
        loop = asyncio.get_running_loop()
        self.mic_source, self.sys_source = await loop.run_in_executor(
            self._executor, self._resolve_sources
        )
        self.transcriber = await loop.run_in_executor(
            self._executor, stt.build, self.config.stt
        )
        self.diarizer = await loop.run_in_executor(
            self._executor, diarize_mod.build, self.config.diarize
        )

    def _resolve_sources(
        self,
    ) -> tuple[devices.CaptureSpec, devices.CaptureSpec | None]:
        """The mic is required; the system stream is not.

        A machine with no monitor source - any stock macOS, mainly - can still
        record the half of the conversation the microphone hears, and a
        one-sided transcript beats refusing to start. The mic is different:
        without it there is nothing to transcribe.
        """
        mic = devices.resolve(self.config.audio.mic_source, monitor=False)
        try:
            system = devices.resolve(self.config.audio.system_source, monitor=True)
        except devices.SystemAudioUnavailable as exc:
            self._warnings.append(f"[system] {exc}")
            system = None
        return mic, system

    async def start(self, title: str | None = None) -> Path:
        if self._running:
            raise RuntimeError("engine already running")
        if self.transcriber is None:
            await self.prepare()

        self.journal = Journal.create(
            Path(self.config.sessions_dir),
            title,
            mic_source=str(self.mic_source),
            sys_source=str(self.sys_source) if self.sys_source else "unavailable",
            backend=self.transcriber.describe,
        )
        self._t0 = time.monotonic()
        self._running = True

        for message in self._warnings:
            self._on_error(message)
        self._warnings.clear()

        streams = [(MIC, self.mic_source)]
        if self.sys_source is not None:
            streams.append((SYSTEM, self.sys_source))

        for label, source in streams:
            segmenter = Segmenter(
                label, self.config.vad, self._queue.put_nowait, clock=self._session_clock
            )
            capture = _capture_for(source)(
                label,
                source,
                on_frame=segmenter.feed,
                on_error=self._on_error,
                wav_path=self._wav_path(label),
            )
            try:
                await capture.start()
            except Exception as exc:
                # The mic is the session; without it there is nothing to
                # transcribe. System audio is the half we can do without - a
                # revoked screen recording permission on macOS should cost the
                # far side of the call, not the recording.
                if label == MIC:
                    raise
                self._on_error(
                    f"[{label}] {exc} - recording the microphone only. "
                    "Run `sttop doctor` for how to enable system audio."
                )
                continue
            # Registered only once it is live, so a stream that never started
            # is not later stopped, metered, or fed by a paused segmenter.
            self._segmenters[label] = segmenter
            self._captures.append(capture)

        self._consumer = asyncio.create_task(self._consume(), name="transcribe")
        return self.journal.path

    async def stop(self, drain_timeout: float = 30.0) -> Path | None:
        """Stop capture, then finish transcribing whatever is still queued.

        Safe to call at any point in the lifecycle, and always releases: a run
        that failed *during* start() has still loaded the models and started
        the executor thread, and those must go back whether or not any audio
        was ever captured.
        """
        if self._running:
            self._running = False
            await self._drain(drain_timeout)
        return self._release()

    async def _drain(self, timeout: float) -> None:
        for capture in self._captures:
            await capture.stop()
        for segmenter in self._segmenters.values():
            segmenter.close()  # flush any utterance still open

        self._queue.put_nowait(None)  # sentinel: drain, then finish
        if self._consumer is not None:
            try:
                await asyncio.wait_for(self._consumer, timeout=timeout)
            except TimeoutError:
                self._on_error(f"[stt] gave up draining after {timeout:.0f}s")
                self._consumer.cancel()
            self._consumer = None

    def _release(self) -> Path | None:
        """Close the journal and hand back the models, executor and captures.

        Leaves the engine prepared-from-scratch rather than half-alive: a
        transcriber whose model has been closed must not look loaded to
        start(), or the next run transcribes against nothing.
        """
        path = None
        if self.journal is not None:
            path = self.journal.close(self._session_clock())
        for resource in (self.transcriber, self.diarizer):
            if resource is not None:
                resource.close()
        self.transcriber = None
        self.diarizer = None
        # The journal goes too, for the same reason: a closed transcript that
        # still looks open makes status() report the finished session's counts
        # and lets rename_speaker() rewrite a file nobody is appending to.
        self.journal = None
        self._executor.shutdown(wait=True)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
        self._captures.clear()
        self._segmenters.clear()
        return path

    def toggle_pause(self) -> bool:
        self._paused = not self._paused
        for segmenter in self._segmenters.values():
            segmenter.paused = self._paused
        return self._paused

    def rename_speaker(self, old: str, new: str) -> int:
        return self.journal.rename_speaker(old, new) if self.journal else 0

    # -- status ------------------------------------------------------------

    def status(self) -> EngineStatus:
        """A snapshot of the run. Valid at any point in the lifecycle -
        before prepare(), mid-session, or after stop()."""
        return EngineStatus(
            elapsed=self._session_clock() if self._t0 else 0.0,
            backlog=self._queue.qsize(),
            levels={c.label: c.level for c in self._captures},
            paused=self._paused,
            running=self._running,
            utterances=self.journal.count if self.journal else 0,
            speakers=self.diarizer.speaker_count if self.diarizer else 0,
            backend=self.transcriber.describe if self.transcriber else "no backend",
            diarizer=self.diarizer.describe if self.diarizer else "diarize off",
        )

    # -- internals ---------------------------------------------------------

    def _session_clock(self) -> float:
        return time.monotonic() - self._t0

    def _wav_path(self, label: str) -> Path | None:
        if not self.config.audio.save_wav or self.journal is None:
            return None
        return Path(self.config.audio_dir) / f"{self.journal.path.stem}-{label}.wav"

    async def _consume(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            segment = await self._queue.get()
            if segment is None:
                return
            try:
                utterance = await loop.run_in_executor(
                    self._executor, self._transcribe, segment
                )
            except Exception as exc:  # one bad segment must not end the run
                self._on_error(f"[stt] {type(exc).__name__}: {exc}")
                continue
            if utterance is None:
                continue
            # Back on the loop: writing and notifying are cheap and ordered.
            self.journal.append(utterance)
            self._on_utterance(utterance)

    def _transcribe(self, segment: Segment) -> Utterance | None:
        """The only code that runs off the event loop."""
        assert self.transcriber is not None

        transcript = self.transcriber.transcribe(segment.pcm)
        if not transcript.text:
            return None  # VAD fired on noise, not speech

        return Utterance(
            source=segment.source,
            speaker=self.diarizer.label(segment, is_mic=segment.source == MIC),
            start=segment.start,
            end=segment.end,
            text=transcript.text,
            language=transcript.language,
            confidence=transcript.confidence,
        )
