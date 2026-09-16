"""Wires capture -> VAD -> transcription -> diarization -> Markdown journal.

Concurrency model during recording - one thread boundary, and it is the model:

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
and serialising it keeps utterances in the order they were spoken. During
startup only, independent speech and speaker models initialize concurrently.

When transcription falls behind, the queue absorbs the lag and `backlog`
reports it rather than dropping audio.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
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
#: (old label, label it was folded into, lines rewritten)
RenameCallback = Callable[[str, str, int], None]


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
    loading: str = ""
    loading_elapsed: float = 0.0
    error: str = ""
    catching_up: bool = False


class Engine:
    def __init__(
        self,
        config: Config,
        on_utterance: UtteranceCallback,
        on_error: ErrorCallback | None = None,
        on_rename: RenameCallback | None = None,
        cipher=None,
    ) -> None:
        self.config = config
        #: crypto.Cipher when storage.encrypt is on; the engine only carries
        #: it to the journal, which does the sealing.
        self._cipher = cipher
        self._on_utterance = on_utterance
        self._on_error = on_error or (lambda message: None)
        self._on_rename = on_rename or (lambda old, new, lines: None)

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
        #: The `pipe` command's process, fed one JSON line per event.
        self._pipe: asyncio.subprocess.Process | None = None
        self._t0: float = 0.0
        self._paused = False
        self._running = False
        self._loading = ""
        self._loading_t0 = 0.0
        self._preparing: asyncio.Task | None = None
        self._stopping = False
        self._model_ready: asyncio.Task | None = None
        self._error = ""
        self._inflight = False
        self._catching_up = False
        self._ended_elapsed: float | None = None

    # -- lifecycle ---------------------------------------------------------

    async def prepare(self) -> None:
        """Load independent models concurrently, keeping the UI responsive."""
        if self._preparing is None:
            self._preparing = asyncio.create_task(self._prepare())
        try:
            await asyncio.shield(self._preparing)
        except asyncio.CancelledError:
            # Cancelling an executor future cannot cancel its model load. Let
            # ownership land on the engine so stop() can close it safely.
            await self._preparing
            raise
        finally:
            self._preparing = None

    async def _prepare(self) -> None:
        loop = asyncio.get_running_loop()
        self._loading_t0 = time.monotonic()
        if self.mic_source is None:
            await self._prepare_sources()
        pending = {"speech model", "speaker model"}

        def refresh() -> None:
            self._loading = "Loading " + " + ".join(sorted(pending))

        async def speech() -> None:
            self.transcriber = await loop.run_in_executor(
                self._executor, stt.build, self.config.stt
            )
            pending.remove("speech model")
            refresh()

        async def speakers() -> None:
            # Only initialization uses another thread; inference stays serial.
            self.diarizer = await asyncio.to_thread(
                diarize_mod.build, self.config.diarize
            )
            pending.remove("speaker model")
            refresh()

        refresh()
        results = await asyncio.gather(speech(), speakers(), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result
        self._loading = ""

    async def _prepare_sources(self) -> None:
        self._loading_t0 = time.monotonic()
        self._loading = "Resolving audio devices"
        loop = asyncio.get_running_loop()
        self.mic_source, self.sys_source = await loop.run_in_executor(
            self._executor, self._resolve_sources
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
        self._stopping = False
        self._error = ""
        if self.mic_source is None:
            await self._prepare_sources()
        if self._stopping:
            raise asyncio.CancelledError
        self._loading = "Starting audio capture"

        self.journal = Journal.create(
            Path(self.config.sessions_dir),
            title,
            mic_source=str(self.mic_source),
            sys_source=str(self.sys_source) if self.sys_source else "unavailable",
            backend=(self.transcriber.describe if self.transcriber else
                     f"parakeet {self.config.stt.model or stt.DEFAULT_MODEL}"),
            cipher=self._cipher,
        )
        self._t0 = time.monotonic()
        self._ended_elapsed = None
        if self.config.pipe:
            # stdout to nowhere: the child writing to the terminal would paint
            # over the UI. stderr is already the session log.
            self._pipe = await asyncio.create_subprocess_shell(
                self.config.pipe,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
            )

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
            segmenter.paused = self._paused
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
                    f"[{label}] Recording the microphone only. {exc}\n"
                    "Help: `sttop doctor`."
                )
                continue
            # Registered only once it is live, so a stream that never started
            # is not later stopped, metered, or fed by a paused segmenter.
            self._segmenters[label] = segmenter
            self._captures.append(capture)
            self._running = True

        # Audio is already feeding the queue before any model imports/downloads.
        if self.transcriber is None or self.diarizer is None:
            self._loading = "Loading speech + speaker models"
            self._model_ready = asyncio.create_task(self.prepare(), name="load-models")
        else:
            self._loading = ""
        self._consumer = asyncio.create_task(self._load_and_consume(), name="transcribe")
        return self.journal.path

    async def stop(self, drain_timeout: float | None = None) -> Path | None:
        """Stop capture, then finish transcribing whatever is still queued.

        Safe to call at any point in the lifecycle, and always releases: a run
        that failed *during* start() has still loaded the models and started
        the executor thread, and those must go back whether or not any audio
        was ever captured.

        By default finish every buffered segment, including a first-download
        backlog; a fixed 30-second deadline can silently lose the opening.
        """
        self._stopping = True
        # Freeze the recording immediately, even if models are still loading.
        await self._stop_capture()
        if self._model_ready is not None:
            with contextlib.suppress(Exception):
                await asyncio.shield(self._model_ready)
        elif self._preparing is not None:
            with contextlib.suppress(Exception):
                await asyncio.shield(self._preparing)
        await self._drain(drain_timeout)
        await self._close_pipe()
        return self._release()

    def _emit(self, event: dict) -> None:
        """One JSON line to the pipe command. A command that died costs the
        stream, never the recording."""
        if self._pipe is None:
            return
        # asyncio swallows a write to a dead pipe, so ask before writing.
        if self._pipe.returncode is not None or self._pipe.stdin.is_closing():
            self._on_error(f"[pipe] `{self.config.pipe}` exited; stopped streaming")
            self._pipe = None
            return
        line = json.dumps(event, ensure_ascii=False) + "\n"
        self._pipe.stdin.write(line.encode())

    async def _close_pipe(self, timeout: float = 5.0) -> None:
        """EOF, then a few seconds to finish - a model call may be in flight."""
        pipe, self._pipe = self._pipe, None
        if pipe is None:
            return
        with contextlib.suppress(Exception):
            pipe.stdin.close()
            await pipe.stdin.wait_closed()
        try:
            await asyncio.wait_for(pipe.wait(), timeout)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                pipe.kill()
            await pipe.wait()

    async def _stop_capture(self) -> None:
        self._running = False
        for capture in self._captures:
            await capture.stop()
        for segmenter in self._segmenters.values():
            segmenter.close()  # flush any utterance still open
        if self._t0 and self._ended_elapsed is None:
            self._ended_elapsed = time.monotonic() - self._t0

    async def _drain(self, timeout: float | None) -> None:
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
        self._loading = ""
        self._model_ready = None
        self._queue = asyncio.Queue()
        self._inflight = False
        self._catching_up = False
        return path

    def toggle_pause(self) -> bool:
        self._paused = not self._paused
        for segmenter in self._segmenters.values():
            segmenter.paused = self._paused
        return self._paused

    def rename_speaker(self, old: str, new: str) -> int:
        self._emit({"type": "rename", "old": old, "new": new})
        return self.journal.rename_speaker(old, new) if self.journal else 0

    def transcript(self) -> str:
        """The session so far as plaintext Markdown - live, mid-recording."""
        return self.journal.snapshot() if self.journal else ""

    # -- status ------------------------------------------------------------

    def status(self) -> EngineStatus:
        """A snapshot of the run. Valid at any point in the lifecycle -
        before prepare(), mid-session, or after stop()."""
        return EngineStatus(
            elapsed=self._session_clock() if self._t0 else 0.0,
            backlog=self._queue.qsize() + int(self._inflight),
            levels={c.label: c.level for c in self._captures},
            paused=self._paused,
            running=self._running,
            utterances=self.journal.count if self.journal else 0,
            speakers=self.diarizer.speaker_count if self.diarizer else 0,
            backend=self.transcriber.describe if self.transcriber else "no backend",
            diarizer=self.diarizer.describe if self.diarizer else "diarize off",
            loading=self._loading,
            error=self._error,
            catching_up=self._catching_up,
            loading_elapsed=(time.monotonic() - self._loading_t0)
            if self._loading else 0.0,
        )

    # -- internals ---------------------------------------------------------

    def _session_clock(self) -> float:
        return (
            self._ended_elapsed if self._ended_elapsed is not None
            else time.monotonic() - self._t0
        )

    def _wav_path(self, label: str) -> Path | None:
        if not self.config.audio.save_wav or self.journal is None:
            return None
        return Path(self.config.audio_dir) / f"{self.journal.path.stem}-{label}.wav"

    async def _load_and_consume(self) -> None:
        try:
            if self._model_ready is not None:
                await asyncio.shield(self._model_ready)
        except Exception as exc:
            self._error = f"Model loading failed: {type(exc).__name__}: {exc}"
            self._loading = ""
            await self._stop_capture()
            self._on_error(
                f"{self._error}. Recording stopped; buffered speech could not be "
                "transcribed."
            )
            return
        self._catching_up = not self._queue.empty()
        await self._consume()

    async def _consume(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            segment = await self._queue.get()
            if segment is None:
                return
            self._inflight = True
            try:
                utterance = await loop.run_in_executor(
                    self._executor, self._transcribe, segment
                )
            except Exception as exc:  # one bad segment must not end the run
                self._on_error(f"[stt] {type(exc).__name__}: {exc}")
                continue
            finally:
                self._inflight = False
                if self._queue.empty():
                    self._catching_up = False
            if utterance is None:
                continue
            # Back on the loop: writing and notifying are cheap and ordered.
            self.journal.append(utterance)
            self._emit({"type": "utterance", **asdict(utterance)})
            self._on_utterance(utterance)
            self._apply_merges()

    def _apply_merges(self) -> None:
        """Rewrite the transcript for speakers the diarizer has since joined.

        Drained here rather than inside the diarizer because the journal is
        the event loop's to touch, and because the merge must land *after* the
        utterance that triggered it is already on disk.
        """
        if self.diarizer is None or self.journal is None:
            return
        for old, new in self.diarizer.take_merges():
            lines = self.journal.rename_speaker(old, new)
            self._emit({"type": "rename", "old": old, "new": new})
            self._on_rename(old, new, lines)

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
