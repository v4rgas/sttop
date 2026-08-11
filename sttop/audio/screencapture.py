"""System audio on macOS, via ScreenCaptureKit.

macOS exposes no monitor source, and for years the answer was a virtual audio
driver: install BlackHole, build a Multi-Output Device by hand in Audio MIDI
Setup, and point the system output at it. That works, but it is a lot of
ceremony, it silently changes which device the machine plays through, and it
breaks the volume keys.

ScreenCaptureKit (macOS 13+) captures audio without any of that. Despite the
name nothing is recorded from the screen: `capturesAudio` with an audio-only
stream output is a supported configuration, and it is how OBS takes system
audio on modern macOS. The user grants screen-recording permission once and
their audio keeps playing out of whatever device it already used.

The Apple API surface lives here and nowhere else. Format conversion is in
`pcm`, which is pure numpy and tested on any platform - this module is the
part that can only be exercised on a Mac.
"""

from __future__ import annotations

import asyncio
import contextlib
import platform
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .devices import AudioError

FrameCallback = Callable[[bytes], None]
ErrorCallback = Callable[[str], None]

#: ScreenCaptureKit's audio capture arrived in Ventura.
MIN_MACOS = (13, 0)

#: What we ask ScreenCaptureKit for. It is free to ignore both - the stream's
#: format description is read per buffer and honoured - but asking for what we
#: want means the usual case needs no conversion at all.
_WANT_CHANNELS = 1

#: A stream still needs a video size even when only audio is consumed, so ask
#: for the smallest legal one and the slowest frame rate: no video output is
#: ever attached, and this keeps the capture from costing a display's worth of
#: pixels per frame.
_MIN_VIDEO_SIDE = 2
_VIDEO_INTERVAL = 60  # seconds between the frames nobody reads


def macos_version() -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in platform.mac_ver()[0].split(".") if part)
    except ValueError:
        return ()


def availability() -> str | None:
    """None when system audio can be captured here, else why it cannot."""
    version = macos_version()
    if version and version < MIN_MACOS:
        pretty = ".".join(str(part) for part in version)
        return (
            f"macOS {pretty} is too old for ScreenCaptureKit audio "
            f"(needs {MIN_MACOS[0]}.{MIN_MACOS[1]}+)"
        )
    try:
        import ScreenCaptureKit  # noqa: F401
    except ImportError as exc:
        return f"the ScreenCaptureKit bindings are missing ({exc})"
    return None


class ScreenAudioCapture:
    """Streams system audio, handing fixed-size PCM frames to a callback.

    Mirrors `SourceCapture`'s interface - same label, level meter and
    start/stop lifecycle - so the engine treats the two interchangeably.
    """

    def __init__(
        self,
        label: str,
        spec,
        on_frame: FrameCallback,
        on_error: ErrorCallback | None = None,
        wav_path: Path | None = None,
    ) -> None:
        from .capture import FrameWriter

        self.label = label
        self.spec = spec
        self._on_error = on_error
        self._writer = FrameWriter(on_frame, wav_path)

        self._stream = None
        self._delegate = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._started = threading.Event()
        self._stopping = False

    # -- the properties the UI reads ---------------------------------------

    @property
    def level(self) -> float:
        return self._writer.level

    @level.setter
    def level(self, value: float) -> None:
        self._writer.level = value

    @property
    def frames_seen(self) -> int:
        return self._writer.frames_seen

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        reason = availability()
        if reason is not None:
            raise AudioError(reason)

        self._loop = asyncio.get_running_loop()
        self._writer.open()
        try:
            await asyncio.get_running_loop().run_in_executor(None, self._start_stream)
        except Exception:
            # Half-started is not a state a caller can clean up.
            await self.stop()
            raise

    def _start_stream(self) -> None:
        """Build and start the SCK stream. Blocking, so it runs off the loop."""
        import ScreenCaptureKit as sc

        content = _shareable_content()
        displays = content.displays()
        if not displays:
            raise AudioError("no display to attach a capture stream to")

        # An audio-only capture still binds to a display; excluding nothing
        # from it keeps every application's audio in the mix, which is the
        # point - the far side of a call is whatever is playing.
        stream_filter = sc.SCContentFilter.alloc().initWithDisplay_excludingWindows_(
            displays[0], []
        )

        config = sc.SCStreamConfiguration.alloc().init()
        config.setCapturesAudio_(True)
        # Our own process makes no sound, but if it ever does, hearing
        # ourselves would be transcribed as a participant.
        config.setExcludesCurrentProcessAudio_(True)
        config.setChannelCount_(_WANT_CHANNELS)
        config.setWidth_(_MIN_VIDEO_SIDE)
        config.setHeight_(_MIN_VIDEO_SIDE)
        config.setMinimumFrameInterval_(_cm_time(_VIDEO_INTERVAL))

        self._delegate = _AudioDelegate(self._deliver, self._report_failure)
        self._stream = sc.SCStream.alloc().initWithFilter_configuration_delegate_(
            stream_filter, config, self._delegate
        )

        ok, error = self._stream.addStreamOutput_type_sampleHandlerQueue_error_(
            self._delegate, sc.SCStreamOutputTypeAudio, None, None
        )
        if not ok:
            raise AudioError(f"could not attach the audio output: {error}")

        _await_callback(
            lambda done: self._stream.startCaptureWithCompletionHandler_(done),
            what="startCapture",
        )
        self._started.set()

    async def stop(self) -> None:
        """Idempotent, and every step runs even if an earlier one failed."""
        self._stopping = True
        stream, self._stream = self._stream, None
        if stream is not None and self._started.is_set():
            # A stream that will not stop must still not block shutdown.
            with contextlib.suppress(Exception):
                await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: _await_callback(
                        lambda done: stream.stopCaptureWithCompletionHandler_(done),
                        what="stopCapture",
                    ),
                )
        self._started.clear()
        self._delegate = None
        self._writer.close()

    # -- the audio callback ------------------------------------------------

    def _deliver(self, pcm: bytes) -> None:
        """Called on ScreenCaptureKit's queue, not the event loop.

        Everything the frame touches - the level meter, the WAV, the VAD -
        belongs to the loop, so the only work done here is the hop.
        """
        if self._stopping or self._loop is None or not pcm:
            return
        self._loop.call_soon_threadsafe(self._writer.feed, pcm)

    def _report_failure(self, detail: str) -> None:
        self._writer.level = 0.0
        if self._on_error is not None:
            self._on_error(f"[{self.label}] {detail}")


class _AudioDelegate:
    """SCStreamOutput + SCStreamDelegate, built at import time on macOS only.

    Defined through objc.python_method-free plain methods on a class created
    by `objc.createClass`-style subclassing at call time, because the base
    protocol only exists when the frameworks are importable.
    """

    def __new__(cls, on_pcm, on_error):
        import objc
        import ScreenCaptureKit as sc  # noqa: F401
        from Foundation import NSObject

        klass = _delegate_class(objc, NSObject)
        instance = klass.alloc().init()
        instance.configure_(on_pcm, on_error)
        return instance


_DELEGATE_CLASS = None


def _delegate_class(objc, NSObject):
    """One ObjC class per process - registering the same name twice raises."""
    global _DELEGATE_CLASS
    if _DELEGATE_CLASS is not None:
        return _DELEGATE_CLASS

    class SttopAudioDelegate(NSObject):
        def configure_(self, on_pcm, on_error):
            self._on_pcm = on_pcm
            self._on_error = on_error

        def stream_didOutputSampleBuffer_ofType_(self, stream, buffer, kind):
            try:
                pcm = sample_buffer_to_pcm16(buffer)
            except Exception as exc:  # a bad buffer must not kill the stream
                self._on_error(f"{type(exc).__name__}: {exc}")
                return
            if pcm:
                self._on_pcm(pcm)

        def stream_didStopWithError_(self, stream, error):
            self._on_error(str(error) if error else "capture stopped")

    _DELEGATE_CLASS = SttopAudioDelegate
    return _DELEGATE_CLASS


def sample_buffer_to_pcm16(buffer) -> bytes:
    """CMSampleBuffer -> 16 kHz mono s16le, whatever it arrived as."""
    from CoreMedia import (
        CMAudioFormatDescriptionGetStreamBasicDescription,
        CMBlockBufferCopyDataBytes,
        CMBlockBufferGetDataLength,
        CMSampleBufferGetDataBuffer,
        CMSampleBufferGetFormatDescription,
    )

    from .pcm import float_to_pcm16, resample, to_mono

    block = CMSampleBufferGetDataBuffer(buffer)
    if block is None:
        return b""
    length = CMBlockBufferGetDataLength(block)
    if not length:
        return b""

    status, raw = CMBlockBufferCopyDataBytes(block, 0, length, None)
    if status != 0:
        raise AudioError(f"CMBlockBufferCopyDataBytes failed ({status})")

    description = CMSampleBufferGetFormatDescription(buffer)
    asbd = CMAudioFormatDescriptionGetStreamBasicDescription(description)
    rate = int(asbd.mSampleRate) or 48_000
    channels = int(asbd.mChannelsPerFrame) or 1

    samples = np.frombuffer(bytes(raw), dtype="<f4")
    return float_to_pcm16(resample(to_mono(samples, channels), rate))


# -- pyobjc plumbing --------------------------------------------------------


def _cm_time(seconds: int):
    from CoreMedia import CMTimeMake

    return CMTimeMake(seconds, 1)


def _await_callback(call, *, what: str, timeout: float = 15.0):
    """Run one of SCK's completion-handler APIs synchronously.

    These are the only blocking calls in the module, and they are why the
    stream is built off the event loop rather than on it.
    """
    done = threading.Event()
    failure: list = []

    def handler(error):
        if error is not None:
            failure.append(error)
        done.set()

    call(handler)
    if not done.wait(timeout):
        raise AudioError(f"{what} timed out after {timeout:.0f}s")
    if failure:
        raise AudioError(f"{what} failed: {failure[0]}")


def _shareable_content():
    """The current displays/windows/apps, fetched synchronously."""
    import ScreenCaptureKit as sc

    result: list = []
    failure: list = []
    done = threading.Event()

    def handler(content, error):
        if error is not None:
            failure.append(error)
        else:
            result.append(content)
        done.set()

    sc.SCShareableContent.getShareableContentWithCompletionHandler_(handler)
    if not done.wait(15.0):
        raise AudioError(
            "ScreenCaptureKit did not answer - screen recording permission is "
            "probably not granted yet. Grant it to your terminal in System "
            "Settings > Privacy & Security > Screen & System Audio Recording."
        )
    if failure:
        raise AudioError(
            f"ScreenCaptureKit refused: {failure[0]}. Grant screen recording "
            "permission to your terminal in System Settings > Privacy & "
            "Security > Screen & System Audio Recording."
        )
    return result[0]
