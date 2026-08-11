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
import time
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
    """None when this machine has the API at all, else why it does not.

    Deliberately cheap - a version test and an import. It runs on the way into
    every session, and it says nothing about permission: see `permission_state`.
    """
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


def permission_state() -> str | None:
    """None when ScreenCaptureKit will really hand over audio here.

    `availability` only establishes that the API exists. Screen recording is a
    permission, and the only way to learn whether it was granted is to ask
    ScreenCaptureKit a question and see whether it answers - so this costs a
    round trip and is used by `sttop doctor`, not on the recording path.
    """
    reason = availability()
    if reason is not None:
        return reason
    try:
        _shareable_content()
    except AudioError as exc:
        return str(exc)
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
        """Also called on ScreenCaptureKit's queue, so it hops like the audio.

        The callback ends up drawing on the TUI, and Textual is no more
        thread-safe than the level meter is.
        """
        self._writer.level = 0.0
        if self._on_error is None:
            return
        message = f"[{self.label}] {detail}"
        if self._loop is None:
            self._on_error(message)
            return
        # A stream that fails as the loop is closing has nowhere to report to.
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._on_error, message)


class _AudioDelegate:
    """SCStreamOutput + SCStreamDelegate, built on macOS only.

    Built at call time rather than at import, because the protocols it conforms
    to only exist once the frameworks are importable.
    """

    def __new__(cls, on_pcm, on_error):
        import objc
        from Foundation import NSObject

        klass = _delegate_class(objc, NSObject)
        instance = klass.alloc().init()
        instance.configure(on_pcm, on_error)
        return instance


_DELEGATE_CLASS = None


def _delegate_class(objc, NSObject):
    """One ObjC class per process - registering the same name twice raises."""
    global _DELEGATE_CLASS
    if _DELEGATE_CLASS is not None:
        return _DELEGATE_CLASS

    import ScreenCaptureKit as sc

    audio_type = sc.SCStreamOutputTypeAudio

    # Conformance is not decoration: it is where pyobjc reads the selectors'
    # type encodings from. `ofType:` is an NSInteger enum, and without the
    # protocol pyobjc assumes every argument is an object and hands the
    # callback a pointer-shaped reading of the number 1.
    protocols = [
        objc.protocolNamed("SCStreamOutput"),
        objc.protocolNamed("SCStreamDelegate"),
    ]

    class SttopAudioDelegate(NSObject, protocols=protocols):
        # python_method, because a two-argument Python method would otherwise
        # be registered as the one-argument selector `configure:` and the
        # class would fail to build at all.
        @objc.python_method
        def configure(self, on_pcm, on_error):
            self._on_pcm = on_pcm
            self._on_error = on_error
            self._decode_failed = False

        def stream_didOutputSampleBuffer_ofType_(self, stream, buffer, kind):
            if kind != audio_type:
                return  # no video output is attached, but say so in code
            try:
                pcm = sample_buffer_to_pcm16(buffer)
            except Exception as exc:  # a bad buffer must not kill the stream
                # Buffers arrive 50 times a second, and whatever is wrong with
                # one is wrong with all of them, so this is said once. The
                # stream ending is reported separately and always.
                if not self._decode_failed:
                    self._decode_failed = True
                    self._on_error(f"{type(exc).__name__}: {exc}")
                return
            if pcm:
                self._on_pcm(pcm)

        def stream_didStopWithError_(self, stream, error):
            self._on_error(str(error) if error else "capture stopped")

    _DELEGATE_CLASS = SttopAudioDelegate
    return _DELEGATE_CLASS


#: `kAudioFormatFlagIsNonInterleaved`, which ScreenCaptureKit sets on every
#: buffer: multi-channel audio arrives as one plane per channel, not as LRLR.
_NON_INTERLEAVED = 0x20

#: Field order of AudioStreamBasicDescription, for the tuple form below.
_ASBD_RATE, _ASBD_FLAGS, _ASBD_CHANNELS = 0, 2, 6
_ASBD_LENGTH = 8

#: What ScreenCaptureKit says it will deliver, and what the unpacking assumes.
_SAMPLE_DTYPE = "<f4"


def describe_format(asbd) -> tuple[int, int, bool]:
    """(sample rate, channels, planar) from an AudioStreamBasicDescription.

    Two shapes, one meaning. pyobjc returns a struct with named fields when the
    CoreAudio bindings are installed and a plain tuple when they are not - and
    they are not, since nothing here depends on them. Reading only the named
    form is how this raised `AttributeError` on every buffer.
    """
    if asbd is None:
        raise AudioError("the sample buffer carries no audio format description")

    if isinstance(asbd, tuple):
        if len(asbd) < _ASBD_LENGTH:
            raise AudioError(f"unreadable audio format description: {asbd!r}")
        rate = asbd[_ASBD_RATE]
        flags = asbd[_ASBD_FLAGS]
        channels = asbd[_ASBD_CHANNELS]
    else:
        rate = asbd.mSampleRate
        flags = asbd.mFormatFlags
        channels = asbd.mChannelsPerFrame

    return (
        int(rate) or 48_000,
        int(channels) or 1,
        bool(int(flags) & _NON_INTERLEAVED),
    )


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
    rate, channels, planar = describe_format(
        CMAudioFormatDescriptionGetStreamBasicDescription(description)
    )

    samples = np.frombuffer(bytes(raw), dtype=_SAMPLE_DTYPE)
    mono = to_mono(samples, channels, planar=planar)
    return float_to_pcm16(resample(mono, rate))


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


#: How long to let a libdispatch thread wind down before the main thread may
#: race on to interpreter shutdown. Measured on macOS 26: 0 s kills the process
#: every time and 0.05 s never does, so this is a comfortable multiple.
_CALLBACK_SETTLE = 0.25


def _let_the_callback_thread_finish() -> None:
    """Yield long enough for ScreenCaptureKit's callback thread to unwind.

    The completion handler runs on a libdispatch worker thread, and running
    Python there gives that thread a thread state. If the interpreter starts
    finalising while the thread is still holding one, CPython terminates it
    with `pthread_exit`, which libdispatch's threads are not allowed to do:

        BUG IN CLIENT OF LIBPTHREAD: pthread_exit() called from a thread
        not created by pthread_create()

    The process dies of SIGKILL and files a crash report - after doing its
    work, so `sttop doctor` printed a perfect report and exited 137. Nothing
    in the Python API can join a thread it did not start, so the fix is to
    stop racing it.
    """
    time.sleep(_CALLBACK_SETTLE)


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
    answered = done.wait(15.0)
    _let_the_callback_thread_finish()
    if not answered:
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
