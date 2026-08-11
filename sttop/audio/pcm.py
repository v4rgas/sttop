"""Turn whatever a platform hands us into sttop's one PCM format.

Every consumer downstream - VAD, the model, the WAV sidecar - assumes 16 kHz
mono signed 16-bit frames of exactly FRAME_BYTES. ffmpeg produces that
directly. ScreenCaptureKit does not: it delivers float32 at the display's
sample rate, in chunks whose size follows the audio callback rather than our
frame size. This module is the adapter, and it is deliberately free of any
Apple API so it can be tested anywhere.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .. import FRAME_BYTES, SAMPLE_RATE


def to_mono(samples: np.ndarray, channels: int) -> np.ndarray:
    """Average interleaved channels down to one.

    Averaging rather than taking the left channel: a call whose remote audio
    is panned - or simply stereo music under a voice - would otherwise lose
    whatever sits on the right.
    """
    if channels <= 1:
        return samples
    usable = len(samples) - (len(samples) % channels)
    return samples[:usable].reshape(-1, channels).mean(axis=1)


def resample(samples: np.ndarray, source_rate: int) -> np.ndarray:
    """Rate-convert to SAMPLE_RATE.

    A box filter over each output sample's input span, which is crude as
    filters go but is exactly the anti-aliasing that plain decimation lacks:
    dropping 2 of every 3 samples at 48 kHz folds everything above 8 kHz back
    down into the speech band as noise the model then tries to read. Speech
    lives well below the 8 kHz Nyquist we are left with, so the passband
    droop this introduces costs nothing that matters here.
    """
    if source_rate == SAMPLE_RATE or samples.size == 0:
        return samples

    ratio = source_rate / SAMPLE_RATE
    out_len = int(len(samples) / ratio)
    if out_len <= 0:
        return np.empty(0, dtype=samples.dtype)

    if ratio > 1:
        # Downsampling: average each output sample's whole input window.
        edges = (np.arange(out_len + 1) * ratio).astype(np.int64)
        sums = np.concatenate(([0.0], np.cumsum(samples, dtype=np.float64)))
        widths = np.maximum(np.diff(edges), 1)
        return ((sums[edges[1:]] - sums[edges[:-1]]) / widths).astype(np.float32)

    # Upsampling has no aliasing to guard against, so interpolate.
    positions = np.arange(out_len) * ratio
    return np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)


def float_to_pcm16(samples: np.ndarray) -> bytes:
    """Clip, scale and pack to little-endian int16.

    Clipping first because float audio is not bounded to [-1, 1]: a sample
    over full scale would otherwise wrap to the opposite sign and click.
    """
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


class Reframer:
    """Cuts a stream of arbitrary-sized PCM chunks into fixed frames.

    The audio callback's chunk size has nothing to do with ours, and the VAD
    rejects anything that is not exactly 20 ms, so the remainder of each chunk
    has to be carried into the next one rather than padded or dropped.
    """

    def __init__(self, on_frame: Callable[[bytes], None]) -> None:
        self._on_frame = on_frame
        self._buffer = bytearray()

    def feed(self, pcm: bytes) -> None:
        self._buffer.extend(pcm)
        while len(self._buffer) >= FRAME_BYTES:
            frame = bytes(self._buffer[:FRAME_BYTES])
            del self._buffer[:FRAME_BYTES]
            self._on_frame(frame)

    @property
    def pending(self) -> int:
        return len(self._buffer)
