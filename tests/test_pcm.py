"""The format adapter between ScreenCaptureKit and everything downstream.

Apple's half cannot be exercised off a Mac, so this half is tested hard: a
mistake here is silent, and shows up as a transcript of noise rather than as
an error anyone can see.
"""

import numpy as np

from sttop import FRAME_BYTES, SAMPLE_RATE
from sttop.audio.pcm import Reframer, float_to_pcm16, resample, to_mono


def tone(freq: float, seconds: float, rate: int) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def dominant_freq(samples: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(samples))
    return float(np.fft.rfftfreq(len(samples), 1 / rate)[np.argmax(spectrum)])


# -- channel handling -------------------------------------------------------


def test_stereo_is_averaged_not_halved():
    """Taking the left channel would drop a panned participant entirely."""
    interleaved = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    assert to_mono(interleaved, 2).tolist() == [0.5, 0.5]


def test_a_trailing_partial_frame_is_dropped_not_misaligned():
    """One stray sample would otherwise shift every channel by one and swap
    left with right for the rest of the buffer."""
    assert to_mono(np.array([1.0, 1.0, 1.0], dtype=np.float32), 2).tolist() == [1.0]


def test_mono_passes_through_untouched():
    samples = np.array([0.1, 0.2], dtype=np.float32)
    assert to_mono(samples, 1) is samples


# -- rate conversion --------------------------------------------------------


def test_48k_speech_survives_the_trip_to_16k():
    samples = resample(tone(1000, 0.5, 48_000), 48_000)
    assert len(samples) == 8000  # 0.5 s at 16 kHz
    assert abs(dominant_freq(samples, SAMPLE_RATE) - 1000) < 20


def test_content_above_nyquist_does_not_fold_into_the_speech_band():
    """Plain decimation would alias 15 kHz down to 1 kHz - right where the
    voice is - and the model would faithfully transcribe the artefact."""
    aliased = resample(tone(15_000, 0.5, 48_000), 48_000)
    naive = tone(15_000, 0.5, 48_000)[::3]

    assert np.abs(aliased).max() < 0.2  # filtered away
    assert np.abs(naive).max() > 0.9  # what we are avoiding


def test_the_native_rate_is_not_touched():
    samples = tone(440, 0.1, SAMPLE_RATE)
    assert resample(samples, SAMPLE_RATE) is samples


def test_upsampling_interpolates():
    samples = resample(tone(400, 0.2, 8000), 8000)
    assert len(samples) == 3200
    assert abs(dominant_freq(samples, SAMPLE_RATE) - 400) < 20


def test_an_empty_buffer_is_not_an_error():
    """Audio callbacks do deliver empty buffers around start and stop."""
    assert resample(np.empty(0, dtype=np.float32), 48_000).size == 0


# -- packing ----------------------------------------------------------------


def test_float_is_packed_little_endian_signed_16():
    packed = float_to_pcm16(np.array([0.0, 1.0, -1.0], dtype=np.float32))
    assert np.frombuffer(packed, dtype="<i2").tolist() == [0, 32767, -32767]


def test_over_full_scale_clips_instead_of_wrapping():
    """Float audio is not bounded to +-1; without the clip, 1.5 wraps to a
    large negative sample and clicks."""
    packed = np.frombuffer(
        float_to_pcm16(np.array([1.5, -1.5], dtype=np.float32)), dtype="<i2"
    )
    assert packed.tolist() == [32767, -32767]


# -- reframing --------------------------------------------------------------


def test_arbitrary_chunks_become_exact_frames():
    """The audio callback's chunk size has nothing to do with ours, and the
    VAD rejects anything that is not exactly one frame."""
    frames = []
    reframer = Reframer(frames.append)

    reframer.feed(b"\x00" * (FRAME_BYTES + 7))
    assert len(frames) == 1
    assert reframer.pending == 7

    reframer.feed(b"\x00" * (FRAME_BYTES - 7))
    assert len(frames) == 2
    assert reframer.pending == 0
    assert all(len(frame) == FRAME_BYTES for frame in frames)


def test_a_chunk_smaller_than_a_frame_emits_nothing_yet():
    frames = []
    Reframer(frames.append).feed(b"\x00" * 16)
    assert frames == []


def test_the_stream_is_carried_across_chunks_in_order():
    """A remainder that is padded or dropped instead of carried puts a click
    at every chunk boundary, ~5 times a second."""
    frames = []
    reframer = Reframer(frames.append)
    payload = bytes(range(256)) * 16  # 4096 bytes, not a frame multiple

    for _ in range(4):
        reframer.feed(payload)

    rebuilt = b"".join(frames) + b"\x00" * 0
    assert rebuilt == (payload * 4)[: len(rebuilt)]
