import numpy as np
import pytest

from sttop.audio.segmenter import Segmenter
from sttop.config import VadConfig

_rng = np.random.default_rng(0)


def frame(sigma: float) -> bytes:
    return _rng.normal(0, sigma, 320).astype(np.int16).tobytes()


def quiet() -> bytes:
    """A realistic room-noise floor.

    Not digital zero: webrtcvad is stateful and behaves oddly on all-zero
    frames, which no real capture device ever produces anyway.
    """
    return frame(30)


def speech() -> bytes:
    return frame(6000)


def make(**overrides):
    # Production defaults, but a shorter silence gap so tests stay quick.
    settings = {"silence_ms": 200, "pad_ms": 100, **overrides}
    out = []
    return Segmenter("t", VadConfig(**settings), out.append, clock=lambda: 0.0), out


def feed(seg, maker, count):
    for _ in range(count):
        seg.feed(maker())


def test_quiet_room_emits_nothing():
    seg, out = make()
    feed(seg, quiet, 100)
    assert out == []


def test_speech_then_silence_emits_one_segment():
    seg, out = make()
    feed(seg, speech, 50)
    feed(seg, quiet, 30)
    assert len(out) == 1
    assert out[0].source == "t"
    assert out[0].duration == pytest.approx(1.0, abs=0.4)


def test_trailing_silence_is_trimmed():
    seg, out = make()
    feed(seg, speech, 50)
    feed(seg, quiet, 200)  # far more silence than needed to close
    assert len(out) == 1
    assert out[0].duration < 1.6


def test_blips_below_the_minimum_are_dropped():
    seg, out = make(min_segment_ms=600)
    feed(seg, speech, 3)  # 60ms of speech, well under the 600ms floor
    feed(seg, quiet, 30)
    assert out == []


def test_long_monologue_is_flushed_at_the_ceiling():
    seg, out = make(max_segment_s=1.0)
    feed(seg, speech, 200)  # 4s of unbroken speech
    assert len(out) >= 3
    assert all(s.duration <= 1.1 for s in out)


def test_pause_discards_audio():
    seg, out = make()
    seg.paused = True
    feed(seg, speech, 50)
    feed(seg, quiet, 30)
    assert out == []


def test_close_emits_the_open_segment():
    seg, out = make()
    feed(seg, speech, 50)
    assert out == []  # still open, no silence yet
    seg.close()
    assert len(out) == 1


def test_segments_are_ordered_on_the_session_clock():
    seg, out = make()
    for _ in range(2):
        feed(seg, speech, 50)
        feed(seg, quiet, 30)
    assert len(out) == 2
    assert out[0].start < out[0].end <= out[1].start < out[1].end
