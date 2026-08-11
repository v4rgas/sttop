"""The parts of the ScreenCaptureKit path that are not Apple's.

The stream itself can only be exercised on a Mac, but the format description
it hands over is just a struct, and misreading it is the failure mode with no
symptom: the capture looks alive and the audio is wrong or absent.
"""

import numpy as np
import pytest

from sttop.audio.devices import AudioError
from sttop.audio.pcm import to_mono
from sttop.audio.screencapture import describe_format

#: What ScreenCaptureKit actually delivers on macOS 26 / pyobjc 12: 48 kHz
#: float32, one plane per channel. Flags 41 = float | packed | non-interleaved.
MONO = (48000.0, 1819304813, 41, 4, 1, 4, 1, 32, 0)
STEREO = (48000.0, 1819304813, 41, 4, 1, 4, 2, 32, 0)


class NamedFields:
    """The struct pyobjc returns when the CoreAudio bindings are installed."""

    def __init__(self, rate, flags, channels):
        self.mSampleRate = rate
        self.mFormatFlags = flags
        self.mChannelsPerFrame = channels


def test_the_tuple_form_is_read_by_position():
    """pyobjc hands back a bare tuple unless the CoreAudio bindings happen to
    be installed - and nothing here depends on them, so it always does.
    Reading `.mSampleRate` off it raised on every single buffer."""
    assert describe_format(MONO) == (48_000, 1, True)
    assert describe_format(STEREO) == (48_000, 2, True)


def test_the_struct_form_is_read_by_name():
    """Installing anything that pulls in CoreAudio must not break capture."""
    assert describe_format(NamedFields(44_100.0, 41, 2)) == (44_100, 2, True)


def test_interleaved_audio_is_reported_as_such():
    """The flag is the only thing that says which layout the bytes are in."""
    interleaved_flags = 41 & ~0x20
    assert describe_format((48000.0, 0, interleaved_flags, 4, 1, 8, 2, 32, 0))[2] is False


def test_a_buffer_with_no_format_description_is_an_error():
    with pytest.raises(AudioError, match="no audio format description"):
        describe_format(None)


def test_a_truncated_description_is_an_error_not_a_guess():
    """Indexing past the end of a short tuple would either raise something
    unhelpful or read the wrong field as the sample rate."""
    with pytest.raises(AudioError, match="unreadable"):
        describe_format((48000.0, 0, 41))


def test_absent_values_fall_back_rather_than_dividing_by_zero():
    assert describe_format((0.0, 0, 41, 4, 1, 4, 0, 32, 0)) == (48_000, 1, True)


# -- the layout the flag selects -------------------------------------------


def test_planar_channels_are_averaged_plane_against_plane():
    """One plane per channel: [all left][all right]. Read as interleaved this
    does not fail - it averages the first half of the buffer against the
    second, which is a comb filter, and the model transcribes the result."""
    planar = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert to_mono(planar, 2, planar=True).tolist() == [0.5, 0.5, 0.5]


def test_the_two_layouts_disagree_so_the_flag_has_to_be_right():
    samples = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    assert to_mono(samples, 2, planar=True).tolist() == [0.5, 0.5]
    assert to_mono(samples, 2).tolist() == [1.0, 0.0]
