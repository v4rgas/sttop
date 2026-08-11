import pytest

from sttop.audio import devices
from sttop.audio.devices import AudioError, Source


def fake_sources(monkeypatch, *names):
    listed = [Source(i, n, "module", "s16le 1ch", "IDLE") for i, n in enumerate(names)]
    monkeypatch.setattr(devices, "list_sources", lambda: listed)


def linux(monkeypatch):
    monkeypatch.setattr(devices, "MACOS", False)


def test_substring_resolves_to_the_one_match(monkeypatch):
    linux(monkeypatch)
    fake_sources(monkeypatch, "alsa_input.pci-0000_00.analog-stereo", "hdmi.monitor")
    spec = devices.resolve("hdmi", monitor=True)
    assert (spec.backend, spec.device) == ("pulse", "hdmi.monitor")


def test_ambiguous_substring_is_an_error(monkeypatch):
    linux(monkeypatch)
    fake_sources(monkeypatch, "usb_mic.analog", "usb_headset.analog")
    with pytest.raises(AudioError, match="ambiguous"):
        devices.resolve("usb", monitor=False)


def test_no_match_is_an_error(monkeypatch):
    linux(monkeypatch)
    fake_sources(monkeypatch, "usb_mic.analog")
    with pytest.raises(AudioError, match="no audio source"):
        devices.resolve("bluetooth", monitor=False)


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_means_default_like_unset(monkeypatch, blank):
    """The documented config uses `mic_source = ""` for "just use the default".
    Treated as a substring it matches everything and fails as ambiguous."""
    linux(monkeypatch)
    fake_sources(monkeypatch, "usb_mic.analog", "usb_headset.analog")
    monkeypatch.setattr(devices, "default_source", lambda: "the-default")
    monkeypatch.setattr(devices, "default_sink_monitor", lambda: "the-default.monitor")

    assert devices.resolve(blank, monitor=False).device == "the-default"
    assert devices.resolve(blank, monitor=True).device == "the-default.monitor"
    assert devices.resolve(None, monitor=False).device == "the-default"


def test_unparseable_pactl_rows_are_skipped(monkeypatch):
    monkeypatch.setattr(devices, "have_pactl", lambda: True)
    monkeypatch.setattr(
        devices,
        "_pactl",
        lambda *a: "some header line\n0\tmic.analog\tmodule\ts16le\tIDLE\n\ntruncated\t1",
    )
    assert [s.name for s in devices._pulse_sources()] == ["mic.analog"]


# -- no pactl ---------------------------------------------------------------


def test_defaults_survive_a_missing_pactl(monkeypatch):
    """`pactl` ships separately from pipewire-pulse on Debian, so plenty of
    working audio stacks lack it. The server resolves these names itself."""
    linux(monkeypatch)
    monkeypatch.setattr(devices, "have_pactl", lambda: False)

    assert devices.resolve(None, monitor=False).device == devices.PULSE_DEFAULT_SOURCE
    assert devices.resolve("", monitor=True).device == devices.PULSE_DEFAULT_MONITOR
    assert devices.list_sources() == []


def test_a_named_source_still_needs_pactl(monkeypatch):
    """Asking for a source by name is the one thing defaults cannot cover."""
    linux(monkeypatch)
    monkeypatch.setattr(devices, "have_pactl", lambda: False)
    with pytest.raises(AudioError, match="no audio source"):
        devices.resolve("hdmi", monitor=True)


# -- macos ------------------------------------------------------------------

AVF_LISTING = """\
[AVFoundation indev @ 0x7f8] AVFoundation video devices:
[AVFoundation indev @ 0x7f8] [0] FaceTime HD Camera
[AVFoundation indev @ 0x7f8] AVFoundation audio devices:
[AVFoundation indev @ 0x7f8] [0] MacBook Pro Microphone
[AVFoundation indev @ 0x7f8] [1] BlackHole 2ch
: Input/output error
"""


def fake_avf(monkeypatch, listing=AVF_LISTING):
    class Result:
        stderr = listing
        returncode = 1

    monkeypatch.setattr(devices, "MACOS", True)
    monkeypatch.setattr(devices.subprocess, "run", lambda *a, **k: Result())
    monkeypatch.setattr(devices, "ffmpeg_bin", lambda: "ffmpeg")


def test_avfoundation_listing_keeps_only_audio_devices(monkeypatch):
    """The camera sits in the same output under its own header - transcribing
    it would be an interesting bug."""
    fake_avf(monkeypatch)
    assert [s.name for s in devices.list_sources()] == [
        "MacBook Pro Microphone",
        "BlackHole 2ch",
    ]


def test_a_loopback_device_becomes_the_system_source(monkeypatch):
    fake_avf(monkeypatch)

    spec = devices.resolve(None, monitor=True)
    assert (spec.backend, spec.device, spec.label) == (
        "avfoundation", ":1", "BlackHole 2ch",
    )


def test_no_loopback_device_is_survivable(monkeypatch):
    """Recording only the mic is worth far more than refusing to start, so
    this failure gets its own class for the engine to catch."""
    fake_avf(monkeypatch, listing="[x] AVFoundation audio devices:\n[x] [0] Built-in\n")

    with pytest.raises(devices.SystemAudioUnavailable, match="sttop doctor"):
        devices.resolve(None, monitor=True)


def test_the_mic_needs_no_configuration_on_macos(monkeypatch):
    fake_avf(monkeypatch)
    spec = devices.resolve(None, monitor=False)
    assert (spec.backend, spec.device) == ("avfoundation", ":default")
