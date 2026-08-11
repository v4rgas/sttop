import pytest

from sttop.audio import devices
from sttop.audio.devices import AudioError, Source


def fake_sources(monkeypatch, *names):
    listed = [Source(i, n, "module", "s16le 1ch", "IDLE") for i, n in enumerate(names)]
    monkeypatch.setattr(devices, "list_sources", lambda: listed)


def test_substring_resolves_to_the_one_match(monkeypatch):
    fake_sources(monkeypatch, "alsa_input.pci-0000_00.analog-stereo", "hdmi.monitor")
    assert devices.resolve("hdmi", monitor=True) == "hdmi.monitor"


def test_ambiguous_substring_is_an_error(monkeypatch):
    fake_sources(monkeypatch, "usb_mic.analog", "usb_headset.analog")
    with pytest.raises(AudioError, match="ambiguous"):
        devices.resolve("usb", monitor=False)


def test_no_match_is_an_error(monkeypatch):
    fake_sources(monkeypatch, "usb_mic.analog")
    with pytest.raises(AudioError, match="no audio source"):
        devices.resolve("bluetooth", monitor=False)


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_means_default_like_unset(monkeypatch, blank):
    """The documented config uses `mic_source = ""` for "just use the default".
    Treated as a substring it matches everything and fails as ambiguous."""
    fake_sources(monkeypatch, "usb_mic.analog", "usb_headset.analog")
    monkeypatch.setattr(devices, "default_source", lambda: "the-default")
    monkeypatch.setattr(devices, "default_sink_monitor", lambda: "the-default.monitor")

    assert devices.resolve(blank, monitor=False) == "the-default"
    assert devices.resolve(blank, monitor=True) == "the-default.monitor"
    assert devices.resolve(None, monitor=False) == "the-default"


def test_unparseable_pactl_rows_are_skipped(monkeypatch):
    monkeypatch.setattr(
        devices,
        "_pactl",
        lambda *a: "some header line\n0\tmic.analog\tmodule\ts16le\tIDLE\n\ntruncated\t1",
    )
    assert [s.name for s in devices.list_sources()] == ["mic.analog"]
