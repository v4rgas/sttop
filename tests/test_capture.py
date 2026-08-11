"""Teardown and failure-reporting checks that need no audio server."""

import asyncio

import pytest

from sttop import SAMPLE_RATE
from sttop.audio import capture as capture_mod
from sttop.audio.capture import SourceCapture, build_command
from sttop.audio.devices import AudioError, CaptureSpec

SPEC = CaptureSpec("pulse", "src", "src")


def capture(**kwargs) -> tuple[SourceCapture, list[str]]:
    errors: list[str] = []
    source = SourceCapture("mic", SPEC, lambda frame: None, errors.append, **kwargs)
    return source, errors


def test_pulse_and_avfoundation_differ_only_in_the_input(monkeypatch):
    monkeypatch.setattr(capture_mod, "ffmpeg_bin", lambda: "/usr/bin/ffmpeg")

    pulse = build_command(SPEC)
    mac = build_command(CaptureSpec("avfoundation", ":1", "BlackHole 2ch"))

    assert pulse[:1] == ["/usr/bin/ffmpeg"]
    assert pulse[pulse.index("-f") : pulse.index("-f") + 4] == [
        "-f", "pulse", "-i", "src",
    ]
    assert mac[mac.index("-f") : mac.index("-f") + 4] == [
        "-f", "avfoundation", "-i", ":1",
    ]
    tail = ["-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
    assert pulse[-len(tail):] == tail
    assert mac[-len(tail):] == tail


def test_an_unknown_backend_is_rejected_before_spawning():
    with pytest.raises(AudioError, match="unknown capture backend"):
        build_command(CaptureSpec("portaudio", "0", "whatever"))


def test_stop_on_a_capture_that_never_started():
    source, errors = capture()
    asyncio.run(source.stop())
    assert errors == []


def test_a_dead_source_does_not_keep_a_live_meter():
    """A frozen meter reads as a working capture, contradicting the warning
    printed next to it."""
    source, errors = capture()
    source.level = 0.9
    source._report_failure("pulse: connection terminated")

    assert source.level == 0.0
    assert errors == ["[mic] pulse: connection terminated"]


def test_teardown_survives_a_reader_that_died(monkeypatch):
    """stop() must still reach the process and WAV cleanup even if the pump
    task raised, or it leaves an ffmpeg running and a truncated WAV."""

    async def scenario():
        source, _ = capture()

        async def doomed():
            raise RuntimeError("disk full")

        source._task = asyncio.create_task(doomed())
        await asyncio.sleep(0)  # let it fail before we tear down
        await source.stop()
        assert source._task is None

    asyncio.run(scenario())
