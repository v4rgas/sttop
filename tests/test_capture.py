"""Teardown and failure-reporting checks that need no audio server."""

import asyncio

from sttop.audio.capture import SourceCapture


def capture(**kwargs) -> tuple[SourceCapture, list[str]]:
    errors: list[str] = []
    source = SourceCapture("mic", "src", lambda frame: None, errors.append, **kwargs)
    return source, errors


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
