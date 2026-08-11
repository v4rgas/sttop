"""Lifecycle checks that need no audio hardware and no models."""

import asyncio
from pathlib import Path

import pytest

from sttop import engine as engine_mod
from sttop.audio.devices import AudioError, CaptureSpec
from sttop.config import Config
from sttop.engine import Engine
from sttop.journal import Journal, Utterance


class FakeModel:
    describe = "fake"
    speaker_count = 0

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def prepared_engine(tmp_path) -> tuple[Engine, FakeModel, FakeModel]:
    """An engine as it stands after prepare() but before a successful start()."""
    config = Config()
    config.sessions_dir = str(tmp_path / "sessions")
    engine = Engine(config, lambda utterance: None)
    engine.transcriber, engine.diarizer = FakeModel(), FakeModel()
    return engine, engine.transcriber, engine.diarizer


def test_stop_releases_models_when_start_never_succeeded(tmp_path):
    """The models and the executor thread are already live once prepare() has
    run, so a session that dies during start() still has to give them back."""
    engine, transcriber, diarizer = prepared_engine(tmp_path)

    assert asyncio.run(engine.stop()) is None
    assert transcriber.closed and diarizer.closed
    assert engine.transcriber is None and engine.diarizer is None


def started_engine(tmp_path):
    """An engine with a live journal, as after start() - without the audio."""
    engine, transcriber, diarizer = prepared_engine(tmp_path)
    engine.journal = Journal.create(Path(engine.config.sessions_dir), "demo")
    engine.journal.append(Utterance("system", "spk1", 0.0, 1.0, "hello"))
    return engine, transcriber, diarizer


def test_stop_hands_back_the_transcript_once(tmp_path):
    engine, _, _ = started_engine(tmp_path)

    path = asyncio.run(engine.stop())
    assert path is not None and "hello" in path.read_text()
    # Nothing was closed the second time, so there is no transcript to report.
    assert asyncio.run(engine.stop()) is None


def test_a_finished_session_is_not_still_reported_as_live(tmp_path):
    engine, _, _ = started_engine(tmp_path)
    assert engine.status().utterances == 1

    asyncio.run(engine.stop())
    assert engine.status().utterances == 0
    # The file is closed; renaming into it would edit a finalised transcript.
    assert engine.rename_speaker("spk1", "Ana") == 0


def test_stop_is_idempotent(tmp_path):
    engine, _, _ = prepared_engine(tmp_path)
    asyncio.run(engine.stop())
    assert asyncio.run(engine.stop()) is None


def test_a_stopped_engine_does_not_look_prepared(tmp_path):
    """A transcriber whose model has been closed must not satisfy start()'s
    "already prepared" check, or the next run transcribes against nothing."""
    engine, _, _ = prepared_engine(tmp_path)
    asyncio.run(engine.stop())
    assert engine.transcriber is None


def test_status_is_valid_before_prepare_and_after_stop(tmp_path):
    engine, _, _ = prepared_engine(tmp_path)
    assert engine.status().backend == "fake"

    asyncio.run(engine.stop())
    status = engine.status()
    assert status.running is False
    assert status.backend == "no backend"
    assert status.utterances == 0


# -- one stream failing to start -------------------------------------------


class FakeCapture:
    """Stands in for ffmpeg and for ScreenCaptureKit, neither of which a test
    can start. `fails` names the labels whose start() raises."""

    fails: set[str] = set()
    made: list["FakeCapture"] = []

    def __init__(self, label, spec, on_frame, on_error=None, wav_path=None):
        self.label = label
        self.level = 0.0
        self.frames_seen = 0
        self.started = False
        self.stopped = False
        FakeCapture.made.append(self)

    async def start(self):
        if self.label in FakeCapture.fails:
            raise AudioError("screen recording permission was not granted")
        self.started = True

    async def stop(self):
        self.stopped = True


def engine_with_two_sources(tmp_path, failing: set[str]):
    engine, _, _ = prepared_engine(tmp_path)
    engine.mic_source = CaptureSpec("avfoundation", ":default", "default input")
    engine.sys_source = CaptureSpec("screencapture", "system", "system audio")
    FakeCapture.fails = set(failing)
    FakeCapture.made = []
    return engine


@pytest.fixture
def fake_captures(monkeypatch):
    monkeypatch.setattr(engine_mod, "_capture_for", lambda spec: FakeCapture)


def test_system_audio_that_will_not_start_costs_only_system_audio(
    tmp_path, fake_captures
):
    """Screen recording permission is granted per app and only read at launch,
    so this is the ordinary state of a fresh mac - and the mic still carries
    your own half of the call, which is worth far more than refusing to run."""
    errors: list[str] = []
    engine = engine_with_two_sources(tmp_path, failing={"system"})
    engine._on_error = errors.append

    async def run():
        path = await engine.start("degraded")
        await engine.stop()
        return path

    assert asyncio.run(run()) is not None
    assert [c.label for c in engine._captures] == []  # released by stop()
    assert [c.label for c in FakeCapture.made if c.started] == ["mic"]

    warning = "\n".join(errors)
    assert "[system]" in warning and "microphone only" in warning
    # The one piece of advice that fixes it, at the moment it is needed.
    assert "sttop doctor" in warning


def test_a_dead_system_stream_is_not_metered_or_drained(tmp_path, fake_captures):
    """Registering a capture that never started leaves a permanently empty
    meter in the UI and a segmenter that flushes a silent utterance on stop."""
    engine = engine_with_two_sources(tmp_path, failing={"system"})

    async def run():
        await engine.start("degraded")
        live = set(engine.status().levels), set(engine._segmenters)
        await engine.stop()
        return live

    metered, segmented = asyncio.run(run())
    assert metered == {"mic"}
    assert segmented == {"mic"}


def test_a_mic_that_will_not_start_still_fails_the_session(tmp_path, fake_captures):
    """There is nothing to transcribe without it, and a session that looks
    like it is recording silence is worse than one that says why it is not."""
    engine = engine_with_two_sources(tmp_path, failing={"mic", "system"})

    with pytest.raises(AudioError, match="permission"):
        asyncio.run(engine.start("doomed"))

    # And stop() still gives back the models and the executor thread.
    assert asyncio.run(engine.stop()) is not None
    assert engine.transcriber is None
