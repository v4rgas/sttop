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
        self.on_frame = on_frame
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


def test_a_diarizer_merge_relabels_the_transcript(tmp_path):
    """The diarizer decides late that two labels were one person; the engine is
    what turns that decision into a rewritten transcript."""
    engine, _, diarizer = started_engine(tmp_path)
    engine.journal.append(Utterance("system", "spk2", 1.0, 2.0, "goodbye"))
    renames = []
    engine._on_rename = lambda old, new, lines: renames.append((old, new, lines))
    diarizer.take_merges = lambda: [("spk2", "spk1")]

    engine._apply_merges()

    text = engine.journal.path.read_text()
    assert "**spk2**" not in text and text.count("**spk1**") == 2
    assert renames == [("spk2", "spk1", 1)]


def test_utterances_stream_to_the_pipe_command(tmp_path, fake_captures):
    import json

    out = tmp_path / "stream.jsonl"
    engine = engine_with_two_sources(tmp_path, failing=set())
    engine.config.pipe = f"cat > {out}"

    async def run():
        await engine.start("piped")
        engine._emit({"type": "utterance", "speaker": "you", "text": "hola"})
        engine.rename_speaker("spk1", "Ana")
        await engine.stop()

    asyncio.run(run())
    events = [json.loads(line) for line in out.read_text().splitlines()]
    assert events == [
        {"type": "utterance", "speaker": "you", "text": "hola"},
        {"type": "rename", "old": "spk1", "new": "Ana"},
    ]


def test_a_pipe_command_that_dies_does_not_end_the_session(tmp_path, fake_captures):
    errors: list[str] = []
    engine = engine_with_two_sources(tmp_path, failing=set())
    engine.config.pipe = "true"
    engine._on_error = errors.append

    async def run():
        await engine.start("piped")
        await engine._pipe.wait()
        engine._emit({"type": "utterance"})
        engine._emit({"type": "utterance"})
        return await engine.stop()

    assert asyncio.run(run()) is not None
    piped = [e for e in errors if "[pipe]" in e]
    assert piped == ["[pipe] `true` exited; stopped streaming"]


def test_model_initialization_overlaps_and_reports_the_remaining_stage(
    tmp_path, monkeypatch
):
    import threading

    speech_entered, speaker_entered = threading.Event(), threading.Event()
    release_speech, release_speaker = threading.Event(), threading.Event()
    engine = Engine(Config(sessions_dir=str(tmp_path)), lambda utterance: None)
    monkeypatch.setattr(engine, "_resolve_sources", lambda: (None, None))

    def speech(config):
        speech_entered.set()
        assert release_speech.wait(5)
        return FakeModel()

    def speakers(config):
        speaker_entered.set()
        assert release_speaker.wait(5)
        return FakeModel()

    monkeypatch.setattr(engine_mod.stt, "build", speech)
    monkeypatch.setattr(engine_mod.diarize_mod, "build", speakers)

    async def wait_until(predicate):
        async with asyncio.timeout(3):
            while not predicate():
                await asyncio.sleep(0.01)

    async def run():
        preparing = asyncio.create_task(engine.prepare())
        try:
            await wait_until(lambda: speech_entered.is_set() and speaker_entered.is_set())
            assert "speech model" in engine.status().loading
            assert "speaker model" in engine.status().loading
            release_speech.set()
            await wait_until(lambda: engine.transcriber is not None)
            assert engine.status().loading == "Loading speaker model"
            assert not engine.status().running
        finally:
            release_speech.set()
            release_speaker.set()
            await preparing
            await engine.stop()

    asyncio.run(run())


class BufferedModel(FakeModel):
    def transcribe(self, pcm):
        from sttop.stt import Transcript

        return Transcript(text=pcm.decode())

    def label(self, segment, is_mic):
        return "you" if is_mic else "spk1"

    def take_merges(self):
        return []


@pytest.fixture
def slow_models(tmp_path, monkeypatch, fake_captures):
    import threading

    config = Config(sessions_dir=str(tmp_path / "sessions"))
    seen, errors = [], []
    engine = Engine(config, seen.append, errors.append)
    source = CaptureSpec("pulse", "default", "mic")
    monkeypatch.setattr(engine, "_resolve_sources", lambda: (source, None))
    entered, release = threading.Event(), threading.Event()
    model = BufferedModel()
    FakeCapture.fails = set()
    FakeCapture.made = []

    def speech(config):
        entered.set()
        assert release.wait(5), "test did not release model loader"
        return model

    monkeypatch.setattr(engine_mod.stt, "build", speech)
    monkeypatch.setattr(engine_mod.diarize_mod, "build", lambda config: BufferedModel())
    yield engine, entered, release, model, seen, errors
    release.set()


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.01)


def test_capture_starts_before_models_and_buffered_speech_catches_up(slow_models):
    from sttop.audio.segmenter import Segment

    engine, entered, release, _, seen, errors = slow_models

    async def run():
        path = await engine.start("early")
        try:
            await until(entered.is_set)
            assert engine.status().running
            assert FakeCapture.made[0].started
            assert engine.transcriber is None
            engine._queue.put_nowait(Segment("mic", b"opening", 0.0, 1.0))
            engine._queue.put_nowait(Segment("system", b"reply", 1.0, 2.0))
            assert engine.status().backlog == 2
            assert not seen
            release.set()
            await until(lambda: len(seen) == 2)
            engine._queue.put_nowait(Segment("mic", b"live", 2.0, 3.0))
            await until(lambda: len(seen) == 3)
            assert [u.text for u in seen] == ["opening", "reply", "live"]
            assert [u.start for u in seen] == [0.0, 1.0, 2.0]
            assert not engine.status().loading
            assert not engine.status().catching_up
        finally:
            release.set()
            await engine.stop()
        assert "opening" in path.read_text()
        assert not errors

    asyncio.run(run())


def test_quit_stops_capture_then_waits_for_models_and_drains_opening(slow_models):
    from sttop.audio.segmenter import Segment

    engine, entered, release, model, seen, _ = slow_models

    async def run():
        await engine.start("early quit")
        await until(entered.is_set)
        engine._queue.put_nowait(Segment("mic", b"opening", 0.0, 1.0))
        stopping = asyncio.create_task(engine.stop())
        try:
            await until(lambda: FakeCapture.made[0].stopped)
            assert not engine.status().running
            assert not stopping.done()
            elapsed = engine.status().elapsed
            await asyncio.sleep(0.02)
            assert engine.status().elapsed == elapsed
        finally:
            release.set()
        path = await stopping
        assert [u.text for u in seen] == ["opening"]
        assert "opening" in path.read_text()
        assert model.closed
        assert engine.journal is None

    asyncio.run(run())


def test_loading_failure_stops_capture_and_reports_untranscribed_audio(
    slow_models, monkeypatch
):
    engine, _, _, _, _, errors = slow_models

    def fail(config):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(engine_mod.stt, "build", fail)

    async def run():
        await engine.start("failed")
        await until(lambda: bool(errors))
        assert FakeCapture.made[0].stopped
        assert not engine.status().running
        assert not engine.status().loading
        assert "model unavailable" in engine.status().error
        assert "buffered speech could not be transcribed" in errors[0]
        await engine.stop()
        assert engine.transcriber is None and engine.diarizer is None

    asyncio.run(run())


def test_audio_frames_during_loading_are_flushed_and_pause_is_respected(slow_models):
    from types import SimpleNamespace

    from sttop import FRAME_BYTES
    from sttop.stt import Transcript

    engine, entered, release, model, seen, _ = slow_models
    audio = []

    def transcribe(pcm):
        audio.append(pcm)
        return Transcript(text="speech")

    model.transcribe = transcribe

    async def run():
        await engine.start("frames")
        await until(entered.is_set)
        engine._segmenters["mic"]._vad = SimpleNamespace(is_speech=lambda *args: True)
        capture = FakeCapture.made[0]
        frame = b"\x01\x00" * (FRAME_BYTES // 2)
        for _ in range(40):
            capture.on_frame(frame)
        engine.toggle_pause()  # flush the opening utterance before pausing
        assert engine.status().backlog == 1
        for _ in range(40):
            capture.on_frame(frame)
        engine.toggle_pause()
        for _ in range(40):
            capture.on_frame(frame)
        assert not seen  # all this audio arrived before the model was ready
        release.set()
        await engine.stop()  # flush the still-open final utterance too
        assert audio == [frame * 40, frame * 40]
        assert len(seen) == 2
        assert seen[1].start - seen[0].start == pytest.approx(1.6)

    asyncio.run(run())
