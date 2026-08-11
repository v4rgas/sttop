"""Lifecycle checks that need no audio hardware and no models."""

import asyncio
from pathlib import Path

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
