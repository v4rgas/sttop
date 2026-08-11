"""Lifecycle checks that need no audio hardware and no models."""

import asyncio

from sttop.config import Config
from sttop.engine import Engine


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
