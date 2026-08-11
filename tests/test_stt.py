import pytest

from sttop import stt
from sttop.config import SttConfig


@pytest.fixture
def fake_backends(monkeypatch):
    """Stand in for both transcribers so nothing loads a model."""
    built = {}

    class Fake:
        def __init__(self, config: SttConfig) -> None:
            built["config"] = config
            self.describe = "fake"

    monkeypatch.setattr("sttop.stt.parakeet.ParakeetTranscriber", Fake)
    monkeypatch.setattr("sttop.stt.local.LocalTranscriber", Fake)
    return built


def test_blank_model_resolves_to_the_backend_default(fake_backends):
    stt.build(SttConfig(backend="whisper"))
    assert fake_backends["config"].model == stt.DEFAULT_MODELS["whisper"]


def test_an_explicit_model_is_kept(fake_backends):
    stt.build(SttConfig(backend="whisper", model="large-v3"))
    assert fake_backends["config"].model == "large-v3"


def test_building_does_not_pin_the_model_in_the_caller_config(fake_backends):
    """`model = ""` means "whatever suits the backend". Writing the resolved
    name back would freeze it, so a later backend change keeps the old
    backend's model."""
    config = SttConfig(backend="parakeet")
    stt.build(config)
    assert config.model == ""

    config.backend = "whisper"
    stt.build(config)
    assert fake_backends["config"].model == stt.DEFAULT_MODELS["whisper"]


def test_unknown_backend_names_the_alternatives():
    with pytest.raises(ValueError, match="parakeet, whisper"):
        stt.build(SttConfig(backend="wisper"))


def test_blank_language_means_autodetect(fake_backends):
    """A TOML file cannot write None, so blank is how a user says "you decide".
    Passed through verbatim it reaches the backend as an invalid language."""
    stt.build(SttConfig(backend="whisper", language=""))
    assert fake_backends["config"].language is None
