import pytest

from sttop import stt
from sttop.config import SttConfig


@pytest.fixture
def fake_backend(monkeypatch):
    """Stand in for Parakeet so nothing loads a model."""
    built = {}

    class Fake:
        def __init__(self, config: SttConfig) -> None:
            built["config"] = config
            self.describe = "fake"

    monkeypatch.setattr("sttop.stt.parakeet.ParakeetTranscriber", Fake)
    monkeypatch.setattr(stt, "apple_silicon", lambda: False)
    return built


def test_blank_model_resolves_to_parakeet_default(fake_backend):
    config = SttConfig()
    stt.build(config)
    assert fake_backend["config"].model == stt.DEFAULT_MODEL
    assert config.model == ""


def test_an_explicit_model_is_kept(fake_backend):
    stt.build(SttConfig(model="nemo-parakeet-tdt-0.6b-v2"))
    assert fake_backend["config"].model == "nemo-parakeet-tdt-0.6b-v2"


def test_blank_language_means_autodetect(fake_backend):
    stt.build(SttConfig(language=""))
    assert fake_backend["config"].language is None


def test_parakeet_does_not_auto_select_coreml_during_loading(monkeypatch):
    import sys
    from types import SimpleNamespace

    from sttop.stt.parakeet import ParakeetTranscriber

    loaded = []

    def load(model, **kwargs):
        loaded.append((model, kwargs))
        return object()

    monkeypatch.setitem(sys.modules, "onnx_asr", SimpleNamespace(load_model=load))
    monkeypatch.setattr("sttop.nativelog.quiet_onnxruntime", lambda: None)
    monkeypatch.setattr("sttop.stt.parakeet.snapshot", lambda *args: "/cached/onnx")
    transcriber = ParakeetTranscriber(SttConfig(model=stt.DEFAULT_MODEL))
    assert loaded == [(stt.DEFAULT_MODEL, {
        "providers": ["CPUExecutionProvider"], "path": "/cached/onnx"
    })]
    assert "cpu" in transcriber.describe


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_auto_uses_mlx_on_apple_silicon(fake_backend, monkeypatch, version):
    monkeypatch.setattr(stt, "apple_silicon", lambda: True)
    loaded = []
    monkeypatch.setattr(
        "sttop.stt.parakeet_mlx.ParakeetMLXTranscriber", loaded.append
    )
    config = SttConfig(model=f"nemo-parakeet-tdt-0.6b-{version}")
    stt.build(config)
    assert loaded[0].model == f"mlx-community/parakeet-tdt-0.6b-{version}"
    assert loaded[0].backend == "mlx"
    assert config.backend == "auto"
    assert fake_backend == {}


def test_onnx_override_and_custom_models_stay_on_cpu(fake_backend, monkeypatch):
    monkeypatch.setattr(stt, "apple_silicon", lambda: True)
    stt.build(SttConfig(backend="onnx"))
    assert fake_backend["config"].backend == "onnx"
    stt.build(SttConfig(model="custom/onnx-model"))
    assert fake_backend["config"].model == "custom/onnx-model"


def test_auto_falls_back_only_for_missing_mlx_dependencies(fake_backend, monkeypatch):
    monkeypatch.setattr(stt, "apple_silicon", lambda: True)

    def missing(config):
        raise ImportError("missing mlx")

    monkeypatch.setattr("sttop.stt.parakeet_mlx.ParakeetMLXTranscriber", missing)
    stt.build(SttConfig())
    assert fake_backend["config"].model == stt.DEFAULT_MODEL
    with pytest.raises(ImportError):
        stt.build(SttConfig(backend="mlx"))
    with pytest.raises(ImportError):
        stt.build(SttConfig(model="mlx-community/custom-model"))

    def broken(config):
        raise RuntimeError("model download failed")

    monkeypatch.setattr("sttop.stt.parakeet_mlx.ParakeetMLXTranscriber", broken)
    with pytest.raises(RuntimeError, match="download failed"):
        stt.build(SttConfig())


def test_mlx_rejects_unsupported_platform_and_unknown_backend(fake_backend):
    with pytest.raises(ValueError, match="Apple Silicon"):
        stt.build(SttConfig(backend="mlx"))
    with pytest.raises(ValueError, match="stt.backend"):
        stt.build(SttConfig(backend="typo"))


def test_mlx_consumes_normalized_pcm_in_memory(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    import numpy as np

    from sttop.stt.parakeet_mlx import ParakeetMLXTranscriber

    monkeypatch.setattr("sttop.stt.parakeet_mlx.snapshot", lambda *args: "/cached/mlx")

    calls = {}
    preprocess = SimpleNamespace(sample_rate=16000)
    model = SimpleNamespace(
        preprocessor_config=preprocess,
        parameters=lambda: "weights",
        generate=lambda mel: [SimpleNamespace(text="  Hola, hello.  ")],
    )

    def get_logmel(audio, config):
        calls["audio"] = audio
        assert config is preprocess
        return audio

    mx = ModuleType("mlx.core")
    mx.array = np.array
    mx.metal = SimpleNamespace(is_available=lambda: True)
    mx.eval = lambda weights: calls.update(weights=weights)
    mx.clear_cache = lambda: calls.update(cleared=True)
    mlx = ModuleType("mlx")
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "parakeet_mlx", SimpleNamespace(
        from_pretrained=lambda name: calls.update(model=name) or model
    ))
    monkeypatch.setitem(sys.modules, "parakeet_mlx.audio", SimpleNamespace(
        get_logmel=get_logmel
    ))
    config = SttConfig(model=stt.MLX_MODELS[stt.DEFAULT_MODEL], language="es")
    transcriber = ParakeetMLXTranscriber(config)
    pcm = np.array([-32768, 0, 16384, 32767], dtype=np.int16).tobytes()
    result = transcriber.transcribe(pcm)
    np.testing.assert_allclose(calls["audio"], [-1, 0, 0.5, 32767 / 32768])
    assert calls["audio"].dtype == np.float32
    assert result.text == "Hola, hello."
    assert result.language == "es"
    assert calls["model"] == "/cached/mlx"
    assert calls["weights"] == "weights"
    assert "mlx" in transcriber.describe
    assert transcriber.transcribe(b"").text == ""
    transcriber.close()
    assert transcriber._model is None
    assert calls["cleared"]
