"""Downloads must stay on pinned revisions, including cache misses."""

import pytest

from sttop.stt.sources import MLX_FILES, MLX_SOURCES, ONNX_FILES, ONNX_SOURCES, snapshot


@pytest.mark.parametrize("cached", ["complete", "partial", "missing"])
def test_pinned_downloads_and_offline_cache(monkeypatch, tmp_path, cached):
    from huggingface_hub.errors import LocalEntryNotFoundError

    repo = "mlx-community/parakeet-tdt-0.6b-v3"
    revision = MLX_SOURCES[repo]
    calls = []
    if cached != "missing":
        (tmp_path / "config.json").touch()
    if cached == "complete":
        (tmp_path / "model.safetensors").touch()

    def download(**kwargs):
        calls.append(kwargs)
        if cached == "missing" and kwargs.get("local_files_only"):
            raise LocalEntryNotFoundError("not cached")
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    assert snapshot(repo, revision, MLX_FILES) == str(tmp_path)
    assert len(calls) == (1 if cached == "complete" else 2)
    for call in calls:
        assert call["repo_id"] == repo
        assert call["revision"] == revision
        assert call["endpoint"] == "https://huggingface.co"
        assert call["allow_patterns"] == ["config.json", "model.safetensors"]


def test_onnx_uses_the_pinned_snapshot(monkeypatch):
    import sys
    from types import SimpleNamespace

    from sttop.config import SttConfig
    from sttop.stt import DEFAULT_MODEL
    from sttop.stt.parakeet import ParakeetTranscriber

    calls = []

    def download(*args):
        assert args == (*ONNX_SOURCES[DEFAULT_MODEL], ONNX_FILES)
        return "/cached/pinned-model"

    monkeypatch.setattr("sttop.stt.parakeet.snapshot", download)
    monkeypatch.setattr("sttop.nativelog.quiet_onnxruntime", lambda: None)
    monkeypatch.setitem(sys.modules, "onnx_asr", SimpleNamespace(
        load_model=lambda model, **kwargs: calls.append(kwargs)
    ))
    ParakeetTranscriber(SttConfig(model=DEFAULT_MODEL))
    assert calls == [{
        "path": "/cached/pinned-model", "providers": ["CPUExecutionProvider"]
    }]
