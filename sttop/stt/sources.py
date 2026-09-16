"""Pinned community conversions of NVIDIA's Parakeet weights.

Updating these revisions is an explicit source change, never a floating download
from a repository's main branch. Custom model overrides are user-selected.
"""

from __future__ import annotations

from pathlib import Path

ONNX_SOURCES = {
    "nemo-parakeet-tdt-0.6b-v3": (
        "istupakov/parakeet-tdt-0.6b-v3-onnx",
        "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce",
    ),
    "nemo-parakeet-tdt-0.6b-v2": (
        "istupakov/parakeet-tdt-0.6b-v2-onnx",
        "0bbb45a3365852604aef28b538a8f066f4ccaa85",
    ),
}
MLX_SOURCES = {
    "mlx-community/parakeet-tdt-0.6b-v3": "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15",
    "mlx-community/parakeet-tdt-0.6b-v2": "8ae155301e23d820d82aa60d24817c900e69e487",
}
ONNX_FILES = (
    "config.json", "encoder-model.onnx", "encoder-model.onnx.data",
    "decoder_joint-model.onnx", "vocab.txt",
)
MLX_FILES = ("config.json", "model.safetensors")


def snapshot(repo: str, revision: str, files: tuple[str, ...]) -> str:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    options = dict(
        repo_id=repo, revision=revision, allow_patterns=list(files),
        endpoint="https://huggingface.co",
    )
    # A complete cached snapshot needs no network, including on subsequent runs.
    try:
        cached = snapshot_download(**options, local_files_only=True)
        if all((Path(cached) / name).is_file() for name in files):
            return cached
    except LocalEntryNotFoundError:
        pass
    return snapshot_download(**options)
