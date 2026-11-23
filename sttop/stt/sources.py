"""Pinned community conversions of NVIDIA's Parakeet weights.

Updating these revisions is an explicit source change, never a floating download
from a repository's main branch. Custom model overrides are user-selected.
"""

from __future__ import annotations

from pathlib import Path

ONNX_SOURCES = {
    # Moondream's post-trained Parakeet TDT 0.6B v3: same architecture and
    # tokenizer, lower WER. This export is istupakov's v3 graphs with the
    # Ultra weights swapped in, so onnx-asr loads it as v3.
    "parakeet-ultra": (
        "Masterx/parakeet-tdt-0.6b-ultra-onnx",
        "99b09f030a5a6efeaa13cf2cf54592100ce2c3f1",
    ),
    "nemo-parakeet-tdt-0.6b-v3": (
        "istupakov/parakeet-tdt-0.6b-v3-onnx",
        "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce",
    ),
    "nemo-parakeet-tdt-0.6b-v2": (
        "istupakov/parakeet-tdt-0.6b-v2-onnx",
        "0bbb45a3365852604aef28b538a8f066f4ccaa85",
    ),
}
#: onnx-asr only knows NVIDIA's own model names; these aliases borrow one.
ONNX_ARCHITECTURES = {"parakeet-ultra": "nemo-parakeet-tdt-0.6b-v3"}
MLX_SOURCES = {
    # Tensor names, shapes and config match mlx-community's v3 (bf16 weights).
    "selcukkubur/parakeet-ultra-mlx": "b554592c50b2a48471add2daa3d46fa9f00fef5e",
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
