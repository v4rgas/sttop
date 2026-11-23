"""onnxruntime settings shared by the speech and speaker models."""

from __future__ import annotations


def cpu_options(threads: int):
    """Session options for a model that shares the CPU with another one.

    onnxruntime's defaults are tuned for a model that has the machine to
    itself: one intra-op thread per core, each busy-waiting between operators.
    sttop runs Parakeet and Nemotron side by side, so those defaults have the
    two pools spinning against each other. On a 16-thread laptop that doubled
    total CPU for the same text and labels.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return options
