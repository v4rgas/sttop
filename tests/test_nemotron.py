"""The Nemotron backend's bookkeeping, with stand-in ONNX sessions.

The model's accuracy is measured against AMI, not here. What these pin down is
what sttop builds around it: utterances become a timeline the model can read,
each utterance gets the probabilities of its own frames back, memory stays
bounded over a long call, and the label follows the dominant speaker.
"""

from __future__ import annotations

import numpy as np
import pytest

from sttop import SAMPLE_RATE
from sttop.audio.segmenter import Segment
from sttop.config import DiarizeConfig
from sttop.diarize import SELF_LABEL, UNKNOWN_LABEL, NemotronDiarizer
from sttop.nemotron import HOP, N_FFT, RIGHT_CONTEXT, NemotronStream

SPEAKERS = 8
HIDDEN = 4


def constants() -> dict[str, np.ndarray]:
    return {
        "hidden_size": np.array(HIDDEN),
        "num_speakers": np.array(SPEAKERS),
        "subsampling_factor": np.array(8),
        "speaker_cache_length": np.array(16),
        "speaker_cache_silence_frames_per_speaker": np.array(1),
        "prediction_score_threshold": np.array(0.25),
        "latest_frames_score_boost": np.array(0.05),
        "strong_boost_rate": np.array(0.75),
        "weak_boost_rate": np.array(1.5),
        "min_positive_scores_rate": np.array(0.5),
        "silence_embeds": np.zeros(HIDDEN, np.float32),
    }


class FakeMel:
    """Shape-faithful preprocessor: one 128-bin frame per hop."""

    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []

    def run(self, _, feeds):
        samples = feeds["preemphasized"]
        self.inputs.append(samples)
        frames = (samples.shape[1] - N_FFT) // HOP + 1
        return [np.zeros((1, frames, 128), np.float32)]


class FakeModel:
    """Says speaker `active` is talking in every frame it is asked about, or
    nobody when it is None."""

    def __init__(self) -> None:
        self.active: int | None = 0

    def run(self, _, feeds):
        context = int(feeds["context_length"])
        mel = feeds["chunk_mel"].shape[1]
        frames = context + -(-mel // 8)
        logits = np.full((1, frames * 8, SPEAKERS), -10.0, np.float32)
        if self.active is not None:
            logits[..., self.active] = 10.0
        return [logits, np.zeros((1, frames, HIDDEN), np.float32)]


def stream() -> tuple[NemotronStream, FakeMel, FakeModel]:
    mel, model = FakeMel(), FakeModel()
    return NemotronStream(mel, model, constants()), mel, model


def tone(seconds: float) -> np.ndarray:
    return np.full(int(seconds * SAMPLE_RATE), 0.1, np.float32)


def test_each_utterance_gets_its_own_frames_back():
    s, _, _ = stream()
    for seconds, gap in ((1.0, 0.0), (2.5, 0.8), (0.4, 30.0)):
        probs = s.push(tone(seconds), gap)
        assert abs(len(probs) - seconds * 100) <= 1
        assert probs.shape[1] == SPEAKERS


def test_every_step_carries_lookahead_past_the_utterance():
    """The right context is silence the segmenter has already heard, so the
    model always gets it - it is where the 0.7 s end-of-turn pause goes."""
    s, mel, _ = stream()
    s.push(tone(1.0), 0.0)
    window = mel.inputs[-1].shape[1]
    assert window >= SAMPLE_RATE * 1.0 + RIGHT_CONTEXT * 8 * HOP


def test_a_long_call_does_not_keep_its_audio():
    s, _, _ = stream()
    for _ in range(200):
        s.push(tone(5.0), 1.0)
    # The mel window of the next step reaches back a fraction of a second.
    assert len(s._audio) < SAMPLE_RATE * 1


def utterance(start: float, seconds: float) -> Segment:
    pcm = (tone(seconds) * 32767).astype(np.int16).tobytes()
    return Segment("system", pcm, start, start + seconds)


def test_label_is_the_dominant_speaker_counted_from_one():
    s, _, model = stream()
    diarizer = NemotronDiarizer(DiarizeConfig(), stream=s)

    model.active = 0
    assert diarizer.label(utterance(0.0, 2.0), is_mic=False) == "spk1"
    model.active = 2
    assert diarizer.label(utterance(3.0, 2.0), is_mic=False) == "spk3"
    assert diarizer.speaker_count == 2
    assert diarizer.take_merges() == []


def test_the_microphone_never_reaches_the_model():
    s, mel, _ = stream()
    diarizer = NemotronDiarizer(DiarizeConfig(), stream=s)
    assert diarizer.label(utterance(0.0, 2.0), is_mic=True) == SELF_LABEL
    assert mel.inputs == []


@pytest.mark.parametrize(("start", "expected"), [(2.5, "spk1"), (30.0, UNKNOWN_LABEL)])
def test_nobody_the_model_knows_falls_back_to_who_just_spoke(start, expected):
    s, _, model = stream()
    diarizer = NemotronDiarizer(DiarizeConfig(), stream=s)
    diarizer.label(utterance(0.0, 2.0), is_mic=False)

    model.active = None  # a cough: every channel quiet
    assert diarizer.label(utterance(start, 1.0), is_mic=False) == expected



def test_expected_speakers_caps_the_labels():
    s, _, model = stream()
    diarizer = NemotronDiarizer(DiarizeConfig(max_speakers=2), stream=s)
    model.active = 0
    diarizer.label(utterance(0.0, 2.0), is_mic=False)
    model.active = 5  # a sixth voice the user said would not be there
    assert diarizer.label(utterance(3.0, 2.0), is_mic=False) in ("spk1", "spk2")
