"""Clustering behaviour, with hand-made embeddings instead of a voice model.

The failure this guards against is over-splitting: one person arriving as a
handful of noisy embeddings and walking away with four labels. Synthetic
vectors let us set the similarities exactly, which real speech never does.
"""

from __future__ import annotations

import numpy as np
import pytest

from sttop.config import DiarizeConfig
from sttop.diarize import Clusters, _windows


def unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def voice(angle: float, jitter: float = 0.0) -> np.ndarray:
    """A point on the unit circle, so cosine similarity is cos(angle diff)."""
    return unit(np.cos(angle + jitter), np.sin(angle + jitter))


@pytest.fixture
def config() -> DiarizeConfig:
    return DiarizeConfig()


def test_noisy_repeats_of_one_voice_stay_one_speaker(config):
    clusters = Clusters(config)
    labels = [clusters.assign(voice(0.0, j)) for j in (0.0, 0.9, -0.8, 0.6, -0.5)]
    assert len(set(labels)) == 1
    assert clusters.count == 1


def test_a_distinct_voice_opens_a_second_speaker(config):
    clusters = Clusters(config)
    first = clusters.assign(voice(0.0))
    second = clusters.assign(voice(np.pi))  # cosine -1, as unlike as it gets
    assert first != second
    assert clusters.count == 2


#: Two voices that start far enough apart to cluster separately, then turn out
#: to overlap - the shape of one person split across two labels.
CONVERGING = [0.0, 0.25, -0.25, 1.5, 1.25, 1.75, 0.55, 0.95, 0.6, 0.9]


def test_settled_speakers_that_converge_are_merged_retroactively(config):
    clusters = Clusters(config)
    labels = [clusters.assign(voice(angle)) for angle in CONVERGING]

    assert len(set(labels[:6])) == 2, "the two voices must split before merging"
    assert clusters.take_merges() == [("spk2", "spk1")]
    assert clusters.count == 1
    assert not clusters.take_merges(), "merges are drained, not repeated"


def test_the_utterance_that_triggers_a_merge_gets_the_surviving_label(config):
    """Its journal line is written before the rename runs, so it has to carry
    the label the rename will leave behind."""
    clusters = Clusters(config)
    labels = [clusters.assign(voice(angle)) for angle in CONVERGING]
    merged_away = {old for old, _ in clusters.take_merges()}
    assert labels[-1] not in merged_away


def test_the_surviving_label_is_the_older_one(config):
    clusters = Clusters(config)
    for angle in CONVERGING:
        clusters.assign(voice(angle))
    assert clusters.assign(voice(0.3)) == "spk1"


def test_unsettled_speakers_are_never_merged(config):
    """A voice heard a couple of times has no centroid worth trusting.
    Splitting one person is recoverable; merging two people is not."""
    config.warmup = 99
    clusters = Clusters(config)
    for angle in CONVERGING:
        clusters.assign(voice(angle))
    assert clusters.count == 2
    assert not clusters.take_merges()


def test_long_audio_is_embedded_as_overlapping_windows():
    audio = np.zeros(16000 * 10, dtype=np.float32)
    windows = _windows(audio, window_s=3.0)
    assert len(windows) > 1
    assert all(len(w) == 16000 * 3 for w in windows)


def test_short_audio_is_embedded_whole():
    audio = np.zeros(16000 * 2, dtype=np.float32)
    assert [len(w) for w in _windows(audio, window_s=3.0)] == [16000 * 2]
