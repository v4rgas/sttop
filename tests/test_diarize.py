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

#: Embeddings live in this many dimensions here - enough that `noise` can hand
#: out directions that are orthogonal to each other and to every voice, which a
#: circle cannot do for more than a handful of points.
DIMS = 8


def unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def voice(angle: float, jitter: float = 0.0) -> np.ndarray:
    """A direction in the first two dimensions, so the cosine between two
    voices is exactly the cosine of the angle between them."""
    vector = np.zeros(DIMS, dtype=np.float32)
    vector[0], vector[1] = np.cos(angle + jitter), np.sin(angle + jitter)
    return unit(vector)


def noise(index: int) -> np.ndarray:
    """A direction orthogonal to every voice and to every other `noise` -
    what two words of speech embed to, and it never lands twice."""
    vector = np.zeros(DIMS, dtype=np.float32)
    vector[2 + index % (DIMS - 2)] = 1.0
    return vector


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


#: An angle far enough from 0.0 to match nothing established there.
OUTLIER = 2.6


def test_short_noise_that_never_recurs_never_mints_a_speaker(config):
    """The bug from a real session: one remote participant became eight
    speakers, and every phantom was a two-word line - "Cool.", "Way.",
    "Oh yeah, I haven't." Two words embed to mostly noise, so they match
    nobody, and the old code read "matches nobody" as "somebody new"."""
    clusters = Clusters(config)
    for angle in (0.0, 0.2, -0.2, 0.1):
        clusters.assign(voice(angle))

    for index in range(6):
        assert clusters.assign(noise(index), may_create=False) is None
    assert clusters.count == 1


def test_a_short_voice_that_recurs_does_become_a_speaker(config):
    """The other half of the trade: someone who only ever says short things is
    still a participant. One unmatched utterance proves nothing; the second
    one that looks like it is a voice."""
    clusters = Clusters(config)
    clusters.assign(voice(0.0))

    assert clusters.assign(noise(0), may_create=False) is None
    assert clusters.assign(noise(0), may_create=False) == "spk2"
    assert clusters.count == 2


def test_a_long_outlier_is_still_allowed_to_be_a_new_speaker(config):
    """The flip side - somebody who actually says something is a participant,
    not noise, and must still be split off."""
    clusters = Clusters(config)
    clusters.assign(voice(0.0))
    assert clusters.assign(voice(OUTLIER), may_create=True) == "spk2"
    assert clusters.count == 2


def test_the_speaker_cap_gives_the_nearest_match_however_far(config):
    """`--speakers 2`: you know the room, so nothing outside it exists."""
    config.max_speakers = 2
    clusters = Clusters(config)
    assert clusters.assign(voice(0.0)) == "spk1"
    assert clusters.assign(voice(OUTLIER)) == "spk2"
    # A third, unrelated voice has nowhere to go but the nearer of the two.
    assert clusters.assign(voice(np.pi)) in {"spk1", "spk2"}
    assert clusters.count == 2
