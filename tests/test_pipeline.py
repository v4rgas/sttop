"""End-to-end check: play speech into the default sink, read it back off the
monitor source, and assert the transcript lands in the Markdown journal.

Needs a running audio server and downloads the `tiny` Whisper model, so it is
opt-in: STTOP_INTEGRATION=1 uv run --extra dev pytest
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

import pytest

SAMPLE_URL = "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac"

pytestmark = pytest.mark.skipif(
    os.environ.get("STTOP_INTEGRATION") != "1",
    reason="set STTOP_INTEGRATION=1 to run (needs audio hardware + model download)",
)


def _have_ffmpeg() -> bool:
    """Enough to prepare the sample - the diarizer needs no audio server."""
    return bool(shutil.which("ffmpeg"))


def _have_audio() -> bool:
    if not (_have_ffmpeg() and shutil.which("pactl")):
        return False
    probe = subprocess.run(["pactl", "get-default-sink"], capture_output=True, text=True)
    return probe.returncode == 0 and bool(probe.stdout.strip())


@pytest.fixture(scope="session")
def speech_wav(tmp_path_factory) -> Path:
    if not _have_ffmpeg():
        pytest.skip("ffmpeg not on PATH")  # a skip, not a fixture error
    directory = tmp_path_factory.mktemp("audio")
    flac, wav = directory / "sample.flac", directory / "sample.wav"
    urllib.request.urlretrieve(SAMPLE_URL, flac)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", str(flac), "-ac", "1", "-ar", "16000", str(wav), "-y"],
        check=True,
    )
    return wav


@pytest.mark.skipif(not _have_audio(), reason="no ffmpeg/pactl or no default sink")
def test_system_audio_reaches_the_transcript(tmp_path, speech_wav):
    from sttop.config import Config
    from sttop.engine import Engine

    config = Config()
    config.diarize.enabled = False  # covered by test_one_voice_stays_one_speaker
    config.sessions_dir = str(tmp_path / "sessions")

    seen = []

    async def run():
        engine = Engine(config, seen.append)
        await engine.start("integration")

        async def play():
            await asyncio.sleep(1.5)
            player = await asyncio.create_subprocess_exec(
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-re",
                "-i", str(speech_wav), "-f", "pulse", "default",
            )
            await player.wait()

        asyncio.create_task(play())
        await asyncio.sleep(16)
        return await engine.stop()

    path = asyncio.run(run())

    text = path.read_text().lower()
    assert "country" in text, f"sample speech missing from transcript:\n{text}"
    assert any(u.source == "system" for u in seen)
    assert "duration:" in text  # footer written on close


def test_one_voice_stays_one_speaker(speech_wav):
    """Online clustering must not split a single speaker across labels."""
    import wave

    import numpy as np

    from sttop.audio.segmenter import Segment
    from sttop.config import DiarizeConfig
    from sttop.diarize import UNKNOWN_LABEL, Diarizer, build

    with wave.open(str(speech_wav)) as handle:
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)

    diarizer = build(DiarizeConfig())
    assert isinstance(diarizer, Diarizer), "diarizer unavailable - install sttop[diarize]"

    def segment(chunk, start):
        return Segment("system", chunk.astype(np.int16).tobytes(), start,
                       start + len(chunk) / 16000)

    step = 32000  # 2s chunks, as the segmenter would produce
    same = [
        diarizer.label(segment(samples[i:i + step], i / 16000), is_mic=False)
        for i in range(0, len(samples) - step // 2, step)
    ]
    # A short opening utterance is `spk?` until a second one corroborates it -
    # the diarizer would rather say it does not know than invent a participant.
    # What it must never do is spread one voice over several speaker labels.
    named = [label for label in same if label != UNKNOWN_LABEL]
    assert named, f"never settled on a speaker: {same}"
    assert len(set(named)) == 1, f"one voice split across labels: {same}"
    assert UNKNOWN_LABEL not in same[same.index(named[0]):], f"unsettled again: {same}"

    # A clearly different timbre must open a second speaker.
    shifted = np.interp(
        np.arange(0, len(samples), 1.45), np.arange(len(samples)),
        samples.astype(np.float32),
    )
    # Long enough to stand on its own, so one utterance is enough.
    other = diarizer.label(segment(shifted[:step * 2], 60.0), is_mic=False)
    assert other != same[0]
    assert diarizer.speaker_count == 2

    # The mic side never needs a model.
    assert diarizer.label(segment(samples[:step], 90.0), is_mic=True) == "you"

    # Short interjections - the "Cool." / "A ver." lines that used to mint a
    # phantom participant each. They may attach to somebody, but they must not
    # invent anybody: real audio, and the count has to hold.
    before = diarizer.speaker_count
    short = int(2.2 * 16000)
    for index in range(6):
        start = 120.0 + index * 30.0  # far apart, so no "still talking" fallback
        chunk = shifted[index * short : index * short + short]
        diarizer.label(segment(chunk, start), is_mic=False)
    assert diarizer.speaker_count == before
