"""sttop - live speech-to-text monitor for the terminal."""

__version__ = "0.1.0"

SAMPLE_RATE = 16_000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 320 samples
FRAME_BYTES = FRAME_SAMPLES * 2  # int16 mono
