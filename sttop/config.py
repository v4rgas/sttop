"""Configuration: dataclass defaults, overlaid with ~/.config/sttop/config.toml."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

from platformdirs import user_config_dir, user_data_dir

CONFIG_DIR = Path(user_config_dir("sttop"))
DATA_DIR = Path(user_data_dir("sttop"))
CONFIG_PATH = CONFIG_DIR / "config.toml"


@dataclass
class AudioConfig:
    #: PulseAudio/PipeWire source name. None resolves to the system default source.
    mic_source: str | None = None
    #: None resolves to the monitor of the default sink (i.e. whatever you hear).
    system_source: str | None = None
    #: Keep the raw 16kHz mono capture alongside the transcript.
    save_wav: bool = False


@dataclass
class VadConfig:
    #: webrtcvad aggressiveness, 0 (permissive) .. 3 (strict).
    aggressiveness: int = 2
    #: Silence needed to close a segment.
    silence_ms: int = 700
    #: Force a flush on long monologues so the transcript stays live.
    max_segment_s: float = 15.0
    #: Drop blips shorter than this.
    min_segment_ms: int = 400
    #: Audio kept from *before* speech onset, so words aren't clipped.
    pad_ms: int = 300


@dataclass
class SttConfig:
    #: "parakeet" (onnxruntime), "whisper" (faster-whisper), or "cloud"
    #: (any OpenAI-compatible /audio/transcriptions endpoint).
    backend: str = "parakeet"
    #: Blank picks the default model for the chosen backend.
    model: str = ""
    #: "auto" | "cpu" | "cuda". Whisper only; parakeet is CPU/onnxruntime.
    device: str = "auto"
    #: "auto" | "int8" | "int8_float16" | "float16" | "float32"
    compute_type: str = "auto"
    #: None lets Whisper autodetect per segment.
    language: str | None = None
    beam_size: int = 1


@dataclass
class CloudConfig:
    #: Any OpenAI-compatible transcription API. Groq is the cheap fast default.
    #: Note: OpenRouter does NOT expose an audio-transcription endpoint.
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "whisper-large-v3-turbo"
    api_key_env: str = "GROQ_API_KEY"
    timeout_s: float = 60.0


@dataclass
class DiarizeConfig:
    enabled: bool = True
    #: Cosine similarity above which a voice is considered a known speaker.
    threshold: float = 0.50
    #: Grey zone below `threshold` where an utterance joins the nearest speaker
    #: without updating that speaker's centroid, instead of opening a new one.
    margin: float = 0.15
    #: Segments shorter than this are too small for a reliable voice embedding.
    min_speech_s: float = 1.5
    model: str = "speechbrain/spkrec-ecapa-voxceleb"


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    diarize: DiarizeConfig = field(default_factory=DiarizeConfig)
    #: One Markdown file per session lands here.
    sessions_dir: str = str(DATA_DIR / "sessions")
    #: Where WAVs land when audio.save_wav is on.
    audio_dir: str = str(DATA_DIR / "audio")

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        path = path or CONFIG_PATH
        cfg = cls()
        if path.is_file():
            with path.open("rb") as fh:
                _merge(cfg, tomllib.load(fh))
        return cfg

    def to_toml(self) -> str:
        return _dump_toml(asdict(self))


def _merge(target, data: dict) -> None:
    """Overlay a parsed TOML dict onto a nested dataclass, ignoring unknown keys."""
    known = {f.name: f for f in fields(target)}
    for key, value in data.items():
        spec = known.get(key)
        if spec is None:
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(target, key, value)


def _dump_toml(data: dict) -> str:
    """Minimal TOML writer - enough for the flat/one-level-nested shape above."""
    scalars, tables = [], []
    for key, value in data.items():
        (tables if isinstance(value, dict) else scalars).append((key, value))

    lines = [f"{k} = {_toml_value(v)}" for k, v in scalars if v is not None]
    for name, table in tables:
        lines.append(f"\n[{name}]")
        lines += [f"{k} = {_toml_value(v)}" for k, v in table.items() if v is not None]
    return "\n".join(lines) + "\n"


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return str(value)


def write_default_config(path: Path | None = None) -> Path:
    path = path or CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(Config().to_toml())
    return path
