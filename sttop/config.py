"""Configuration: dataclass defaults, overlaid with ~/.config/sttop/config.toml."""

from __future__ import annotations

import inspect
import re
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

from platformdirs import user_config_dir, user_data_dir

CONFIG_DIR = Path(user_config_dir("sttop"))
DATA_DIR = Path(user_data_dir("sttop"))
CONFIG_PATH = CONFIG_DIR / "config.toml"


class ConfigError(ValueError):
    """A config file that cannot be applied.

    Raised at load time, naming the offending key. The alternative - letting a
    string sit where the rest of sttop expects a section - surfaces as an
    AttributeError somewhere far away, long after the cause has scrolled off.
    """


@dataclass
class AudioConfig:
    #: PulseAudio/PipeWire source name. None resolves to the system default source.
    mic_source: str | None = None
    #: None resolves to the monitor of the default sink (i.e. whatever you hear),
    #: or on macOS to a loopback device such as BlackHole if one is installed.
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
    #: "parakeet" (onnxruntime) or "whisper" (faster-whisper).
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
class UiConfig:
    #: "auto" follows the terminal's background (ansi-dark / ansi-light), so the
    #: UI uses your own palette. Any Textual theme name also works, e.g.
    #: "gruvbox", "nord", "textual-dark".
    theme: str = "auto"


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
    diarize: DiarizeConfig = field(default_factory=DiarizeConfig)
    ui: UiConfig = field(default_factory=UiConfig)
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
        return _dump_toml(self)


def _merge(target, data: dict, prefix: str = "") -> None:
    """Overlay a parsed TOML dict onto a nested dataclass, ignoring unknown keys.

    Unknown keys are ignored so a config written by a newer sttop still loads,
    but a key with the *wrong shape* is rejected here, where the file and line
    are still in hand.
    """
    known = {f.name: f for f in fields(target)}
    for key, value in data.items():
        if key not in known:
            continue
        current = getattr(target, key)
        where = f"{prefix}{key}"
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ConfigError(f"{where} is a section, not a single value")
            _merge(current, value, f"{where}.")
        else:
            setattr(target, key, _checked(value, current, where))


def _checked(value, current, where: str):
    """Reject a value that would poison a field with the wrong type."""
    if current is None:  # an optional field accepts whatever it is given
        return value
    if isinstance(current, bool) != isinstance(value, bool):
        # bool is a subclass of int, so this pair needs its own test.
        raise ConfigError(f"{where} expects {type(current).__name__}")
    if isinstance(value, type(current)):
        return value
    if isinstance(current, float) and isinstance(value, int):
        return float(value)  # TOML `15` for a float field is not a mistake
    raise ConfigError(
        f"{where} expects {type(current).__name__}, got {type(value).__name__}"
    )


#: A `#:` comment above a field documents it. Picked up by `_field_docs` so the
#: generated config file explains its own knobs from this single source.
_FIELD_DOC = re.compile(r"#:\s?(.*)")


def _field_docs(cls) -> dict[str, list[str]]:
    """Map each field of a dataclass to its `#:` comment lines."""
    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError):  # pragma: no cover - source-less install
        return {}

    docs: dict[str, list[str]] = {}
    pending: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if match := _FIELD_DOC.match(stripped):
            pending.append(match.group(1).strip())
            continue
        name, sep, _ = stripped.partition(":")
        if sep and pending and name.isidentifier():
            docs[name] = pending
        pending = []
    return docs


def _dump_toml(config) -> str:
    """Minimal TOML writer - enough for the flat/one-level-nested shape above."""
    scalars, tables = [], []
    for spec in fields(config):
        value = getattr(config, spec.name)
        (tables if is_dataclass(value) else scalars).append((spec.name, value))

    lines = _entries(config, [name for name, _ in scalars])
    for name, table in tables:
        lines += [f"\n[{name}]", *_entries(table, [f.name for f in fields(table)])]
    return "\n".join(lines).lstrip("\n") + "\n"


def _entries(owner, names: list[str]) -> list[str]:
    docs = _field_docs(type(owner))
    lines: list[str] = []
    for name in names:
        comment = docs.get(name, [])
        lines += ["", *(f"# {line}" for line in comment)] if comment else []
        value = getattr(owner, name)
        # An unset field is written commented-out rather than dropped: a knob
        # you cannot see in the generated file is a knob you never find.
        lines.append(
            f"# {name} =" if value is None else f"{name} = {_toml_value(value)}"
        )
    return lines


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
