"""PulseAudio/PipeWire source discovery via pactl."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


class AudioError(RuntimeError):
    pass


@dataclass(frozen=True)
class Source:
    index: int
    name: str
    driver: str
    spec: str
    state: str

    @property
    def is_monitor(self) -> bool:
        return self.name.endswith(".monitor")


def _pactl(*args: str) -> str:
    if not shutil.which("pactl"):
        raise AudioError("pactl not found - sttop needs PipeWire or PulseAudio")
    try:
        out = subprocess.run(
            ["pactl", *args], capture_output=True, text=True, timeout=5, check=True
        )
    except subprocess.CalledProcessError as exc:
        raise AudioError(f"pactl {' '.join(args)} failed: {exc.stderr.strip()}") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioError("pactl timed out - is the audio server running?") from exc
    return out.stdout.strip()


def list_sources() -> list[Source]:
    sources = []
    for line in _pactl("list", "short", "sources").splitlines():
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        index, name, driver, spec, state = parts[:5]
        sources.append(Source(int(index), name, driver, spec, state))
    return sources


def default_sink_monitor() -> str:
    """The monitor source carrying whatever is currently playing on this machine."""
    sink = _pactl("get-default-sink")
    if not sink or sink == "@DEFAULT_SINK@":
        raise AudioError("no default sink - cannot capture system audio")
    return f"{sink}.monitor"


def default_source() -> str:
    """The default recording source, i.e. the active microphone."""
    source = _pactl("get-default-source")
    if not source or source == "@DEFAULT_SOURCE@":
        raise AudioError("no default source - cannot capture the microphone")
    return source


def resolve(requested: str | None, *, monitor: bool) -> str:
    """Resolve a configured source name, falling back to the sensible default.

    A requested name may be a full source name or a unique substring of one,
    so you can write `system_source = "hdmi"` instead of the full alsa id.
    """
    if requested is None:
        return default_sink_monitor() if monitor else default_source()

    names = [s.name for s in list_sources()]
    if requested in names:
        return requested

    matches = [n for n in names if requested in n]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise AudioError(f"no audio source matching {requested!r}")
    raise AudioError(f"{requested!r} is ambiguous, matches: {', '.join(matches)}")
