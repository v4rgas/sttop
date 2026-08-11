"""Audio source discovery, per platform.

Two capture worlds, one interface. On Linux a PulseAudio/PipeWire server hands
out both the microphone and a *monitor* of whatever is playing, so both halves
of the two-stream design come from the same place. macOS has no monitor: the
mic arrives through AVFoundation, and system audio comes from ScreenCaptureKit,
which taps the machine's output with no virtual driver and no device switching.
Everything below exists to hide that difference behind `resolve()`, which hands
back a `CaptureSpec` describing which reader to use.

`pactl` is used on Linux when it is there, for listing and substring matching,
but is no longer required: the audio server resolves `@DEFAULT_SOURCE@` and
`@DEFAULT_MONITOR@` itself, so a PipeWire box without `pulseaudio-utils`
records with defaults instead of failing at startup.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

from .ffmpeg import FFmpegMissing, ffmpeg_bin

MACOS = sys.platform == "darwin"

#: Pulse resolves these server-side, so they work with no `pactl` present.
PULSE_DEFAULT_SOURCE = "@DEFAULT_SOURCE@"
PULSE_DEFAULT_MONITOR = "@DEFAULT_MONITOR@"

class AudioError(RuntimeError):
    pass


class SystemAudioUnavailable(AudioError):
    """No way to hear what the machine is playing.

    Its own class because it is survivable: the session can run mic-only,
    which is worth much more than refusing to start.
    """


@dataclass(frozen=True)
class Source:
    index: int
    name: str
    driver: str
    spec: str
    state: str
    monitor: bool | None = None

    @property
    def is_monitor(self) -> bool:
        if self.monitor is not None:
            return self.monitor
        return self.name.endswith(".monitor")


@dataclass(frozen=True)
class CaptureSpec:
    """How to capture one stream, and what to call it.

    `backend` picks the reader; `device` is that backend's device string;
    `label` is what the journal header and the UI show.
    """

    backend: str  # "pulse" | "avfoundation" | "screencapture"
    device: str
    label: str

    def __str__(self) -> str:
        return self.label


# -- linux / pulseaudio ----------------------------------------------------


def have_pactl() -> bool:
    return shutil.which("pactl") is not None


def _pactl(*args: str) -> str:
    if not have_pactl():
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


def _pulse_sources() -> list[Source]:
    sources = []
    for line in _pactl("list", "short", "sources").splitlines():
        parts = line.split("\t")
        if len(parts) < 5 or not parts[0].isdigit():
            continue  # a header or some other row shape we do not understand
        index, name, driver, spec, state = parts[:5]
        sources.append(Source(int(index), name, driver, spec, state))
    return sources


def default_sink_monitor() -> str:
    """The monitor source carrying whatever is currently playing on this machine."""
    if not have_pactl():
        return PULSE_DEFAULT_MONITOR
    sink = _pactl("get-default-sink")
    if not sink or sink == "@DEFAULT_SINK@":
        return PULSE_DEFAULT_MONITOR
    return f"{sink}.monitor"


def default_source() -> str:
    """The default recording source, i.e. the active microphone."""
    if not have_pactl():
        return PULSE_DEFAULT_SOURCE
    source = _pactl("get-default-source")
    if not source or source == "@DEFAULT_SOURCE@":
        return PULSE_DEFAULT_SOURCE
    return source


# -- macos / avfoundation ---------------------------------------------------

_AVF_AUDIO_HEADER = "AVFoundation audio devices:"
_AVF_DEVICE = re.compile(r"\[(\d+)\]\s+(.+?)\s*$")


def _avfoundation_sources() -> list[Source]:
    """Parse `ffmpeg -f avfoundation -list_devices true`.

    ffmpeg prints the list to stderr and then exits non-zero, having been
    asked to open a device it was never given - so the exit status says
    nothing and only the text matters.
    """
    proc = subprocess.run(
        [ffmpeg_bin(), "-hide_banner", "-nostdin",
         "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True, text=True, timeout=15,
    )
    sources: list[Source] = []
    in_audio = False
    for raw in proc.stderr.splitlines():
        line = raw.split("] ", 1)[-1].strip()
        if line.endswith("devices:"):
            in_audio = line.endswith(_AVF_AUDIO_HEADER)
            continue
        match = _AVF_DEVICE.match(line) if in_audio else None
        if match:
            index, name = int(match.group(1)), match.group(2)
            sources.append(
                Source(index, name, "avfoundation", "", "AVAILABLE", monitor=False)
            )
    return sources


def _mac_availability() -> str | None:
    """Indirection so tests can stand in for the Apple frameworks."""
    from .screencapture import availability

    return availability()


def _mac_system_spec() -> CaptureSpec:
    reason = _mac_availability()
    if reason is not None:
        raise SystemAudioUnavailable(f"{reason} - see `sttop doctor`")
    return CaptureSpec("screencapture", "system", "system audio (ScreenCaptureKit)")


# -- platform-neutral API --------------------------------------------------


def list_sources() -> list[Source]:
    if MACOS:
        return _avfoundation_sources()
    if not have_pactl():
        return []  # defaults still work; there is just nothing to enumerate
    return _pulse_sources()


def resolve(requested: str | None, *, monitor: bool) -> CaptureSpec:
    """Resolve a configured source name, falling back to the sensible default.

    A requested name may be a full source name or a unique substring of one,
    so you can write `system_source = "hdmi"` instead of the full alsa id.

    Blank means "no preference", the same as unset - otherwise the empty string
    goes on to match every source and resolution fails as ambiguous, which is a
    baffling way to punish someone for writing `mic_source = ""`.
    """
    if requested is None or not requested.strip():
        return _default_spec(monitor=monitor)

    matched = _match(requested)
    if MACOS:
        return CaptureSpec("avfoundation", f":{matched.index}", matched.name)
    return CaptureSpec("pulse", matched.name, matched.name)


def _default_spec(*, monitor: bool) -> CaptureSpec:
    if MACOS:
        if monitor:
            return _mac_system_spec()
        return CaptureSpec("avfoundation", ":default", "default input")
    name = default_sink_monitor() if monitor else default_source()
    label = "default monitor" if name == PULSE_DEFAULT_MONITOR else name
    label = "default input" if name == PULSE_DEFAULT_SOURCE else label
    return CaptureSpec("pulse", name, label)


def _match(requested: str) -> Source:
    sources = list_sources()
    exact = [s for s in sources if s.name == requested]
    if exact:
        return exact[0]

    matches = [s for s in sources if requested.lower() in s.name.lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise AudioError(f"no audio source matching {requested!r}")
    names = ", ".join(s.name for s in matches)
    raise AudioError(f"{requested!r} is ambiguous, matches: {names}")


def diagnose() -> list[tuple[str, str]]:
    """(check, verdict) rows for `sttop doctor`. Never raises."""
    from . import ffmpeg as ffmpeg_mod

    rows = [("platform", sys.platform), ("ffmpeg", ffmpeg_mod.describe())]
    if MACOS:
        rows.append(("screencapturekit", _mac_availability() or "available"))
    else:
        rows.append(
            ("pactl", shutil.which("pactl") or "not found (defaults still work)")
        )
    for label, monitor in (("mic", False), ("system", True)):
        try:
            spec = resolve(None, monitor=monitor)
            rows.append((label, f"{spec.label}  [{spec.backend} {spec.device}]"))
        except (AudioError, FFmpegMissing) as exc:
            rows.append((label, f"unavailable - {exc}"))
    return rows
