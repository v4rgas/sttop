"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import CONFIG_PATH, Config, ConfigError, write_default_config
from .stt import BACKENDS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sttop",
        description="Live speech-to-text monitor. Taps mic + system audio, "
        "transcribes and labels speakers in real time, writes Markdown.",
    )
    parser.add_argument("--version", action="version", version=f"sttop {__version__}")
    parser.add_argument("-c", "--config", type=Path, help=f"default: {CONFIG_PATH}")

    # `--config` reads as a global option, so accept it on either side of the
    # subcommand. SUPPRESS is what makes that work: without it the subcommand's
    # own default would overwrite a value already given before the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-c", "--config", type=Path, default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )

    sub = parser.add_subparsers(dest="command")

    def command(name: str, summary: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=summary, parents=[common])

    record = command("record", "start a session (default)")
    record.add_argument("-t", "--title", help="session title, used in the filename")
    record.add_argument("--mic", help="mic source name or substring")
    record.add_argument("--system", help="system/monitor source name or substring")
    record.add_argument("-m", "--model", help="override the backend's default model")
    record.add_argument("--backend", choices=list(BACKENDS))
    record.add_argument(
        "--language", help="force a language, e.g. es (default: autodetect)"
    )
    record.add_argument("--no-diarize", action="store_true", help="skip speaker id")
    record.add_argument("--save-wav", action="store_true", help="keep the raw audio")

    devices_cmd = command("devices", "list audio sources")
    devices_cmd.add_argument(
        "--test", action="store_true", help="record 1s from each and report levels"
    )

    command("doctor", "check audio deps and explain anything missing")
    command("sessions", "list recorded sessions")
    command("theme", "show the detected terminal colour scheme")
    command("config", "write a default config file")

    return parser


#: Global options that may appear before the subcommand, and whether the option
#: swallows the token after it.
_GLOBAL_OPTIONS = {"-c": True, "--config": True, "-h": False, "--help": False,
                   "--version": False}


def with_default_command(argv: list[str], commands: set[str]) -> list[str]:
    """Insert `record` when no subcommand was given.

    `sttop -t standup` means `sttop record -t standup`. Rather than parse twice
    and hope the first attempt fails cleanly - it does not, since `-t` is a
    record option and argparse rejects it outright - the command is filled in
    before parsing, so there is only ever one well-formed parse.
    """
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in commands:
            return argv
        option, joined, _ = token.partition("=")
        if option not in _GLOBAL_OPTIONS:
            break  # a record option, or a positional - record starts here
        # Skip the global option, plus its value when given as a separate
        # token (`-c path`) rather than joined on (`--config=path`).
        index += 2 if _GLOBAL_OPTIONS[option] and not joined else 1
    return [*argv[:index], "record", *argv[index:]]


def cmd_devices(config: Config, args) -> int:
    from .audio import capture, devices

    def selected(requested, *, monitor: bool):
        """The resolved spec, or the reason there is none. Never raises: one
        broken half must not hide the listing that explains why."""
        try:
            return devices.resolve(requested, monitor=monitor), None
        except devices.AudioError as exc:
            return None, str(exc)

    mic, mic_error = selected(config.audio.mic_source, monitor=False)
    system, sys_error = selected(config.audio.system_source, monitor=True)
    if mic is None and system is None:
        print(f"error: {mic_error}", file=sys.stderr)
        return 1

    print(f"mic     -> {mic.label if mic else f'unavailable - {mic_error}'}")
    print(f"system  -> {system.label if system else f'unavailable - {sys_error}'}\n")

    try:
        sources = devices.list_sources()
    except devices.AudioError as exc:
        print(f"cannot list sources: {exc}", file=sys.stderr)
        sources = []

    for source in sources:
        role = "mic" if mic and source.name == mic.label else (
            "sys" if system and source.name == system.label else "   "
        )
        kind = "monitor" if source.is_monitor else "input  "
        line = f"{role} {kind} {source.state:<10} {source.name}"
        if args.test:
            try:
                spec = _spec_for(devices, source)
                peak = capture.check_source(spec, seconds=1.0)
                line += f"   peak {peak:.3f}" + ("" if peak > 0.001 else "  (silent)")
            except Exception as exc:
                line += f"   [failed: {str(exc).splitlines()[0][:40]}]"
        print(line)

    if system is not None and system.label not in {s.name for s in sources}:
        print(f"\nsystem audio via {system.backend}: {system.device}")
    elif system is None:
        print("\nrun `sttop doctor` for how to enable system audio here")
    return 0


def _spec_for(devices, source):
    """A capture spec for one listed source, in that platform's terms."""
    if devices.MACOS:
        return devices.CaptureSpec("avfoundation", f":{source.index}", source.name)
    return devices.CaptureSpec("pulse", source.name, source.name)


#: Printed by `sttop doctor` when macOS has no system-audio source. There is
#: no Apple-provided monitor device, so this needs a loopback driver.
_MACOS_SYSTEM_AUDIO_HELP = """
macOS has no equivalent of a PulseAudio monitor, so system audio needs a
loopback device:

  brew install blackhole-2ch

Then open Audio MIDI Setup, create a Multi-Output Device containing both your
speakers and BlackHole, and select it as the system output - that is what
lets you keep hearing the call while sttop records it. sttop picks BlackHole
up automatically once it exists; nothing to configure here.

Until then sttop records the microphone only - your side of the call.
"""

_LINUX_SYSTEM_AUDIO_HELP = """
System audio comes from the monitor of your default sink. If it is missing,
check that PipeWire or PulseAudio is running (`systemctl --user status pipewire`).
`pactl` (Debian/Ubuntu: `sudo apt install pulseaudio-utils`) is optional - it is
only needed to list sources by name; defaults work without it.
"""


def cmd_doctor(config: Config) -> int:
    from .audio import devices

    for check, verdict in devices.diagnose():
        print(f"{check:<16}{verdict}")

    try:
        devices.resolve(config.audio.system_source, monitor=True)
    except devices.AudioError:
        print(_MACOS_SYSTEM_AUDIO_HELP if devices.MACOS else _LINUX_SYSTEM_AUDIO_HELP)
    return 0


def cmd_theme(config: Config) -> int:
    from .terminal import detect_theme, theme_sources

    for name, verdict in theme_sources():
        print(f"{name:<12}{verdict or '<no answer>'}")
    print(f"configured  {config.ui.theme}")
    print(f"\nusing       {detect_theme(config.ui.theme)}")
    return 0


def cmd_sessions(config: Config) -> int:
    from .journal import list_sessions

    directory = Path(config.sessions_dir)
    paths = list_sessions(directory)
    if not paths:
        print(f"no sessions yet in {directory}")
        return 0
    for path in paths:
        size = path.stat().st_size
        print(f"{path.name:<52} {size / 1024:6.1f} KiB")
    print(f"\n{len(paths)} session(s) in {directory}")
    return 0


def cmd_record(config: Config, args) -> int:
    from .tui import SttopApp

    if args.mic:
        config.audio.mic_source = args.mic
    if args.system:
        config.audio.system_source = args.system
    if args.model:
        config.stt.model = args.model
    if args.backend:
        config.stt.backend = args.backend
    if args.language:
        config.stt.language = args.language
    if args.no_diarize:
        config.diarize.enabled = False
    if args.save_wav:
        config.audio.save_wav = True

    path = SttopApp(config, args.title).run()
    if path:
        print(f"transcript: {path}")
    return 0


COMMANDS = {
    "record", "devices", "doctor", "sessions", "theme", "config",
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(with_default_command(argv, COMMANDS))

    if args.command == "config":  # writing a config must not require a valid one
        path = write_default_config(args.config)
        print(f"wrote {path}")
        return 0

    try:
        config = Config.load(args.config)
    except ConfigError as exc:
        print(f"error: bad config: {exc}", file=sys.stderr)
        return 1

    if args.command == "devices":
        return cmd_devices(config, args)
    if args.command == "doctor":
        return cmd_doctor(config)
    if args.command == "sessions":
        return cmd_sessions(config)
    if args.command == "theme":
        return cmd_theme(config)
    return cmd_record(config, args)


if __name__ == "__main__":
    sys.exit(main())
