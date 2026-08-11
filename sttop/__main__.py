"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import CONFIG_PATH, Config, write_default_config
from .stt import BACKENDS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sttop",
        description="Live speech-to-text monitor. Taps mic + system audio, "
        "transcribes and labels speakers in real time, writes Markdown.",
    )
    parser.add_argument("--version", action="version", version=f"sttop {__version__}")
    parser.add_argument("-c", "--config", type=Path, help=f"default: {CONFIG_PATH}")

    sub = parser.add_subparsers(dest="command")

    record = sub.add_parser("record", help="start a session (default)")
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

    devices_cmd = sub.add_parser("devices", help="list audio sources")
    devices_cmd.add_argument(
        "--test", action="store_true", help="record 1s from each and report levels"
    )

    sub.add_parser("sessions", help="list recorded sessions")
    sub.add_parser("theme", help="show the detected terminal colour scheme")
    sub.add_parser("config", help="write a default config file")

    return parser


def cmd_devices(config: Config, args) -> int:
    from .audio import capture, devices

    try:
        sources = devices.list_sources()
        mic = devices.resolve(config.audio.mic_source, monitor=False)
        system = devices.resolve(config.audio.system_source, monitor=True)
    except devices.AudioError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"mic     -> {mic}")
    print(f"system  -> {system}\n")
    for source in sources:
        role = "mic" if source.name == mic else "sys" if source.name == system else "   "
        kind = "monitor" if source.is_monitor else "input  "
        line = f"{role} {kind} {source.state:<10} {source.name}"
        if args.test:
            try:
                peak = capture.check_source(source.name, seconds=1.0)
                line += f"   peak {peak:.3f}" + ("" if peak > 0.001 else "  (silent)")
            except Exception as exc:
                line += f"   [failed: {str(exc).splitlines()[0][:40]}]"
        print(line)
    return 0


def cmd_theme(config: Config) -> int:
    import os

    from .terminal import detect_theme, from_colorfgbg, query_osc11

    colorfgbg = os.environ.get("COLORFGBG")
    print(f"COLORFGBG   {colorfgbg or '<unset>'} -> {from_colorfgbg(colorfgbg)}")
    print(f"OSC 11      {query_osc11() or '<no reply>'}")
    print(f"configured  {config.ui.theme}")
    print(f"\nusing      {detect_theme(config.ui.theme)}")
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = Config.load(args.config)

    if args.command == "devices":
        return cmd_devices(config, args)
    if args.command == "sessions":
        return cmd_sessions(config)
    if args.command == "theme":
        return cmd_theme(config)
    if args.command == "config":
        path = write_default_config(args.config)
        print(f"wrote {path}")
        return 0

    if args.command is None:  # bare `sttop` records with defaults
        args = parser.parse_args([*(argv or sys.argv[1:]), "record"])
    return cmd_record(config, args)


if __name__ == "__main__":
    sys.exit(main())
