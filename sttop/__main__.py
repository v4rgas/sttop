"""Command line entry point."""

from __future__ import annotations

import argparse
import getpass
import os
import secrets
import subprocess
import sys
from pathlib import Path

from . import __version__
from .config import CONFIG_PATH, DATA_DIR, Config, ConfigError, write_default_config
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
    record.add_argument(
        "--speakers",
        type=int,
        metavar="N",
        help="how many voices to expect besides your own; caps the clustering",
    )
    record.add_argument("--save-wav", action="store_true", help="keep the raw audio")
    record.add_argument(
        "--pipe",
        metavar="CMD",
        help="stream each utterance as a JSON line to this command's stdin",
    )

    devices_cmd = command("devices", "list audio sources")
    devices_cmd.add_argument(
        "--test", action="store_true", help="record 1s from each and report levels"
    )

    command("doctor", "check audio deps and explain anything missing")
    sub.add_parser(
        "sessions", aliases=["ls"], help="list recorded sessions", parents=[common]
    )

    read = command("read", "open a transcript, decrypting it if needed")
    read.add_argument(
        "session",
        nargs="?",
        help="session filename or a substring of one; default: the latest",
    )

    command(
        "sync",
        "push sessions to a private repo, encrypted; first run sets everything up",
    )
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


#: Printed by `sttop doctor` when macOS system audio is unavailable. Nothing
#: to install - it is a permission, or a macOS too old for ScreenCaptureKit.
_MACOS_SYSTEM_AUDIO_HELP = """
System audio uses ScreenCaptureKit, which needs no driver and does not change
your output device - but it does need permission, granted to the terminal
sttop runs in:

  System Settings > Privacy & Security > Screen & System Audio Recording

Enable your terminal there, then restart it - macOS only re-reads that
permission when the app launches. macOS 13 (Ventura) or newer is required.

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

    # Asked once, and used for both the row and the advice below it.
    permission = devices.mac_permission() if devices.MACOS else None
    for check, verdict in devices.diagnose(permission=permission):
        print(f"{check:<18}{verdict}")

    if _system_audio_problem(devices, config, permission):
        print(_MACOS_SYSTEM_AUDIO_HELP if devices.MACOS else _LINUX_SYSTEM_AUDIO_HELP)
    return 0


def _system_audio_problem(devices, config: Config, permission: str | None) -> bool:
    """Whether to print the how-to-fix-it text below the checks.

    Resolving is not enough on macOS: the source resolves fine when screen
    recording is denied, and the session then starts mic-only with the one
    piece of advice that would have helped left unprinted.
    """
    try:
        devices.resolve(config.audio.system_source, monitor=True)
    except devices.AudioError:
        return True
    # Linux has no permission to check, and must not be shown the macOS help
    # however the caller filled this in.
    return bool(devices.MACOS and permission)


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


# -- vault -------------------------------------------------------------------


def _passphrase(prompt: str = "vault passphrase: ") -> str:
    """From STTOP_PASSPHRASE when set (scripts, the skill), the tty otherwise."""
    return os.environ.get("STTOP_PASSPHRASE") or getpass.getpass(prompt)


def _unlock(config: Config, generate: bool = False):
    """The sessions vault's cipher, creating the vault on first use.

    With `generate` (the sync-mode setup), an empty answer means "make one for
    me": a random passphrase, kept in the sessions' .env - the only place it
    exists, and where another machine copies it from.

    Never runs while Textual holds the terminal: getpass needs the tty to
    itself, and a wrong passphrase should cost one line, not a TUI teardown.
    """
    from . import crypto

    directory = Path(config.sessions_dir)
    if crypto.vault_exists(directory):
        return crypto.open_vault(directory, _passphrase())

    passphrase = os.environ.get("STTOP_PASSPHRASE")
    if not passphrase:
        print("\npick a vault passphrase - reading sessions elsewhere needs it,")
        print("and a lost passphrase is unrecoverable")
        hint = " (empty = generate one)" if generate else ""
        passphrase = getpass.getpass(f"vault passphrase{hint}: ")
        if passphrase:
            if getpass.getpass("repeat: ") != passphrase:
                raise crypto.VaultError("passphrases do not match")
        elif generate:
            passphrase = secrets.token_urlsafe(24)
            cipher = crypto.open_vault(directory, passphrase)
            saved = crypto.save_passphrase(directory, passphrase)
            print(f"  generated one → {_tilde(saved)}")
            return cipher
        else:
            raise crypto.VaultError("an empty passphrase protects nothing")
    return crypto.open_vault(directory, passphrase)


def _tilde(path: Path) -> str:
    """~-shortened for display: full paths make one-line messages unreadable."""
    text = str(path)
    home = str(Path.home())
    return "~" + text[len(home):] if text.startswith(home + "/") else text


def _vault_cipher(config: Config):
    """Sync mode's cipher: remembered in the sessions' .env, so the passphrase
    is a one-time setup cost, not a per-sync toll. The .env can never enter
    the repo - the whitelist .gitignore admits only ciphertext."""
    from . import crypto

    directory = Path(config.sessions_dir)
    cipher = crypto.load_key(directory)
    if cipher is None:
        cipher = _unlock(config, generate=True)
        crypto.save_key(directory, cipher)
    return cipher


def cmd_read(config: Config, args) -> int:
    from . import crypto
    from .journal import list_sessions

    directory = Path(config.sessions_dir)
    if args.session and Path(args.session).is_file():
        path = Path(args.session)
    else:
        paths = list_sessions(directory, limit=1000)
        if args.session:
            paths = [p for p in paths if args.session in p.name]
        if not paths:
            what = f"no session matching {args.session!r}" if args.session else (
                "no sessions yet"
            )
            print(f"{what} in {directory}", file=sys.stderr)
            return 1
        path = paths[0]  # newest first, so a bare `sttop read` opens the latest

    if crypto.is_encrypted(path):
        # The remembered key first (sync mode: no prompt, ever); the
        # passphrase only when there is no key or the file outgrew it.
        cipher = crypto.load_key(directory)
        if cipher is not None:
            try:
                text = crypto.decrypt_session(path, cipher)
            except crypto.VaultError:
                cipher = None
        if cipher is None:
            try:
                text = crypto.decrypt_session(path, _passphrase())
            except crypto.VaultError as exc:
                print(f"error: {path.name}: {exc}", file=sys.stderr)
                return 1
    else:
        text = path.read_text(encoding="utf-8")

    _display(text)
    return 0


def _display(text: str) -> None:
    """Through the pager on a tty, plain on a pipe - so `sttop read` both
    *opens* a meeting interactively and feeds `grep` or an LLM cleanly."""
    if sys.stdout.isatty():
        pager = os.environ.get("PAGER", "less")
        try:
            subprocess.run([pager], input=text, text=True)
            return
        except OSError:
            pass  # no pager on this machine; printing still works
    print(text, end="")


def _sync(config: Config) -> str:
    from .sync import pull_sessions, sync_sessions

    directory = Path(config.sessions_dir)
    remote = config.storage.git_remote
    if remote:
        # Pull before the vault is even looked at: a second PC joining an
        # existing repo must adopt the remote's vault, not mint its own.
        pulled = pull_sessions(directory, remote)
        if "session" in pulled:
            print(pulled)
    # "always" already has ciphertext on disk; "sync" seals on the way out.
    cipher = _vault_cipher(config) if config.storage.encrypt == "sync" else None
    return sync_sessions(directory, remote, cipher=cipher)


def cmd_sync(config: Config, config_path: Path | None) -> int:
    """One flow: the first run *is* the setup.

    Unconfigured and on a tty, it asks for the repo and the passphrase, saves
    both, and pushes - after which this run and every later one are the same
    command doing the same thing with nothing to answer.
    """
    from .crypto import VaultError
    from .sync import SyncError

    first_run = not (config.storage.git_sync or config.storage.git_remote)
    if first_run:
        if not sys.stdin.isatty():
            print(
                "git sync is not set up - run `sttop sync` in a terminal once,\n"
                f"or set storage.git_remote in {config_path or CONFIG_PATH}",
                file=sys.stderr,
            )
            return 1
        from .sync import create_github_repo, gh_ready

        print("encrypted cloud sync setup - the repo only ever sees ciphertext\n")
        offer_gh = gh_ready()
        hint = "create one with gh" if offer_gh else "keep history local-only"
        remote = input(f"git remote to push to (empty = {hint}): ").strip()
        if not remote and offer_gh:
            name = input("new private repo name (empty = local-only): ").strip()
            if name:
                try:
                    remote = create_github_repo(name)
                    print(f"  created {remote}")
                except SyncError as exc:
                    print(f"warn: {exc} - keeping history local-only", file=sys.stderr)
        config.storage.git_remote = remote
        config.storage.git_sync = True

    try:
        print(_sync(config))  # first time, this also asks for the passphrase
    except VaultError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except SyncError as exc:
        message = str(exc)
        if "could not read Username" in message:
            message += " - run `gh auth setup-git`, or use an ssh remote"
        print(f"error: git sync failed: {message}", file=sys.stderr)
        return 1

    if first_run:
        # Persisted only after the sync worked: a config that says sync is on
        # should mean a sync has actually succeeded once.
        path = config_path or CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(config.to_toml())
        print(f"config saved ({_tilde(path)}) - sessions now sync after recording")
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
    if args.speakers is not None:
        config.diarize.max_speakers = max(0, args.speakers)
    if args.save_wav:
        config.audio.save_wav = True
    if args.pipe:
        config.pipe = args.pipe

    cipher = None
    if config.storage.encrypt == "always":
        from .crypto import VaultError

        try:
            cipher = _unlock(config)
        except VaultError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    from .nativelog import quiet_onnxruntime, stderr_to, tail

    quiet_onnxruntime()
    log_path = DATA_DIR / "session.log"
    # fd 2 belongs to the log for the length of the run: a native library
    # writing to it mid-session paints over the UI, which is unreadable and
    # cannot be redrawn away.
    with stderr_to(log_path):
        path = SttopApp(config, args.title, cipher=cipher).run()

    if path:
        print(f"transcript: {path}")
        _sync_after_record(config)
    else:
        # Nothing to show for the run: whatever went wrong is in the log, and
        # this is the only time anyone would want to see it.
        captured = tail(log_path)
        if captured:
            print(f"{captured}\n\n(full log: {log_path})", file=sys.stderr)
    return 0


def _sync_after_record(config: Config) -> None:
    """Best-effort: an unreachable remote must not eat the transcript line
    above, so failure is a warning, and `sttop sync` retries it later."""
    if not (config.storage.git_sync or config.storage.git_remote):
        return
    from .crypto import VaultError
    from .sync import SyncError

    try:
        print(_sync(config))
    except (SyncError, VaultError) as exc:
        print(f"warn: git sync failed ({exc}) - run `sttop sync` to retry",
              file=sys.stderr)


COMMANDS = {
    "record", "devices", "doctor", "sessions", "read", "sync", "theme", "config",
    "ls",
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
    if config.storage.encrypt not in ("sync", "always"):
        print(
            'error: bad config: storage.encrypt must be "sync" or "always", '
            f"got {config.storage.encrypt!r}",
            file=sys.stderr,
        )
        return 1

    if args.command == "devices":
        return cmd_devices(config, args)
    if args.command == "doctor":
        return cmd_doctor(config)
    if args.command in ("sessions", "ls"):
        return cmd_sessions(config)
    if args.command == "read":
        return cmd_read(config, args)
    if args.command == "sync":
        return cmd_sync(config, args.config)
    if args.command == "theme":
        return cmd_theme(config)
    return cmd_record(config, args)


if __name__ == "__main__":
    sys.exit(main())
