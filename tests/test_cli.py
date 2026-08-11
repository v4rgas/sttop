import os

import pytest

from sttop.__main__ import COMMANDS, build_parser, main, with_default_command


def parse(*argv):
    return build_parser().parse_args(with_default_command(list(argv), COMMANDS))


def test_the_command_list_matches_the_parser():
    """`with_default_command` has to know the subcommand names before argparse
    parses anything, so the two lists are guarded against drift here."""
    subcommands = [a for a in build_parser()._actions if a.dest == "command"]
    assert set(subcommands[0].choices) == COMMANDS


def test_config_option_is_accepted_after_the_subcommand(tmp_path):
    args = parse("config", "-c", str(tmp_path / "c.toml"))
    assert args.config == tmp_path / "c.toml"


def test_config_option_before_the_subcommand_survives(tmp_path):
    """The subcommand's own copy must not overwrite a value already given."""
    args = parse("-c", str(tmp_path / "c.toml"), "config")
    assert args.config == tmp_path / "c.toml"


def test_bare_invocation_records():
    assert parse().command == "record"


def test_record_options_do_not_need_the_subcommand():
    """`sttop -t standup` is the documented shorthand for `sttop record -t ...`."""
    args = parse("-t", "standup")
    assert (args.command, args.title) == ("record", "standup")


def test_backend_shorthand():
    args = parse("--backend", "whisper", "-m", "small")
    assert (args.command, args.backend, args.model) == ("record", "whisper", "small")


def test_subcommands_are_left_alone():
    assert parse("devices", "--test").command == "devices"
    assert parse("sessions").command == "sessions"


@pytest.mark.parametrize("flag", ["-c", "--config"])
def test_global_config_option_keeps_its_place(flag, tmp_path):
    args = parse(flag, str(tmp_path / "c.toml"), "-t", "standup")
    assert (args.command, args.title) == ("record", "standup")
    assert args.config == tmp_path / "c.toml"


def test_joined_global_option(tmp_path):
    args = parse(f"--config={tmp_path / 'c.toml'}", "-t", "x")
    assert args.command == "record" and args.config == tmp_path / "c.toml"


def test_empty_argv_is_not_the_process_argv(monkeypatch, tmp_path, capsys):
    """main([]) must mean "no arguments", not "go read sys.argv" - otherwise a
    caller's own flags (pytest's, say) leak into sttop's parse."""
    monkeypatch.setattr("sys.argv", ["pytest", "--nonsense", "sessions"])
    recorded = []
    monkeypatch.setattr("sttop.__main__.cmd_record", lambda *a: recorded.append(a) or 0)

    assert main(["-c", str(tmp_path / "absent.toml")]) == 0
    assert len(recorded) == 1


def test_bad_config_is_reported_not_raised(tmp_path, capsys):
    path = tmp_path / "config.toml"
    path.write_text('stt = "whisper"\n')  # a section given a scalar
    assert main(["-c", str(path), "sessions"]) == 1
    assert "bad config" in capsys.readouterr().err


def test_writing_a_config_does_not_require_a_valid_one(tmp_path, capsys):
    path = tmp_path / "config.toml"
    path.write_text("vad = 3\n")
    assert main(["-c", str(path), "config"]) == 0
    assert "vad = 3" not in path.read_text()


# -- native log capture -----------------------------------------------------


class _RecordArgs:
    """The record subcommand's arguments, all left at their defaults."""

    mic = system = model = backend = language = title = speakers = None
    no_diarize = save_wav = False


def run_record(monkeypatch, tmp_path, app_run):
    import sttop.tui
    from sttop import __main__ as cli
    from sttop.config import Config

    class FakeApp:
        def __init__(self, config, title):
            pass

        run = staticmethod(app_run)

    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sttop.tui, "SttopApp", FakeApp)
    return cli.cmd_record(Config(), _RecordArgs())


def test_native_noise_cannot_paint_over_the_ui(monkeypatch, tmp_path, capsys):
    """onnxruntime warns about its CoreML provider while the model loads,
    which is mid-session, straight to fd 2 - the terminal Textual is drawing
    on. No redraw clears that, so fd 2 goes to a file for the run."""
    transcript = tmp_path / "session.md"

    def app_run():
        os.write(2, b"W:onnxruntime coreml_execution_provider noise\n")
        return transcript

    assert run_record(monkeypatch, tmp_path, app_run) == 0

    captured = capsys.readouterr()
    assert "coreml_execution_provider" not in captured.err
    assert str(transcript) in captured.out
    assert "coreml_execution_provider" in (tmp_path / "session.log").read_text()


def test_a_run_with_nothing_to_show_surfaces_the_log(monkeypatch, tmp_path, capsys):
    """Having hidden fd 2, a native crash would otherwise leave no trace at
    all - and this is the one moment anyone wants to see it."""

    def app_run():
        os.write(2, b"libc++abi: terminating\n")
        return None

    assert run_record(monkeypatch, tmp_path, app_run) == 0
    assert "libc++abi: terminating" in capsys.readouterr().err
