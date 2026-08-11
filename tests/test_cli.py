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
