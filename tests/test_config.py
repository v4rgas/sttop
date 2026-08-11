import tomllib

from sttop.config import Config


def test_defaults_are_local_first():
    config = Config()
    assert config.stt.backend == "parakeet"
    assert config.diarize.enabled is True


def test_toml_overlay_is_partial(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        '[stt]\nmodel = "medium"\n\n[vad]\nsilence_ms = 300\n\nunknown_key = 1\n'
    )
    config = Config.load(path)
    assert config.stt.model == "medium"
    assert config.vad.silence_ms == 300
    # Untouched fields keep their defaults.
    assert config.stt.backend == "parakeet"
    assert config.vad.aggressiveness == 2


def test_missing_file_yields_defaults(tmp_path):
    assert Config.load(tmp_path / "absent.toml").stt.backend == Config().stt.backend


def test_dumped_config_round_trips():
    original = Config()
    original.stt.model = "medium"
    original.diarize.enabled = False
    original.audio.save_wav = True

    parsed = tomllib.loads(original.to_toml())
    assert parsed["stt"]["model"] == "medium"
    assert parsed["diarize"]["enabled"] is False
    assert parsed["audio"]["save_wav"] is True
    assert parsed["vad"]["max_segment_s"] == 15.0
