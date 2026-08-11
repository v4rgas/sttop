import tomllib

import pytest

from sttop.config import Config, ConfigError


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


def test_a_section_given_a_scalar_is_rejected(tmp_path):
    """Left to itself this leaves a string where a section belongs, and only
    fails much later as an AttributeError deep in the backend loader."""
    path = tmp_path / "config.toml"
    path.write_text('stt = "whisper"\n')
    with pytest.raises(ConfigError, match="stt"):
        Config.load(path)


def test_a_field_given_the_wrong_type_is_rejected(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[vad]\nsilence_ms = "fast"\n')
    with pytest.raises(ConfigError, match="vad.silence_ms"):
        Config.load(path)


def test_whole_numbers_are_accepted_for_float_fields(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[vad]\nmax_segment_s = 20\n")
    assert Config.load(path).vad.max_segment_s == 20.0


def test_generated_config_documents_and_shows_unset_fields():
    """`sttop config` is the discovery surface: a knob missing from the file is
    a knob nobody finds, and the docs must come from the dataclass itself."""
    text = Config().to_toml()
    assert "# mic_source =" in text  # unset, but visible
    assert "# system_source =" in text
    assert "# Keep the raw 16kHz mono capture alongside the transcript." in text


def test_generated_config_reloads_as_the_defaults(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(Config().to_toml())
    assert Config.load(path) == Config()


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
