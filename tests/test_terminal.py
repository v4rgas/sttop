import pytest

from sttop.terminal import (
    DARK,
    LIGHT,
    detect_theme,
    from_colorfgbg,
    luminance,
    parse_osc11,
)


def test_parses_16_bit_components():
    assert parse_osc11(b"\033]11;rgb:0000/0000/0000\033\\") == 0.0
    assert parse_osc11(b"\033]11;rgb:ffff/ffff/ffff\033\\") == pytest.approx(1.0)


def test_parses_narrower_components():
    """Component width varies by terminal; 8-bit replies are common."""
    assert parse_osc11(b"rgb:00/00/00") == 0.0
    assert parse_osc11(b"rgb:ff/ff/ff") == pytest.approx(1.0)


def test_bel_terminated_reply():
    assert parse_osc11(b"\033]11;rgb:ffff/ffff/ffff\a") == pytest.approx(1.0)


def test_non_replies_are_rejected():
    assert parse_osc11(b"") is None
    assert parse_osc11(b"\033]11;?\033\\") is None  # our own query echoed back


def test_luminance_weights_green_highest():
    assert luminance(0, 1, 0) > luminance(1, 0, 0) > luminance(0, 0, 1)


def test_colorfgbg():
    assert from_colorfgbg("15;0") == DARK       # white on black
    assert from_colorfgbg("0;15") == LIGHT      # black on white
    assert from_colorfgbg("15;default;0") == DARK  # rxvt's three-part form
    assert from_colorfgbg("") is None
    assert from_colorfgbg(None) is None
    assert from_colorfgbg("nonsense") is None


def test_explicit_preference_wins_without_probing(monkeypatch):
    monkeypatch.setenv("COLORFGBG", "0;15")
    assert detect_theme("gruvbox") == "gruvbox"


def test_auto_prefers_colorfgbg(monkeypatch):
    monkeypatch.setenv("COLORFGBG", "0;15")
    assert detect_theme("auto") == LIGHT


def test_auto_falls_back_to_dark(monkeypatch):
    monkeypatch.delenv("COLORFGBG", raising=False)
    monkeypatch.setattr("sttop.terminal.query_osc11", lambda *a, **k: None)
    assert detect_theme("auto") == DARK


def test_colorfgbg_answer_skips_the_tty_query(monkeypatch):
    """The OSC 11 query costs a timeout and takes over the terminal, so it must
    not run when the environment has already answered."""
    monkeypatch.setenv("COLORFGBG", "0;15")
    asked = []
    monkeypatch.setattr("sttop.terminal.query_osc11", lambda *a, **k: asked.append(1))
    assert detect_theme("auto") == LIGHT
    assert asked == []


def test_theme_sources_reports_every_source(monkeypatch):
    from sttop.terminal import theme_sources

    monkeypatch.setenv("COLORFGBG", "0;15")
    monkeypatch.setattr("sttop.terminal.query_osc11", lambda *a, **k: DARK)
    assert list(theme_sources()) == [("COLORFGBG", LIGHT), ("OSC 11", DARK)]
