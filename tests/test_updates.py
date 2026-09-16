"""Release versions, update decisions and failure isolation (no network)."""

import json
import threading

import pytest

from sttop import updates

CURRENT = "0.9.0"
LATEST = "0.10.0"


@pytest.fixture
def check(monkeypatch, tmp_path):
    monkeypatch.setattr(updates, "installed_version", lambda: CURRENT)
    cache = tmp_path / "cache.json"
    return lambda: updates.check_for_update(cache_path=cache)


def release(version):
    return {"releases": {version: [{"yanked": False}]}}


@pytest.mark.parametrize("latest,expected", [
    ("0.10.0", True), ("0.9.0", False), ("0.8.0", False),
    ("0.10.0rc1", False), ("0.10.0.dev1", False),
])
def test_version_comparison(check, monkeypatch, latest, expected):
    monkeypatch.setattr(updates, "_request", lambda: release(latest))
    notice = check()
    assert bool(notice) is expected
    if notice:
        assert "uvx sttop@latest" in notice.message
        assert "uv tool upgrade sttop" in notice.message
        assert "git" not in notice.message


def test_yanked_empty_and_invalid_releases_are_ignored(check, monkeypatch):
    data = release(LATEST)
    data["releases"].update({
        "99.0": [{"yanked": True}], "98.0": [], "invalid": [{"yanked": False}],
        "100.0rc1": [{"yanked": False}],
    })
    monkeypatch.setattr(updates, "_request", lambda: data)
    assert check() == updates.UpdateNotice(CURRENT, LATEST)


def test_matching_version_and_cache_avoid_network(check, monkeypatch):
    calls = []
    monkeypatch.setattr(
        updates, "_request", lambda: calls.append(True) or release(CURRENT)
    )
    assert check() is None
    assert check() is None
    assert calls == [True]


def test_cached_update_is_reused(check, monkeypatch):
    monkeypatch.setattr(
        updates, "_request", lambda: release(LATEST)
    )
    assert check() == updates.UpdateNotice(CURRENT, LATEST)
    monkeypatch.setattr(
        updates, "_request", lambda: pytest.fail("unexpected network")
    )
    assert check() == updates.UpdateNotice(CURRENT, LATEST)


@pytest.mark.parametrize(
    "data",
    [
        "broken json",
        "[]",
        '{"current": "a"}',
        '{"current": "' + CURRENT + '", "checked_at": "bad"}',
    ],
)
def test_bad_cache_does_not_prevent_checks(check, monkeypatch, tmp_path, data):
    (tmp_path / "cache.json").write_text(data)
    monkeypatch.setattr(updates, "_request", lambda: release(CURRENT))
    assert check() is None
    assert json.loads((tmp_path / "cache.json").read_text())["latest"] == CURRENT


def test_new_build_invalidates_cache(check, monkeypatch):
    calls = []
    monkeypatch.setattr(
        updates, "_request", lambda: calls.append(True) or release(CURRENT)
    )
    check()
    monkeypatch.setattr(updates, "installed_version", lambda: LATEST)
    check()
    assert len(calls) == 2


def test_expired_cache_is_refreshed(check, monkeypatch):
    calls = []
    monkeypatch.setattr(updates.time, "time", lambda: 100)
    monkeypatch.setattr(
        updates, "_request", lambda: calls.append(True) or release(CURRENT)
    )
    check()
    monkeypatch.setattr(updates.time, "time", lambda: 100 + updates.CACHE_TTL)
    check()
    assert len(calls) == 2


def test_offline_is_silent(check, monkeypatch):
    def offline():
        raise OSError("offline")

    monkeypatch.setattr(updates, "_request", offline)
    assert check() is None


def test_unknown_identity_skips_network(check, monkeypatch):
    monkeypatch.setattr(updates, "installed_version", lambda: None)
    monkeypatch.setattr(
        updates, "_request", lambda: pytest.fail("unexpected network")
    )
    assert check() is None


def test_checker_returns_while_network_is_blocked(monkeypatch):
    started, release, delivered = threading.Event(), threading.Event(), threading.Event()

    def slow_check():
        started.set()
        release.wait(2)
        return updates.UpdateNotice(CURRENT, LATEST)

    monkeypatch.setattr(updates, "check_for_update", slow_check)
    thread = updates.start_update_check(lambda notice: delivered.set())
    try:
        assert started.wait(1)
        assert thread.daemon and thread.is_alive()
        assert not delivered.is_set()
    finally:
        release.set()
        thread.join(2)
    assert delivered.is_set()


def test_version_comes_from_installed_metadata(monkeypatch):
    monkeypatch.setattr(updates.metadata, "version", lambda name: "1.2.3")
    assert updates.installed_version() == "1.2.3"


def test_source_checkout_version_fallback(monkeypatch):
    def missing(name):
        raise updates.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(updates.metadata, "version", missing)
    assert updates.installed_version() == updates.__version__
