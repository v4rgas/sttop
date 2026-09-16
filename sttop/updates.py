"""Best-effort PyPI release checks. All I/O runs off the recording/UI threads."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from urllib.request import Request, urlopen

from packaging.version import InvalidVersion, Version
from platformdirs import user_cache_path

from . import __version__

API_URL = "https://pypi.org/pypi/sttop/json"
CACHE_TTL = 6 * 60 * 60
TIMEOUT = 3


def installed_version() -> str:
    try:
        return metadata.version("sttop")
    except metadata.PackageNotFoundError:
        return __version__


def _version(value: object) -> Version | None:
    if not isinstance(value, str):
        return None
    try:
        return Version(value)
    except InvalidVersion:
        return None


def _latest_release(data: dict) -> Version | None:
    releases = data.get("releases", {})
    if not isinstance(releases, dict):
        return None
    versions = []
    for name, files in releases.items():
        version = _version(name)
        if version is None or version.is_prerelease or version.is_devrelease:
            continue
        if isinstance(files, list) and any(
            isinstance(file, dict) and file.get("yanked") is False for file in files
        ):
            versions.append(version)
    return max(versions, default=None)


@dataclass(frozen=True)
class UpdateNotice:
    current: str
    latest: str

    @property
    def message(self) -> str:
        return (
            f"sttop update available: {self.current} → {self.latest}. "
            "Run: uvx sttop@latest\n"
            "Installed with uv tool? Run: uv tool upgrade sttop"
        )


def _request() -> dict:
    request = Request(
        API_URL,
        headers={"Accept": "application/json", "User-Agent": "sttop-update-check"},
    )
    with urlopen(request, timeout=TIMEOUT) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError("unexpected PyPI response")
    return value



def check_for_update(*, cache_path: Path | None = None) -> UpdateNotice | None:
    """Compare installed and published versions, silently tolerating offline use."""
    current = _version(installed_version())
    if current is None:
        return None
    if cache_path is None:
        cache_path = user_cache_path("sttop") / "pypi-update-check.json"
    now = time.time()
    try:
        cached = json.loads(cache_path.read_text())
        if (
            cached.get("current") == str(current)
            and 0 <= now - cached["checked_at"] < CACHE_TTL
        ):
            latest = _version(cached.get("latest"))
            if latest is not None and not latest.is_prerelease:
                return (
                    UpdateNotice(str(current), str(latest)) if latest > current else None
                )
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    try:
        latest = _latest_release(_request())
        if latest is None:
            return None
    except (OSError, ValueError, TypeError):
        return None
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "current": str(current), "latest": str(latest), "checked_at": now,
        }))
    except OSError:
        pass
    return UpdateNotice(str(current), str(latest)) if latest > current else None


def start_update_check(callback: Callable[[UpdateNotice], None]) -> threading.Thread:
    """Fire and forget; a slow network must not hold recording or exit open."""

    def check() -> None:
        notice = check_for_update()
        if notice:
            callback(notice)

    thread = threading.Thread(target=check, name="sttop-update-check", daemon=True)
    thread.start()
    return thread
