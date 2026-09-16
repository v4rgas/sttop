"""Keep automatic background network checks out of widget tests."""

import pytest


@pytest.fixture(autouse=True)
def no_background_update_requests(monkeypatch):
    monkeypatch.setattr(
        "sttop.tui.start_update_check", lambda callback: None, raising=False
    )
