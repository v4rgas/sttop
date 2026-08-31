import subprocess

import pytest

from sttop.crypto import Cipher, decrypt_session
from sttop.sync import SyncError, seal_sessions, sync_sessions


def git(directory, *args) -> str:
    return subprocess.run(
        ["git", "-C", str(directory), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture()
def sessions(tmp_path):
    directory = tmp_path / "sessions"
    directory.mkdir()
    (directory / ".sttop-vault").write_text("sttop-vault/1 c2FsdA==\n")
    (directory / "2026-08-31-1000-standup.md.enc").write_text("sttop-enc/1 x\nAAAA\n")
    return directory


def test_first_sync_creates_the_repo_and_commits(sessions):
    message = sync_sessions(sessions)
    assert "committed" in message
    assert (sessions / ".git").is_dir()
    tracked = git(sessions, "ls-files").splitlines()
    assert "2026-08-31-1000-standup.md.enc" in tracked
    assert ".sttop-vault" in tracked


def test_plaintext_never_enters_the_repo(sessions):
    """The whole point of the repo: it can go to a remote, so nothing human-
    readable may ever be tracked - transcripts, audio, logs, and above all
    the .env holding the vault key."""
    (sessions / "2026-08-31-0900-secret.md").write_text("# the plaintext one")
    (sessions / "2026-08-31-0900-secret-mic.wav").write_bytes(b"RIFF")
    (sessions / ".env").write_text("STTOP_KEY=c2FsdA==.a2V5\n")
    sync_sessions(sessions)
    tracked = set(git(sessions, "ls-files").splitlines())
    assert tracked == {".gitignore", ".sttop-vault", "2026-08-31-1000-standup.md.enc"}


def test_nothing_new_is_not_a_new_commit(sessions):
    sync_sessions(sessions)
    assert "nothing new" in sync_sessions(sessions)
    assert git(sessions, "rev-list", "--count", "HEAD") == "1"


def test_a_drifted_gitignore_is_repaired(sessions):
    """The whitelist is the security boundary; a widened copy must not survive
    the next sync."""
    sync_sessions(sessions)
    (sessions / ".gitignore").write_text("!*.md\n")
    sync_sessions(sessions)
    assert "!*.md\n" not in (sessions / ".gitignore").read_text()


def test_push_to_a_remote(sessions, tmp_path):
    remote = tmp_path / "cloud.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)

    message = sync_sessions(sessions, str(remote))
    assert "pushed" in message
    assert "standup.md.enc" in git(remote, "ls-tree", "-r", "--name-only", "HEAD")

    # A second machine pushed meanwhile: sync rebases and still lands.
    other = tmp_path / "other"
    subprocess.run(["git", "clone", "-q", str(remote), str(other)], check=True)
    (other / "2026-08-31-1100-retro.md.enc").write_text("sttop-enc/1 x\nBBBB\n")
    git(other, "add", "-A")
    git(other, "-c", "user.name=o", "-c", "user.email=o@x", "commit", "-q", "-m", "x")
    git(other, "push", "-q")

    (sessions / "2026-08-31-1200-plan.md.enc").write_text("sttop-enc/1 x\nCCCC\n")
    assert "pushed" in sync_sessions(sessions, str(remote))
    names = git(remote, "ls-tree", "-r", "--name-only", "HEAD")
    assert "retro.md.enc" in names and "plan.md.enc" in names


def test_an_unreachable_remote_raises_with_gits_words(sessions, tmp_path):
    with pytest.raises(SyncError):
        sync_sessions(sessions, str(tmp_path / "does-not-exist.git"))


# -- sync mode: seal on the way out ------------------------------------------


CIPHER = Cipher(b"k" * 32, b"s" * 16)


def test_seal_gives_every_plaintext_session_a_twin(tmp_path):
    (tmp_path / "2026-08-31-1000-standup.md").write_text("# standup\nhello\n")
    assert seal_sessions(tmp_path, CIPHER) == 1
    twin = tmp_path / "2026-08-31-1000-standup.md.enc"
    assert decrypt_session(twin, CIPHER) == "# standup\nhello\n"
    assert b"hello" not in twin.read_bytes()


def test_seal_skips_an_unchanged_session(tmp_path):
    source = tmp_path / "a.md"
    source.write_text("# a\n")
    seal_sessions(tmp_path, CIPHER)
    before = (tmp_path / "a.md.enc").read_bytes()
    assert seal_sessions(tmp_path, CIPHER) == 0  # no rewrite, no git delta
    assert (tmp_path / "a.md.enc").read_bytes() == before


def test_seal_refreshes_a_stale_twin(tmp_path):
    """A rename or a crash-recovered session changes the .md; the twin in the
    repo must follow, or the cloud keeps the version before the fix."""
    source = tmp_path / "a.md"
    source.write_text("# a\n**spk1** said it\n")
    seal_sessions(tmp_path, CIPHER)
    source.write_text("# a\n**Ana** said it\n")
    assert seal_sessions(tmp_path, CIPHER) == 1
    assert "Ana" in decrypt_session(tmp_path / "a.md.enc", CIPHER)


def test_sync_with_a_cipher_ships_the_twin_not_the_plaintext(tmp_path):
    directory = tmp_path / "sessions"
    directory.mkdir()
    (directory / "2026-08-31-1000-standup.md").write_text("# standup\n")
    sync_sessions(directory, cipher=CIPHER)
    tracked = git(directory, "ls-files").splitlines()
    assert "2026-08-31-1000-standup.md.enc" in tracked
    assert "2026-08-31-1000-standup.md" not in tracked


def test_a_github_https_remote_is_wired_to_gh_credentials(sessions, monkeypatch):
    """gh hands out https remotes; plain git would interrogate the user for a
    username and a dead password. The repo-local helper hands the job to gh."""
    import shutil as _shutil

    from sttop.sync import _ensure_gh_credentials

    sync_sessions(sessions)  # materialise the repo
    monkeypatch.setattr(_shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    _ensure_gh_credentials(sessions, "https://github.com/you/meetings.git")
    helper = git(sessions, "config", "--local", "credential.helper")
    assert helper == "!gh auth git-credential"

    _ensure_gh_credentials(sessions, "https://github.com/you/meetings.git")
    probe = subprocess.run(
        ["git", "-C", str(sessions), "config", "--get-all", "credential.helper"],
        capture_output=True, text=True,
    )
    assert len(probe.stdout.splitlines()) == 1  # idempotent, not accumulating


def test_an_ssh_remote_leaves_credentials_alone(sessions):
    from sttop.sync import _ensure_gh_credentials

    sync_sessions(sessions)
    _ensure_gh_credentials(sessions, "git@github.com:you/meetings.git")
    probe = subprocess.run(
        ["git", "-C", str(sessions), "config", "--local", "credential.helper"],
        capture_output=True, text=True,
    )
    assert probe.returncode != 0  # nothing set


def test_an_empty_directory_bootstraps_only_the_whitelist(tmp_path):
    directory = tmp_path / "sessions"
    sync_sessions(directory)
    assert git(directory, "ls-files").splitlines() == [".gitignore"]
