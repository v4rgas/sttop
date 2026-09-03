import subprocess

import pytest

from sttop.crypto import Cipher, decrypt_session
from sttop.sync import SyncError, pull_sessions, seal_sessions, sync_sessions


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
    assert "standup.md.enc" in git(remote, "ls-tree", "-r", "--name-only", "main")

    # A second machine pushed meanwhile: sync rebases and still lands.
    other = tmp_path / "other"
    subprocess.run(
        ["git", "clone", "-q", "-b", "main", str(remote), str(other)], check=True
    )
    (other / "2026-08-31-1100-retro.md.enc").write_text("sttop-enc/1 x\nBBBB\n")
    git(other, "add", "-A")
    git(other, "-c", "user.name=o", "-c", "user.email=o@x", "commit", "-q", "-m", "x")
    git(other, "push", "-q")

    (sessions / "2026-08-31-1200-plan.md.enc").write_text("sttop-enc/1 x\nCCCC\n")
    assert "pushed" in sync_sessions(sessions, str(remote))
    names = git(remote, "ls-tree", "-r", "--name-only", "main")
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


# -- multiple machines --------------------------------------------------------


def bare(tmp_path, name="cloud.git"):
    remote = tmp_path / name
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    return str(remote)


def machine(tmp_path, name, *files):
    directory = tmp_path / name
    directory.mkdir(exist_ok=True)
    for filename, content in files:
        (directory / filename).write_text(content)
    return directory


def test_pull_adopts_an_existing_repo_wholesale(tmp_path):
    """The second-PC story: pulling into a fresh directory brings down the
    sessions *and* the vault, before any local vault could be minted."""
    remote = bare(tmp_path)
    first = machine(
        tmp_path, "pc1",
        (".sttop-vault", "sttop-vault/1 c2FsdA==\n"),
        ("2026-08-31-1000-standup.md.enc", "sttop-enc/1 x\nAAAA\n"),
    )
    sync_sessions(first, remote)

    second = tmp_path / "pc2"
    message = pull_sessions(second, remote)
    assert "adopted 1 session" in message
    assert (second / "2026-08-31-1000-standup.md.enc").is_file()
    assert (second / ".sttop-vault").is_file()
    assert "up to date" in pull_sessions(second, remote)


def test_pull_on_an_empty_remote_is_a_shrug(tmp_path):
    assert "remote is empty" in pull_sessions(tmp_path / "pc2", bare(tmp_path))


def test_machines_taking_turns_converge(tmp_path):
    """A records, B records, A records again - nobody pulls by hand, yet
    every machine's sync leaves the remote holding everything."""
    remote = bare(tmp_path)
    a = machine(tmp_path, "a", ("s1.md.enc", "sttop-enc/1 x\nA1\n"))
    sync_sessions(a, remote)

    b = tmp_path / "b"
    pull_sessions(b, remote)
    (b / "s2.md.enc").write_text("sttop-enc/1 x\nB1\n")
    sync_sessions(b, remote)

    (a / "s3.md.enc").write_text("sttop-enc/1 x\nA2\n")
    sync_sessions(a, remote)  # must integrate B's s2 before pushing s3
    names = git(remote, "ls-tree", "-r", "--name-only", "main")
    assert {"s1.md.enc", "s2.md.enc", "s3.md.enc"} <= set(names.splitlines())
    assert (a / "s2.md.enc").is_file()  # and A now has B's session locally


def test_a_machine_with_local_history_can_join_the_shared_remote(tmp_path):
    """A laptop may record before sync is configured. Its independently
    rooted history must be grafted onto the desktop's without losing either
    machine's sessions."""
    remote = bare(tmp_path)
    desktop = machine(tmp_path, "desktop", ("desktop.md.enc", "desktop\n"))
    sync_sessions(desktop, remote)

    laptop = machine(tmp_path, "laptop", ("laptop.md.enc", "laptop\n"))
    sync_sessions(laptop)  # creates an independent local root commit
    assert "pushed" in sync_sessions(laptop, remote)

    names = set(git(remote, "ls-tree", "-r", "--name-only", "main").splitlines())
    assert {"desktop.md.enc", "laptop.md.enc"} <= names
    assert (laptop / "desktop.md.enc").is_file()


def test_a_conflicting_seal_goes_to_the_machine_with_the_plaintext(tmp_path):
    """Both machines rewrote the same session (a rename on each side, say).
    The one holding the plaintext .md is the authority - it can always
    re-seal - so its version must win on the remote."""
    remote = bare(tmp_path)
    a = machine(tmp_path, "a", ("x.md.enc", "sttop-enc/1 x\nV1\n"))
    sync_sessions(a, remote)
    b = tmp_path / "b"
    pull_sessions(b, remote)

    (b / "x.md.enc").write_text("sttop-enc/1 x\nB-VERSION\n")
    sync_sessions(b, remote)  # B pushes its rewrite first

    (a / "x.md").write_text("# the plaintext, owned by A")
    (a / "x.md.enc").write_text("sttop-enc/1 x\nA-VERSION\n")
    sync_sessions(a, remote)  # conflict: A owns x.md, so A-VERSION wins

    assert "A-VERSION" in git(remote, "show", "main:x.md.enc")
    assert "A-VERSION" in (a / "x.md.enc").read_text()


def test_a_conflicting_seal_without_the_plaintext_defers_to_the_remote(tmp_path):
    remote = bare(tmp_path)
    a = machine(tmp_path, "a", ("x.md.enc", "sttop-enc/1 x\nV1\n"))
    sync_sessions(a, remote)
    b = tmp_path / "b"
    pull_sessions(b, remote)

    (a / "x.md.enc").write_text("sttop-enc/1 x\nREMOTE-TRUTH\n")
    sync_sessions(a, remote)

    (b / "x.md.enc").write_text("sttop-enc/1 x\nLOCAL-GUESS\n")  # no x.md on B
    sync_sessions(b, remote)

    assert "REMOTE-TRUTH" in git(remote, "show", "main:x.md.enc")
    assert "REMOTE-TRUTH" in (b / "x.md.enc").read_text()
