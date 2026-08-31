"""Git sync for the sessions directory.

The repo is sttop's, not the user's: created on first sync, committed after
every session, pushed when `storage.git_remote` names somewhere to push. The
point is off-machine history that stays unreadable - so the .gitignore is a
whitelist, and only ciphertext ever enters it. A plaintext transcript, a WAV,
a stray log: all unsyncable by default, instead of leaking the first time
someone forgets to add an ignore rule.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

from .crypto import ENCRYPTED_SUFFIX, VAULT_NAME, Cipher, VaultError, decrypt_session


class SyncError(RuntimeError):
    """git failed; the message carries git's most interesting line."""


GITIGNORE = f"""# managed by sttop - only encrypted sessions belong in this repo
*
!.gitignore
!{VAULT_NAME}
!*{ENCRYPTED_SUFFIX}
"""


def _git(directory: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(directory), *args], capture_output=True, text=True
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise SyncError(detail[-1] if detail else f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _identity(directory: Path) -> list[str]:
    """A fallback author, only when the machine has none configured - the
    commit must not fail somewhere `git config user.email` was never run."""
    probe = subprocess.run(
        ["git", "-C", str(directory), "config", "user.email"],
        capture_output=True,
        text=True,
    )
    if probe.stdout.strip():
        return []
    return ["-c", "user.name=sttop", "-c", "user.email=sttop@localhost"]


def seal_sessions(directory: Path, cipher: Cipher) -> int:
    """Give every plaintext session an up-to-date sealed twin.

    This is "sync" mode's whole trick: the .md files stay where they are, and
    only these .md.enc copies are commit-eligible. A twin that already holds
    the current text is left untouched, so an unchanged session costs neither
    a rewrite nor a git delta.
    """
    sealed = 0
    for source in directory.glob("*.md"):
        twin = source.with_name(source.name + ".enc")
        text = source.read_text(encoding="utf-8")
        if twin.is_file():
            try:
                if decrypt_session(twin, cipher) == text:
                    continue
            except VaultError:
                pass  # foreign or corrupted twin: replace it with a good one
        temporary = twin.with_name(twin.name + ".tmp")
        temporary.write_text(cipher.header + cipher.seal(text), encoding="utf-8")
        temporary.replace(twin)
        sealed += 1
    return sealed


def sync_sessions(directory: Path, remote: str = "", cipher: Cipher | None = None) -> str:
    """Commit whatever is new; push when a remote is given. Returns a one-line
    summary for the terminal. Raises SyncError with git's own words otherwise.

    With a cipher, plaintext sessions are sealed first ("sync" mode); without
    one, whatever encrypted files already exist are synced ("always" mode).
    """
    directory.mkdir(parents=True, exist_ok=True)
    if cipher is not None:
        seal_sessions(directory, cipher)
    if not (directory / ".git").is_dir():
        _git(directory, "init", "-q")

    # Rewritten, not appended: the whitelist *is* the security boundary, and
    # an old copy that has drifted from GITIGNORE would silently widen it.
    gitignore = directory / ".gitignore"
    if not gitignore.is_file() or gitignore.read_text() != GITIGNORE:
        gitignore.write_text(GITIGNORE)

    _git(directory, "add", "-A")
    committed = False
    if _git(directory, "diff", "--cached", "--name-only"):
        message = f"sttop: sessions as of {datetime.now():%Y-%m-%d %H:%M}"
        _git(directory, *_identity(directory), "commit", "-q", "-m", message)
        committed = True

    try:  # a repo with no commits yet has nothing to push and no branch name
        _git(directory, "rev-parse", "--verify", "-q", "HEAD")
    except SyncError:
        return "git: nothing to commit yet (no encrypted sessions)"

    if not remote:
        return (
            "git: committed (local history only - set storage.git_remote to push)"
            if committed
            else "git: nothing new to commit"
        )

    _set_origin(directory, remote)
    branch = _git(directory, "rev-parse", "--abbrev-ref", "HEAD")
    try:
        _git(directory, "push", "-q", "-u", "origin", branch)
    except SyncError:
        # The usual cause is another machine having pushed first. Take its
        # history - every file is append-only ciphertext, so a rebase is
        # conflict-free unless the same session was edited twice - and retry.
        _git(directory, "pull", "-q", "--rebase", "origin", branch)
        _git(directory, "push", "-q", "-u", "origin", branch)
    return f"git: pushed to {remote}"


def _set_origin(directory: Path, remote: str) -> None:
    try:
        current = _git(directory, "remote", "get-url", "origin")
    except SyncError:
        _git(directory, "remote", "add", "origin", remote)
        return
    if current != remote:  # the config changed; the config wins
        _git(directory, "remote", "set-url", "origin", remote)
