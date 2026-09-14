"""Git sync for the sessions directory.

The repo is sttop's, not the user's: created on first sync, committed after
every session, pushed when `storage.git_remote` names somewhere to push. The
point is off-machine history that stays unreadable - so the .gitignore is a
whitelist, and only ciphertext ever enters it. A plaintext transcript, a WAV,
a stray log: all unsyncable by default, instead of leaking the first time
someone forgets to add an ignore rule.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from .crypto import ENCRYPTED_SUFFIX, VAULT_NAME, Cipher, VaultError, decrypt_session


class SyncError(RuntimeError):
    """git failed; the message carries git's most interesting line."""


GITIGNORE = f"""# managed by sttop - only encrypted sessions belong in this repo
*
!*/
!.gitignore
!{VAULT_NAME}
!*{ENCRYPTED_SUFFIX}
"""


def _git(directory: Path, *args: str) -> str:
    # Prompts disabled: an auth problem must become an error line and a
    # retry hint, not a surprise interrogation mid-flow (or a hang when the
    # sync runs right after a recording).
    result = subprocess.run(
        ["git", "-C", str(directory), *args],
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
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
    for source in directory.rglob("*.md"):
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


def _init_repo(directory: Path) -> None:
    if not (directory / ".git").is_dir():
        try:
            _git(directory, "init", "-q", "-b", "main")
        except SyncError:  # git too old for -b; the branch name then hardly matters
            _git(directory, "init", "-q")


def _ensure_whitelist(directory: Path) -> None:
    """Rewritten, not appended: the whitelist *is* the security boundary, and
    an old copy that has drifted from GITIGNORE would silently widen it."""
    gitignore = directory / ".gitignore"
    if not gitignore.is_file() or gitignore.read_text() != GITIGNORE:
        gitignore.write_text(GITIGNORE)


def _has_commits(directory: Path) -> bool:
    probe = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "--verify", "-q", "HEAD"],
        capture_output=True,
    )
    return probe.returncode == 0


def _remote_branch(directory: Path) -> str | None:
    """The remote's branch to share, or None for an empty remote.

    HEAD's symref when it points at a real branch - but a bare repo whose
    HEAD names a branch nobody ever pushed (init said master, sttop pushed
    main) must not read as empty, so the actual heads have the final say.
    """
    head, heads = None, []
    for line in _git(directory, "ls-remote", "--symref", "origin").splitlines():
        if line.startswith("ref: refs/heads/"):
            head = line.split()[1].removeprefix("refs/heads/")
            continue
        _, _, name = line.partition("\t")
        if name.startswith("refs/heads/"):
            heads.append(name.removeprefix("refs/heads/"))
    if head in heads:
        return head
    for preferred in ("main", "master", *heads):
        if preferred in heads:
            return preferred
    return None


def _align_branch(directory: Path, remote_branch: str) -> str:
    """Rename the local branch to the remote's, so two machines whose git
    defaults disagree (main vs master) still share one history."""
    local = _git(directory, "symbolic-ref", "--short", "HEAD")
    if local != remote_branch:
        _git(directory, "branch", "-m", local, remote_branch)
    return remote_branch


def _resolve(directory: Path, name: str) -> None:
    """One conflicted path, decided without a human.

    During a rebase, --theirs is the local commit being replayed and --ours
    the remote history underneath it. A session whose plaintext lives on this
    machine is this machine's to state - it can always re-seal it - so the
    local seal wins; anything else, the remote knows better.
    """
    owned = name.endswith(ENCRYPTED_SUFFIX) and (
        directory / name.removesuffix(".enc")
    ).is_file()
    side = "--theirs" if owned else "--ours"
    with contextlib.suppress(SyncError):
        # An add/delete conflict has only one side to check out; when that
        # fails, whatever is in the worktree stands.
        _git(directory, "checkout", side, "--", name)


def _rebase_onto(directory: Path, branch: str) -> None:
    """Rebase local commits onto origin/<branch>, resolving every conflict by
    policy. Aborts cleanly rather than leaving a half-done rebase behind."""
    remote = f"origin/{branch}"
    common = subprocess.run(
        ["git", "-C", str(directory), "merge-base", "HEAD", remote],
        capture_output=True,
    )
    # A machine may have recorded and committed locally before it was pointed
    # at the shared remote. There is then no merge base, but the histories are
    # still perfectly reconcilable: replay the complete local history on top
    # of the remote instead of rejecting it as "unrelated".
    arguments = ["rebase", remote] if common.returncode == 0 else (
        ["rebase", "--onto", remote, "--root"]
    )
    for _ in range(100):  # one iteration per conflicted commit, bounded
        try:
            _git(directory, *_identity(directory), "-c", "core.editor=true", *arguments)
            return
        except SyncError as failure:
            conflicted = _git(
                directory, "diff", "--name-only", "--diff-filter=U"
            ).splitlines()
            if not conflicted:  # not a conflict stop: bail out with a clean tree
                subprocess.run(
                    ["git", "-C", str(directory), "rebase", "--abort"],
                    capture_output=True,
                )
                raise SyncError(f"could not reconcile histories: {failure}") from None
            for name in conflicted:
                _resolve(directory, name)
            _git(directory, "add", "-A")
            arguments = ["rebase", "--continue"]
    subprocess.run(
        ["git", "-C", str(directory), "rebase", "--abort"], capture_output=True
    )
    raise SyncError("gave up reconciling histories - run `sttop sync` again")


def pull_sessions(directory: Path, remote: str) -> str:
    """Bring other machines' sessions down, before anything else happens.

    Safe on a machine that has never synced: it adopts the remote's history
    wholesale - vault included, which is what makes joining an existing repo
    from a second PC work, and why this must run *before* a vault could be
    created locally.
    """
    if not remote:
        return "git: no remote configured"
    directory.mkdir(parents=True, exist_ok=True)
    _init_repo(directory)
    _set_origin(directory, remote)
    _ensure_gh_credentials(directory, remote)
    _git(directory, "fetch", "-q", "origin")
    branch = _remote_branch(directory)
    if branch is None:
        return "git: remote is empty"

    before = {p.name for p in directory.rglob(f"*{ENCRYPTED_SUFFIX}")}
    if _has_commits(directory):
        _rebase_onto(directory, _align_branch(directory, branch))
        verb = "pulled"
    else:
        _git(directory, "checkout", "-q", "-B", branch, f"origin/{branch}")
        verb = "adopted"
    new = {p.name for p in directory.rglob(f"*{ENCRYPTED_SUFFIX}")} - before
    return f"git: {verb} {len(new)} session(s) from the remote" if new else (
        "git: up to date"
    )


def sync_sessions(directory: Path, remote: str = "", cipher: Cipher | None = None) -> str:
    """Seal, commit, reconcile with the remote, push. Returns a one-line
    summary for the terminal; raises SyncError with git's own words otherwise.

    With a cipher, plaintext sessions are sealed first ("sync" mode); without
    one, whatever encrypted files already exist are synced ("always" mode).
    The remote is always integrated before pushing, so machines can take
    turns freely; conflicting seals of one session resolve to the machine
    that owns its plaintext.
    """
    directory.mkdir(parents=True, exist_ok=True)
    _init_repo(directory)
    _ensure_whitelist(directory)
    if cipher is not None:
        seal_sessions(directory, cipher)

    _git(directory, "add", "-A")
    committed = False
    if _git(directory, "diff", "--cached", "--name-only"):
        message = f"sttop: sessions as of {datetime.now():%Y-%m-%d %H:%M}"
        _git(directory, *_identity(directory), "commit", "-q", "-m", message)
        committed = True

    if not _has_commits(directory):
        return "git: nothing to commit yet (no encrypted sessions)"

    if not remote:
        return (
            "git: committed (local history only - set storage.git_remote to push)"
            if committed
            else "git: nothing new to commit"
        )

    _set_origin(directory, remote)
    _ensure_gh_credentials(directory, remote)
    _git(directory, "fetch", "-q", "origin")
    upstream = _remote_branch(directory)
    branch = (
        _align_branch(directory, upstream)
        if upstream
        else _git(directory, "symbolic-ref", "--short", "HEAD")
    )
    if upstream:
        _rebase_onto(directory, branch)
    try:
        _git(directory, "push", "-q", "-u", "origin", branch)
    except SyncError:
        # Another machine pushed in the window since the fetch: integrate
        # its commits and try once more.
        _git(directory, "fetch", "-q", "origin")
        _rebase_onto(directory, branch)
        _git(directory, "push", "-q", "-u", "origin", branch)
    return f"git: pushed to {remote}"


def gh_ready() -> bool:
    """Whether the GitHub CLI is installed and logged in - the two things
    creating a repo on the user's behalf needs."""
    if shutil.which("gh") is None:
        return False
    probe = subprocess.run(["gh", "auth", "status"], capture_output=True)
    return probe.returncode == 0


def create_github_repo(name: str) -> str:
    """A *private* repo via gh, returning the URL to push to.

    The URL follows the user's configured git protocol, so whatever auth gh
    already set up keeps working. If the repo already exists - an earlier
    setup attempt that failed later on - it is simply reused: the URL is the
    answer either way.
    """
    made = subprocess.run(
        ["gh", "repo", "create", name, "--private"], capture_output=True, text=True
    )
    url = _repo_url(name)
    if url is None:
        detail = (made.stderr or made.stdout).strip().splitlines()
        raise SyncError(detail[-1] if detail else f"gh could not create {name}")
    return url


def _repo_url(name: str) -> str | None:
    protocol = subprocess.run(
        ["gh", "config", "get", "git_protocol"], capture_output=True, text=True
    ).stdout.strip()
    field = "sshUrl" if protocol == "ssh" else "url"
    view = subprocess.run(
        ["gh", "repo", "view", name, "--json", field, "-q", f".{field}"],
        capture_output=True,
        text=True,
    )
    url = view.stdout.strip()
    if view.returncode != 0 or not url:
        return None
    return url if url.endswith(".git") else url + ".git"


def _ensure_gh_credentials(directory: Path, remote: str) -> None:
    """Let gh answer the https credential prompt, repo-locally.

    gh hands out https remotes, but plain git does not know to ask gh for
    the token - without this, the first push interrogates the user for a
    username and a password GitHub no longer even accepts. Local config
    only: nothing outside this repo changes.
    """
    if not remote.startswith("https://github.com/") or shutil.which("gh") is None:
        return
    probe = subprocess.run(
        ["git", "-C", str(directory), "config", "--local", "credential.helper"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:  # nothing set for this repo yet
        _git(directory, "config", "credential.helper", "!gh auth git-credential")


def _set_origin(directory: Path, remote: str) -> None:
    try:
        current = _git(directory, "remote", "get-url", "origin")
    except SyncError:
        _git(directory, "remote", "add", "origin", remote)
        return
    if current != remote:  # the config changed; the config wins
        _git(directory, "remote", "set-url", "origin", remote)
