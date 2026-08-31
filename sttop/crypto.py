"""Passphrase encryption for session transcripts.

Turned on with `storage.encrypt`, so a synced copy of the sessions directory -
a git remote, a backup, a stolen laptop - holds ciphertext that only sttop plus
the passphrase can read.

The format has to survive the journal's write-and-flush-per-utterance habit,
so it is a log, not a blob: a header line naming the salt, then one base64
line per sealed chunk (AES-256-GCM, random nonce). Appending a chunk never
touches earlier ones, which keeps the crash-safety story identical to the
plaintext journal - kill sttop mid-meeting and everything already flushed
still decrypts.

One salt serves the whole sessions directory, recorded in `.sttop-vault` next
to the sessions - and synced with them, so a fresh clone of the repo carries
everything decryption needs except the passphrase. That buys a single scrypt
run to read any number of files, and one early "wrong passphrase" instead of
a decryption failure per file - but each session file repeats the salt in its
own header, so a file that escapes the directory is still decryptable on its
own.

The derived key is remembered in a `.env` beside the sessions, readable only
by the owner, so the passphrase is typed once ever. That is safe *because* of
"sync" mode's threat model: the plaintext lives on the same disk anyway, so
the stored key gives an attacker with this disk nothing the .md files did
not - it only protects the remote. The repo's whitelist .gitignore is what
keeps `.env` (and the plaintext) out of every commit.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

MAGIC = "sttop-enc/1"
VAULT_MAGIC = "sttop-vault/1"
#: Encrypted sessions sit next to plaintext ones, distinguishable at a glance.
ENCRYPTED_SUFFIX = ".md.enc"
#: Salt + passphrase verifier, in the sessions directory and in the repo.
VAULT_NAME = ".sttop-vault"
#: The remembered key, in the sessions directory and NEVER in the repo.
KEY_FILE = ".env"
KEY_VAR = "STTOP_KEY"
#: A generated passphrase is kept here too - it exists nowhere else, and the
#: user needs to copy it to read the sessions on another machine.
PASS_VAR = "STTOP_PASSPHRASE"

_CHECK = "sttop vault check"
#: scrypt cost: ~32 MiB and well under a second on anything that can also run
#: a speech model. High enough that an offline guess costs real time.
_N, _R, _P = 2**15, 8, 1


class VaultError(ValueError):
    """A vault or session that cannot be opened - wrong passphrase, or a file
    that is not one of ours. The message is safe to show the user."""


def _derive(passphrase: str, salt: bytes) -> bytes:
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

    return Scrypt(salt=salt, length=32, n=_N, r=_R, p=_P).derive(
        passphrase.encode("utf-8")
    )


class Cipher:
    """A derived key, ready to seal chunks into lines and open them again."""

    def __init__(self, key: bytes, salt: bytes) -> None:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        self._gcm = AESGCM(key)
        self.key = key
        self.salt = salt

    @property
    def header(self) -> str:
        """First line of every encrypted session file: magic + salt."""
        return f"{MAGIC} {base64.b64encode(self.salt).decode()}\n"

    def seal(self, text: str) -> str:
        """One chunk of plaintext as one appendable base64 line."""
        nonce = os.urandom(12)
        sealed = nonce + self._gcm.encrypt(nonce, text.encode("utf-8"), None)
        return base64.b64encode(sealed).decode() + "\n"

    def open(self, line: str) -> str:
        from cryptography.exceptions import InvalidTag

        try:
            raw = base64.b64decode(line.strip(), validate=True)
            return self._gcm.decrypt(raw[:12], raw[12:], None).decode("utf-8")
        except (InvalidTag, ValueError):
            raise VaultError("wrong passphrase, or a corrupted file") from None


def _parse_header(line: str, magic: str) -> bytes:
    name, sep, encoded = line.strip().partition(" ")
    if name != magic or not sep:
        raise VaultError(f"not a {magic} file")
    try:
        return base64.b64decode(encoded, validate=True)
    except ValueError:
        raise VaultError(f"corrupted {magic} header") from None


def vault_exists(directory: Path) -> bool:
    return (directory / VAULT_NAME).is_file()


def open_vault(directory: Path, passphrase: str) -> Cipher:
    """The directory's cipher, creating the vault on first use.

    The vault file is just the salt plus a sealed known string, so a wrong
    passphrase is caught here - once, up front - rather than as a mysterious
    failure on whichever session file is opened first.
    """
    path = directory / VAULT_NAME
    if not path.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        salt = os.urandom(16)
        cipher = Cipher(_derive(passphrase, salt), salt)
        header = f"{VAULT_MAGIC} {base64.b64encode(salt).decode()}\n"
        path.write_text(header + cipher.seal(_CHECK), encoding="utf-8")
        return cipher

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise VaultError(f"empty vault file: {path}")
    salt = _parse_header(lines[0], VAULT_MAGIC)
    cipher = Cipher(_derive(passphrase, salt), salt)
    try:
        if len(lines) < 2 or cipher.open(lines[1]) != _CHECK:
            raise VaultError("corrupted vault file")
    except VaultError:
        raise VaultError("wrong passphrase") from None
    return cipher


def is_encrypted(path: Path) -> bool:
    return path.name.endswith(ENCRYPTED_SUFFIX)


def _set_env_var(path: Path, name: str, value: str) -> None:
    """Set one variable in a dotenv file, owner-readable only.

    Written as dotenv lines so scripts and the skill can source them too. Any
    other variables someone keeps in the file survive the rewrite.
    """
    kept = []
    if path.is_file():
        kept = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if not line.startswith(f"{name}=")
        ]
    body = "\n".join([*kept, f"{name}={value}"]) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as fh:
        fh.write(body)


def save_key(directory: Path, cipher: Cipher) -> Path:
    """Remember the derived key in <sessions>/.env."""
    path = directory / KEY_FILE
    salt64 = base64.b64encode(cipher.salt).decode()
    key64 = base64.b64encode(cipher.key).decode()
    _set_env_var(path, KEY_VAR, f"{salt64}.{key64}")
    return path


def save_passphrase(directory: Path, passphrase: str) -> Path:
    """Keep a *generated* passphrase in <sessions>/.env - the user never saw
    it typed, so this file is the only place it exists to be copied from."""
    path = directory / KEY_FILE
    _set_env_var(path, PASS_VAR, passphrase)
    return path


def load_key(directory: Path) -> Cipher | None:
    """The remembered cipher, or None - never an error, because every failure
    mode here (no file, stale key, another vault's key) has the same answer:
    fall back to asking for the passphrase."""
    path = directory / KEY_FILE
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.partition("=")
        if name.strip() != KEY_VAR or not sep:
            continue
        try:
            salt64, _, key64 = value.strip().partition(".")
            cipher = Cipher(
                base64.b64decode(key64, validate=True),
                base64.b64decode(salt64, validate=True),
            )
        except ValueError:
            return None
        return cipher if _matches_vault(directory, cipher) else None
    return None


def _matches_vault(directory: Path, cipher: Cipher) -> bool:
    """A remembered key is only trusted against its own vault - a key left
    behind from another vault must prompt, not seal files nobody can open."""
    vault = directory / VAULT_NAME
    if not vault.is_file():
        return False
    try:
        lines = vault.read_text(encoding="utf-8").splitlines()
        return (
            len(lines) >= 2
            and _parse_header(lines[0], VAULT_MAGIC) == cipher.salt
            and cipher.open(lines[1]) == _CHECK
        )
    except VaultError:
        return False


def decrypt_session(path: Path, key: str | Cipher) -> str:
    """The plaintext Markdown of one encrypted session.

    Accepts either a passphrase or an already-derived Cipher; the cipher is
    reused only when its salt matches the file's own header, so a stray file
    from another vault still decrypts correctly (with its own scrypt run).
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise VaultError(f"empty session file: {path}")
    salt = _parse_header(lines[0], MAGIC)
    if isinstance(key, Cipher) and key.salt == salt:
        cipher = key
    elif isinstance(key, Cipher):
        raise VaultError(f"{path.name} belongs to a different vault")
    else:
        cipher = Cipher(_derive(key, salt), salt)
    return "".join(cipher.open(line) for line in lines[1:] if line.strip())
