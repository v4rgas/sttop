import pytest

from sttop.crypto import (
    VAULT_NAME,
    Cipher,
    VaultError,
    decrypt_session,
    is_encrypted,
    open_vault,
    vault_exists,
)
from sttop.journal import Journal, Utterance

# The scrypt parameters are the product's; the tests pay them once per vault.


def test_open_vault_creates_then_reopens(tmp_path):
    assert not vault_exists(tmp_path)
    first = open_vault(tmp_path, "hunter2")
    assert vault_exists(tmp_path)
    second = open_vault(tmp_path, "hunter2")
    # Same salt both times, so files sealed by either open with the other.
    assert first.salt == second.salt
    assert second.open(first.seal("hola")) == "hola"


def test_wrong_passphrase_is_caught_at_the_vault(tmp_path):
    """One clear error up front, not a decryption failure per session file."""
    open_vault(tmp_path, "right")
    with pytest.raises(VaultError, match="wrong passphrase"):
        open_vault(tmp_path, "wrong")


def test_seal_round_trip_and_tamper_detection(tmp_path):
    cipher = open_vault(tmp_path, "pw")
    line = cipher.seal("- `00:03` **you** — hello\n")
    assert cipher.open(line) == "- `00:03` **you** — hello\n"
    with pytest.raises(VaultError):
        cipher.open("AAAA" + line[4:])


def test_decrypt_session_from_passphrase_alone(tmp_path):
    """A session file carries its own salt: it decrypts without the .vault,
    so a file copied off the machine is not orphaned."""
    cipher = open_vault(tmp_path, "pw")
    journal = Journal.create(tmp_path, "demo", cipher=cipher)
    journal.append(Utterance("mic", "you", 3.0, 4.0, "hello there"))
    journal.close(10.0)
    (tmp_path / VAULT_NAME).unlink()

    text = decrypt_session(journal.path, "pw")
    assert "- `00:03` **you** — hello there" in text
    assert "# demo" in text


def test_decrypt_session_reuses_a_matching_cipher(tmp_path):
    cipher = open_vault(tmp_path, "pw")
    journal = Journal.create(tmp_path, "demo", cipher=cipher)
    journal.append(Utterance("mic", "you", 0.0, 1.0, "hi"))
    journal.close(1.0)
    assert "hi" in decrypt_session(journal.path, cipher)


def test_a_foreign_cipher_is_rejected_not_misused(tmp_path):
    ours = open_vault(tmp_path / "a", "pw")
    theirs = open_vault(tmp_path / "b", "pw")  # same passphrase, other salt
    journal = Journal.create(tmp_path / "a", "demo", cipher=ours)
    journal.append(Utterance("mic", "you", 0.0, 1.0, "hi"))
    journal.close(1.0)
    with pytest.raises(VaultError, match="different vault"):
        decrypt_session(journal.path, theirs)


def test_is_encrypted_by_suffix(tmp_path):
    assert is_encrypted(tmp_path / "x.md.enc")
    assert not is_encrypted(tmp_path / "x.md")


def test_plaintext_never_reaches_the_disk(tmp_path):
    cipher = open_vault(tmp_path, "pw")
    journal = Journal.create(tmp_path, "secret meeting", cipher=cipher)
    journal.append(Utterance("mic", "you", 0.0, 1.0, "the launch is friday"))
    raw = journal.path.read_bytes()
    journal.close(1.0)
    for secret in (b"launch", b"friday", b"secret meeting", b"you"):
        assert secret not in raw
    # ...while the header stays greppable enough to identify the format.
    assert raw.startswith(b"sttop-enc/1 ")


def test_cipher_lines_are_not_deterministic(tmp_path):
    """A fresh nonce per chunk: equal plaintexts must not reveal themselves."""
    cipher = Cipher(b"k" * 32, b"s" * 16)
    assert cipher.seal("same") != cipher.seal("same")


# -- the remembered key (.env) ------------------------------------------------


def test_save_key_round_trips_without_the_passphrase(tmp_path):
    from sttop.crypto import load_key, save_key

    cipher = open_vault(tmp_path, "pw")
    save_key(tmp_path, cipher)
    remembered = load_key(tmp_path)
    assert remembered is not None
    assert remembered.open(cipher.seal("hola")) == "hola"


def test_the_env_file_is_owner_only(tmp_path):
    from sttop.crypto import save_key

    path = save_key(tmp_path, open_vault(tmp_path, "pw"))
    assert path.name == ".env"
    assert path.stat().st_mode & 0o777 == 0o600


def test_save_key_keeps_other_env_variables(tmp_path):
    from sttop.crypto import save_key

    (tmp_path / ".env").write_text("OTHER=thing\nSTTOP_KEY=stale\n")
    save_key(tmp_path, open_vault(tmp_path, "pw"))
    lines = (tmp_path / ".env").read_text().splitlines()
    assert "OTHER=thing" in lines
    assert sum(line.startswith("STTOP_KEY=") for line in lines) == 1


def test_a_key_from_another_vault_is_not_trusted(tmp_path):
    """load_key answers None for every bad state - missing, corrupt, or a key
    that does not open *this* vault - because the fallback is always the same:
    ask for the passphrase."""
    from sttop.crypto import load_key, save_key

    assert load_key(tmp_path / "nowhere") is None

    ours = tmp_path / "ours"
    theirs = tmp_path / "theirs"
    open_vault(ours, "pw")
    save_key(ours, open_vault(theirs, "pw"))  # a foreign key in our .env
    assert load_key(ours) is None

    (ours / ".env").write_text("STTOP_KEY=not.base64!\n")
    assert load_key(ours) is None
