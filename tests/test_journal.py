from sttop.journal import Journal, Utterance, clock, list_sessions, slugify


def utterance(speaker: str, start: float, text: str) -> Utterance:
    return Utterance("system", speaker, start, start + 1.0, text)


def test_clock_formats_hours_only_when_needed():
    assert clock(0) == "00:00"
    assert clock(63) == "01:03"
    assert clock(3723) == "1:02:03"
    assert clock(-5) == "00:00"


def test_slugify():
    assert slugify("Standup / Q3 planning!") == "standup-q3-planning"
    assert slugify("!!!") == "session"


def test_lines_are_readable_before_close(tmp_path):
    journal = Journal.create(tmp_path, "demo", backend="test")
    journal.append(utterance("you", 3.0, "hello"))

    # Flushed per line: the file is complete without closing it.
    text = journal.path.read_text()
    assert "- `00:03` **you** — hello" in text
    assert "# demo" in text
    journal.close(10.0)


def test_close_writes_the_footer(tmp_path):
    journal = Journal.create(tmp_path, "demo")
    journal.append(utterance("you", 0.0, "hi"))
    path = journal.close(65.0)
    text = path.read_text()
    assert "duration: 01:05" in text
    assert "utterances: 1" in text


def test_speaker_turns_are_separated_by_blank_lines(tmp_path):
    journal = Journal.create(tmp_path, "demo")
    journal.append(utterance("you", 0.0, "a"))
    journal.append(utterance("you", 1.0, "b"))
    journal.append(utterance("spk1", 2.0, "c"))
    body = journal.path.read_text().split("## Transcript\n\n")[1]
    journal.close(3.0)
    assert body == (
        "- `00:00` **you** — a\n"
        "- `00:01` **you** — b\n"
        "\n"
        "- `00:02` **spk1** — c\n"
    )


def test_rename_rewrites_past_lines_and_keeps_writing(tmp_path):
    journal = Journal.create(tmp_path, "demo")
    journal.append(utterance("spk1", 0.0, "one"))
    journal.append(utterance("spk1", 1.0, "two"))

    assert journal.rename_speaker("spk1", "Ana") == 2

    journal.append(utterance("spk1", 2.0, "three"))
    text = journal.close(3.0).read_text()
    assert text.count("**Ana**") == 2
    assert "**spk1** — three" in text
    assert "# demo" in text  # the header survived the rewrite


def test_rename_of_the_current_speaker_does_not_reprint_the_header(tmp_path):
    journal = Journal.create(tmp_path, "demo")
    journal.append(utterance("spk1", 0.0, "one"))
    journal.rename_speaker("spk1", "Ana")
    journal.append(utterance("Ana", 1.0, "two"))
    body = journal.close(2.0).read_text()
    # Same speaker before and after the rename: no blank-line turn break.
    assert "**Ana** — one\n- `00:01` **Ana** — two" in body


def test_filenames_do_not_collide(tmp_path):
    first = Journal.create(tmp_path, "same")
    second = Journal.create(tmp_path, "same")
    assert first.path != second.path
    first.close(1.0)
    second.close(1.0)
    assert len(list_sessions(tmp_path)) == 2


def test_list_sessions_on_a_missing_directory(tmp_path):
    assert list_sessions(tmp_path / "nope") == []


def test_snapshot_is_the_file_so_far(tmp_path):
    """What `y` copies mid-meeting must equal what is on disk, byte for byte -
    otherwise the clipboard and the transcript disagree about the meeting."""
    journal = Journal.create(tmp_path, "demo", backend="test")
    journal.append(utterance("you", 0.0, "hello"))
    journal.append(utterance("spk1", 1.0, "hi"))
    assert journal.snapshot() == journal.path.read_text()
    journal.rename_speaker("spk1", "Ana")
    assert journal.snapshot() == journal.path.read_text()
    journal.close(2.0)
    assert journal.snapshot() == journal.path.read_text()


def test_encrypted_journal_round_trips(tmp_path):
    from sttop.crypto import Cipher, decrypt_session

    cipher = Cipher(b"k" * 32, b"s" * 16)
    journal = Journal.create(tmp_path, "demo", cipher=cipher)
    journal.append(utterance("spk1", 0.0, "one"))
    journal.append(utterance("spk1", 1.0, "two"))
    assert journal.rename_speaker("spk1", "Ana") == 2
    journal.append(utterance("Ana", 2.0, "three"))
    path = journal.close(3.0)

    assert path.name.endswith(".md.enc")
    text = decrypt_session(path, cipher)
    assert text == journal.snapshot()
    assert text.count("**Ana**") == 3
    assert "duration: 00:03" in text


def test_list_sessions_sees_encrypted_and_plain_alike(tmp_path):
    from sttop.crypto import Cipher

    Journal.create(tmp_path, "plain").close(1.0)
    Journal.create(tmp_path, "sealed", cipher=Cipher(b"k" * 32, b"s" * 16)).close(1.0)
    names = [p.name for p in list_sessions(tmp_path)]
    assert len(names) == 2
    assert any(n.endswith(".md.enc") for n in names)


def test_a_sealed_twin_is_the_same_session_and_plaintext_wins(tmp_path):
    """Sync mode leaves a .md.enc beside every .md; listing both would show
    each meeting twice, and `sttop read` would open the unreadable one."""
    (tmp_path / "2026-01-01-0900-a.md").write_text("# a")
    (tmp_path / "2026-01-01-0900-a.md.enc").write_text("sttop-enc/1 x\n")
    (tmp_path / "2026-01-02-0900-b.md.enc").write_text("sttop-enc/1 x\n")  # clone-only
    names = [p.name for p in list_sessions(tmp_path)]
    assert names == ["2026-01-02-0900-b.md.enc", "2026-01-01-0900-a.md"]


def test_rename_leaves_no_debris(tmp_path):
    """The rewrite goes via a temporary file so a kill mid-rename cannot
    truncate the session. Nothing of it may survive in the sessions dir."""
    journal = Journal.create(tmp_path, "demo")
    journal.append(utterance("spk1", 0.0, "one"))
    journal.rename_speaker("spk1", "Ana")
    journal.close(1.0)

    assert [p.name for p in tmp_path.iterdir()] == [journal.path.name]
    assert list_sessions(tmp_path) == [journal.path]


def test_a_session_is_filed_under_its_name_with_its_stamp(tmp_path):
    from sttop.journal import file_session, list_sessions, name_candidates

    first = tmp_path / "2026-09-14-1030-session.md"
    first.write_text("x")
    filed = file_session(first, tmp_path, "micelio/daily")
    assert filed == tmp_path / "micelio" / "daily.2026-09-14-1030.md"
    assert list_sessions(tmp_path) == [filed]

    (tmp_path / "2026-09-14-1030-session.md.enc").write_text("x")
    sealed = tmp_path / "2026-09-14-1030-session.md.enc"
    again = file_session(sealed, tmp_path, "micelio/daily")
    assert again.name == "daily.2026-09-14-1030.md.enc"
    (tmp_path / "micelio" / "dev").mkdir()

    assert name_candidates(tmp_path, "m") == ["micelio"]
    assert name_candidates(tmp_path, "micelio/d") == ["micelio/daily", "micelio/dev"]
    assert name_candidates(tmp_path, "../") == []
    assert file_session(filed, tmp_path, "../..") == filed  # nothing escapes
