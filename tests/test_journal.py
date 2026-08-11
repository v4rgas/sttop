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
