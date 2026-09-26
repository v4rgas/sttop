from sttop.follow import follow, live_session, utterances
from sttop.journal import Journal, Utterance, file_session


def read(path):
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def say(journal, text, speaker="you", at=0.0):
    journal.append(Utterance("mic", speaker, at, at + 1, text))


def test_only_utterance_lines_count():
    text = (
        "# weekly\n\n- started: 2026-09-26 10:00:00\n- mic: `m`\n\n## Transcript\n\n"
        "- `00:01` **you** — hola\n\n- `01:02:03` **speaker 1** — chao\n"
        "\n---\n\n- ended: 2026-09-26 11:00:00\n- utterances: 2\n"
    )
    assert utterances(text) == [
        "- `00:01` **you** — hola",
        "- `01:02:03` **speaker 1** — chao",
    ]


def test_a_line_still_being_written_waits():
    assert utterances("- `00:01` **you** — hola\n- `00:02` **you** — ch") == [
        "- `00:01` **you** — hola"
    ]


def test_new_lines_come_out_in_batches_until_the_session_ends(tmp_path):
    journal = Journal.create(tmp_path, "weekly")
    say(journal, "first")
    events = []

    def tick(_seconds):  # the meeting goes on between batches
        ticks = len(events)
        if ticks == 2:
            say(journal, "second")
            say(journal, "third", speaker="speaker 1")
        elif ticks == 3:
            journal.close(60.0)

    follow(journal.path, tmp_path, read, events.append, sleep=tick)

    assert events[0].startswith("[sttop] following ")
    assert events[1] == "- `00:00` **you** — first"
    assert events[2] == "- `00:00` **you** — second\n- `00:00` **speaker 1** — third"
    assert events[3].startswith("[sttop] session ended: ")


def test_a_rename_does_not_repeat_old_lines(tmp_path):
    journal = Journal.create(tmp_path, "weekly")
    say(journal, "hola", speaker="speaker 1")
    events = []

    def tick(_seconds):
        if len(events) == 2:
            journal.rename_speaker("speaker 1", "Ana")
            say(journal, "chao", speaker="Ana")
        else:
            journal.close(1.0)

    follow(journal.path, tmp_path, read, events.append, sleep=tick)
    assert events[2] == "- `00:00` **Ana** — chao"


def test_a_session_filed_away_mid_batch_is_found_again(tmp_path):
    """Naming a session on quit moves it into a folder; the lines written
    since the last batch must still come out."""
    journal = Journal.create(tmp_path, "weekly")
    events = []

    def tick(_seconds):
        say(journal, "last words")
        path = journal.close(1.0)
        file_session(path, tmp_path, "revi/weekly")

    follow(journal.path, tmp_path, read, events.append, sleep=tick)
    assert events[1] == "- `00:00` **you** — last words"
    assert "revi" in events[2]


def test_a_deleted_session_ends_the_stream(tmp_path):
    journal = Journal.create(tmp_path, "weekly")
    events = []
    follow(journal.path, tmp_path, read, events.append,
           sleep=lambda _: journal.path.unlink())
    assert events[-1].startswith("[sttop] session file is gone")


def test_waits_for_a_recording_to_start(tmp_path):
    Journal.create(tmp_path, "yesterday").close(1.0)
    waited = []
    started = []

    def tick(_seconds):
        # A new minute, so the newer session sorts first.
        started.append(Journal(tmp_path / "2099-01-01-0900-weekly.md", "weekly"))
        started[0]._open("", "", "")

    path = live_session(tmp_path, read, waiting=lambda: waited.append(1), sleep=tick)
    assert path.name == "2099-01-01-0900-weekly.md"
    assert waited == [1]
