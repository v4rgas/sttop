# sttop

sttop transcribes your mic and system audio locally, right in your terminal.
It labels who's speaking and writes a Markdown transcript as you talk.

## Start recording now

```bash
uvx sttop
```

![sttop recording a standup](https://raw.githubusercontent.com/v4rgas/sttop/main/docs/sttop.svg)

It runs on Linux and macOS and downloads models on the first run. You don't need
API keys. On macOS 13+, give your terminal permission in System Settings →
Privacy & Security → Screen & System Audio Recording, then restart the terminal
to capture system audio.

<details>
<summary>Installation options</summary>

On Linux, use CPU-only PyTorch wheels for a smaller download:

```bash
uvx --index https://download.pytorch.org/whl/cpu sttop
```

To install permanently:

```bash
uv tool install sttop
```

</details>

## Usage

```bash
uvx sttop -t "standup"               # name the session
uvx sttop sessions                   # list transcripts
uvx sttop read                       # open the latest transcript
uvx sttop read standup               # find a session by name
uvx sttop devices --test             # check audio sources
uvx sttop doctor                     # troubleshoot setup
uvx sttop --help                     # all commands and options
```

| Key | Action |
| --- | --- |
| `q` | Name the session and quit |
| `ctrl+c` | Quit without naming |
| `space` | Pause / resume |
| `r` | Rename a speaker throughout the transcript |
| `y` | Copy the transcript so far |

sttop saves each line to `~/.local/share/sttop/sessions/` as it's transcribed.
It labels your mic as "you" and assigns speaker labels to the system audio.

## Customize

Run `uvx sttop config` to create `~/.config/sttop/config.toml`. The comments
explain the settings for audio sources, models, speaker labels, themes, and storage.

sttop uses Parakeet by default. To use Whisper:

```bash
uvx sttop --backend whisper -m small
```

Use `--pipe 'your-command'` to stream utterances and speaker renames as JSON lines
to another program's stdin.

## Encrypted backup

```bash
uvx sttop sync
```

The first run sets up a private Git remote and a vault passphrase. sttop keeps
Markdown files on your machine and syncs encrypted copies after each recording.
Use the same remote and passphrase on another machine to share your transcripts.

Keep a copy of the passphrase. You can't recover the remote backups without it.
Set `storage.encrypt = "always"` to encrypt local transcripts too.

## Development

```bash
git clone https://github.com/v4rgas/sttop
cd sttop
uv sync --extra dev
uv run sttop
uv run pytest
```

## License

MIT
