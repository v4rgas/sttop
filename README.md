# sttop

Live speech-to-text monitor for the terminal — `htop`, but for what is being said.

Taps your **microphone** and your **system audio output** as two independent streams,
transcribes both in real time, labels who is speaking, and appends every line to a
Markdown file as it happens. Fully local: no network, no API keys, nothing leaves
the machine.

```
 sttop  ● rec 04:12   mic ███·······  sys ██████····   parakeet-tdt/cpu onnx · ecapa @0.50 · queue 0 · 37 lines · 3 voices
 writing → ~/.local/share/sttop/sessions/2026-08-10-1432-standup.md

 03:58  you    so the migration lands friday?
 04:02  spk1   friday is tight, monday is safer
 04:09  spk2   +1 on monday

 q quit   space pause   r rename speaker
```

## Why two streams

Capturing the mic and the speaker output separately means **you** are identified for
free — anything on the mic is you, no model required, never wrong. Voice embeddings
then only have to split the *remote* side into individual participants, which is a
much easier problem than diarizing a single mixed track.

## Install

Needs `ffmpeg` and PipeWire or PulseAudio (`pactl`).

```bash
git clone <this repo> && cd sttop
uv sync
```

## Use

```bash
uv run sttop                       # record with defaults
uv run sttop -t "standup"          # title the session (used in the filename)
uv run sttop --backend whisper -m small
uv run sttop devices --test        # list audio sources, record 1s from each
uv run sttop sessions              # list past transcripts
uv run sttop config                # write ~/.config/sttop/config.toml
uv run sttop theme                 # show the detected terminal colour scheme
```

Keys: `q` quit · `space` pause · `r` rename a speaker (`spk1=Ana`, rewrites past lines too).

## Output

One Markdown file per session in `~/.local/share/sttop/sessions/`, flushed after every
line — kill it mid-meeting and the transcript so far is already on disk.

```markdown
# standup

- started: 2026-08-10 14:32:01 -04
- mic: `alsa_input.pci-0000_08_00.6.analog-stereo`
- system: `alsa_output.pci-0000_08_00.6.analog-stereo.monitor`
- backend: `parakeet-tdt/cpu onnx`

## Transcript

- `03:58` **you** — so the migration lands friday?

- `04:02` **spk1** — friday is tight, monday is safer
```

## How it works

```
ffmpeg -f pulse (mic)      ─┐
                            ├─ webrtcvad segmenter ─→ queue ─→ parakeet ─→ ecapa ─→ journal.md
ffmpeg -f pulse (monitor)  ─┘
```

Audio is cut into utterances by voice-activity detection (a segment closes after
700 ms of silence, or at 15 s for a monologue), and only speech reaches the model.

**One thread boundary, and it is the model.** The two capture readers and the
consumer are asyncio tasks — they are blocking pipe I/O, which is what an event
loop is for — while transcription and voice embedding run in a single-worker
`ThreadPoolExecutor`. So the UI needs no cross-thread marshalling, shutdown is
ordinary task cancellation, and utterances stay in the order they were spoken. The
executor is single-worker on purpose: transcription is CPU-bound and already
internally parallel, so a second worker would only thrash the cache. When it falls
behind, the queue absorbs the lag — visible as `queue N` in the status bar — rather
than dropping audio.

Speaker labels come from online clustering of ECAPA-TDNN voice embeddings: each
utterance is matched against running centroids by cosine similarity. A confident
match (≥ `threshold`) joins that speaker and updates its centroid; a near miss
(within `margin` below it) joins without touching the centroid; only a clearly
distant voice opens a new speaker. That hysteresis matters — without it a single
noisy embedding mints a phantom participant, and one person ends up spread across
`spk1`/`spk2`/`spk3`. Being online means labels are assigned as audio arrives and are
never revised, which is the price of real time. Segments under 1.5 s are too short to
embed reliably and inherit the previous speaker, or show as `spk?`.

## Backends

**parakeet** (default) — NVIDIA Parakeet TDT 0.6b v3 through onnxruntime. Multilingual
across 25 European languages with autodetection, punctuated output, and roughly 19×
real time on CPU. Needs neither torch nor the NeMo toolkit, since onnx-asr runs the
exported graph directly. On the same 11 s clip where `whisper tiny` produced a
hallucinated lead-in and lost its punctuation, Parakeet returned the sentence verbatim.

**whisper** — faster-whisper/CTranslate2, if you want Whisper's language coverage.
CTranslate2 ships **CUDA and CPU backends only — there is no ROCm build**, so on an AMD
GPU this runs on CPU no matter what torch reports. The device is detected at startup
(`cuda` if CTranslate2 sees one, else `cpu`) and shown in the status bar.

To push a Radeon card at the *diarization* half, resync torch against the ROCm index
(see the comment in `pyproject.toml`).

## Config

`~/.config/sttop/config.toml`. Run `sttop config` to write a default with every
knob and its documentation in it; the comments come from the source, so the file
never drifts from the code. Anything you leave out keeps its default, and blank
means "you decide" wherever a default is picked for you.

```toml
sessions_dir = "~/.local/share/sttop/sessions"

[audio]
mic_source = ""        # substring match against pactl source names; blank = default
system_source = ""     # blank = monitor of the default sink
save_wav = false

[vad]
aggressiveness = 2     # 0 permissive .. 3 strict
silence_ms = 700
max_segment_s = 15.0

[stt]
backend = "parakeet"   # parakeet | whisper
model = ""             # blank = the backend's default model
device = "auto"        # whisper only
language = ""          # blank = autodetect

[ui]
theme = "auto"         # auto follows your terminal; or gruvbox, nord, ...

[diarize]
enabled = true
threshold = 0.50       # lower = fewer, broader speakers
margin = 0.15          # grey zone that attaches instead of opening a speaker
```

## Theming

By default sttop paints with the `ansi-dark` / `ansi-light` Textual themes, which
use only the terminal's own 16 ANSI colours — so it inherits whatever palette you
already have rather than imposing its own. Which of the two is picked by reading
`COLORFGBG`, and failing that by asking the terminal for its background colour over
OSC 11 (supported by ghostty, kitty, alacritty, wezterm, foot, xterm). If nothing
answers, it assumes dark. Run `sttop theme` to see what was detected.

Set `ui.theme` to any Textual theme name (`gruvbox`, `nord`, `catppuccin-mocha`,
`solarized-light`, …) to override the terminal-following behaviour.

## Tests

```bash
uv run --extra dev pytest
```

The audio-dependent path is exercised by `tests/test_pipeline.py`, which plays a speech
sample into the default sink and reads it back off the monitor. It needs real audio
hardware and downloads a model, so it is opt-in:

```bash
STTOP_INTEGRATION=1 uv run --extra dev pytest
```

## License

MIT
