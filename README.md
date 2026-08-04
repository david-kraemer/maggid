# maggid

Local text-to-speech for Claude Code over [MCP](https://modelcontextprotocol.io/).
Runs [Chatterbox Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) on
Apple Silicon through [MLX-audio](https://github.com/Blaizzy/mlx-audio). Claude
speaks its notifications and narration aloud, with a distinct voice per workspace
when several agents run at once.

A *maggid* is an itinerant preacher — one who tells, as against one who
expounds. Fitting for a server whose only job is to say what happened.

**Requirements:** macOS on Apple Silicon, Python 3.12 or later.

## Setup

```bash
cd ~/projects/maggid
uv sync

# Download the model (about 3 GB) and write a default config:
uv run maggid init
```

Use `uv sync`, not `uv pip install -e .`. The former installs the versions in
`uv.lock`; the latter re-resolves against the looser bounds in `pyproject.toml`
and can leave you on a different `fastmcp`. If the environment ever ends up in a
mixed state — a stale package tree from a downgrade shows up as
`ImportError: cannot import name ... (unknown location)` — delete `.venv` and
sync again rather than installing over it.

### Shared daemon (recommended)

One process serves every Claude Code session. Run this mode if you keep more
than one or two sessions open.

```bash
sed "s|__VENV__|$PWD/.venv|g" launchd/com.maggid.plist \
  > ~/Library/LaunchAgents/com.maggid.plist
launchctl load ~/Library/LaunchAgents/com.maggid.plist

claude mcp add --transport http --scope user maggid http://127.0.0.1:8765/mcp
```

Per-session servers each load their own copy of the model and own their own
speaker. Two agents that talk at once then produce overlapping audio you cannot
parse. One daemon gives you one model in memory, one warmup, and a single
playback queue, so an urgent `permission` message in one workspace preempts
`narrate` in another instead of racing it.

The daemon warms the model at load. You pay the 20-second startup once at login,
not on the first message.

### Per-session stdio (fallback)

```bash
claude mcp add --transport stdio --scope user maggid -- \
  /path/to/maggid/.venv/bin/maggid
```

Simpler, with no lifecycle to manage. The server lives exactly as long as its
session. But each session pays its own 3 GB and its own warmup, and audio from
concurrent sessions overlaps.

Verify either mode with `claude mcp list`.

## Tools

| Tool                                 | Purpose                                            |
| ------------------------------------ | -------------------------------------------------- |
| `notify(message, channel?)`          | Short task-completion alert                        |
| `speak(text, ref_audio?, channel?)`  | Longer narration, optionally in a cloned voice     |
| `interrupt()`                        | Stop playback and discard the backlog              |

`ref_audio` is a path to a WAV clip to clone. It **must be longer than 5
seconds**. It overrides the workspace voice. `channel` selects a queue priority;
see below.

Use `interrupt` when queued narration has been overtaken by events. It clears
the whole backlog, not just the last utterance.

## Channels

A channel is a priority class for the shared queue, so urgent speech jumps ahead
of narration that already waits.

| Channel      | Priority    |
| ------------ | ----------- |
| `permission` | 1 (highest) |
| `question`   | 2           |
| `notify`     | 10          |
| `narrate`    | 15          |

A message with no channel gets the `notify` priority.

Everything is spoken at one rate, 1.1, applied by resampling at playback.
Per-channel rates were a Kokoro-era idea that never earned its keep: the whole
1.0-to-1.3 range saved under half a second on a typical notification. Priority
and wording carry a channel's meaning better than speed does.

Override priorities in `~/.config/maggid/channels.toml`:

```toml
[narrate]
priority = 20
```

The backlog holds 32 utterances. Past that the server drops new audio instead of
queueing it behind speech that will be stale before it plays.

## Per-workspace identity

With several agents running, the useful question is not "what did it say" but
"which one said it". The daemon answers that two ways at once.

**A spoken label.** Each message starts with the workspace directory name.
`notify("Tests passed")` from `~/projects/spade` is heard as *"spade. Tests
passed."* Set `prefix = false` to turn this off.

**A distinct voice per workspace.** The server reads the client's advertised root
through MCP `roots` and assigns a voice from a pool of nine on first contact.
Assignments live in `~/.config/maggid/assignments.json`, keyed on the
root path, so they survive a daemon restart.

Pin one if you would rather not take what you are given:

```toml
[voices]
spade = "bm_daniel"
```

**Concurrent sessions on one root get names, not numbers.** Several terminals on
one directory is the normal case. The second session on `~/projects/cfd` is heard
as *"cfd Bonnie"*, with its own voice. The name follows the gender of that voice,
and the first session keeps the bare workspace name. A pinned voice applies to
the first session only.

**Why the label does the heavy lifting.** The nine pooled voices come from rating
every cloned candidate 1 to 10, then solving for the subset with the largest
minimum pairwise separation (speaker-embedding cosine). Even so, the best nine
still hold one pair at 0.86 cosine. Chatterbox pulls everything it clones toward
its own character and compresses the whole set into 0.74 to 0.90. Voice alone
separates four or five workspaces, not nine. The label is what scales. The voice
reinforces it.

Past nine workspaces the pool wraps deterministically and two projects share a
voice. Only the label tells them apart at that point.

## Voices

Chatterbox has no voice presets. It clones from a reference clip, which is what
lifts the ceiling, but it means you supply the clips.

A reference clip must be **longer than 5 seconds**. The easiest source that
involves nobody's actual voice is to synthesize the clips from Kokoro's presets.

This is a one-time bootstrap, so its dependencies are deliberately not project
dependencies. Kokoro needs an English grapheme-to-phoneme stack that pulls in
spaCy and torch, and mlx-audio dropped it from its own requirements in 0.4.
Install it into a throwaway environment:

```bash
uv pip install 'mlx-audio>=0.4.6' 'misaki[en]>=0.9' soundfile \
  'en_core_web_sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'
```

```python
import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model

k = load_model("mlx-community/Kokoro-82M-bf16")
text = "A sentence long enough to run past the five second minimum, and one more to be safe."
chunks = [
    r.audio for r in k.generate(text=text, voice="am_adam", speed=1.0, lang_code="a")
]
audio = mx.concatenate(chunks) if len(chunks) > 1 else chunks[0]
mx.eval(audio)
sf.write("voices/refs/am_adam.wav", np.asarray(audio), 24000)
```

The server caches encoded conditionals per clip, so a cloned utterance costs
about 300 ms rather than the 950 ms it takes to re-encode the reference on every
call.

**On cloning real voices:** a reference clip is impersonation capability, whoever
it came from and whatever it was made for. Your own voice and synthesized clips
are unproblematic. A third party's voice needs their agreement. Note also that
upstream Chatterbox watermarks its output through `resemble-perth` and the MLX
port does not. Audio from this server carries no watermark.

## Performance

Measured on an M5 Max with a short phrase ("Tests passed."), best of three warm
runs:

| Model             | Latency | RTF   | Peak RSS |
| ----------------- | ------- | ----- | -------- |
| Kokoro-82M        | 37 ms   | 0.025 | 0.78 GB  |
| Chatterbox base   | 660 ms  | 0.412 | 3.64 GB  |
| Turbo fp16        | 330 ms  | 0.235 | 2.95 GB  |
| **Turbo 8-bit**   | 190 ms  | 0.164 | 2.98 GB  |
| Turbo 4-bit       | 200 ms  | 0.158 | 2.98 GB  |

8-bit is the default. It ties 4-bit on speed and memory, so nothing is gained by
quantizing further. RTF 0.164 means synthesis runs about 6 times faster than
playback, so the speaker is the bottleneck, not the model. More agents do not
make synthesis the constraint.

## Architecture

```
N × Claude Code  ──http──>  one daemon  ──>  Chatterbox Turbo  ──>  audio device
                                 │
                        global priority queue
```

| Module        | Responsibility                                       |
| ------------- | ---------------------------------------------------- |
| `config.py`   | The TOML file: reference voice, channel priorities   |
| `identity.py` | Workspace roots, voice assignment, session slots     |
| `device.py`   | Which output device is current, and telling PortAudio |
| `synth.py`    | Model loading, conditionals, inference               |
| `playback.py` | Priority queue, audio device, resampling             |
| `server.py`   | MCP tools, lifecycle, command line                   |

- **Serialized synthesis.** MLX's Metal command buffer aborts the process if two
  threads evaluate at once, so all inference runs on one dedicated thread. This
  costs nothing at about 190 ms per utterance.
- **Persistent audio stream.** One `sounddevice` output stream stays open for the
  life of the process. Shelling out to `afplay` cost about 1.0 s of process and
  CoreAudio startup per utterance, five times the synthesis time, paid on every
  notification. Removing it also made `interrupt` near-instant, about 50 ms.
- **Rate at playback.** Chatterbox ignores a speed argument, so the server
  resamples before the device write instead of running inference again.
- **The stream follows the output device.** See below — holding one stream open
  is what made this need saying at all.

### Following the output device

Connect a headset to a running daemon and speech has to move to it. Under
`afplay` this was free: a new process picked up the current default output every
time. One stream held open for the life of the daemon gives that up, because
PortAudio decides two things when it initializes and never revisits either —
what hardware exists, and which of it is "the default output". A stream opened at
login therefore keeps writing to the speakers, and the headset is not even in the
device list to switch to.

Re-initializing PortAudio refreshes both. It cannot be what *detects* the change,
though, because it leaves every open stream a dangling pointer. So the two jobs
go to the two APIs that can do them:

- **CoreAudio says whether the output moved.** One HAL property read for the
  default device's id, about 15 µs, and harmless to an open stream. Cheap enough
  to ask before every utterance, which is the right cadence: between utterances
  there is no playback for a reopen to interrupt.
- **PortAudio gets re-initialized, then asked for its own default.** It reads the
  same CoreAudio property, so after the re-scan the two agree by construction and
  there is no device name to match up.

The reopen — abort, close, re-scan, open — costs 130 to 190 ms depending on the
device, paid only when it actually changed. The id is a token compared for
equality and nothing else. PortAudio names the device for the log line, which is
the same string CoreAudio would give and a fortieth of the cost.

**The id is a token and not a name because names are not unique.** One soundcore
headset registers twice under a single name, and both entries are outputs
PortAudio will happily open. Matching a CoreAudio name against PortAudio's device
list would be a coin flip on exactly the hardware this feature exists for.

A short write with no interrupt pending means the device went away mid-utterance:
the headset walked out of range. That reopens once and finishes the rest of the
utterance on whatever is default now, rather than dropping it.

`device.py` calls `sounddevice._terminate` and `_initialize`, which are private.
There is no public way to make PortAudio re-read the hardware.

**Why not drop the persistent stream instead?** Opening a fresh stream per
utterance would delete this whole module: PortAudio re-reads its default every
time, so audio lands on the current device with no probe, no token, and no
CoreAudio. Measured, it costs 247 ms an utterance on Bluetooth against 42 ms for
a write to a stream already open. That is more than the 190 ms of synthesis in
front of it, on every notification, to save fifty lines. `afplay`'s 1.0 s was
process and framework startup; a fresh in-process stream is far cheaper than that
and still not cheap. The held-open stream earns its complexity, and it earns it on
the Bluetooth headset rather than in the general case.

**Why not a CoreAudio property listener?** Push instead of poll would replace a
15 µs read with a flag, and cost a C callback into Python, a run loop, and
mutable state written from a CoreAudio thread. There is no latency to win: the
poll is already free at the only cadence that matters.

**Why not reopen speculatively, while idle?** It would hide the 190 ms from the
first utterance after a switch. It also needs a timer and a background task to
own it, which is real structure for a saving David sees a few times a day.

## Development

```bash
uv run --group dev pytest
uv run --group dev ruff check .
uv run --group dev ruff format .
```

The tests cover config loading, voice and slot assignment, spoken labels, queue
ordering, resampling, and output-device selection. None of them load the model.

`test_device.py` is the one file that touches real hardware, and has to: every
claim the device-following design rests on is a claim about what CoreAudio and
PortAudio actually do, and a mock would only restate the assumption. It opens a
stream, times the HAL probe, and asserts that a re-scan invalidates an open
stream. Those tests skip off macOS.


## Troubleshooting

**Daemon not reachable.** Run `launchctl list | grep maggid`, then read
`/tmp/maggid.err`. Claude Code reconnects on its next tool call. Only the
utterance in flight is lost.

**No audio from the daemon.** Confirm it is a LaunchAgent, not a LaunchDaemon.
Only a LaunchAgent can reach the audio device. `sounddevice` errors appear in
`/tmp/maggid.err`.

**"Audio prompt must be longer than 5 seconds".** Your `ref_audio` clip is too
short. The server checks this before it calls the model.

**First call slow.** The model is loading, which takes about 20 s. The daemon
warms at startup. stdio mode loads on the first call.

## License

MIT
