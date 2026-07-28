# tts-mcp-server

Local text-to-speech for Claude Code via [MCP](https://modelcontextprotocol.io/).
Runs [Chatterbox Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) on
Apple Silicon through [MLX-audio](https://github.com/Blaizzy/mlx-audio), so Claude
can speak notifications and narration aloud — with a distinct voice per workspace
when several agents are running at once.

**Requirements:** macOS on Apple Silicon, Python 3.12+.

## Setup

```bash
cd ~/projects/tts-mcp-server
uv venv && uv pip install -e .

# Pre-download the model (~3 GB) and write a default config:
uv run tts-mcp-server init
```

### Shared daemon (recommended)

One process serves every Claude Code session. This is the mode worth running if
you keep more than one or two sessions open.

```bash
sed "s|__VENV__|$PWD/.venv|g" launchd/com.tts-mcp-server.plist \
  > ~/Library/LaunchAgents/com.tts-mcp-server.plist
launchctl load ~/Library/LaunchAgents/com.tts-mcp-server.plist

claude mcp add --transport http --scope user tts http://127.0.0.1:8765/mcp
```

Why bother: with per-session servers, each session loads its own copy of the
model and owns its own speaker, so two agents talking at once produce overlapping
audio you can't parse. One daemon means one model in memory, one warmup, and a
single playback queue — so an urgent `permission` message in one workspace
preempts `narrate` in another instead of racing it.

The daemon warms the model at load, so the ~20 s startup is paid once at login
rather than charged to whichever agent speaks first.

### Per-session stdio (fallback)

```bash
claude mcp add --transport stdio --scope user tts -- \
  /path/to/tts-mcp-server/.venv/bin/tts-mcp-server
```

Simpler, no lifecycle to manage, and the server lives exactly as long as its
session. But each session pays its own ~3 GB and its own warmup, and audio from
concurrent sessions overlaps.

Verify either mode with `claude mcp list`.

## Tools

### `notify(message, speed?, channel?)`

Short task-completion alert. Uses the configured voice.

### `speak(text, ref_audio?, speed?, channel?)`

Longer narration, optionally in a cloned voice.

| Parameter   | Default      | Notes                                    |
| ----------- | ------------ | ---------------------------------------- |
| `text`      | _(required)_ | Any string                               |
| `ref_audio` | built-in     | Path to a WAV to clone; **must be >5 s** |
| `speed`     | `1.2`        | 0.5 – 2.0, applied at playback           |
| `channel`   | _(none)_     | See [channels](#channels)                |

### `interrupt()`

Stops what's playing and discards the backlog. Takes no arguments. Useful when
queued narration has been overtaken by events.

## Channels

Priority classes for the shared queue, so urgent speech jumps ahead of
narration already waiting. Defaults:

| Channel      | Priority    |
| ------------ | ----------- |
| `permission` | 1 (highest) |
| `question`   | 2           |
| `notify`     | 10          |
| `narrate`    | 15          |

Speed is uniform at 1.1. Per-channel rates were a Kokoro-era idea that never
earned its keep — the whole 1.0–1.3 range saved under half a second on a typical
notification. A channel's meaning is carried by priority and by how the message
is worded, not by how fast it is read.

Override in `~/.config/tts-mcp-server/channels.toml`:

```toml
# ref_audio = "/Users/you/.config/tts-mcp-server/voices/mine.wav"

[narrate]
priority = 20
```

The backlog is capped at 32 utterances; past that, new audio is dropped rather
than queued behind speech that will be stale by the time it plays.

## Per-workspace identity

With several agents running, the useful question is not "what did it say" but
"which one said it". The daemon answers that two ways at once.

**A spoken label.** Each message is prefixed with the workspace directory name,
so `notify("Tests passed")` from `~/projects/spade` is heard as *"spade. Tests
passed."* Disable with `prefix = false`.

**A distinct voice per workspace.** The server reads the client's advertised
root via MCP `roots`, and assigns a voice from a pool of nine on first contact.
Assignments persist in `~/.config/tts-mcp-server/assignments.json` and are keyed
on the root path, so they survive daemon restarts.

Pin one explicitly if you'd rather not take what you're given:

```toml
[voices]
spade = "bm_daniel"
```

**Why the label does the heavy lifting.** The nine pooled voices were picked by
rating every cloned candidate 1–10 and then solving for the subset with maximum
minimum pairwise separation (speaker-embedding cosine). Even so, the best
possible nine still contain a pair at 0.86 cosine — Chatterbox pulls everything
it clones toward its own character, compressing the whole set into 0.74–0.90.
Voice alone reliably distinguishes four or five workspaces, not nine. The label
is what actually scales; the voice reinforces it.

Past nine workspaces the pool wraps deterministically, and two projects share a
voice. Only the label tells them apart at that point.

## Voices

Chatterbox has no voice presets. It clones from a reference clip — which is what
lifts the ceiling — but it means you supply the clips.

Reference clips must be **longer than 5 seconds**. The easiest source that
involves no one's actual voice is to synthesize them from Kokoro's presets:

```python
import soundfile as sf, numpy as np, mlx.core as mx
from mlx_audio.tts.utils import load_model

k = load_model("mlx-community/Kokoro-82M-bf16")
text = "A sentence long enough to run past the five second minimum, plus another to be safe."
chunks = [r.audio for r in k.generate(text=text, voice="am_adam", speed=1.0, lang_code="a")]
audio = mx.concatenate(chunks) if len(chunks) > 1 else chunks[0]
mx.eval(audio)
sf.write("voices/am_adam.wav", np.asarray(audio), 24000)
```

Encoded conditionals are cached per clip, so a cloned utterance costs ~300 ms
rather than the ~950 ms it takes to re-encode the reference every call.

**On cloning real voices:** a reference clip is impersonation capability, whoever
it came from and whatever it was made for. Your own voice and synthesized clips
are unproblematic; a third party's needs their agreement. Note also that upstream
Chatterbox watermarks its output via `resemble-perth`, and the MLX port does not —
audio from this server carries no watermark.

## Performance

Measured on an M5 Max, short phrase ("Tests passed."), best of three warm runs:

| Model             | Latency | RTF   | Peak RSS |
| ----------------- | ------- | ----- | -------- |
| Kokoro-82M        | 37 ms   | 0.025 | 0.78 GB  |
| Chatterbox base   | 660 ms  | 0.412 | 3.64 GB  |
| Turbo fp16        | 330 ms  | 0.235 | 2.95 GB  |
| **Turbo 8-bit**   | 190 ms  | 0.164 | 2.98 GB  |
| Turbo 4-bit       | 200 ms  | 0.158 | 2.98 GB  |

8-bit is the default: it ties 4-bit on speed and memory, so there's nothing to
gain from quantizing further. RTF 0.164 means synthesis runs ~6× faster than
playback, so the speaker is the bottleneck, not the model — adding more agents
doesn't make synthesis the constraint.

## Architecture

```
N × Claude Code  ──http──>  one daemon  ──>  Chatterbox Turbo  ──>  afplay
                                 │
                        global priority queue
```

- **Serialized synthesis.** MLX's Metal command buffer aborts the process if two
  threads evaluate concurrently, so all inference runs on a single dedicated
  thread. Costs nothing at ~190 ms per utterance.
- **Persistent audio stream.** One `sounddevice` output stream stays open for
  the process lifetime. Shelling out to `afplay` cost ~1.0 s of process and
  CoreAudio startup per utterance — five times the synthesis time, paid on every
  notification. Removing it also made `interrupt` near-instant (~50 ms).
- **Speed at playback.** Chatterbox ignores a speed argument, so rate is applied
  by resampling before the device write, not by re-running inference.

## Troubleshooting

**Daemon not reachable:** `launchctl list | grep tts`, then check
`/tmp/tts-mcp-server.err`. Claude Code reconnects automatically once it's back;
only the utterance in flight is lost.

**Must be a LaunchAgent, not a LaunchDaemon.** LaunchDaemons run outside your GUI
session and can't reach the audio device, so `afplay` fails silently.

**`Audio prompt must be longer than 5 seconds`:** your `ref_audio` clip is too
short. The server checks this up front.

**First call slow:** the model is loading (~20 s). The daemon warms at startup;
stdio mode loads lazily on first call.

**No audio from the daemon:** confirm it is a LaunchAgent, not a LaunchDaemon —
only the former can reach the audio device. `sounddevice` errors surface in
`/tmp/tts-mcp-server.err`.

## License

MIT
