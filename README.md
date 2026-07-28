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

Named profiles for speed and priority, so urgent speech jumps the queue.
Defaults:

| Channel      | Speed | Priority    |
| ------------ | ----- | ----------- |
| `permission` | 1.0   | 1 (highest) |
| `question`   | 1.0   | 2           |
| `notify`     | 1.2   | 10          |
| `narrate`    | 1.3   | 15          |

Override in `~/.config/tts-mcp-server/channels.toml`:

```toml
# ref_audio = "/Users/you/.config/tts-mcp-server/voices/mine.wav"

[narrate]
speed = 1.4
priority = 20
```

The backlog is capped at 32 utterances; past that, new audio is dropped rather
than queued behind speech that will be stale by the time it plays.

## Voices

Chatterbox has no voice presets. It clones from a reference clip, which is what
lifts the per-workspace-voice ceiling — but it means you supply the clips.

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
- **Speed at playback.** Chatterbox ignores a speed argument, so rate is applied
  via `afplay -r`. Changing speed doesn't re-run inference.
- **Temp files.** Audio is written to a temp WAV, played, then deleted.

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

**`afplay` not found:** you're not on macOS. Replace the `afplay` call in
`PlaybackQueue._play()` with your platform's player.

## License

MIT
