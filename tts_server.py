"""TTS MCP Server for Claude Code notifications.

Provides speech synthesis via MLX-audio and Kokoro-82M on Apple Silicon.
Exposes a ``notify`` tool for task completion alerts, a ``speak`` tool for
general-purpose TTS with voice/speed control, and an ``interrupt`` tool to
cut off playback and discard the backlog.
"""

import argparse
import asyncio
import concurrent.futures
import dataclasses
import functools
import itertools
import json
import logging
import pathlib
import signal
import tempfile
import time
import tomllib
import urllib.parse
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import mlx.core as mx
import soundfile as sf
from fastmcp import Context, FastMCP
from mlx.nn.layers import Module
from mlx_audio.tts.utils import load_model as mlx_load_model
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)


# Chatterbox Turbo, 8-bit. Measured on an M5 Max against the alternatives:
# 190 ms for a short phrase vs 330 ms at fp16 and 660 ms for base Chatterbox.
# 4-bit ties 8-bit on speed and memory, so there is nothing to buy by going
# lower. Kokoro was 37 ms but capped us at a handful of distinguishable voices;
# Turbo clones from a reference clip instead, which is what lets each workspace
# sound different.
SAMPLE_RATE = 24000
SPEED = 1.2
HUGGINGFACE_REPO = "mlx-community/Chatterbox-Turbo-TTS-8bit"

# Chatterbox ignores a speed argument, so rate lives at playback via `afplay -r`.
# That decouples it from synthesis: changing speed no longer re-runs the model.
MIN_SPEED, MAX_SPEED = 0.5, 2.0

# Chatterbox rejects shorter reference clips outright ("Audio prompt must be
# longer than 5 seconds!"), so catch it up front with a legible message.
MIN_REF_SECONDS = 5.0

# Backlog cap. Past this, a chatty fleet of agents is queueing audio that will
# still be playing long after the work it describes is done, so we drop instead.
MAX_BACKLOG = 32

CHANNELS_CONFIG = pathlib.Path.home() / ".config" / "tts-mcp-server" / "channels.toml"

# Shared-daemon defaults. One process serving every session means one model in
# memory instead of one per session, one warmup instead of N, and — the point —
# a single playback queue, so an urgent channel in one workspace preempts
# narration in another. Loopback only: an open port here lets any local process
# make the machine talk.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


# ---------------------------------------------------------------------------
# Channels — named voice/speed/priority profiles
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Channel:
    speed: float
    priority: int


DEFAULT_CHANNELS: dict[str, Channel] = {
    "notify": Channel(speed=1.2, priority=10),
    "permission": Channel(speed=1.0, priority=1),
    "question": Channel(speed=1.0, priority=2),
    "narrate": Channel(speed=1.3, priority=15),
}


def load_config(
    path: pathlib.Path = CHANNELS_CONFIG,
) -> tuple[str | None, dict[str, Channel]]:
    """Load reference voice and channel config from TOML, falling back to defaults.

    ``ref_audio`` is a path to a WAV clip to clone; None uses Chatterbox's
    built-in voice.

    :returns: (ref_audio, channels)
    """
    ref_audio = None
    channels = dict(DEFAULT_CHANNELS)
    if not path.is_file():
        logger.info("No config at %s — using defaults.", path)
        return ref_audio, channels
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    ref_audio = raw.get("ref_audio")
    if ref_audio is not None:
        try:
            _validate_ref_audio(ref_audio)
        except ValueError as e:
            logger.warning("%s — falling back to built-in voice.", e)
            ref_audio = None

    global _prefix_enabled, _voice_overrides
    _prefix_enabled = bool(raw.get("prefix", True))
    _voice_overrides = dict(raw.get("voices", {}))

    for name, overrides in raw.items():
        if name == "voices" or not isinstance(overrides, dict):
            continue
        base = DEFAULT_CHANNELS.get(name)
        channels[name] = Channel(
            speed=overrides.get("speed", base.speed if base else SPEED),
            priority=overrides.get("priority", base.priority if base else 10),
        )
    logger.info(
        "Loaded config from %s (ref_audio=%s, %d channel(s)).",
        path,
        ref_audio or "<built-in>",
        len(channels),
    )
    return ref_audio, channels


def _resolve(
    channels: dict[str, Channel],
    channel: str | None,
    speed: float | None,
) -> tuple[float, int]:
    """Merge explicit speed over channel defaults.

    :returns: (speed, priority)
    """
    if channel is not None:
        ch = channels.get(channel)
        if ch is None:
            raise ValueError(
                f"Unknown channel {channel!r}. Available: {', '.join(sorted(channels))}"
            )
        return (
            speed if speed is not None else ch.speed,
            ch.priority,
        )
    return (
        speed if speed is not None else SPEED,
        10,
    )


_ref_audio: str | None = None
_channels: dict[str, Channel] = {}
_warm_on_start: bool = False


# ---------------------------------------------------------------------------
# Identity — which workspace is speaking
# ---------------------------------------------------------------------------

VOICES_DIR = pathlib.Path.home() / ".config" / "tts-mcp-server" / "voices" / "refs"
ASSIGNMENTS = pathlib.Path.home() / ".config" / "tts-mcp-server" / "assignments.json"

# Auditioned set: every voice rated 8+, then the exact maximin subset by
# speaker-embedding cosine. Ordered best-rated first, which is also assignment
# order, so the first workspace seen gets the best voice.
#
# Nine voices is the ceiling and it is a soft one — the most separated nine of
# any selection still contain a pair at 0.86 cosine, because Chatterbox pulls
# everything it clones toward its own character. Voice alone reliably carries
# maybe four or five workspaces, which is why the spoken label does the real
# work and the voice reinforces it.
VOICE_POOL = (
    "af_heart",
    "af_jessica",
    "af_sarah",
    "am_liam",
    "bf_isabella",
    "am_fenrir",
    "am_puck",
    "bf_alice",
    "bm_daniel",
)


class VoiceRegistry:
    """Assigns a stable voice per workspace, surviving daemon restarts.

    Keyed on the workspace root rather than the MCP session id: a session id
    rotates whenever the client reconnects, which would reshuffle every voice
    the first time the daemon bounced.
    """

    def __init__(self, path: pathlib.Path = ASSIGNMENTS) -> None:
        self._path = path
        self._map: dict[str, str] = {}
        try:
            self._map = json.loads(path.read_text())
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            logger.warning("Could not read %s — starting fresh.", path)

    def voice(self, root: str) -> str:
        """Preset assigned to this workspace, allocating one on first sight."""
        if root not in self._map:
            used = set(self._map.values())
            free = [v for v in VOICE_POOL if v not in used]
            # Past nine workspaces the pool wraps. Deterministic so the
            # assignment is at least stable, but two roots now share a voice
            # and only the spoken label tells them apart.
            self._map[root] = (
                free[0] if free else VOICE_POOL[len(self._map) % len(VOICE_POOL)]
            )
            logger.info("Assigned voice %s to %s", self._map[root], root)
            self._save()
        return self._map[root]

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._map, indent=2, sort_keys=True))
        except OSError:
            logger.warning("Could not persist assignments to %s", self._path)


_registry = VoiceRegistry()
_voice_overrides: dict[str, str] = {}
_prefix_enabled: bool = True


async def identity(ctx: Context) -> tuple[str | None, str]:
    """Resolve the caller's workspace into a voice and a spoken label.

    :returns: (ref_audio path or None, label — empty when unidentifiable)
    """
    root = await workspace_root(ctx)
    if root is None:
        return _ref_audio, ""
    label = pathlib.PurePath(root).name or "workspace"
    preset = _voice_overrides.get(label) or _registry.voice(root)
    clip = VOICES_DIR / f"{preset}.wav"
    if not clip.is_file():
        logger.warning("Voice clip %s missing — using configured default.", clip)
        return _ref_audio, label
    return str(clip), label


async def workspace_root(ctx: Context) -> str | None:
    """Filesystem path the client advertises as its root, if it advertises one."""
    try:
        roots = await ctx.list_roots()
    except Exception as e:  # noqa: BLE001 - identity is best-effort
        # Clients need not support roots, and a transport hiccup here should
        # cost the label, not the notification.
        logger.debug("No workspace root available: %s", e)
        return None
    if not roots:
        return None
    uri = str(roots[0].uri)
    if not uri.startswith("file://"):
        return uri or None
    return urllib.parse.unquote(urllib.parse.urlparse(uri).path) or None


def announce(label: str, text: str) -> str:
    """Prepend the workspace label so the listener knows who is talking.

    A period rather than a dash: the engine reads it as a pause, where a dash
    is either swallowed or spoken.
    """
    if not label or not _prefix_enabled:
        return text
    return f"{label}. {text}"


# ---------------------------------------------------------------------------
# Playback queue — serializes audio output from concurrent tool calls
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, order=True)
class PlaybackItem:
    """Priority queue entry. Lower priority = more urgent (heapq convention)."""

    priority: int
    seq: int
    audio: mx.array = dataclasses.field(compare=False)
    speed: float = dataclasses.field(default=1.0, compare=False)


class PlaybackQueue:
    """Async priority queue with a background worker that plays audio sequentially."""

    def __init__(self, maxsize: int = MAX_BACKLOG) -> None:
        self._counter = itertools.count()
        self._queue: asyncio.PriorityQueue[PlaybackItem] = asyncio.PriorityQueue(
            maxsize=maxsize
        )
        self._worker: asyncio.Task[None] | None = None
        self._current: asyncio.subprocess.Process | None = None

    def start(self) -> None:
        """Start the background drain task."""
        self._worker = asyncio.create_task(self._drain())
        logger.info("Playback worker started.")

    async def stop(self) -> None:
        """Cancel the worker and clean up."""
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        logger.info("Playback worker stopped.")

    def enqueue(self, audio: mx.array, priority: int = 10, speed: float = 1.0) -> bool:
        """Add audio to the queue. Non-blocking.

        :returns: False if the backlog is saturated and the audio was dropped.
        """
        try:
            self._queue.put_nowait(
                PlaybackItem(priority, next(self._counter), audio, speed)
            )
        except asyncio.QueueFull:
            logger.warning("Backlog full (%d) — dropping audio.", self._queue.maxsize)
            return False
        return True

    def clear(self) -> int:
        """Discard the backlog and stop whatever is playing.

        :returns: Number of queued items dropped, excluding the one playing.
        """
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._queue.task_done()
            dropped += 1
        if self._current is not None and self._current.returncode is None:
            self._current.terminate()
        return dropped

    async def _drain(self) -> None:
        """Pull items and play them one at a time."""
        while True:
            item = await self._queue.get()
            try:
                await self._play(item.audio, item.speed)
            except Exception:
                logger.exception("Playback failed")
            finally:
                self._queue.task_done()

    async def _play(self, audio: mx.array, speed: float = 1.0) -> None:
        """Write audio to a temp WAV and play via afplay at the given rate."""
        fd, name = tempfile.mkstemp(suffix=".wav", prefix="tts_")
        path = pathlib.Path(name)
        try:
            await asyncio.to_thread(_write_wav, fd, audio)
            proc = await asyncio.create_subprocess_exec(
                "afplay",
                "-r",
                str(speed),
                str(path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            self._current = proc
            _, stderr = await proc.communicate()
            # -SIGTERM is how interrupt() stops playback; not a failure.
            if proc.returncode not in (0, -signal.SIGTERM):
                raise RuntimeError(f"afplay failed: {stderr.decode()}")
        finally:
            self._current = None
            path.unlink(missing_ok=True)


_playback: PlaybackQueue | None = None


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_server: FastMCP) -> AsyncIterator[dict]:
    """Start/stop the playback worker with the server lifecycle."""
    global _playback, _ref_audio, _channels
    _ref_audio, _channels = load_config()
    _playback = PlaybackQueue()
    _playback.start()
    # Warm the model off the critical path. Loading takes ~20 s, which a
    # long-lived daemon should absorb at login rather than charging to whoever
    # sends the first notification. The transport is already listening, so the
    # handshake stays instant either way.
    warm = asyncio.create_task(_warmup()) if _warm_on_start else None
    try:
        yield {}
    finally:
        if warm is not None:
            warm.cancel()
        await _playback.stop()
        _playback = None


async def _warmup() -> None:
    """Load the model in the synthesis thread, logging rather than raising."""
    try:
        t0 = time.monotonic()
        await asyncio.get_running_loop().run_in_executor(_synthesis, load_model)
        logger.info("Warmup finished in %.1fs.", time.monotonic() - t0)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Warmup failed; first call will pay the load cost")


mcp = FastMCP("TTS Notification Server", lifespan=lifespan)


@mcp.tool()
async def notify(
    message: str,
    ctx: Context,
    speed: float | None = None,
    channel: str | None = None,
) -> str:
    """Speak a short task-completion notification.

    :param message: Notification text (e.g. "Build finished").
    :param speed: Playback speed multiplier, 0.5–2.0.
    :param channel: Named channel for speed/priority defaults.
    """
    playback = _require_playback()
    speed_, priority = _resolve(_channels, channel, speed)
    _validate_speed(speed_)
    ref, label = await identity(ctx)
    audio = await synthesize(announce(label, message), ref_audio=ref)
    if not playback.enqueue(audio, priority=priority, speed=speed_):
        return f"Dropped (backlog full): {message}"
    return f"Notified{f' as {label}' if label else ''}: {message}"


@mcp.tool()
async def speak(
    text: str,
    ctx: Context,
    ref_audio: str | None = None,
    speed: float | None = None,
    channel: str | None = None,
) -> str:
    """Generate and play speech, optionally cloning a reference voice.

    :param text: Text to speak.
    :param ref_audio: Path to a WAV clip to clone (overrides the workspace voice).
    :param speed: Playback speed multiplier, 0.5–2.0.
    :param channel: Named channel for speed/priority defaults.
    """
    playback = _require_playback()
    if ref_audio is not None:
        _validate_ref_audio(ref_audio)
        ref, label = ref_audio, ""
    else:
        ref, label = await identity(ctx)
    speed_, priority = _resolve(_channels, channel, speed)
    _validate_speed(speed_)
    audio = await synthesize(announce(label, text), ref_audio=ref)
    if not playback.enqueue(audio, priority=priority, speed=speed_):
        return f"Dropped (backlog full): {len(text)} chars"
    dur = len(audio) / SAMPLE_RATE / speed_
    who = label or pathlib.PurePath(ref).stem if ref else "built-in"
    return f"Spoke {len(text)} chars in {dur:.1f}s (voice={who}, speed={speed_}x)"


@mcp.tool()
async def interrupt() -> str:
    """Stop what is playing now and discard everything still queued."""
    dropped = _require_playback().clear()
    return f"Interrupted playback, discarded {dropped} queued item(s)."


# MLX's Metal command buffer is not safe for concurrent evaluation — two
# threads calling generate() at once aborts the process with
# "Completed handler provided after commit call". The lru_cache on load_model
# is likewise not atomic, so a concurrent cold start races inside transformers.
# One synthesis thread fixes both, and costs nothing: inference is ~60 ms.
_synthesis = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="tts-synth"
)


async def synthesize(text: str, ref_audio: str | None = None) -> mx.array:
    """Run TTS inference off the event loop, serialized against other callers."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _synthesis, functools.partial(generate, text, ref_audio=ref_audio)
    )


@functools.lru_cache
def load_model(path: str = HUGGINGFACE_REPO) -> Module:
    logger.info("Loading model %s ...", path)
    model = mlx_load_model(path)
    # Warmup: compile Metal shaders so the first real call is fast.
    list(model.generate("warmup"))
    # Stash the built-in voice before any clone overwrites the slot, so a call
    # with no ref_audio doesn't inherit whichever voice was cloned last.
    model._builtin_conds = model._conds
    logger.info("Model loaded and warmed up.")
    return model


@functools.lru_cache
def conditionals(ref_audio: str):
    """Encode a reference clip once and keep it.

    Passing ``ref_audio`` to generate() re-encodes the clip on every call —
    950 ms per utterance instead of 190 ms. Encoding once per voice and swapping
    the model's slot gets that back. Safe because all synthesis is serialized on
    a single thread, so no two callers touch the slot at once.
    """
    model = load_model()
    model.prepare_conditionals(ref_audio)
    return model._conds


def generate(text: str, ref_audio: str | None = None) -> mx.array:
    """Run TTS inference, return raw audio array.

    Runs in the synthesis thread, so the lazy MLX graph is forced here —
    otherwise evaluation would land back on the caller's thread. Speed is not a
    parameter: Chatterbox ignores it, and playback applies it instead.
    """
    model = load_model()
    model._conds = (
        conditionals(ref_audio) if ref_audio is not None else model._builtin_conds
    )
    chunks = [r.audio for r in model.generate(text=text)]
    if not chunks:
        raise RuntimeError("No audio generated")
    audio = mx.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    mx.eval(audio)
    return audio


def _write_wav(fd: int, audio: mx.array) -> None:
    """Write audio as WAV to an open descriptor, closing it afterwards."""
    with open(fd, "wb") as f:
        sf.write(f, audio, SAMPLE_RATE, format="WAV")


def _require_playback() -> PlaybackQueue:
    """Fetch the live queue, or fail with something legible."""
    if _playback is None:
        raise RuntimeError(
            "Playback queue is not running; the server is starting or shutting down."
        )
    return _playback


def _validate_ref_audio(path: str) -> None:
    """Reject a reference clip before it fails deep inside the model."""
    p = pathlib.Path(path)
    if not p.is_file():
        raise ValueError(f"ref_audio not found: {path}")
    info = sf.info(str(p))
    if info.duration <= MIN_REF_SECONDS:
        raise ValueError(
            f"ref_audio must be longer than {MIN_REF_SECONDS}s, "
            f"got {info.duration:.1f}s: {path}"
        )


def _validate_speed(speed: float) -> None:
    if not MIN_SPEED <= speed <= MAX_SPEED:
        raise ValueError(f"Speed must be {MIN_SPEED}–{MAX_SPEED}, got {speed}")


def main(argv: list[str] | None = None) -> None:
    """Run the server. Argument parsing lives here so the console script sees it."""
    global _warm_on_start
    args = _parse_args(argv)

    if args.command == "init":
        logger.info("Preloading model for faster first response...")
        init()
        return

    # Warm eagerly when we're the shared daemon, lazily when we're a per-session
    # stdio child that may never be asked to speak at all.
    _warm_on_start = args.warm if args.warm is not None else args.transport == "http"

    if args.transport == "http":
        logger.info("Starting shared TTS daemon on %s:%d ...", args.host, args.port)
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        logger.info("Starting TTS MCP server (stdio) ...")
        mcp.run(transport="stdio")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tts-mcp-server",
        description="TTS MCP server for Claude Code — Chatterbox Turbo on Apple Silicon.",
        epilog="Run without arguments for a per-session stdio server.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["init"],
        default=None,
        help="'init' to pre-download the TTS model and write a default config.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="'http' runs one shared daemon for every session (default: stdio).",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="HTTP bind address.")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="HTTP port to listen on."
    )
    warm = parser.add_mutually_exclusive_group()
    warm.add_argument(
        "--warm",
        dest="warm",
        action="store_true",
        default=None,
        help="Load the model at startup instead of on first call.",
    )
    warm.add_argument(
        "--no-warm", dest="warm", action="store_false", help="Always load lazily."
    )
    return parser.parse_args(argv)


def write_default_config(path: pathlib.Path = CHANNELS_CONFIG) -> bool:
    """Write default channels.toml if it doesn't exist.

    :returns: True if file was created, False if it already existed.
    """
    if path.is_file():
        logger.info("Config already exists at %s — skipping.", path)
        return False
    lines = [
        "# Speak the workspace name before each message, so you can tell which",
        "# agent is talking. Voice alone does not reliably carry more than a few.",
        "prefix = true",
        "",
        "# Fallback clip when the workspace can't be identified. Omit for the",
        "# built-in voice.",
        '# ref_audio = "/Users/you/.config/tts-mcp-server/voices/refs/af_heart.wav"',
        "",
        "# Pin a workspace to a voice. Keys are directory names; values are",
        "# presets under voices/refs/. Unpinned workspaces are assigned",
        "# automatically on first contact and remembered in assignments.json.",
        "# [voices]",
        '# spade = "af_heart"',
        "",
    ]
    for name, ch in DEFAULT_CHANNELS.items():
        lines += [f"[{name}]", f"speed = {ch.speed}", f"priority = {ch.priority}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    logger.info("Wrote default config to %s.", path)
    return True


def init():
    write_default_config()
    load_model()


if __name__ == "__main__":
    main()
