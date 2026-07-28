"""TTS MCP Server for Claude Code notifications.

Provides speech synthesis via MLX-audio and Kokoro-82M on Apple Silicon.
Exposes a ``notify`` tool for task completion alerts, a ``speak`` tool for
general-purpose TTS with voice/speed control, and an ``interrupt`` tool to
cut off playback and discard the backlog.
"""

import argparse
import asyncio
import dataclasses
import functools
import itertools
import logging
import pathlib
import signal
import tempfile
import tomllib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import mlx.core as mx
import soundfile as sf
from fastmcp import FastMCP
from mlx.nn.layers import Module
from mlx_audio.tts.utils import load_model as mlx_load_model
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)


# NOTE: ATM highest quality Kokoro voice, cf. https://tinyurl.com/32hcr3b7
DEFAULT_VOICE = "af_heart"
SAMPLE_RATE = 24000
SPEED = 1.2
HUGGINGFACE_REPO = "mlx-community/Kokoro-82M-bf16"

# Backlog cap. Past this, a chatty fleet of agents is queueing audio that will
# still be playing long after the work it describes is done, so we drop instead.
MAX_BACKLOG = 32

CHANNELS_CONFIG = pathlib.Path.home() / ".config" / "tts-mcp-server" / "channels.toml"


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


def load_config(path: pathlib.Path = CHANNELS_CONFIG) -> tuple[str, dict[str, Channel]]:
    """Load voice and channel config from TOML, falling back to defaults.

    :returns: (voice, channels)
    """
    voice = DEFAULT_VOICE
    channels = dict(DEFAULT_CHANNELS)
    if not path.is_file():
        logger.info("No config at %s — using defaults.", path)
        return voice, channels
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    voice = raw.get("voice", DEFAULT_VOICE)
    for name, overrides in raw.items():
        if not isinstance(overrides, dict):
            continue
        base = DEFAULT_CHANNELS.get(name)
        channels[name] = Channel(
            speed=overrides.get("speed", base.speed if base else SPEED),
            priority=overrides.get("priority", base.priority if base else 10),
        )
    logger.info(
        "Loaded config from %s (voice=%s, %d channel(s)).", path, voice, len(channels)
    )
    return voice, channels


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


_voice: str = DEFAULT_VOICE
_channels: dict[str, Channel] = {}


# ---------------------------------------------------------------------------
# Playback queue — serializes audio output from concurrent tool calls
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, order=True)
class PlaybackItem:
    """Priority queue entry. Lower priority = more urgent (heapq convention)."""

    priority: int
    seq: int
    audio: mx.array = dataclasses.field(compare=False)


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

    def enqueue(self, audio: mx.array, priority: int = 10) -> bool:
        """Add audio to the queue. Non-blocking.

        :returns: False if the backlog is saturated and the audio was dropped.
        """
        try:
            self._queue.put_nowait(PlaybackItem(priority, next(self._counter), audio))
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
                await self._play(item.audio)
            except Exception:
                logger.exception("Playback failed")
            finally:
                self._queue.task_done()

    async def _play(self, audio: mx.array) -> None:
        """Write audio to a temp WAV and play via afplay."""
        fd, name = tempfile.mkstemp(suffix=".wav", prefix="tts_")
        path = pathlib.Path(name)
        try:
            await asyncio.to_thread(_write_wav, fd, audio)
            proc = await asyncio.create_subprocess_exec(
                "afplay",
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
    global _playback, _voice, _channels
    _voice, _channels = load_config()
    _playback = PlaybackQueue()
    _playback.start()
    try:
        yield {}
    finally:
        await _playback.stop()
        _playback = None


mcp = FastMCP("TTS Notification Server", lifespan=lifespan)


@mcp.tool()
async def notify(
    message: str,
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
    audio = await asyncio.to_thread(generate, message, voice=_voice, speed=speed_)
    if not playback.enqueue(audio, priority=priority):
        return f"Dropped (backlog full): {message}"
    return f"Notified: {message}"


@mcp.tool()
async def speak(
    text: str,
    voice: str | None = None,
    speed: float | None = None,
    channel: str | None = None,
) -> str:
    """Generate and play speech with voice and speed control.

    :param text: Text to speak.
    :param voice: Kokoro voice preset (overrides instance default).
    :param speed: Playback speed multiplier, 0.5–2.0.
    :param channel: Named channel for speed/priority defaults.
    """
    playback = _require_playback()
    voice_ = voice if voice is not None else _voice
    speed_, priority = _resolve(_channels, channel, speed)
    _validate_speed(speed_)
    audio = await asyncio.to_thread(generate, text, voice=voice_, speed=speed_)
    if not playback.enqueue(audio, priority=priority):
        return f"Dropped (backlog full): {len(text)} chars"
    dur = len(audio) / SAMPLE_RATE
    return f"Spoke {len(text)} chars in {dur:.1f}s (voice={voice_}, speed={speed_}x)"


@mcp.tool()
async def interrupt() -> str:
    """Stop what is playing now and discard everything still queued."""
    dropped = _require_playback().clear()
    return f"Interrupted playback, discarded {dropped} queued item(s)."


@functools.lru_cache
def load_model(path: str = HUGGINGFACE_REPO) -> Module:
    logger.info("Loading model %s ...", path)
    model = mlx_load_model(path)
    # Warmup: compile Metal shaders so the first real call is fast.
    list(model.generate("warmup"))
    logger.info("Model loaded and warmed up.")
    return model


def generate(text: str, voice: str = DEFAULT_VOICE, speed: float = SPEED) -> mx.array:
    """Run TTS inference, return raw audio array.

    Runs off the event loop via ``asyncio.to_thread``, so the lazy MLX graph is
    forced here — otherwise evaluation would land back on the caller's thread.
    """
    model = load_model()
    chunks = [
        r.audio
        for r in model.generate(text=text, voice=voice, speed=speed, lang_code="a")
    ]
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


def _validate_speed(speed: float) -> None:
    if not 0.5 <= speed <= 2.0:
        raise ValueError(f"Speed must be 0.5–2.0, got {speed}")


def main():
    logger.info("Starting TTS MCP server ...")
    # Start the MCP transport immediately so the handshake succeeds,
    # then warm up the model lazily on first tool call.
    mcp.run(transport="stdio")


def write_default_config(path: pathlib.Path = CHANNELS_CONFIG) -> bool:
    """Write default channels.toml if it doesn't exist.

    :returns: True if file was created, False if it already existed.
    """
    if path.is_file():
        logger.info("Config already exists at %s — skipping.", path)
        return False
    lines = [f'voice = "{DEFAULT_VOICE}"', ""]
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
    parser = argparse.ArgumentParser(
        description="TTS MCP server for Claude Code — Kokoro-82M on Apple Silicon.",
        epilog="Run without arguments to start the MCP server.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["init"],
        default=None,
        help="'init' to pre-download the TTS model (~200 MB).",
    )
    args = parser.parse_args()

    if args.command == "init":
        logger.info("Preloading model for faster first response...")
        init()
    else:
        main()
