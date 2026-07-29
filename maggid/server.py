"""MCP server: tools, lifecycle, and command line."""

import argparse
import asyncio
import dataclasses
import logging
import pathlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastmcp import Context, FastMCP
from rich.logging import RichHandler

from . import identity, synth
from .config import Config, validate_ref_audio
from .playback import SAMPLE_RATE, SPEED, PlaybackQueue

__all__ = ["init", "main"]

logger = logging.getLogger(__name__)

# One process serving every session means one model in memory instead of one per
# session, one warmup instead of N, and — the point — a single playback queue, so
# an urgent channel in one workspace preempts narration in another. Loopback
# only: an open port here lets any local process make the machine talk.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


@dataclasses.dataclass(frozen=True)
class Runtime:
    """Live server state. Built at startup, torn down at exit."""

    config: Config
    playback: PlaybackQueue
    voices: identity.VoiceRegistry
    slots: identity.SessionSlots


_runtime: Runtime | None = None
_warm_at_start = False


@asynccontextmanager
async def lifespan(_server: FastMCP) -> AsyncIterator[dict]:
    """Build the runtime, then tear it down."""
    global _runtime
    playback = PlaybackQueue()
    playback.start()
    _runtime = Runtime(
        config=Config.read(),
        playback=playback,
        voices=identity.VoiceRegistry(),
        slots=identity.SessionSlots(),
    )
    # Warm off the critical path. The load takes about 20 s. A long-lived daemon
    # should pay that at login, not charge it to the first notification. The
    # transport already listens, so the handshake stays instant either way.
    warm = asyncio.create_task(synth.preload()) if _warm_at_start else None
    try:
        yield {}
    finally:
        if warm is not None:
            warm.cancel()
        await playback.stop()
        _runtime = None


mcp = FastMCP("Maggid", lifespan=lifespan)


@mcp.tool()
async def notify(message: str, ctx: Context, channel: str | None = None) -> str:
    """Speak a short task-completion notification.

    :param message: Notification text, such as "Build finished".
    :param channel: Named channel. Sets the queue priority.
    """
    spoken = await _say(message, ctx, channel)
    if not spoken.queued:
        return f"Dropped (backlog full): {message}"
    return f"Notified{f' as {spoken.label}' if spoken.label else ''}: {message}"


@mcp.tool()
async def speak(
    text: str,
    ctx: Context,
    ref_audio: str | None = None,
    channel: str | None = None,
) -> str:
    """Speak longer narration, optionally in a cloned voice.

    :param text: Text to speak.
    :param ref_audio: WAV clip to clone. Overrides the workspace voice.
    :param channel: Named channel. Sets the queue priority.
    """
    spoken = await _say(text, ctx, channel, ref_audio)
    if not spoken.queued:
        return f"Dropped (backlog full): {len(text)} chars"
    return f"Spoke {len(text)} chars in {spoken.seconds:.1f}s (voice={spoken.voice})"


@mcp.tool()
async def interrupt() -> str:
    """Stop what is playing now and discard everything still queued."""
    dropped = _live().playback.clear()
    return f"Interrupted playback, discarded {dropped} queued item(s)."


def main(argv: list[str] | None = None) -> None:
    """Run the server. Argument parsing lives here, so the console script sees it."""
    global _warm_at_start
    _configure_logging()
    args = _parse_args(argv)

    if args.command == "init":
        init()
        return

    # Warm eagerly as the shared daemon. Warm lazily as a per-session stdio
    # child, which may never be asked to speak at all.
    _warm_at_start = args.warm if args.warm is not None else args.transport == "http"

    if args.transport == "http":
        logger.info(
            "Starting the shared maggid daemon on %s:%d ...", args.host, args.port
        )
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        logger.info("Starting maggid on stdio ...")
        mcp.run(transport="stdio")


def init() -> None:
    """Pre-download the model and write a default config."""
    _configure_logging()
    Config.write_default()
    logger.info("Preloading the model for a faster first response ...")
    synth.load_model()


@dataclasses.dataclass(frozen=True, slots=True)
class Spoken:
    """What was said, and whether it reached the queue."""

    label: str
    voice: str
    seconds: float
    queued: bool


async def _say(
    text: str, ctx: Context, channel: str | None, ref_audio: str | None = None
) -> Spoken:
    """Synthesize one utterance and queue it. Shared by notify and speak."""
    runtime = _live()
    priority = runtime.config.priority(channel)
    if ref_audio is not None:
        clip, label = validate_ref_audio(pathlib.Path(ref_audio)), ""
    else:
        clip, label = await identity.speaker(
            ctx, runtime.config, runtime.voices, runtime.slots
        )
    audio = await synth.synthesize(
        identity.announce(label, text, runtime.config.prefix), ref_audio=clip
    )
    return Spoken(
        label=label,
        voice=label or (clip.stem if clip else "built-in"),
        seconds=len(audio) / SAMPLE_RATE / SPEED,
        queued=runtime.playback.enqueue(audio, priority),
    )


def _live() -> Runtime:
    """The running server state, or a legible error."""
    if _runtime is None:
        raise RuntimeError("The server is starting or shutting down. Try again.")
    return _runtime


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="maggid",
        description="TTS MCP server for Claude Code. Chatterbox Turbo on Apple "
        "Silicon.",
        epilog="Run without arguments for a per-session stdio server.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["init"],
        default=None,
        help="'init' pre-downloads the TTS model and writes a default config.",
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
        help="Load the model at startup instead of on the first call.",
    )
    warm.add_argument(
        "--no-warm", dest="warm", action="store_false", help="Always load lazily."
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
