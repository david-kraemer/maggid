"""Configuration: the reference voice, channel priorities, and the default file."""

import dataclasses
import logging
import pathlib
import tomllib
from collections.abc import Mapping
from typing import Self

import soundfile

from .paths import StrPath, resolve_path, writable_path

logger = logging.getLogger(__name__)

CONFIG_DIR = pathlib.Path.home() / ".config" / "maggid"
CONFIG_FILE = CONFIG_DIR / "channels.toml"


@dataclasses.dataclass(frozen=True)
class Config:
    """Everything the TOML file controls."""

    ref_audio: pathlib.Path | None = None
    prefix: bool = True
    voices: Mapping[str, str] = dataclasses.field(default_factory=dict)
    priorities: Mapping[str, int] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_PRIORITIES)
    )

    @classmethod
    def read(cls, path: StrPath = CONFIG_FILE) -> Self:
        """Read the TOML file. A missing or partial file falls back to defaults."""
        path = resolve_path(path)
        if not path.is_file():
            logger.info("No config at %s. Using defaults.", path)
            return cls()
        raw = tomllib.loads(path.read_text())
        # A table with a priority key is a channel. The [voices] table has none, so it
        # needs no special case.
        priorities = dict(DEFAULT_PRIORITIES) | {
            name: table["priority"]
            for name, table in raw.items()
            if isinstance(table, dict) and "priority" in table
        }
        config = cls(
            ref_audio=usable_ref(raw.get("ref_audio")),
            prefix=bool(raw.get("prefix", True)),
            voices=dict(raw.get("voices", {})),
            priorities=priorities,
        )
        logger.info(
            "Loaded %s. Voice: %s. Channels: %d.",
            path,
            config.ref_audio or "built-in",
            len(priorities),
        )
        return config

    def priority(self, channel: str | None) -> int:
        """Queue priority for a channel. Lower plays first."""
        name = FALLBACK_CHANNEL if channel is None else channel
        try:
            return self.priorities[name]
        except KeyError:
            raise ValueError(
                f"Unknown channel {name!r}. "
                f"Available: {', '.join(sorted(self.priorities))}"
            ) from None

    @classmethod
    def write_default(cls, path: StrPath = CONFIG_FILE) -> bool:
        """Write the default TOML file if none exists.

        :returns: True if this call created the file.
        """
        path = writable_path(path)
        if path.is_file():
            logger.info("Config exists at %s. Skipped.", path)
            return False
        path.write_text(CONFIG_TEMPLATE)
        logger.info("Wrote %s.", path)
        return True


@dataclasses.dataclass(frozen=True, slots=True)
class Channel:
    """A priority class for the shared queue."""

    name: str
    priority: int
    purpose: str


# Urgent speech goes ahead of narration that already waits. A lower number plays
# first. DEFAULT_PRIORITIES and the generated config file both come from this table,
# so a new channel is one new row.
CHANNELS: list[Channel] = [
    Channel("permission", 1, "Blocks until you answer."),
    Channel("question", 2, "Answer whenever you like."),
    Channel("notify", 10, "Stage transitions."),
    Channel("narrate", 15, "Reasoning aloud."),
]
DEFAULT_PRIORITIES: Mapping[str, int] = {c.name: c.priority for c in CHANNELS}

# A message with no channel gets this priority. A test asserts the row exists.
FALLBACK_CHANNEL = "notify"


def usable_ref(path: StrPath | None) -> pathlib.Path | None:
    """The clip path if Chatterbox can clone it, else None."""
    if path is None:
        return None
    try:
        return validate_ref_audio(path)
    except ValueError as e:
        logger.warning("%s. Using the built-in voice.", e)
        return None


def validate_ref_audio(path: StrPath) -> pathlib.Path:
    """The clip path. Raises if the clip is missing or too short.

    The check happens here, not deep inside the model.
    """
    # Chatterbox refuses a shorter clip: "Audio prompt must be longer than 5 seconds!".
    min_ref_seconds = 5.0

    path = resolve_path(path)
    if not path.is_file():
        raise ValueError(f"ref_audio not found: {path}")
    if (seconds := soundfile.info(path).duration) <= min_ref_seconds:
        raise ValueError(
            f"ref_audio must be longer than {min_ref_seconds}s, "
            f"got {seconds:.1f}s: {path}"
        )
    return path


_CONFIG_PREAMBLE = """\
# Speak the workspace name before each message, so you know which agent talks.
# Voice alone separates only four or five workspaces.
prefix = true

# Fallback clip for an unidentified workspace. Omit it for the built-in voice.
# ref_audio = "/Users/you/.config/maggid/voices/refs/af_heart.wav"

# Pin a workspace to a voice. Keys are directory names. Values are presets in
# voices/refs/. An unpinned workspace gets a voice on first contact, kept in
# assignments.json.
# [voices]
# spade = "af_heart"

# Queue priority per channel. Lower plays first.
"""

# Emitted from CHANNELS, not written beside it. The file cannot document a priority
# that the code does not use.
CONFIG_TEMPLATE = (
    _CONFIG_PREAMBLE
    + "\n"
    + "\n".join(
        f"# {c.purpose}\n[{c.name}]\npriority = {c.priority}\n" for c in CHANNELS
    )
)
