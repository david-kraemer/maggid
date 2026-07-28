"""Configuration: the reference voice, channel priorities, and the default file."""

import dataclasses
import logging
import pathlib
import tomllib
from collections.abc import Mapping

import soundfile

logger = logging.getLogger(__name__)

CONFIG_DIR = pathlib.Path.home() / ".config" / "maggid"
CONFIG_FILE = CONFIG_DIR / "channels.toml"

# Chatterbox refuses a shorter clip: "Audio prompt must be longer than 5
# seconds!". Check it here to give a legible error.
MIN_REF_SECONDS = 5.0

# Channels set queue priority. Lower plays first. A message with no channel gets
# the notify priority.
DEFAULT_PRIORITIES: Mapping[str, int] = {
    "permission": 1,
    "question": 2,
    "notify": 10,
    "narrate": 15,
}
FALLBACK_CHANNEL = "notify"


@dataclasses.dataclass(frozen=True)
class Config:
    """Everything the TOML file controls."""

    ref_audio: str | None = None
    prefix: bool = True
    voices: Mapping[str, str] = dataclasses.field(default_factory=dict)
    priorities: Mapping[str, int] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_PRIORITIES)
    )


def load_config(path: pathlib.Path = CONFIG_FILE) -> Config:
    """Read the TOML file. A missing or partial file falls back to defaults."""
    if not path.is_file():
        logger.info("No config at %s. Using defaults.", path)
        return Config()
    raw = tomllib.loads(path.read_text())
    # A table with a priority key is a channel. The [voices] table has none, so
    # it needs no special case.
    priorities = dict(DEFAULT_PRIORITIES) | {
        name: table["priority"]
        for name, table in raw.items()
        if isinstance(table, dict) and "priority" in table
    }
    config = Config(
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


def channel_priority(config: Config, channel: str | None) -> int:
    """Queue priority for a channel. Lower plays first."""
    name = FALLBACK_CHANNEL if channel is None else channel
    try:
        return config.priorities[name]
    except KeyError:
        raise ValueError(
            f"Unknown channel {name!r}. "
            f"Available: {', '.join(sorted(config.priorities))}"
        ) from None


def usable_ref(path: str | None) -> str | None:
    """The clip path if Chatterbox can clone it, else None."""
    if path is None:
        return None
    try:
        validate_ref_audio(path)
    except ValueError as e:
        logger.warning("%s. Using the built-in voice.", e)
        return None
    return path


def validate_ref_audio(path: str) -> None:
    """Raise if the clip is missing or too short.

    Catches the failure here rather than deep inside the model.
    """
    clip = pathlib.Path(path)
    if not clip.is_file():
        raise ValueError(f"ref_audio not found: {path}")
    seconds = soundfile.info(str(clip)).duration
    if seconds <= MIN_REF_SECONDS:
        raise ValueError(
            f"ref_audio must be longer than {MIN_REF_SECONDS}s, "
            f"got {seconds:.1f}s: {path}"
        )


# The channel tables here must match DEFAULT_PRIORITIES. A test asserts it.
CONFIG_TEMPLATE = """\
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
[permission]
priority = 1

[question]
priority = 2

[notify]
priority = 10

[narrate]
priority = 15
"""


def write_default_config(path: pathlib.Path = CONFIG_FILE) -> bool:
    """Write the default TOML file if none exists.

    :returns: True if this call created the file.
    """
    if path.is_file():
        logger.info("Config exists at %s. Skipped.", path)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CONFIG_TEMPLATE)
    logger.info("Wrote %s.", path)
    return True
