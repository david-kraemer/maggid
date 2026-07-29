"""Which workspace speaks: a voice and a spoken label."""

import dataclasses
import itertools
import json
import logging
import pathlib
import threading
import time
import urllib.parse
from collections.abc import Mapping
from typing import Literal

from fastmcp import Context

from .config import CONFIG_DIR, Config
from .paths import StrPath, resolve_path, writable_path

logger = logging.getLogger(__name__)

VOICES_DIR = CONFIG_DIR / "voices" / "refs"
ASSIGNMENTS = CONFIG_DIR / "assignments.json"

type Sex = Literal["male", "female"]

# Kokoro puts the sex in the second character of a preset id: af_, am_, bf_, bm_.
SEX_BY_LETTER: Mapping[str, Sex] = {"f": "female", "m": "male"}


def preset_sex(preset: str) -> Sex | None:
    """The sex a Kokoro preset id encodes. None if the id does not follow it.

    This is the only place that reads the convention.
    """
    return SEX_BY_LETTER.get(preset[1:2])


# Each candidate clone has a rating. These are the ones at 8 or better, in the subset
# with the largest minimum pairwise distance (speaker-embedding cosine). The best
# rating comes first, and assignment follows the same order.
#
# Nine voices is the practical limit. The best nine still hold one pair at 0.86
# cosine, because Chatterbox moves every clone toward its own character. Voice alone
# separates four or five workspaces. The label separates the rest.
VOICE_IDS: tuple[str, ...] = (
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

# Concurrent sessions on one root get names, not numbers. "cfd Bonnie" and "cfd Colin"
# are easier to tell apart than "cfd two" and "cfd three". The numbers differ only in
# an unstressed final syllable.
NAMES_BY_SEX: Mapping[Sex, tuple[str, ...]] = {
    "female": (
        "Bonnie",
        "Danielle",
        "Julia",
        "Lisa",
        "Nicole",
        "Paula",
        "Farrah",
        "Hermine",
        "Shary",
        "Virginie",
    ),
    "male": (
        "Colin",
        "Earl",
        "Martin",
        "Owen",
        "Richard",
        "Karl",
        "Walter",
        "Gaston",
        "Idris",
        "Tobias",
    ),
}

# A slot is free when its session is quiet for this long. Sessions do not report that
# they close, and a reconnect arrives under a new id, so expiry is the only signal.
SESSION_TTL = 1800.0


def _names_for_pool() -> Mapping[str, str]:
    """One name per pooled voice, matched by sex.

    The zip truncates on purpose. A lopsided pool loses a name and falls back to the
    slot number.
    """
    return {
        preset: spoken
        for sex, names in NAMES_BY_SEX.items()
        for preset, spoken in zip(
            (p for p in VOICE_IDS if preset_sex(p) == sex), names, strict=False
        )
    }


VOICE_NAMES = _names_for_pool()


def next_voice(assigned: Mapping[str, str]) -> str:
    """An unused voice. Wraps to a used one when the pool is empty.

    Past nine workspaces, two roots share a voice and only the label separates them.
    The wrap is deterministic, so it stays stable.
    """
    used = set(assigned.values())
    free = [v for v in VOICE_IDS if v not in used]
    return free[0] if free else VOICE_IDS[len(assigned) % len(VOICE_IDS)]


class VoiceRegistry:
    """A stable voice per workspace. Survives a daemon restart.

    The key is the workspace root, not the MCP session id. A session id changes on
    each reconnect, which would reshuffle every voice when the daemon restarts.
    """

    def __init__(self, path: StrPath = ASSIGNMENTS) -> None:
        self._path = resolve_path(path)
        self._map: dict[str, str] = {}
        try:
            self._map = json.loads(self._path.read_text())
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            logger.warning("Could not read %s. Starting fresh.", self._path)

    def voice(self, root: str) -> str:
        """The preset for this workspace. Assigns one on first sight."""
        if root not in self._map:
            self._map[root] = next_voice(self._map)
            logger.info("Assigned voice %s to %s", self._map[root], root)
            self._save()
        return self._map[root]

    def _save(self) -> None:
        try:
            writable_path(self._path).write_text(
                json.dumps(self._map, indent=2, sort_keys=True)
            )
        except OSError:
            logger.warning("Could not persist assignments to %s", self._path)


@dataclasses.dataclass(frozen=True, slots=True)
class Session:
    """One client connection to one workspace root."""

    root: str
    id: str


@dataclasses.dataclass(frozen=True, slots=True)
class Slot:
    """The number a session holds on its root, and when it last spoke."""

    number: int
    at: float


type Slots = Mapping[Session, Slot]


def unexpired(slots: Slots, now: float, ttl: float) -> Slots:
    """The table without the sessions that are quiet."""
    return {session: slot for session, slot in slots.items() if now - slot.at <= ttl}


def lowest_free(slots: Slots, root: str) -> int:
    """The lowest number the root does not use. Starts at 1."""
    taken = {slot.number for session, slot in slots.items() if session.root == root}
    return next(i for i in itertools.count(1) if i not in taken)


def assign_slot(
    slots: Slots, session: Session, now: float, ttl: float
) -> tuple[Slots, int]:
    """The number this session holds, and the table that results.

    The function is pure. The caller owns the clock and the table.

    :returns: (the table with this session included; the number)
    """
    live = unexpired(slots, now, ttl)
    number = (
        live[session].number if session in live else lowest_free(live, session.root)
    )
    return {**live, session: Slot(number, now)}, number


class SessionSlots:
    """Separates concurrent sessions on one workspace root.

    Several terminals on one directory is normal. Without slots they share a voice and
    a label. This class is a lock and a clock around assign_slot.
    """

    def __init__(self, ttl: float = SESSION_TTL) -> None:
        self._ttl = ttl
        self._slots: Slots = {}
        self._lock = threading.Lock()

    def slot(self, root: str, session_id: str | None) -> int:
        """The number of this concurrent session on the root. Starts at 1."""
        if session_id is None:
            return 1
        with self._lock:
            self._slots, number = assign_slot(
                self._slots, Session(root, session_id), time.monotonic(), self._ttl
            )
            return number


async def speaker(
    ctx: Context, config: Config, voices: VoiceRegistry, slots: SessionSlots
) -> tuple[pathlib.Path | None, str]:
    """The voice clip and spoken label for the calling session.

    :returns: (the clip path, or None for the built-in voice; the label, or "" when
        the workspace has no identity)
    """
    root = await workspace_root(ctx)
    if root is None:
        return config.ref_audio, ""
    slot = slots.slot(root, ctx.session_id)
    name = workspace_name(root)
    # Voices are per slot, so the second terminal on a root sounds different from the
    # first. A [voices] pin applies to the first slot only.
    pinned = config.voices.get(name) if slot == 1 else None
    preset = pinned or voices.voice(voice_key(root, slot))
    # The label comes after the voice, because the name must match the voice sex.
    label = workspace_label(name, slot, preset)
    clip = VOICES_DIR / f"{preset}.wav"
    if not clip.is_file():
        logger.warning("Voice clip %s is missing. Using the default voice.", clip)
        return config.ref_audio, label
    return clip, label


def announce(label: str, text: str, prefix: bool = True) -> str:
    """The text with the workspace label in front, so the listener knows who speaks.

    Use a period, not a dash. The engine reads a period as a pause. It either omits a
    dash or speaks it.
    """
    if not label or not prefix:
        return text
    return f"{label}. {text}"


def workspace_name(root: str) -> str:
    """The last component of a workspace root."""
    return pathlib.PurePath(root).name or "workspace"


def voice_key(root: str, slot: int) -> str:
    """The registry key for a session. Slot 1 uses the bare root.

    assignments.json holds this shape. A change to it orphans every voice already
    assigned.
    """
    return root if slot == 1 else f"{root}#{slot}"


def workspace_label(name: str, slot: int, preset: str) -> str:
    """The label the listener hears before the message.

    Slot 1 keeps the bare workspace name. A lone session is never "spade Colin", and
    no session gets a new label when a second one starts.
    """
    return name if slot == 1 else f"{name} {session_name(preset, slot)}"


def session_name(preset: str, slot: int) -> str:
    """A name for a session that matches the sex of its voice.

    Falls back to the slot number when the sex is unknown, such as a [voices] pin
    outside the pool. A wrong name is worse than a number.
    """
    if preset in VOICE_NAMES:
        return VOICE_NAMES[preset]
    if (sex := preset_sex(preset)) is not None:
        names = NAMES_BY_SEX[sex]
        return names[slot % len(names)]
    return str(slot)


async def workspace_root(ctx: Context) -> str | None:
    """The path the client advertises as its root, if it advertises one.

    This stays a str. It is an opaque key for VoiceRegistry, and assignments.json
    holds the exact text the client sent. Path normalization would orphan it.
    """
    try:
        roots = await ctx.list_roots()
    except Exception as e:  # noqa: BLE001 - identity is best-effort
        # A client need not support roots. A transport fault costs the label, not the
        # notification.
        logger.debug("No workspace root available: %s", e)
        return None
    if not roots:
        return None
    uri = str(roots[0].uri)
    if not uri.startswith("file://"):
        return uri or None
    return urllib.parse.unquote(urllib.parse.urlparse(uri).path) or None
