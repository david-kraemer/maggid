"""Which workspace is speaking: a voice and a spoken label."""

import itertools
import json
import logging
import pathlib
import threading
import time
import urllib.parse
from collections.abc import Callable, Mapping
from typing import Literal, TypedDict

from fastmcp import Context

from .config import CONFIG_DIR, Config
from .typing_ import guard_type

logger = logging.getLogger(__name__)

VOICES_DIR = CONFIG_DIR / "voices" / "refs"
ASSIGNMENTS = CONFIG_DIR / "assignments.json"

type Sex = Literal["male", "female"]
type Dialect = Literal["american", "british"]


class Voice(TypedDict):
    id: str
    sex: Sex
    dialect: Dialect


class Name(TypedDict):
    name: str
    sex: Sex


# Kokoro encodes sex in the second character of a preset id: af_, am_, bf_, bm_.
SEX_BY_LETTER: Mapping[str, Sex] = {"f": "female", "m": "male"}


# Rated every cloned candidate, kept the ones at 8 or better, then took the subset
# with the largest minimum pairwise distance (speaker-embedding cosine). Best-rated
# first, which is also assignment order, so the first workspace seen gets the best
# voice.
#
# Nine is the practical ceiling. Even the best nine hold one pair at 0.86 cosine,
# because Chatterbox pulls every clone toward its own character. Voice alone separates
# four or five workspaces. The spoken label does the real work.
#
# Kokoro ids encode the same sex and dialect in their first two characters. The table
# is authoritative and a test asserts the two agree, so a typo here cannot quietly
# hand a male voice a female name.
_VOICES: list[Voice] = [
    {"id": "af_heart", "sex": "female", "dialect": "american"},
    {"id": "af_jessica", "sex": "female", "dialect": "american"},
    {"id": "af_sarah", "sex": "female", "dialect": "american"},
    {"id": "am_liam", "sex": "male", "dialect": "american"},
    {"id": "bf_isabella", "sex": "female", "dialect": "british"},
    {"id": "am_fenrir", "sex": "male", "dialect": "american"},
    {"id": "am_puck", "sex": "male", "dialect": "american"},
    {"id": "bf_alice", "sex": "female", "dialect": "british"},
    {"id": "bm_daniel", "sex": "male", "dialect": "british"},
]
VOICES: Mapping[str, Voice] = {v["id"]: v for v in _VOICES}
voice: Callable[[str], Voice] = VOICES.__getitem__
is_voice = guard_type(voice)

VOICE_IDS = list(VOICES.keys())

# Concurrent sessions on one root get names, not numbers. "cfd Bonnie" and
# "cfd Colin" are easier to tell apart than "cfd two" and "cfd three", which
# differ only in an unstressed final syllable.
_NAMES: list[Name] = [
    {"name": "Bonnie", "sex": "female"},
    {"name": "Danielle", "sex": "female"},
    {"name": "Julia", "sex": "female"},
    {"name": "Lisa", "sex": "female"},
    {"name": "Nicole", "sex": "female"},
    {"name": "Paula", "sex": "female"},
    {"name": "Farrah", "sex": "female"},
    {"name": "Hermine", "sex": "female"},
    {"name": "Shary", "sex": "female"},
    {"name": "Virginie", "sex": "female"},
    {"name": "Colin", "sex": "male"},
    {"name": "Earl", "sex": "male"},
    {"name": "Martin", "sex": "male"},
    {"name": "Owen", "sex": "male"},
    {"name": "Richard", "sex": "male"},
    {"name": "Karl", "sex": "male"},
    {"name": "Walter", "sex": "male"},
    {"name": "Gaston", "sex": "male"},
    {"name": "Idris", "sex": "male"},
    {"name": "Tobias", "sex": "male"},
]

NAMES: Mapping[str, Name] = {n["name"]: n for n in _NAMES}
name: Callable[[str], Name] = NAMES.__getitem__
is_name = guard_type(name)

# Names by sex, in table order, so a voice can draw one that agrees with it. Derived
# from _NAMES rather than kept alongside it: two lists to edit is one list too many.
NAMES_BY_SEX: Mapping[Sex, tuple[str, ...]] = {
    sex: tuple(n["name"] for n in _NAMES if n["sex"] == sex)
    for sex in {n["sex"] for n in _NAMES}
}


# A slot is free once its session goes quiet this long. Sessions do not announce
# that they close, and a reconnect arrives under a fresh id, so expiry is the
# only signal.
SESSION_TTL = 1800.0


def _names_for_pool() -> Mapping[str, str]:
    """One sex-matched name per pooled voice.

    A name drawn from the voice's own row can never disagree with it, and distinct
    voices give distinct names at no cost. The zip truncates on purpose, so a lopsided
    pool loses a name and falls back to the slot number instead of failing.
    """
    return {
        v["id"]: spoken
        for sex, names in NAMES_BY_SEX.items()
        for v, spoken in zip(
            (v for v in _VOICES if v["sex"] == sex), names, strict=False
        )
    }


VOICE_NAMES = _names_for_pool()


def next_voice(assigned: Mapping[str, str]) -> str:
    """An unused voice, or a wrapped one once the pool runs out.

    Past nine workspaces two roots share a voice, and only the spoken label tells
    them apart. The wrap is deterministic, so it is at least stable.
    """
    used = set(assigned.values())
    free = [v for v in VOICE_IDS if v not in used]
    return free[0] if free else VOICE_IDS[len(assigned) % len(VOICE_IDS)]


class VoiceRegistry:
    """Assigns a stable voice per workspace. Survives a daemon restart.

    Keyed on the workspace root, not the MCP session id. A session id rotates
    whenever the client reconnects, which would reshuffle every voice the first
    time the daemon bounced.
    """

    def __init__(self, path: pathlib.Path = ASSIGNMENTS) -> None:
        self._path = path
        self._map: dict[str, str] = {}
        try:
            self._map = json.loads(path.read_text())
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            logger.warning("Could not read %s. Starting fresh.", path)

    def voice(self, root: str) -> str:
        """Preset for this workspace. Allocates one on first sight."""
        if root not in self._map:
            self._map[root] = next_voice(self._map)
            logger.info("Assigned voice %s to %s", self._map[root], root)
            self._save()
        return self._map[root]

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._map, indent=2, sort_keys=True))
        except OSError:
            logger.warning("Could not persist assignments to %s", self._path)


# Slot table: which numbered session holds a root, and when it last spoke.
type Seen = Mapping[tuple[str, str], tuple[int, float]]


def unexpired(seen: Seen, now: float, ttl: float) -> Seen:
    """The table without slots whose session has gone quiet."""
    return {key: value for key, value in seen.items() if now - value[1] <= ttl}


def lowest_free(seen: Seen, root: str) -> int:
    """Lowest slot number unused on the root. Starts at 1."""
    taken = {slot for (r, _), (slot, _) in seen.items() if r == root}
    return next(i for i in itertools.count(1) if i not in taken)


def assign_slot(
    seen: Seen, root: str, session_id: str, now: float, ttl: float
) -> tuple[Seen, int]:
    """The slot this session holds, and the table it leaves behind.

    Pure, so the caller owns both the clock and the state. Expiry happens here
    rather than on a timer because sessions do not announce that they close, and a
    reconnect arrives under a fresh id.

    :returns: (table including this session's slot; the slot number)
    """
    live = unexpired(seen, now, ttl)
    key = (root, session_id)
    slot = live[key][0] if key in live else lowest_free(live, root)
    return {**live, key: (slot, now)}, slot


class SessionSlots:
    """Separates concurrent sessions that share one workspace root.

    Several terminals on one directory is the normal case, not an edge case.
    Without slots they share a voice and a label, which defeats the purpose.

    A lock and a clock around assign_slot. All the reasoning lives there.
    """

    def __init__(self, ttl: float = SESSION_TTL) -> None:
        self._ttl = ttl
        self._seen: Seen = {}
        self._lock = threading.Lock()

    def slot(self, root: str, session_id: str | None) -> int:
        """Which concurrent session this is for the root. Starts at 1."""
        if session_id is None:
            return 1
        with self._lock:
            self._seen, slot = assign_slot(
                self._seen, root, session_id, time.monotonic(), self._ttl
            )
            return slot


async def speaker(
    ctx: Context, config: Config, voices: VoiceRegistry, slots: SessionSlots
) -> tuple[pathlib.Path | None, str]:
    """Voice clip and spoken label for the calling session.

    :returns: (clip path, or None for the built-in voice; label, or "" when the
        workspace is unidentifiable)
    """
    root = await workspace_root(ctx)
    if root is None:
        return config.ref_audio, ""
    slot = slots.slot(root, ctx.session_id)
    name = workspace_name(root)
    # Voices are per slot, so the second terminal on a root sounds different from
    # the first. A [voices] pin applies to the first slot only.
    pinned = config.voices.get(name) if slot == 1 else None
    preset = pinned or voices.voice(voice_key(root, slot))
    # The label comes after the voice, because the name must match the voice sex.
    label = workspace_label(name, slot, preset)
    clip = VOICES_DIR / f"{preset}.wav"
    if not clip.is_file():
        logger.warning("Voice clip %s is missing. Using the default voice.", clip)
        return config.ref_audio, label
    return clip, label


def workspace_name(root: str) -> str:
    """Spoken stem of a workspace root."""
    return pathlib.PurePath(root).name or "workspace"


def voice_key(root: str, slot: int) -> str:
    """Registry key for a session. Slot 1 owns the root; later slots hang off it.

    This shape reaches assignments.json, so changing it orphans every voice a user
    has already been given.
    """
    return root if slot == 1 else f"{root}#{slot}"


def workspace_label(name: str, slot: int, preset: str) -> str:
    """What the listener hears before the message.

    Slot 1 keeps the bare workspace name, so a lone session is never called "spade
    Colin", and no session is renamed once a second one appears.
    """
    return name if slot == 1 else f"{name} {session_name(preset, slot)}"


def session_name(preset: str, slot: int) -> str:
    """Spoken name for a session, matching the sex of its voice.

    Falls back to the slot number for a preset whose sex cannot be established, such
    as a [voices] pin outside the pool. A mismatched name is worse than a number.
    """
    if preset in VOICE_NAMES:
        return VOICE_NAMES[preset]
    if (sex := preset_sex(preset)) is not None:
        names = NAMES_BY_SEX[sex]
        return names[slot % len(names)]
    return str(slot)


def preset_sex(preset: str) -> Sex | None:
    """Sex of a Kokoro preset: from the table if pooled, else from the id.

    Kokoro encodes sex in the second character — af_, am_, bf_, bm_ — which is the
    only handle available for a [voices] pin the pool has never seen.
    """
    if is_voice(preset):
        return voice(preset)["sex"]
    return SEX_BY_LETTER.get(preset[1:2])


async def workspace_root(ctx: Context) -> str | None:
    """Path the client advertises as its root, if it advertises one."""
    try:
        roots = await ctx.list_roots()
    except Exception as e:  # noqa: BLE001 - identity is best-effort
        # A client need not support roots. A transport fault here must cost the
        # label, not the notification.
        logger.debug("No workspace root available: %s", e)
        return None
    if not roots:
        return None
    uri = str(roots[0].uri)
    if not uri.startswith("file://"):
        return uri or None
    return urllib.parse.unquote(urllib.parse.urlparse(uri).path) or None


def announce(label: str, text: str, prefix: bool = True) -> str:
    """Text with the workspace label first, so the listener knows who talks.

    A period, not a dash. The engine reads a period as a pause. It either
    swallows a dash or speaks it.
    """
    if not label or not prefix:
        return text
    return f"{label}. {text}"
