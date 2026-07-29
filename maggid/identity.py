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
            self._map[root] = self._allocate()
            logger.info("Assigned voice %s to %s", self._map[root], root)
            self._save()
        return self._map[root]

    def _allocate(self) -> str:
        """An unused voice, or a wrapped one once the pool runs out.

        Past nine workspaces two roots share a voice, and only the spoken label
        tells them apart. The wrap is deterministic, so it is at least stable.
        """
        used = set(self._map.values())
        free = [v for v in VOICE_IDS if v not in used]
        return free[0] if free else VOICE_IDS[len(self._map) % len(VOICE_IDS)]

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._map, indent=2, sort_keys=True))
        except OSError:
            logger.warning("Could not persist assignments to %s", self._path)


class SessionSlots:
    """Separates concurrent sessions that share one workspace root.

    Several terminals on one directory is the normal case, not an edge case.
    Without slots they share a voice and a label, which defeats the purpose.
    """

    def __init__(self, ttl: float = SESSION_TTL) -> None:
        self._ttl = ttl
        self._seen: dict[tuple[str, str], tuple[int, float]] = {}
        self._lock = threading.Lock()

    def slot(self, root: str, session_id: str | None) -> int:
        """Which concurrent session this is for the root. Starts at 1."""
        if session_id is None:
            return 1
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            key = (root, session_id)
            slot = self._seen[key][0] if key in self._seen else self._free(root)
            self._seen[key] = (slot, now)
            return slot

    def _expire(self, now: float) -> None:
        """Drop slots whose session went quiet. The caller holds the lock."""
        self._seen = {
            key: value
            for key, value in self._seen.items()
            if now - value[1] <= self._ttl
        }

    def _free(self, root: str) -> int:
        """Lowest slot number unused on the root. The caller holds the lock."""
        taken = {slot for (r, _), (slot, _) in self._seen.items() if r == root}
        return next(i for i in itertools.count(1) if i not in taken)


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
    name = pathlib.PurePath(root).name or "workspace"
    # Voices are per slot, so the second terminal on a root sounds different from
    # the first. A [voices] pin applies to the first slot only.
    pinned = config.voices.get(name) if slot == 1 else None
    preset = pinned or voices.voice(root if slot == 1 else f"{root}#{slot}")
    # The label comes after the voice, because the name must match the voice
    # gender. Slot 1 keeps the bare workspace name, so a lone session is never
    # called "spade Colin", and no session is renamed later.
    label = name if slot == 1 else f"{name} {session_name(preset, slot)}"
    clip = VOICES_DIR / f"{preset}.wav"
    if not clip.is_file():
        logger.warning("Voice clip %s is missing. Using the default voice.", clip)
        return config.ref_audio, label
    return clip, label


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
    return {"f": "female", "m": "male"}.get(preset[1:2])


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
