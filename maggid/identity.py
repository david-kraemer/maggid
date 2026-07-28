"""Which workspace is speaking: a voice and a spoken label."""

import itertools
import json
import logging
import pathlib
import threading
import time
import urllib.parse

from fastmcp import Context

from maggid.config import CONFIG_DIR, Config

logger = logging.getLogger(__name__)

VOICES_DIR = CONFIG_DIR / "voices" / "refs"
ASSIGNMENTS = CONFIG_DIR / "assignments.json"

# Rated every cloned candidate, kept the ones at 8 or better, then took the
# subset with the largest minimum pairwise distance (speaker-embedding cosine).
# Best-rated first, which is also assignment order, so the first workspace seen
# gets the best voice.
#
# Nine is the practical ceiling. Even the best nine hold one pair at 0.86
# cosine, because Chatterbox pulls every clone toward its own character. Voice
# alone separates four or five workspaces. The spoken label does the real work.
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

# Concurrent sessions on one root get names, not numbers. "cfd Bonnie" and
# "cfd Colin" are easier to tell apart than "cfd two" and "cfd three", which
# differ only in an unstressed final syllable.
#
# From the 2028 Atlantic list, split by gender, most familiar first. The name
# follows the gender of the voice, because a male name in a female voice
# confuses more than a bare number does. "Alex" is dropped as ambiguous.
FEMALE_NAMES = (
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
)
MALE_NAMES = (
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
)

# A slot is free once its session goes quiet this long. Sessions do not announce
# that they close, and a reconnect arrives under a fresh id, so expiry is the
# only signal.
SESSION_TTL = 1800.0


def _names_for_pool() -> dict[str, str]:
    """One gender-matched name per pooled voice.

    Kokoro presets encode gender in the second character: af_, am_, bf_, bm_. A
    name taken from the voice can never disagree with it, and distinct voices
    give distinct names at no cost. The zip truncates on purpose, so a lopsided
    pool loses a name and falls back to the slot number instead of failing.
    """
    return {
        voice: name
        for names, letter in ((FEMALE_NAMES, "f"), (MALE_NAMES, "m"))
        for voice, name in zip(
            (v for v in VOICE_POOL if v[1] == letter), names, strict=False
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
        free = [v for v in VOICE_POOL if v not in used]
        return free[0] if free else VOICE_POOL[len(self._map) % len(VOICE_POOL)]

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
) -> tuple[str | None, str]:
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
    return str(clip), label


def session_name(preset: str, slot: int) -> str:
    """Spoken name for a session, matching the gender of its voice.

    Falls back to the slot number for a voice outside the pool, such as a
    [voices] pin. A mismatched name is worse than a number.
    """
    if preset in VOICE_NAMES:
        return VOICE_NAMES[preset]
    if len(preset) > 1 and preset[1] in "fm":
        pool = FEMALE_NAMES if preset[1] == "f" else MALE_NAMES
        return pool[slot % len(pool)]
    return str(slot)


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
