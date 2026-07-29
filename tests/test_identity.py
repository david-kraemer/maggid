"""Tests for voice assignment, session slots, and spoken labels."""

import asyncio
import pathlib

import pytest

from maggid import identity
from maggid.config import Config

# --- names ---------------------------------------------------------------


def test_the_voice_table_agrees_with_the_kokoro_ids():
    """Regression: a hand-edited row gave am_puck a female name."""
    encoded = {"f": "female", "m": "male"}
    for preset in identity.VOICE_IDS:
        assert identity.voice(preset)["sex"] == encoded[preset[1]]


def test_every_pooled_voice_has_a_distinct_name():
    names = identity.VOICE_NAMES
    assert set(names) == set(identity.VOICE_IDS)
    assert len(set(names.values())) == len(identity.VOICE_IDS)


def test_names_match_voice_sex():
    for preset, spoken in identity.VOICE_NAMES.items():
        assert spoken in identity.NAMES_BY_SEX[identity.voice(preset)["sex"]]


def test_a_lopsided_pool_loses_a_name_instead_of_failing(monkeypatch):
    """Regression: two exhausted iterators used to raise StopIteration at import."""
    lopsided = [
        {"id": f"af_{i}", "sex": "female", "dialect": "american"} for i in range(99)
    ]
    monkeypatch.setattr(identity, "_VOICES", lopsided)
    assert len(identity._names_for_pool()) == len(identity.NAMES_BY_SEX["female"])


def test_session_name_falls_back_to_the_slot_number():
    assert identity.session_name("custom", 3) == "3"


def test_session_name_uses_sex_for_an_unpooled_preset():
    assert identity.session_name("bm_unknown", 1) == identity.NAMES_BY_SEX["male"][1]
    assert identity.session_name("bf_unknown", 1) == identity.NAMES_BY_SEX["female"][1]


# --- announce ------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "prefix", "expected"),
    [
        ("spade", True, "spade. Done."),
        ("spade", False, "Done."),
        ("", True, "Done."),
    ],
)
def test_announce(label, prefix, expected):
    assert identity.announce(label, "Done.", prefix) == expected


# --- voice registry ------------------------------------------------------


def test_voices_are_stable_and_persist(tmp_path):
    path = tmp_path / "assignments.json"
    registry = identity.VoiceRegistry(path)
    assigned = registry.voice("/a")
    assert registry.voice("/a") == assigned
    assert identity.VoiceRegistry(path).voice("/a") == assigned


def test_the_pool_is_exhausted_before_it_wraps(tmp_path):
    registry = identity.VoiceRegistry(tmp_path / "assignments.json")
    assigned = [registry.voice(f"/root{i}") for i in range(len(identity.VOICE_IDS))]
    assert set(assigned) == set(identity.VOICE_IDS)
    assert registry.voice("/one-too-many") in identity.VOICE_IDS


def test_unreadable_assignments_start_fresh(tmp_path):
    path = tmp_path / "assignments.json"
    path.write_text("not json")
    assert identity.VoiceRegistry(path).voice("/a") == identity.VOICE_IDS[0]


# --- session slots -------------------------------------------------------


def test_one_session_keeps_slot_one():
    slots = identity.SessionSlots()
    assert slots.slot("/a", "s1") == 1
    assert slots.slot("/a", "s1") == 1


def test_concurrent_sessions_on_one_root_get_separate_slots():
    slots = identity.SessionSlots()
    assert [slots.slot("/a", f"s{i}") for i in range(1, 4)] == [1, 2, 3]


def test_slots_are_per_root():
    slots = identity.SessionSlots()
    assert slots.slot("/a", "s1") == 1
    assert slots.slot("/b", "s2") == 1


def test_a_missing_session_id_is_slot_one():
    assert identity.SessionSlots().slot("/a", None) == 1


def test_an_expired_slot_is_reclaimed():
    slots = identity.SessionSlots(ttl=-1.0)
    assert slots.slot("/a", "s1") == 1
    assert slots.slot("/a", "s2") == 1


def test_a_freed_slot_is_reused_at_the_lowest_number():
    slots = identity.SessionSlots()
    for session in ("s1", "s2", "s3"):
        slots.slot("/a", session)
    slots._seen.pop(("/a", "s2"))
    assert slots.slot("/a", "s4") == 2


# --- speaker -------------------------------------------------------------


def test_no_root_gives_no_label(context):
    clip, label = asyncio.run(
        identity.speaker(
            context(None),
            Config(ref_audio=pathlib.Path("/fallback.wav")),
            identity.VoiceRegistry(),
            identity.SessionSlots(),
        )
    )
    assert (clip, label) == (pathlib.Path("/fallback.wav"), "")


def test_slot_one_keeps_the_bare_workspace_name(context, voices_dir):
    clip, label = asyncio.run(
        identity.speaker(
            context("/projects/spade"),
            Config(),
            identity.VoiceRegistry(voices_dir / "assignments.json"),
            identity.SessionSlots(),
        )
    )
    assert label == "spade"
    assert clip == voices_dir / f"{identity.VOICE_IDS[0]}.wav"


def test_a_second_session_is_named_and_sounds_different(context, voices_dir):
    registry = identity.VoiceRegistry(voices_dir / "assignments.json")
    slots = identity.SessionSlots()
    first, one = asyncio.run(
        identity.speaker(context("/projects/cfd", "s1"), Config(), registry, slots)
    )
    second, two = asyncio.run(
        identity.speaker(context("/projects/cfd", "s2"), Config(), registry, slots)
    )
    assert one == "cfd"
    assert two.startswith("cfd ") and two != "cfd"
    assert first != second


def test_a_pinned_voice_applies_to_the_first_slot_only(context, voices_dir):
    config = Config(voices={"spade": "bm_daniel"})
    registry = identity.VoiceRegistry(voices_dir / "assignments.json")
    slots = identity.SessionSlots()
    clip, _ = asyncio.run(
        identity.speaker(context("/projects/spade", "s1"), config, registry, slots)
    )
    assert clip == voices_dir / "bm_daniel.wav"
    other, _ = asyncio.run(
        identity.speaker(context("/projects/spade", "s2"), config, registry, slots)
    )
    assert other != clip


def test_a_missing_clip_keeps_the_label(context, tmp_path, monkeypatch):
    """The label must survive a missing clip. Identity is the whole point."""
    monkeypatch.setattr(identity, "VOICES_DIR", tmp_path / "empty")
    clip, label = asyncio.run(
        identity.speaker(
            context("/projects/spade"),
            Config(),
            identity.VoiceRegistry(tmp_path / "assignments.json"),
            identity.SessionSlots(),
        )
    )
    assert clip is None
    assert label == "spade"
