"""Tests for the tool wiring: what gets reported back to the caller."""

import asyncio

import mlx.core as mx
import pytest

from maggid import identity, server
from maggid.config import Config
from maggid.playback import PlaybackQueue


@pytest.fixture
def runtime(voices_dir, monkeypatch):
    """A live Runtime with synthesis stubbed out and no audio device."""

    async def fake_synthesize(text, ref_audio=None):
        return mx.zeros(24000)

    monkeypatch.setattr(server.synth, "synthesize", fake_synthesize)
    live = server.Runtime(
        config=Config(),
        playback=PlaybackQueue(),
        voices=identity.VoiceRegistry(voices_dir / "assignments.json"),
        slots=identity.SessionSlots(),
    )
    monkeypatch.setattr(server, "_runtime", live)
    return live


def test_the_label_names_the_voice(runtime, context):
    """Regression: the label used to be discarded whenever no clip resolved."""
    spoken = asyncio.run(server._say("hi", context("/projects/spade"), None))
    assert spoken.voice == "spade"


def test_a_missing_clip_still_names_the_workspace(
    runtime, context, tmp_path, monkeypatch
):
    monkeypatch.setattr(identity, "VOICES_DIR", tmp_path / "empty")
    spoken = asyncio.run(server._say("hi", context("/projects/spade"), None))
    assert spoken.voice == "spade"


def test_an_unidentified_caller_reports_the_built_in_voice(runtime, context):
    spoken = asyncio.run(server._say("hi", context(None), None))
    assert spoken.voice == "built-in"
    assert spoken.label == ""


def test_an_explicit_clip_names_the_clip(runtime, context, voices_dir, monkeypatch):
    monkeypatch.setattr(server, "validate_ref_audio", lambda path: path)
    clip = str(voices_dir / "am_puck.wav")
    spoken = asyncio.run(server._say("hi", context("/x/spade"), None, clip))
    assert spoken.voice == "am_puck"
    assert spoken.label == ""


def test_duration_accounts_for_the_playback_rate(runtime, context):
    spoken = asyncio.run(server._say("hi", context(None), None))
    assert spoken.seconds == pytest.approx(1.0 / server.SPEED, rel=1e-6)


def test_a_full_backlog_is_reported_not_raised(runtime, context, monkeypatch):
    monkeypatch.setattr(runtime.playback, "enqueue", lambda audio, priority: False)
    spoken = asyncio.run(server._say("hi", context(None), None))
    assert spoken.queued is False


def test_an_unknown_channel_is_rejected(runtime, context):
    with pytest.raises(ValueError, match="Unknown channel"):
        asyncio.run(server._say("hi", context(None), "shout"))


def test_the_channel_sets_the_queue_priority(runtime, context):
    asyncio.run(server._say("hi", context(None), "narrate"))
    asyncio.run(server._say("hi", context(None), "permission"))
    assert runtime.playback._queue.get_nowait().priority == 1


def test_tools_fail_legibly_before_startup(monkeypatch):
    monkeypatch.setattr(server, "_runtime", None)
    with pytest.raises(RuntimeError, match="starting or shutting down"):
        server._live()
