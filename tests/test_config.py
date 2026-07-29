"""Tests for config loading and channel priorities."""

import tomllib

import pytest

from maggid import config as cfg


def test_missing_file_gives_defaults(tmp_path):
    loaded = cfg.Config.read(tmp_path / "absent.toml")
    assert loaded == cfg.Config()
    assert loaded.prefix is True
    assert loaded.ref_audio is None
    # Regression: an empty priorities default made every channel unknown, so a
    # fresh install with no config file could not speak at all.
    assert loaded.priorities == dict(cfg.DEFAULT_PRIORITIES)


def test_the_generated_template_round_trips(tmp_path):
    """The emitted file must be TOML this same module can read back.

    CHANNELS is now the only source of the priorities, so agreement is structural.
    What is still worth asserting is that the emitter produces a parseable file
    whose tables land where Config.read looks for them.
    """
    path = tmp_path / "channels.toml"
    cfg.Config.write_default(path)
    assert cfg.Config.read(path).priorities == dict(cfg.DEFAULT_PRIORITIES)
    assert tomllib.loads(cfg.CONFIG_TEMPLATE)["prefix"] is True


def test_every_channel_is_documented_in_the_template():
    """A channel with no purpose line would ship an unexplained table."""
    for channel in cfg.CHANNELS:
        assert f"# {channel.purpose}\n[{channel.name}]" in cfg.CONFIG_TEMPLATE


def test_the_fallback_channel_is_a_real_channel():
    assert cfg.FALLBACK_CHANNEL in cfg.DEFAULT_PRIORITIES


def test_write_default_config_is_idempotent(tmp_path):
    path = tmp_path / "channels.toml"
    assert cfg.Config.write_default(path) is True
    assert cfg.Config.write_default(path) is False
    assert cfg.Config.read(path).priorities == dict(cfg.DEFAULT_PRIORITIES)


def test_partial_override_keeps_other_defaults(tmp_path):
    path = tmp_path / "channels.toml"
    path.write_text("[narrate]\npriority = 20\n")
    loaded = cfg.Config.read(path)
    assert loaded.priorities["narrate"] == 20
    assert loaded.priorities["permission"] == cfg.DEFAULT_PRIORITIES["permission"]


def test_voices_table_is_not_a_channel(tmp_path):
    path = tmp_path / "channels.toml"
    path.write_text('[voices]\nspade = "af_heart"\n')
    loaded = cfg.Config.read(path)
    assert loaded.voices == {"spade": "af_heart"}
    assert loaded.priorities == dict(cfg.DEFAULT_PRIORITIES)


def test_unusable_ref_audio_falls_back(tmp_path):
    path = tmp_path / "channels.toml"
    path.write_text(f'ref_audio = "{tmp_path / "gone.wav"}"\n')
    assert cfg.Config.read(path).ref_audio is None


def test_prefix_can_be_disabled(tmp_path):
    path = tmp_path / "channels.toml"
    path.write_text("prefix = false\n")
    assert cfg.Config.read(path).prefix is False


@pytest.mark.parametrize(
    ("channel", "expected"), [("permission", 1), ("narrate", 15), (None, 10)]
)
def test_channel_priority(channel, expected):
    assert cfg.Config().priority(channel) == expected


def test_no_channel_follows_the_notify_override():
    config = cfg.Config(priorities=dict(cfg.DEFAULT_PRIORITIES) | {"notify": 7})
    assert config.priority(None) == 7


def test_unknown_channel_lists_the_known_ones():
    with pytest.raises(ValueError, match=r"Unknown channel 'shout'.*narrate"):
        cfg.Config().priority("shout")


def test_validate_ref_audio_rejects_a_missing_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        cfg.validate_ref_audio(tmp_path / "gone.wav")


def test_validate_ref_audio_rejects_a_short_clip(tmp_path):
    import numpy as np
    import soundfile

    clip = tmp_path / "short.wav"
    soundfile.write(clip, np.zeros(24000, dtype=np.float32), 24000)
    with pytest.raises(ValueError, match="longer than"):
        cfg.validate_ref_audio(clip)


def test_validate_ref_audio_accepts_a_long_clip(tmp_path):
    import numpy as np
    import soundfile

    clip = tmp_path / "long.wav"
    soundfile.write(clip, np.zeros(24000 * 6, dtype=np.float32), 24000)
    assert cfg.validate_ref_audio(clip) == clip
