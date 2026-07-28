"""Tests for config loading and channel priorities."""

import tomllib

import pytest

from tts_server import config as cfg


def test_missing_file_gives_defaults(tmp_path):
    loaded = cfg.load_config(tmp_path / "absent.toml")
    assert loaded == cfg.Config()
    assert loaded.prefix is True
    assert loaded.ref_audio is None


def test_template_matches_default_priorities():
    """The generated file must agree with the code it documents."""
    tables = tomllib.loads(cfg.CONFIG_TEMPLATE)
    written = {
        name: table["priority"]
        for name, table in tables.items()
        if isinstance(table, dict)
    }
    assert written == dict(cfg.DEFAULT_PRIORITIES)


def test_write_default_config_is_idempotent(tmp_path):
    path = tmp_path / "channels.toml"
    assert cfg.write_default_config(path) is True
    assert cfg.write_default_config(path) is False
    assert cfg.load_config(path).priorities == dict(cfg.DEFAULT_PRIORITIES)


def test_partial_override_keeps_other_defaults(tmp_path):
    path = tmp_path / "channels.toml"
    path.write_text("[narrate]\npriority = 20\n")
    loaded = cfg.load_config(path)
    assert loaded.priorities["narrate"] == 20
    assert loaded.priorities["permission"] == cfg.DEFAULT_PRIORITIES["permission"]


def test_voices_table_is_not_a_channel(tmp_path):
    path = tmp_path / "channels.toml"
    path.write_text('[voices]\nspade = "af_heart"\n')
    loaded = cfg.load_config(path)
    assert loaded.voices == {"spade": "af_heart"}
    assert loaded.priorities == dict(cfg.DEFAULT_PRIORITIES)


def test_unusable_ref_audio_falls_back(tmp_path):
    path = tmp_path / "channels.toml"
    path.write_text(f'ref_audio = "{tmp_path / "gone.wav"}"\n')
    assert cfg.load_config(path).ref_audio is None


def test_prefix_can_be_disabled(tmp_path):
    path = tmp_path / "channels.toml"
    path.write_text("prefix = false\n")
    assert cfg.load_config(path).prefix is False


@pytest.mark.parametrize(
    ("channel", "expected"), [("permission", 1), ("narrate", 15), (None, 10)]
)
def test_channel_priority(channel, expected):
    assert cfg.channel_priority(cfg.Config(), channel) == expected


def test_no_channel_follows_the_notify_override():
    config = cfg.Config(priorities=dict(cfg.DEFAULT_PRIORITIES) | {"notify": 7})
    assert cfg.channel_priority(config, None) == 7


def test_unknown_channel_lists_the_known_ones():
    with pytest.raises(ValueError, match=r"Unknown channel 'shout'.*narrate"):
        cfg.channel_priority(cfg.Config(), "shout")


def test_validate_ref_audio_rejects_a_missing_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        cfg.validate_ref_audio(str(tmp_path / "gone.wav"))


def test_validate_ref_audio_rejects_a_short_clip(tmp_path):
    import numpy as np
    import soundfile

    clip = tmp_path / "short.wav"
    soundfile.write(clip, np.zeros(24000, dtype=np.float32), 24000)
    with pytest.raises(ValueError, match="longer than"):
        cfg.validate_ref_audio(str(clip))


def test_validate_ref_audio_accepts_a_long_clip(tmp_path):
    import numpy as np
    import soundfile

    clip = tmp_path / "long.wav"
    soundfile.write(clip, np.zeros(24000 * 6, dtype=np.float32), 24000)
    cfg.validate_ref_audio(str(clip))
