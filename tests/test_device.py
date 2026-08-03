"""Tests for the CoreAudio probe and the PortAudio re-scan.

The only tests in the suite that touch real hardware. They have to: every claim
the device-following design rests on is a claim about what CoreAudio and
PortAudio actually do, and a mock would only restate the assumption.
"""

import sys
import time

import numpy as np
import pytest
import sounddevice

from maggid import device

darwin_only = pytest.mark.skipif(sys.platform != "darwin", reason="CoreAudio")


@darwin_only
def test_the_probe_answers_with_a_device():
    assert device.default_output() is not None


@darwin_only
def test_a_re_scan_makes_portaudio_agree_with_the_probe():
    """The load-bearing claim: PortAudio reads the same property the probe does.

    Which is why `_open` can pass no device at all and land on the right one, and
    why the name can come from PortAudio. If this ever failed, following the
    output would have to match a CoreAudio name against PortAudio's device list
    -- which is not even well defined, since names are not unique.
    """
    device.rescan()
    default = sounddevice.query_devices(sounddevice.default.device[1])
    assert device.output_name() == default["name"]
    assert default["max_output_channels"] > 0


@darwin_only
def test_the_probe_leaves_an_open_stream_playing():
    """The whole reason not to re-scan per utterance. Contrast the next test."""
    stream = sounddevice.OutputStream(samplerate=24000, channels=1, dtype="float32")
    stream.start()
    try:
        assert device.default_output() is not None
        assert stream.active
    finally:
        stream.stop()
        stream.close()


@darwin_only
def test_a_re_scan_invalidates_an_open_stream():
    """Why `_open` closes before it re-scans: the old handle dangles."""
    stream = sounddevice.OutputStream(samplerate=24000, channels=1, dtype="float32")
    stream.start()
    try:
        device.rescan()
        with pytest.raises(sounddevice.PortAudioError):
            stream.write(np.zeros(2048, dtype=np.float32))
    finally:
        stream.close()


@darwin_only
def test_the_probe_is_cheap_enough_to_call_every_utterance():
    """Guards the design: costing anything near a reopen would sink it."""
    start = time.perf_counter()
    for _ in range(50):
        device.default_output()
    assert (time.perf_counter() - start) / 50 < 0.001


@darwin_only
def test_one_headset_can_register_twice_under_one_name():
    """Why the token is an id. Not a hypothetical: soundcore does exactly this.

    Skipped when no duplicate happens to be attached, since it asserts about the
    machine rather than about maggid. It is here to record what the id is for.
    """
    names = [
        info["name"]
        for info in sounddevice.query_devices()
        if info["max_output_channels"] > 0
    ]
    if len(names) == len(set(names)):
        pytest.skip("no duplicate output names attached right now")
    assert device.default_output() is not None, "an id still identifies one of them"


def test_the_probe_declines_politely_off_darwin(monkeypatch):
    monkeypatch.setattr(device.sys, "platform", "linux")
    assert device.default_output() is None


def test_a_failing_probe_declines_rather_than_raising(monkeypatch):
    """A HAL error must degrade to PortAudio's default, never break playback."""
    monkeypatch.setattr(device, "_core_audio", None)
    assert device.default_output() is None


def test_an_unnameable_default_still_gives_the_log_line_something(monkeypatch):
    def explode(**_kwargs):
        raise ValueError("no default output device")

    monkeypatch.setattr(device.sounddevice, "query_devices", explode)
    assert device.output_name() == "an unnamed device"
