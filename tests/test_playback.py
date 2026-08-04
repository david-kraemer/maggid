"""Tests for queue ordering, the backlog cap, and resampling."""

import asyncio

import mlx.core as mx
import numpy as np
import sounddevice as sd

from maggid import playback


def _audio(n: int = 8) -> mx.array:
    return mx.zeros(n)


def _drain(queue: playback.PlaybackQueue) -> list[int]:
    """Priorities in the order they would play."""
    order = []
    while True:
        try:
            order.append(queue._queue.get_nowait().priority)
        except asyncio.QueueEmpty:
            return order


def test_urgent_audio_plays_first():
    queue = playback.PlaybackQueue()
    for priority in (15, 1, 10, 2):
        queue.enqueue(_audio(), priority)
    assert _drain(queue) == [1, 2, 10, 15]


def test_equal_priority_keeps_arrival_order():
    queue = playback.PlaybackQueue()
    items = [playback.PlaybackItem(10, seq, _audio()) for seq in range(5)]
    for item in reversed(items):
        queue._queue.put_nowait(item)
    played = [queue._queue.get_nowait().seq for _ in items]
    assert played == list(range(5))


def test_a_full_backlog_drops_instead_of_queueing():
    queue = playback.PlaybackQueue(maxsize=2)
    assert queue.enqueue(_audio(), 10) is True
    assert queue.enqueue(_audio(), 10) is True
    assert queue.enqueue(_audio(), 10) is False


def test_clear_reports_what_it_dropped():
    queue = playback.PlaybackQueue()
    for _ in range(3):
        queue.enqueue(_audio(), 10)
    assert queue.clear() == 3
    assert queue.clear() == 0


def test_clear_cuts_the_utterance_in_flight():
    queue = playback.PlaybackQueue()
    assert not queue._cut.is_set()
    queue.clear()
    assert queue._cut.is_set()


class _FakeStream:
    """An output stream that records what it was written, and can go away."""

    def __init__(self, fail_after: int | None = None) -> None:
        self.written = 0
        self.closed = False
        self._fail_after = fail_after

    def write(self, data: np.ndarray) -> None:
        if self._fail_after is not None and self.written >= self._fail_after:
            raise sd.PortAudioError("device unplugged")
        self.written += len(data)

    def abort(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


SPEAKERS, HEADPHONES = 99, 113  # CoreAudio device ids, as the probe returns them


def _wired(queue, monkeypatch, current: int | None, opened: list) -> None:
    """Point the queue at a device id and record every stream it opens."""
    monkeypatch.setattr(playback.device, "default_output", lambda: current)

    def open_stream():
        queue._close()
        queue._stream, queue._device = _FakeStream(), playback.device.default_output()
        opened.append(queue._device)
        return queue._stream

    monkeypatch.setattr(queue, "_open", open_stream)


def test_a_steady_device_reuses_the_open_stream(monkeypatch):
    """The reopen must not be paid when nothing moved."""
    queue = playback.PlaybackQueue()
    opened = []
    _wired(queue, monkeypatch, SPEAKERS, opened)
    first = queue._current_stream()
    for _ in range(5):
        assert queue._current_stream() is first
    assert opened == [SPEAKERS]


def test_a_switched_device_reopens_the_stream(monkeypatch):
    """Headphones connect between utterances: follow them."""
    queue = playback.PlaybackQueue()
    opened = []
    _wired(queue, monkeypatch, SPEAKERS, opened)
    speakers = queue._current_stream()

    monkeypatch.setattr(playback.device, "default_output", lambda: HEADPHONES)
    headphones = queue._current_stream()

    assert headphones is not speakers
    assert speakers.closed
    assert opened == [SPEAKERS, HEADPHONES]


def test_two_devices_sharing_a_name_are_still_two_devices(monkeypatch):
    """Why the token is an id. A soundcore headset registers twice, as one name."""
    queue = playback.PlaybackQueue()
    opened = []
    _wired(queue, monkeypatch, 115, opened)
    queue._current_stream()
    monkeypatch.setattr(playback.device, "default_output", lambda: 121)
    queue._current_stream()
    assert opened == [115, 121]


def test_an_unreadable_probe_still_opens_once(monkeypatch):
    """No CoreAudio answer means one stream on PortAudio's default, not a churn."""
    queue = playback.PlaybackQueue()
    opened = []
    _wired(queue, monkeypatch, None, opened)
    first = queue._current_stream()
    assert queue._current_stream() is first
    assert opened == [None]


def test_opening_closes_before_it_re_scans(monkeypatch):
    """Ordering is load-bearing: a re-scan leaves an open stream dangling."""
    order: list[str] = []

    class _Noisy(_FakeStream):
        def close(self) -> None:
            order.append("close")
            super().close()

        def start(self) -> None:
            order.append("start")

    queue = playback.PlaybackQueue()
    queue._stream, queue._device = _Noisy(), SPEAKERS
    monkeypatch.setattr(playback.device, "default_output", lambda: HEADPHONES)
    monkeypatch.setattr(playback.device, "rescan", lambda: order.append("rescan"))
    monkeypatch.setattr(playback.sounddevice, "OutputStream", lambda **_kw: _Noisy())

    queue._open()

    assert order == ["close", "rescan", "start"]
    assert queue._device == HEADPHONES


def test_a_full_utterance_reports_every_sample(monkeypatch):
    queue = playback.PlaybackQueue()
    stream = _FakeStream()
    data = np.zeros(playback.WRITE_CHUNK * 3, dtype=np.float32)
    assert queue._write(stream, data) == len(data)
    assert stream.written == len(data)


def test_a_lost_device_reports_how_far_it_got():
    queue = playback.PlaybackQueue()
    stream = _FakeStream(fail_after=playback.WRITE_CHUNK)
    data = np.zeros(playback.WRITE_CHUNK * 4, dtype=np.float32)
    assert queue._write(stream, data) == playback.WRITE_CHUNK


def test_an_interrupt_reports_how_far_it_got():
    queue = playback.PlaybackQueue()
    queue._cut.set()
    data = np.zeros(playback.WRITE_CHUNK * 4, dtype=np.float32)
    assert queue._write(_FakeStream(), data) == 0


def test_a_device_lost_mid_utterance_finishes_on_the_new_one(monkeypatch):
    """Headphones die while speaking: play the rest, don't repeat the start."""
    queue = playback.PlaybackQueue()
    data = np.zeros(playback.WRITE_CHUNK * 4, dtype=np.float32)
    monkeypatch.setattr(playback.device, "default_output", lambda: SPEAKERS)
    streams = [_FakeStream(fail_after=playback.WRITE_CHUNK), _FakeStream()]

    def open_stream():
        queue._stream, queue._device = streams[len(opened)], SPEAKERS
        opened.append(SPEAKERS)
        return queue._stream

    opened: list = []
    monkeypatch.setattr(queue, "_open", open_stream)

    queue._deliver(data)

    assert len(opened) == 2, "should have reopened once"
    assert streams[0].written + streams[1].written == len(data)


def test_an_interrupt_does_not_look_like_a_lost_device(monkeypatch):
    """A cut is a short write too, but must not trigger a reopen."""
    queue = playback.PlaybackQueue()
    queue._cut.set()
    opened: list = []
    _wired(queue, monkeypatch, SPEAKERS, opened)

    queue._deliver(np.zeros(playback.WRITE_CHUNK * 4, dtype=np.float32))

    assert len(opened) == 1, "reopened on an interrupt"


def test_closing_tolerates_a_device_that_already_vanished(monkeypatch):
    queue = playback.PlaybackQueue()

    class _Gone(_FakeStream):
        def abort(self):
            raise sd.PortAudioError("device is gone")

    queue._stream, queue._device = _Gone(), SPEAKERS
    queue._close()
    assert queue._stream is None
    assert queue._device is None


def test_resampled_shortens_by_the_rate():
    audio = np.zeros(1000, dtype=np.float32)
    assert len(playback.resampled(audio, 2.0)) == 500
    assert len(playback.resampled(audio, 0.5)) == 2000


def test_resampled_passes_unit_rate_through_untouched():
    audio = np.arange(10, dtype=np.float32)
    assert playback.resampled(audio, 1.0) is audio


def test_resampled_survives_degenerate_input():
    for audio in (np.zeros(0, dtype=np.float32), np.zeros(1, dtype=np.float32)):
        assert len(playback.resampled(audio, playback.SPEED)) == len(audio)


def test_resampled_preserves_the_signal_shape():
    ramp = np.linspace(0.0, 1.0, 1000, dtype=np.float32)
    out = playback.resampled(ramp, 2.0)
    assert out.dtype == np.float32
    assert out[0] == 0.0
    assert abs(out[-1] - 1.0) < 1e-5
