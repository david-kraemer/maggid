"""Tests for queue ordering, the backlog cap, and resampling."""

import asyncio

import mlx.core as mx
import numpy as np

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
