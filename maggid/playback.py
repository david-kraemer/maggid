"""Playback queue. Serializes audio from concurrent tool calls."""

import asyncio
import contextlib
import dataclasses
import itertools
import logging
import threading

import mlx.core as mx
import numpy as np
import sounddevice

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000

# One rate for everything, applied by resampling at playback rather than by
# re-running the model. Per-channel rates were a Kokoro-era idea that never
# earned its keep: the whole 1.0-to-1.3 range saved under half a second on a
# typical notification. Wording carries a channel's meaning better than speed.
SPEED = 1.1

# Samples per device write. Small enough that interrupt cuts in within about
# 85 ms. Large enough not to churn.
WRITE_CHUNK = 2048

# Backlog cap. Past this, a chatty fleet queues audio that will still be playing
# long after the work it describes is done. Drop it instead.
MAX_BACKLOG = 32


@dataclasses.dataclass(frozen=True, order=True, slots=True)
class PlaybackItem:
    """Priority queue entry. Lower priority plays first, as heapq expects."""

    priority: int
    seq: int
    audio: mx.array = dataclasses.field(compare=False)


class PlaybackQueue:
    """Priority queue with a worker that plays audio one item at a time."""

    def __init__(self, maxsize: int = MAX_BACKLOG) -> None:
        self._counter = itertools.count()
        self._queue: asyncio.PriorityQueue[PlaybackItem] = asyncio.PriorityQueue(
            maxsize=maxsize
        )
        self._worker: asyncio.Task[None] | None = None
        self._stream: sounddevice.OutputStream | None = None
        self._cut = threading.Event()

    def start(self) -> None:
        """Open the audio device and start the drain task."""
        self._stream = sounddevice.OutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32"
        )
        self._stream.start()
        self._worker = asyncio.create_task(self._drain())
        logger.info("Playback worker started.")

    async def stop(self) -> None:
        """Cancel the worker and close the device."""
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("Playback worker stopped.")

    def enqueue(self, audio: mx.array, priority: int) -> bool:
        """Add audio to the queue. Does not block.

        :returns: False if the backlog is full and the audio was dropped.
        """
        try:
            self._queue.put_nowait(PlaybackItem(priority, next(self._counter), audio))
        except asyncio.QueueFull:
            logger.warning("Backlog full (%d). Dropped audio.", self._queue.maxsize)
            return False
        return True

    def clear(self) -> int:
        """Discard the backlog and stop what is playing.

        :returns: Number of queued items dropped, not counting the one playing.
        """
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._queue.task_done()
            dropped += 1
        self._cut.set()
        return dropped

    async def _drain(self) -> None:
        """Pull items and play them one at a time."""
        while True:
            item = await self._queue.get()
            try:
                await self._play(item.audio)
            except Exception:
                logger.exception("Playback failed")
            finally:
                self._queue.task_done()

    async def _play(self, audio: mx.array) -> None:
        """Push samples to the open device.

        This used to shell out to `afplay`, which cost about 1.0 s of process and
        CoreAudio startup per utterance. That is five times the synthesis time,
        paid on every notification. One stream held open removes it.
        """
        stream = self._stream
        if stream is None:
            raise RuntimeError("Audio stream is not open")
        data = resampled(np.asarray(audio, dtype=np.float32).reshape(-1))
        self._cut.clear()
        await asyncio.to_thread(self._write, stream, data)

    def _write(self, stream: sounddevice.OutputStream, data: np.ndarray) -> None:
        """Write in chunks, so interrupt can cut in mid-utterance."""
        for start in range(0, len(data), WRITE_CHUNK):
            if self._cut.is_set():
                logger.debug("Cut playback after %d samples.", start)
                return
            stream.write(data[start : start + WRITE_CHUNK])


def resampled(audio: np.ndarray, speed: float = SPEED) -> np.ndarray:
    """Audio at the playback rate.

    A plain rate change, so it shifts pitch a little, exactly as `afplay -r`
    did. The shift is not audible at 1.1. A phase vocoder would be the fix for a
    large pitch-preserving change.
    """
    if len(audio) < 2 or abs(speed - 1.0) < 1e-3:
        return audio
    count = max(1, int(len(audio) / speed))
    return np.interp(
        np.linspace(0, len(audio) - 1, count), np.arange(len(audio)), audio
    ).astype(np.float32)
