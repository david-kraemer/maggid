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

from . import device

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
        self._device: int | None = None
        self._cut = threading.Event()

    def start(self) -> None:
        """Open the audio device and start the drain task."""
        self._open()
        self._worker = asyncio.create_task(self._drain())
        logger.info("Playback worker started.")

    async def stop(self) -> None:
        """Cancel the worker and close the device."""
        # Cut first. Cancelling the task does not interrupt the thread it handed
        # the utterance to, so without this the close below can land while that
        # thread is still inside a blocking write.
        self._cut.set()
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None
        self._close()
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
        """Push samples to the device, whichever one that currently is.

        This used to shell out to `afplay`, which cost about 1.0 s of process and
        CoreAudio startup per utterance. That is five times the synthesis time,
        paid on every notification. One stream held open removes it.

        Holding it open costs something the `afplay` version got for free, and
        the whole of `_deliver` is about buying that back: a new process picked up
        the default output device every time, so plugging in headphones simply
        worked. See `_open`.
        """
        data = resampled(np.asarray(audio, dtype=np.float32).reshape(-1))
        self._cut.clear()
        # Threaded because the writes block for as long as the utterance takes to
        # play. Holding the event loop for those seconds would stall every
        # concurrent tool call behind whatever is speaking.
        await asyncio.to_thread(self._deliver, data)

    def _deliver(self, data: np.ndarray) -> None:
        """Play a whole utterance, following the output device if it moves."""
        written = self._write(self._current_stream(), data)
        if written == len(data) or self._cut.is_set():
            return
        # Short write with nothing cutting in means the device went away
        # mid-utterance: the headset walked out of range, the dock was pulled.
        # Finish on whatever is default now, and only try once -- a second
        # failure is a real fault, not a device change, and retrying it would
        # spin.
        #
        # Closed explicitly, because the probe may still name this device: macOS
        # can take a moment to promote a replacement, and `_current_stream` would
        # otherwise see no change and hand back the same dead stream.
        logger.info("Output device went away mid-utterance. Resuming on the new one.")
        self._close()
        self._write(self._current_stream(), data, written)

    def _current_stream(self) -> sounddevice.OutputStream:
        """The open stream, bound to the device macOS would play on now.

        Checked before every utterance rather than on a timer. The check is about
        15 us and the reopen it guards 130 to 190 ms depending on the device, so
        checking less often saves nothing measurable, and a timer would go on
        playing to a device David stopped listening to until it next fired. Per
        utterance is also the only cadence at which a reopen is free of
        consequence: between utterances there is nothing playing to interrupt.
        """
        if self._stream is None or device.default_output() != self._device:
            return self._open()
        return self._stream

    def _open(self) -> sounddevice.OutputStream:
        """Bind a fresh stream to whatever macOS currently plays on.

        By PortAudio's own default, but only ever just after a `rescan` -- which
        is the whole trick, because that is what makes "PortAudio's default"
        current. It re-reads the same CoreAudio property the probe reads, so the
        two agree by construction and there is no name to match up.
        """
        # Probed before the re-scan, so a switch during the reopen reads as a
        # difference on the next utterance rather than as one already handled.
        wanted = device.default_output()
        self._close()
        device.rescan()
        self._stream = sounddevice.OutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32"
        )
        self._stream.start()
        self._device = wanted
        logger.info("Playback on %s.", device.output_name())
        return self._stream

    def _close(self) -> None:
        """Release the device, tolerating one that has already disappeared.

        Aborts rather than stops. `stop` waits for the device to drain, which is
        about 90 ms of the reopen and is spent playing the tail of the last
        utterance to the speakers David just walked away from -- or, when the
        device has physically gone, waiting on hardware that will never drain.
        Nothing here ever closes a device it means to keep listening to.
        """
        if self._stream is not None:
            with contextlib.suppress(sounddevice.PortAudioError):
                self._stream.abort()
                self._stream.close()
        self._stream = None
        self._device = None

    def _write(
        self, stream: sounddevice.OutputStream, data: np.ndarray, start: int = 0
    ) -> int:
        """Write in chunks, so interrupt can cut in mid-utterance.

        :returns: How far it got. Short of `len(data)` means either an interrupt
            or a device that stopped accepting samples.
        """
        for offset in range(start, len(data), WRITE_CHUNK):
            if self._cut.is_set():
                logger.debug("Cut playback after %d samples.", offset)
                return offset
            try:
                stream.write(data[offset : offset + WRITE_CHUNK])
            except sounddevice.PortAudioError as error:
                logger.warning("Device stopped accepting samples: %s", error)
                return offset
        return len(data)


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
