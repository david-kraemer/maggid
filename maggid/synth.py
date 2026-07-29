"""Speech synthesis. All inference runs on one thread."""

import asyncio
import concurrent.futures
import functools
import logging
import pathlib
import time

import mlx.core as mx
from mlx.nn.layers import Module
from mlx_audio.tts.utils import load_model as mlx_load_model

logger = logging.getLogger(__name__)

# Chatterbox Turbo, 8-bit. Measured on an M5 Max: 190 ms for a short phrase,
# against 330 ms at fp16 and 660 ms for base Chatterbox. 4-bit ties 8-bit on
# speed and memory, so nothing is gained below 8 bits. Kokoro took 37 ms but
# gave only a handful of distinguishable voices. Turbo clones from a reference
# clip, which is what lets each workspace sound different.
#
# Turbo has no emotion control, whatever the API suggests. Both generate() and
# prepare_conditionals() accept `exaggeration` and do nothing with it, because
# T3Config.turbo() sets emotion_adv=False and the emotion_adv_fc projection is
# never built. Measured: varying it from 0.0 to 1.0 changes the audio less than
# two identical calls differ from each other. Base Chatterbox does support it,
# but costs 660 ms against 190 ms. Not worth it. Priority and wording separate
# the channels; voice and label separate the workspaces.
HUGGINGFACE_REPO = "mlx-community/Chatterbox-Turbo-TTS-8bit"

# MLX's Metal command buffer is not safe for concurrent evaluation. Two threads
# in generate() abort the process with "Completed handler provided after commit
# call". The lru_cache on load_model is not atomic either, so a concurrent cold
# start races inside transformers. One thread fixes both and costs nothing,
# because inference takes about 190 ms.
_synthesis = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="tts-synth"
)


async def synthesize(text: str, ref_audio: pathlib.Path | None = None) -> mx.array:
    """Run inference off the event loop, serialized against other callers."""
    return await asyncio.get_running_loop().run_in_executor(
        _synthesis, functools.partial(generate, text, ref_audio=ref_audio)
    )


async def preload() -> None:
    """Load the model in the synthesis thread. Logs a failure, does not raise."""
    start = time.monotonic()
    try:
        await asyncio.get_running_loop().run_in_executor(_synthesis, load_model)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Warmup failed. The first call pays the load cost.")
        return
    logger.info("Warmup finished in %.1fs.", time.monotonic() - start)


def generate(text: str, ref_audio: pathlib.Path | None = None) -> mx.array:
    """Raw audio for the text.

    Runs in the synthesis thread, so the lazy MLX graph is forced here.
    Otherwise evaluation lands back on the caller's thread. Speed is not a
    parameter, because Chatterbox ignores it and playback applies it instead.
    """
    model = load_model()
    model._conds = (
        conditionals(ref_audio) if ref_audio is not None else model._builtin_conds
    )
    chunks = [r.audio for r in model.generate(text=text)]
    if not chunks:
        raise RuntimeError("No audio generated")
    audio = mx.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    mx.eval(audio)
    return audio


@functools.lru_cache
def load_model(path: str = HUGGINGFACE_REPO) -> Module:
    """The loaded model, warmed up. Loaded once per process."""
    logger.info("Loading model %s ...", path)
    model = mlx_load_model(path)
    # Compile the Metal shaders, so the first real call is fast.
    list(model.generate("warmup"))
    # Keep the built-in voice before a clone overwrites the slot. A call with no
    # ref_audio must not inherit whichever voice was cloned last.
    model._builtin_conds = model._conds
    logger.info("Model loaded and warmed up.")
    return model


@functools.lru_cache
def conditionals(ref_audio: pathlib.Path):
    """The encoded reference clip. Encoded once per clip and kept.

    Passing ref_audio to generate() re-encodes the clip every call: 950 ms per
    utterance instead of 190 ms. Encoding once per voice and swapping the model's
    slot gets that back. This is safe because all synthesis is serialized on one
    thread, so no two callers touch the slot at once.
    """
    model = load_model()
    # mlx-audio wants a str here, so the Path stops at the boundary.
    model.prepare_conditionals(str(ref_audio))
    return model._conds
