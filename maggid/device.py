"""Whether the output device moved, and making PortAudio notice that it did.

PortAudio decides two things when it initializes and never revisits either: what
hardware exists, and which of it is "the default output". A stream held open
across a Bluetooth headset connecting therefore keeps writing to the speakers it
was born on, and the headset is not even in the device list to switch to.

Re-initializing PortAudio refreshes both, because it re-reads the same CoreAudio
property. But it invalidates every open stream, so it cannot be what *notices* a
change -- and holding one stream open is worth about 200 ms an utterance on
Bluetooth, so giving that up to keep this module simple is not a trade available.
CoreAudio answers the other half directly: one property read, harmless to an open
stream, cheap enough to do before every utterance.

So the split is: CoreAudio says *whether* the output moved, PortAudio gets kicked
so it can follow. Only the first needs the HAL, and it needs one integer from it.
Naming the device is PortAudio's job -- after a re-scan the two agree by
construction, and PortAudio knows the name of the device it actually opened.
"""

import ctypes
import ctypes.util
import logging
import struct
import sys

import sounddevice

logger = logging.getLogger(__name__)


def default_output() -> int | None:
    """CoreAudio's id for the device macOS would play on right now.

    An opaque token: compared for equality against the one a stream was opened
    on, and nothing else. The id rather than the display name because names are
    not unique and this is the one place that would silently do the wrong thing
    about it -- one Bluetooth headset can register twice, under one name, and
    both entries can be outputs PortAudio will happily open.

    :returns: None when the platform, the probe, or the machine cannot answer,
        which callers should read as "no opinion" and leave to PortAudio.
    """
    if sys.platform != "darwin" or _core_audio is None:
        return None
    address = _PropertyAddress(_DEFAULT_OUTPUT_DEVICE, _SCOPE_GLOBAL, _ELEMENT_MAIN)
    into = ctypes.c_uint32()
    size = ctypes.c_uint32(ctypes.sizeof(into))
    status = _core_audio.AudioObjectGetPropertyData(
        _SYSTEM_OBJECT,
        ctypes.byref(address),
        0,
        None,
        ctypes.byref(size),
        ctypes.byref(into),
    )
    if status != 0:
        logger.debug("Could not read the default output device: %d", status)
        return None
    # kAudioObjectUnknown is zero, which is how a machine with no output device
    # at all answers. Same meaning as a failed read.
    return into.value or None


def output_name() -> str:
    """What PortAudio calls the device it just opened, for the log line.

    Meaningful only straight after a `rescan`, since that is the only moment
    PortAudio's idea of "the default output" is current. That is also the only
    place it is called, which is why this reads a cache rather than the hardware.

    Asked of PortAudio and not the HAL: after the re-scan it is the same answer,
    it costs 1 us against 43 us, and it names the device actually opened rather
    than the one a probe saw a moment earlier. Getting it from CoreAudio instead
    would mean a second framework, a CFString to own and release, and a Boolean
    return whose upper bits arm64 leaves undefined -- all to duplicate a string
    PortAudio is already holding.
    """
    try:
        return sounddevice.query_devices(kind="output")["name"]
    except (sounddevice.PortAudioError, ValueError, KeyError) as error:
        logger.debug("PortAudio would not name its default output: %s", error)
        return "an unnamed device"


def rescan() -> None:
    """Make PortAudio re-read the hardware, and with it "the default output".

    The only way to refresh either, and it leaves every open stream a dangling
    pointer that raises on the next write. Close first.
    """
    sounddevice._terminate()
    sounddevice._initialize()


# --- CoreAudio, via ctypes -------------------------------------------------
#
# One property read is not enough of the HAL to justify a dependency on pyobjc,
# which would pull in the whole bridge to fetch a single integer.


def _fourcc(code: str) -> int:
    """A CoreAudio selector, which are four-byte ASCII constants."""
    return struct.unpack(">I", code.encode())[0]


_SYSTEM_OBJECT = 1  # kAudioObjectSystemObject
_DEFAULT_OUTPUT_DEVICE = _fourcc("dOut")
_SCOPE_GLOBAL = _fourcc("glob")
_ELEMENT_MAIN = 0


class _PropertyAddress(ctypes.Structure):
    """AudioObjectPropertyAddress: what to read, in which scope."""

    _fields_ = [
        ("selector", ctypes.c_uint32),
        ("scope", ctypes.c_uint32),
        ("element", ctypes.c_uint32),
    ]


def _core_audio_framework() -> ctypes.CDLL | None:
    if sys.platform != "darwin":
        return None
    path = ctypes.util.find_library("CoreAudio")
    return ctypes.CDLL(path) if path else None


_core_audio = _core_audio_framework()

if _core_audio is not None:
    _core_audio.AudioObjectGetPropertyData.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(_PropertyAddress),
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
    ]
    _core_audio.AudioObjectGetPropertyData.restype = ctypes.c_int32
