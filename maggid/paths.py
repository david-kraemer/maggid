"""Path conversion. One function to read a path, one to write to it."""

import os
import pathlib

__all__ = ["StrPath", "resolve_path", "writable_path"]

type StrPath = str | os.PathLike[str]


def resolve_path(path: StrPath) -> pathlib.Path:
    """The argument as a Path, with a leading ~ expanded.

    Every path that enters this package goes through here, so no other function has
    to accept more than one representation.
    """
    return pathlib.Path(path).expanduser()


def writable_path(path: StrPath) -> pathlib.Path:
    """The argument as a Path, with its parent directory in place.

    The directory creation is a no-op when the directory exists. Call this at a write
    site. Use resolve_path to read.
    """
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
