"""Typing helpers. The trailing underscore keeps the name off `typing`."""

from collections.abc import Callable
from typing import Any, TypeGuard

__all__ = ["guard_type"]


def guard_type[S, T](get: Callable[[S], T]) -> Callable[[Any], TypeGuard[T]]:
    """A membership test built from a lookup. Narrows the type on success.

    `is_voice = guard_type(VOICES.__getitem__)` reads as a predicate, and the mapping
    stays the one record of what exists.
    """

    def guard(o: Any) -> TypeGuard[T]:
        try:
            get(o)
        except KeyError:
            return False
        return True

    return guard
