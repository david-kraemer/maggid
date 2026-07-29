"""Typing helpers. Named with a trailing underscore to leave `typing` alone."""

from collections.abc import Callable
from typing import Any, TypeGuard

__all__ = ["guard_type"]


def guard_type[S, T](get: Callable[[S], T]) -> Callable[[Any], TypeGuard[T]]:
    """A membership test from a lookup, narrowing the type when it succeeds.

    Turns a total-looking accessor into a partial one the type checker understands:
    `is_voice = guard_type(VOICES.__getitem__)` reads as a predicate at the call site
    while the mapping stays the single source of what exists.
    """

    def guard(o: Any) -> TypeGuard[T]:
        try:
            get(o)
        except KeyError:
            return False
        return True

    return guard
