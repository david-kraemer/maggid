"""Tests for path conversion."""

import pathlib

from maggid import paths


def test_resolve_path_accepts_a_string(tmp_path):
    assert paths.resolve_path(str(tmp_path)) == tmp_path


def test_resolve_path_expands_a_leading_tilde():
    assert paths.resolve_path("~/x.wav") == pathlib.Path.home() / "x.wav"


def test_resolve_path_makes_no_directory(tmp_path):
    """Reading a path must not touch the filesystem."""
    target = tmp_path / "absent" / "x.json"
    paths.resolve_path(target)
    assert not target.parent.exists()


def test_writable_path_makes_the_parent(tmp_path):
    target = paths.writable_path(tmp_path / "deep" / "nested" / "x.json")
    assert target.parent.is_dir()


def test_writable_path_is_a_no_op_on_an_existing_parent(tmp_path):
    first = paths.writable_path(tmp_path / "x.json")
    assert paths.writable_path(first) == first
