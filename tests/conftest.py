"""Shared fixtures."""

import types

import pytest

from maggid import identity


class FakeContext:
    """Minimal stand-in for an MCP context."""

    def __init__(self, root: str | None, session_id: str | None = "s1") -> None:
        self._root = root
        self.session_id = session_id

    async def list_roots(self):
        if self._root is None:
            raise RuntimeError("client does not support roots")
        return [types.SimpleNamespace(uri=f"file://{self._root}")]


@pytest.fixture
def context():
    """Factory for a fake MCP context: context(root, session_id)."""
    return FakeContext


@pytest.fixture
def voices_dir(tmp_path, monkeypatch):
    """A clip on disk for every pooled voice."""
    monkeypatch.setattr(identity, "VOICES_DIR", tmp_path)
    for preset in identity.VOICE_IDS:
        (tmp_path / f"{preset}.wav").touch()
    return tmp_path
