"""Maggid: a TTS MCP server for Claude Code.

A maggid is an itinerant preacher — one who tells, rather than one who rules on
the law. The name fits a server whose whole job is to say what happened.

Speaks notifications and narration through Chatterbox Turbo on Apple Silicon,
via MLX-audio. Provides `notify` for short alerts, `speak` for longer narration,
and `interrupt` to stop playback and discard the backlog. Runs as a per-session
stdio server or as one shared daemon.
"""

from maggid.server import init, main

__all__ = ["init", "main"]
