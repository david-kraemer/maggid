"""Maggid: a TTS MCP server for Claude Code.

A maggid is an itinerant preacher — one who tells, rather than one who expounds. The
name fits a server whose whole job is to say what happened.

Speaks notifications and narration through Chatterbox Turbo on Apple Silicon, via
MLX-audio. Provides `notify` for short alerts, `speak` for longer narration, and
`interrupt` to stop playback and discard the backlog. Runs as a per-session stdio server
or as one shared daemon.
"""

# server.__all__ is the one list of package exports. Ruff cannot check names through a
# star import, so the rule is off here.
from .server import *  # noqa: F403
