"""Pytest bootstrap: make the REAL Hermes ABC importable during tests.

beaglemem's plugin tests verify ABC conformance (isinstance checks, lifecycle
hooks, tool-call JSON contract). Those tests only exercise the REAL contract
when `agent.memory_provider` (Hermes core) is importable — otherwise the
plugin's standalone fallback (`MemoryProvider = object`) is tested instead,
which proves nothing about Hermes compatibility.

Hermes convention: the source tree lives at `~/.hermes/hermes-agent/` (see
the hermes-agent skill, "Key Paths"). We add it to sys.path when present so
`from agent.memory_provider import MemoryProvider` succeeds and the
conformance tests run against the genuine ABC.

When Hermes is NOT installed (e.g. public CI, standalone contributor), the
path is absent and tests fall back to the plugin's standalone degradation —
`test_register_standalone_is_safe` covers that path explicitly.
"""
import os
import sys

_HERMES_SOURCE = os.path.expanduser("~/.hermes/hermes-agent")
if os.path.isdir(_HERMES_SOURCE) and _HERMES_SOURCE not in sys.path:
    sys.path.insert(0, _HERMES_SOURCE)
