"""Pytest bootstrap: make the REAL Hermes ABC importable during tests.

beaglemem's plugin tests verify ABC conformance (isinstance checks, lifecycle
hooks, tool-call JSON contract). Those tests only exercise the REAL contract
when `agent.memory_provider` (Hermes core) is importable — otherwise the
plugin's standalone fallback (`MemoryProvider = object`) is tested instead,
which proves nothing about Hermes compatibility.

Hermes can be installed three ways; this conftest handles all of them:

1. **Site-packages / pip install** — `agent` is already importable; nothing
   to do.
2. **Git checkout at $HERMES_HOME/hermes-agent** (default
   ~/.hermes/hermes-agent, the convention documented in the hermes-agent
   skill "Key Paths") — injected onto sys.path.
3. **Not installed** (public CI, standalone contributor) — no injection,
   `_HAS_HERMES` stays False, and the two Hermes-dependent tests skip with
   an explicit reason. The other 85 tests still prove standalone behavior.

The skip is honest, not a gap: we do NOT fake conformance with a mock copy
of the ABC (a mock would go stale the moment Hermes changes its contract and
produce false confidence). When Hermes is absent we cannot claim
compatibility, so we don't test it.
"""
import os
import sys


def _try_import_agent() -> bool:
    """Return True if agent.memory_provider is already importable."""
    try:
        import agent.memory_provider  # noqa: F401
        return True
    except ImportError:
        return False


if not _try_import_agent():
    # Not on sys.path yet. Try the git-checkout convention, honoring
    # $HERMES_HOME when set, falling back to the default location.
    hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    candidates = [
        os.path.join(hermes_home, "hermes-agent"),
        os.path.expanduser("~/.hermes/hermes-agent"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)
            if _try_import_agent():
                break
