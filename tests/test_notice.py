"""Tests for the user-visible rebuild notice."""
import os
import tempfile

import pytest

from beaglemem import BeagleMemoryProvider


def _make_provider(tmpdir):
    """Create a provider initialized in a temp data dir with a store."""
    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=tmpdir)
    # force a store so prefetch has something to query
    p._store = __import__("beaglemem.store", fromlist=["BeagleStore"]).BeagleStore(
        os.path.join(tmpdir, "test.db"), create=True
    )
    return p


def test_notice_surfaced_only_via_status_tool(tmpdir):
    """A pending rebuild notice is NOT injected into prefetch; it is exposed
    only through the on-demand status tool."""
    p = _make_provider(tmpdir)
    p._pending_notice = "⚠️ test rebuild notice"

    # prefetch stays pure (no pollution)
    out1 = p.prefetch("something that matches nothing")
    assert "test rebuild notice" not in out1

    # status tool surfaces it
    out2 = p.handle_tool_call("beaglemem_status", {})
    assert "test rebuild notice" in out2
    # notice is cleared after being reported once
    assert p._pending_notice is None


def test_notice_not_shown_without_notice(tmpdir):
    """No notice → prefetch output unchanged."""
    p = _make_provider(tmpdir)
    p._pending_notice = None
    out = p.prefetch("nothing matches")
    assert "⚠️" not in out
