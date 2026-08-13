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


def test_notice_surfaced_once_in_prefetch(tmpdir):
    """A pending rebuild notice is prepended to prefetch output exactly once."""
    p = _make_provider(tmpdir)
    p._pending_notice = "⚠️ test rebuild notice"

    out1 = p.prefetch("something that matches nothing")
    assert "test rebuild notice" in out1
    # cleared after first surfacing
    assert p._pending_notice is None

    out2 = p.prefetch("still nothing")
    assert "test rebuild notice" not in out2


def test_notice_not_shown_without_notice(tmpdir):
    """No notice → prefetch output unchanged."""
    p = _make_provider(tmpdir)
    p._pending_notice = None
    out = p.prefetch("nothing matches")
    assert "⚠️" not in out
