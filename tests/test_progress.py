"""Tests for the vector-build progress bar."""
import os

from beaglemem import BeagleMemoryProvider
from beaglemem.__init__ import _render_progress


def test_render_progress_midpoint():
    out = _render_progress(42, 100)
    assert "42%" in out
    assert "42" in out and "100" in out


def test_render_progress_zero():
    out = _render_progress(0, 100)
    assert "0%" in out


def test_render_progress_complete_clamped():
    out = _render_progress(120, 100)
    assert "100%" in out  # clamped, never over 100


def test_render_progress_empty_total():
    assert _render_progress(0, 0) == ""


def test_progress_bar_surfaced_in_prefetch(tmpdir):
    """While a build is in progress, prefetch output includes the bar."""
    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmpdir))
    p._store = __import__("beaglemem.store", fromlist=["BeagleStore"]).BeagleStore(
        os.path.join(str(tmpdir), "t.db"), create=True
    )
    p._build_progress = (42, 100)
    out = p.prefetch("query")
    assert "Building memory vectors" in out
    assert "42%" in out


def test_no_progress_bar_after_build(tmpdir):
    """No build in progress → no progress bar."""
    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmpdir))
    p._store = __import__("beaglemem.store", fromlist=["BeagleStore"]).BeagleStore(
        os.path.join(str(tmpdir), "t.db"), create=True
    )
    p._build_progress = None
    out = p.prefetch("query")
    assert "Building memory vectors" not in out
