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


def test_progress_bar_via_status_tool(tmpdir):
    """Progress bar is reported via beaglemem_status, NOT prefetch."""
    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmpdir))
    p._store = __import__("beaglemem.store", fromlist=["BeagleStore"]).BeagleStore(
        os.path.join(str(tmpdir), "t.db"), create=True
    )
    p._build_progress = (42, 100)
    # prefetch: pure (no bar)
    out_prefetch = p.prefetch("query")
    assert "Building memory vectors" not in out_prefetch
    # status tool: has bar
    out_status = p.handle_tool_call("beaglemem_status", {})
    assert "Building memory vectors" in out_status
    assert "42%" in out_status


def test_no_progress_bar_after_build(tmpdir):
    """No build in progress → no progress bar anywhere."""
    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmpdir))
    p._store = __import__("beaglemem.store", fromlist=["BeagleStore"]).BeagleStore(
        os.path.join(str(tmpdir), "t.db"), create=True
    )
    p._build_progress = None
    out = p.prefetch("query")
    assert "Building memory vectors" not in out
