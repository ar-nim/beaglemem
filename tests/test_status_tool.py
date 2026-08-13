"""Tests for the on-demand status tool + prefetch purity.

The key design decision: build progress and rebuild notices must NOT be
injected into prefetch context (that pollutes the model context on every
turn). Instead, a `beaglemem_status` tool reports them on demand.
"""
import os

from beaglemem import BeagleMemoryProvider
from beaglemem.__init__ import _render_progress


def test_render_progress_midpoint():
    out = _render_progress(42, 100)
    assert "42%" in out


def test_render_progress_zero():
    out = _render_progress(0, 100)
    assert "0%" in out


def test_render_progress_empty_total():
    assert _render_progress(0, 0) == ""


def test_prefetch_never_injects_progress_bar(tmpdir):
    """Prefetch output must be pure memory context — no progress bar."""
    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmpdir))
    p._store = __import__("beaglemem.store", fromlist=["BeagleStore"]).BeagleStore(
        os.path.join(str(tmpdir), "t.db"), create=True
    )
    p._build_progress = (42, 100)
    out = p.prefetch("query")
    assert "Building memory vectors" not in out
    assert "42%" not in out


def test_prefetch_never_injects_rebuild_notice(tmpdir):
    """The rebuild notice must NOT ride prefetch either."""
    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmpdir))
    p._store = __import__("beaglemem.store", fromlist=["BeagleStore"]).BeagleStore(
        os.path.join(str(tmpdir), "t.db"), create=True
    )
    p._pending_notice = "⚠️ rebuild happening"
    out = p.prefetch("query")
    assert "rebuild" not in out


def test_status_tool_reports_progress(tmpdir):
    """beaglemem_status reports the in-flight build progress."""
    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmpdir))
    p._build_progress = (42, 100)
    out = p.handle_tool_call("beaglemem_status", {})
    assert "42%" in out
    assert "Building memory vectors" in out


def test_status_tool_reports_idle(tmpdir):
    """beaglemem_status says not_built when no build has run."""
    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmpdir))
    p._build_progress = None
    out = p.handle_tool_call("beaglemem_status", {})
    assert "not_built" in out


def test_status_tool_reports_unknown_tool(tmpdir):
    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmpdir))
    out = p.handle_tool_call("not_a_tool", {})
    assert "unknown" in out.lower()


def test_status_tool_schema_exposed(tmpdir):
    """The status tool must be in get_tool_schemas."""
    p = BeagleMemoryProvider()
    names = [s["name"] for s in p.get_tool_schemas()]
    assert "beaglemem_status" in names
