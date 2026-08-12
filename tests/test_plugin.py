"""Hermes MemoryProvider ABC conformance. Tests the self-contained plugin.

The plugin package is `beaglemem` (single word, valid Python module name).
It imports the MemoryProvider ABC at runtime when Hermes is present, and
degrades gracefully when standalone (tests, library users).
"""
import importlib
import json
import os


def test_plugin_module_importable():
    """beaglemem/__init__.py exposes register() and the provider class."""
    from beaglemem import BeagleMemoryProvider, register
    assert callable(register)


def test_plugin_yaml_valid():
    path = os.path.join(os.path.dirname(__file__), "..", "beaglemem", "plugin.yaml")
    with open(path) as fh:
        import yaml
        meta = yaml.safe_load(fh)
    assert meta["name"] == "beaglemem"
    assert "hooks" in meta
    for h in ("prefetch", "sync_turn", "on_memory_write", "on_session_end", "shutdown"):
        assert h in meta["hooks"], f"missing hook: {h}"


def test_provider_name_is_property():
    """name must be a @property per the ABC convention."""
    from beaglemem import BeagleMemoryProvider
    p = BeagleMemoryProvider()
    assert p.name == "beaglemem"
    # Verify it's a property, not a plain class attribute. Class-level access
    # to a property returns the descriptor object itself (plan's original
    # `type(Cls).name` looked on the METACLASS, which never has it).
    assert isinstance(BeagleMemoryProvider.name, property)


def test_provider_is_available_no_deps():
    """is_available() returns True with no network/API key."""
    from beaglemem import BeagleMemoryProvider
    assert BeagleMemoryProvider().is_available() is True


def test_provider_exposes_three_tools():
    """Tool surface: beaglemem_add, beaglemem_search, beaglemem_feedback.
    beaglemem_probe dropped — fused search is always better than semantic-only."""
    from beaglemem import BeagleMemoryProvider
    p = BeagleMemoryProvider()
    schemas = p.get_tool_schemas()
    names = {s["name"] for s in schemas}
    assert names == {"beaglemem_add", "beaglemem_search", "beaglemem_feedback"}


def test_handle_tool_call_returns_json_string():
    """ABC contract: handle_tool_call must return a JSON string
    (agent/memory_provider.py:186 — 'Must return a JSON string')."""
    from beaglemem import BeagleMemoryProvider
    p = BeagleMemoryProvider()
    result = p.handle_tool_call("beaglemem_search", {"query": "x"})
    assert isinstance(result, str)
    parsed = json.loads(result)  # must parse as JSON
    assert "results" in parsed


def test_register_standalone_is_safe():
    """register() degrades gracefully when Hermes core is absent."""
    from beaglemem import register, _HAS_HERMES
    if not _HAS_HERMES:
        # Standalone: no ctx, just confirm no crash on a dummy ctx
        class DummyCtx:
            def __init__(self):
                self.provider = None
            def register_memory_provider(self, p):
                self.provider = p
        ctx = DummyCtx()
        register(ctx)
        # In standalone mode register() is a no-op (guard), so provider stays None
        assert ctx.provider is None
