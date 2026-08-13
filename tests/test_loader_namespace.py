"""Regression test: beaglemem must load under the Hermes user-plugin namespace.

The bug (v0.2): submodules used absolute imports (``from beaglemem.corpus ...``).
When Hermes' plugin loader imports the package, it is registered as
``_hermes_user_memory.beaglemem`` — there is NO top-level ``beaglemem`` on
sys.path. ``__init__.py`` survived via its relative-import fallback, but the
submodules (probe, idf, corpus, adapters, cli) did ``from beaglemem.X import``
and crashed with ModuleNotFoundError.

CI never caught it because the conformance test imports ``from beaglemem ...``
with the repo-root on sys.path, which makes ``beaglemem`` top-level — the real
loader path (synthetic namespace) was never exercised.

This test reproduces the REAL loader: it registers the package under the
synthetic ``_hermes_user_memory.beaglemem`` namespace and asserts every
submodule imports. It does NOT add the repo root to sys.path under the
``beaglemem`` name.
"""
import importlib
import importlib.util
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG_DIR = REPO_ROOT / "beaglemem"

USER_NAMESPACE = "_hermes_user_memory"
MODULE_NAME = f"{USER_NAMESPACE}.beaglemem"


def _register_synthetic_package(name, search_locations):
    """Mirror plugins/memory/__init__.py:_register_synthetic_package."""
    if name in sys.modules:
        return
    spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
    spec.submodule_search_locations = search_locations
    sys.modules[name] = importlib.util.module_from_spec(spec)


def _load_under_user_namespace():
    """Reproduce _load_provider_from_dir()'s synthetic-namespace load."""
    # Ensure the synthetic parent exists.
    _register_synthetic_package(USER_NAMESPACE, [])

    init_file = PKG_DIR / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        MODULE_NAME, str(init_file), submodule_search_locations=[str(PKG_DIR)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = mod

    # Pre-register submodules exactly like the loader does (so relative imports
    # inside the plugin resolve), then exec each — this is where absolute
    # intra-package imports in a submodule would fail.
    for sub_file in sorted(PKG_DIR.glob("*.py")):
        if sub_file.name == "__init__.py":
            continue
        full_sub = f"{MODULE_NAME}.{sub_file.stem}"
        if full_sub not in sys.modules:
            sub_spec = importlib.util.spec_from_file_location(full_sub, str(sub_file))
            sub_mod = importlib.util.module_from_spec(sub_spec)
            sys.modules[full_sub] = sub_mod
            sub_spec.loader.exec_module(sub_mod)  # ← absolute import would raise here

    # Also register + exec the adapters subpackage (imported by corpus.py).
    adapters_dir = PKG_DIR / "adapters"
    if adapters_dir.is_dir():
        adapters_name = f"{MODULE_NAME}.adapters"
        if adapters_name not in sys.modules:
            _register_synthetic_package(adapters_name, [str(adapters_dir)])
        for sub_file in sorted(adapters_dir.glob("*.py")):
            if sub_file.name == "__init__.py":
                continue
            full_sub = f"{adapters_name}.{sub_file.stem}"
            if full_sub not in sys.modules:
                sub_spec = importlib.util.spec_from_file_location(full_sub, str(sub_file))
                sub_mod = importlib.util.module_from_spec(sub_spec)
                sys.modules[full_sub] = sub_mod
                sub_spec.loader.exec_module(sub_mod)

    # Finally exec the top-level __init__.py (its relative imports must work).
    spec.loader.exec_module(mod)
    return mod


def test_loads_under_hermes_user_namespace():
    """beaglemem must load as _hermes_user_memory.beaglemem (no top-level 'beaglemem')."""
    # Remove any sys.path entry that lets 'beaglemem' resolve (the repo root,
    # or its parent). Save and restore after. Also pop any cached top-level
    # 'beaglemem' module — other tests may have imported it (test_plugin.py
    # does `from beaglemem import ...`), and a cached top-level module would
    # mask the submodule-import failure the same way a sys.path entry would.
    saved_path = list(sys.path)
    saved_beaglemem = sys.modules.pop("beaglemem", None)
    filtered = [p for p in sys.path if
                (pathlib.Path(p).resolve() != REPO_ROOT.resolve()
                 and not pathlib.Path(p).resolve().is_relative_to(REPO_ROOT.resolve()))]
    sys.path[:] = filtered

    try:
        mod = _load_under_user_namespace()
    finally:
        sys.path[:] = saved_path
        if saved_beaglemem is not None:
            sys.modules["beaglemem"] = saved_beaglemem

    assert hasattr(mod, "BeagleMemoryProvider")
    assert hasattr(mod, "register")

    p = mod.BeagleMemoryProvider()
    assert p.name == "beaglemem"
    assert p.is_available() is True
    tools = {s["name"] for s in p.get_tool_schemas()}
    assert tools == {"beaglemem_add", "beaglemem_search", "beaglemem_feedback", "beaglemem_status"}

    # Clean up the synthetic modules so they don't leak into other tests.
    for name in list(sys.modules):
        if name == MODULE_NAME or name.startswith(MODULE_NAME + ".") or name == USER_NAMESPACE:
            del sys.modules[name]
