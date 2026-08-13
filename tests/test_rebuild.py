"""Rebuild-on-mismatch tests — Phase 0 Task 0.4.1.

The riskiest behavior in v0.2: when the tokenizer fingerprint changes (e.g.
upgrading v0.1 → v0.2), the plugin must:
1. Clear BOTH the model AND the stale fact cache (not just the model).
2. Force a rebuild even on a small corpus (<100 messages) — a tokenizer
   change is mandatory, not the opportunistic first-run auto-build.
"""
import json
import os
import sqlite3

import numpy as np

from beaglemem import BeagleMemoryProvider
from beaglemem.vectors import BeagleModel


def _make_legacy_model(data_dir: str, dim: int = 64, window: int = 2) -> None:
    """Build + save a model with a NON-matching fingerprint (simulates v0.1)."""
    os.makedirs(data_dir, exist_ok=True)
    model = BeagleModel(dim=dim, window=window, min_count=2)
    model.add_sentence(["hello", "world", "test"])
    model.tokenizer_fingerprint = "LEGACY-00000000000000"
    model.save(data_dir)


def _make_stale_fact_cache(data_dir: str) -> None:
    """Write a fact_vectors.npy + fact_ids.json (stale — built vs old model)."""
    np.save(os.path.join(data_dir, "fact_vectors.npy"),
            np.zeros((2, 64), dtype=np.float32))
    with open(os.path.join(data_dir, "fact_ids.json"), "w") as fh:
        json.dump([1, 2], fh)


def _make_small_corpus(corpus_db: str, n: int = 10) -> None:
    """Create a state.db with a small number of messages (<100)."""
    conn = sqlite3.connect(corpus_db)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS messages "
        "(id INTEGER PRIMARY KEY, role TEXT, content TEXT)"
    )
    for i in range(n):
        conn.execute(
            "INSERT INTO messages (role, content) VALUES (?, ?)",
            ("user", f"test message {i}"),
        )
    conn.commit()
    conn.close()


def test_tokenizer_change_clears_model_and_fact_cache(tmp_path):
    """A tokenizer fingerprint mismatch must clear BOTH the model and the
    stale fact cache — not just the model. Otherwise the plugin serves
    semantic recall built against the OLD tokenizer."""
    data_dir = tmp_path / "beaglemem-data"
    _make_legacy_model(str(data_dir))
    _make_stale_fact_cache(str(data_dir))

    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmp_path))

    # Model cleared (fingerprint mismatch → rebuild needed)
    assert p._model is None
    # Fact cache cleared TOO (the Q5 bug: it was left loaded before)
    assert p._fact_vectors is None


def test_tokenizer_change_rebuilds_small_corpus(tmp_path):
    """A tokenizer change forces a rebuild even when the corpus is small
    (<100 messages). It must NOT silently skip via the auto-build guard."""
    data_dir = tmp_path / "beaglemem-data"
    _make_legacy_model(str(data_dir))
    _make_small_corpus(str(tmp_path / "state.db"), n=10)

    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmp_path))

    # Model was cleared (mismatch) and a rebuild was scheduled despite <100 msgs
    assert p._model is None
    assert p._initial_build_started is True
