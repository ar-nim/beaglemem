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


# --- Phase 5 (corpus-lifecycle refactor): stub detection ---

def _make_stub_model(data_dir: str) -> None:
    """A tiny model stamped as archive-sourced (the 2026-08-13 stub)."""
    os.makedirs(data_dir, exist_ok=True)
    model = BeagleModel(dim=64, window=2, min_count=1)
    model.add_sentence(["restart", "gateway", "delete"])
    model.corpus_source = "corpus_archive"
    model.save(data_dir)


def _make_big_corpus(corpus_db: str, n: int = 50) -> None:
    conn = sqlite3.connect(corpus_db)
    conn.executescript(
        "CREATE TABLE IF NOT EXISTS messages "
        "(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, "
        " content TEXT, active INTEGER DEFAULT 1, compacted INTEGER DEFAULT 0)"
    )
    for i in range(n):
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES ('s','user',?)",
            (f"unique topic word {i} about severance and termination",),
        )
    conn.commit()
    conn.close()


def test_stub_model_detected_and_flagged(tmp_path):
    """A model stamped corpus_source=corpus_archive against a real state.db
    is a stub — it must be cleared + a rebuild notice set."""
    data_dir = tmp_path / "beaglemem-data"
    _make_stub_model(str(data_dir))
    # >100 messages: the source-mismatch branch is gated on _msg_count > 100
    # (avoid false-flagging tiny fresh corpora) — the corpus must exceed it.
    _make_big_corpus(str(tmp_path / "state.db"), n=150)

    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmp_path))

    # Stub cleared so the full auto-build (scheduled) rebuilds from state.db
    assert p._model is None
    assert p._pending_notice is not None
    assert "rebuild" in p._pending_notice.lower()


# --- Phase 6 (corpus-lifecycle refactor): full-build source stamp ---

def test_initial_build_stamps_source_and_consumed(tmp_path):
    """A full build from state.db must stamp corpus_source='state_db' and a
    consumed_sentences > 0, so a later load is not false-flagged as a stub."""
    data_dir = tmp_path / "beaglemem-data"
    data_dir.mkdir()
    _make_big_corpus(str(tmp_path / "state.db"), n=50)
    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmp_path))
    p._initial_build(str(tmp_path / "state.db"), "state_db", str(data_dir), str(data_dir / "beaglemem.db"))
    loaded = BeagleModel.load(str(data_dir))
    assert loaded.corpus_source == "state_db"
    assert loaded.consumed_sentences > 0


def test_initial_build_writes_watermark(tmp_path):
    """_initial_build must stamp last_update.json with the max message id so
    the next on_session_end does NOT re-read the whole corpus and
    double-count co-occurrence into the fresh model."""
    import json
    data_dir = tmp_path / "beaglemem-data"
    data_dir.mkdir()
    _make_big_corpus(str(tmp_path / "state.db"), n=50)
    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmp_path))
    p._initial_build(str(tmp_path / "state.db"), "state_db", str(data_dir), str(data_dir / "beaglemem.db"))
    stamp_path = os.path.join(str(data_dir), "last_update.json")
    assert os.path.exists(stamp_path)
    with open(stamp_path) as fh:
        stamp = json.load(fh)
    assert stamp["last_seen_id"] == 50  # max id of the 50-message corpus
