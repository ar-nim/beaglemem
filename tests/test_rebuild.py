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
    consumed_sentences > 0 in the DB meta, so a later load is not
    false-flagged as a stub. Zero JSON sidecars."""
    data_dir = tmp_path / "beaglemem-data"
    data_dir.mkdir()
    _make_big_corpus(str(tmp_path / "state.db"), n=50)
    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmp_path))
    p._initial_build(str(tmp_path / "state.db"), "state_db", str(data_dir), str(data_dir / "beaglemem.db"))
    meta = p._store.all_meta()
    assert meta.get("corpus_source") == "state_db"
    assert int(meta.get("consumed_sentences", 0)) > 0
    assert len(p._store.vocab_words()) > 0
    assert os.path.exists(str(data_dir / "beagle_mem.npy"))
    assert not os.path.exists(str(data_dir / "beagle_vocab.json"))


def test_initial_build_writes_watermark(tmp_path):
    """_initial_build must stamp the DB meta last_seen_id with the max message
    id so the next on_session_end does NOT re-read the whole corpus and
    double-count co-occurrence into the fresh model."""
    data_dir = tmp_path / "beaglemem-data"
    data_dir.mkdir()
    _make_big_corpus(str(tmp_path / "state.db"), n=50)
    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmp_path))
    p._initial_build(str(tmp_path / "state.db"), "state_db", str(data_dir), str(data_dir / "beaglemem.db"))
    assert p._store.get_meta("last_seen_id") == 50  # max id of the 50-message corpus
    assert not os.path.exists(str(data_dir / "last_update.json"))


# --- Phase 7 (v0.3): load path — DB meta, fingerprint, encoder version ---

def _make_built_state(tmp_path, encoder_version=None, dim=64, window=2,
                      fingerprint_override=None, raw_override=None,
                      n_facts=2, n_words=3):
    """Create a full valid v0.3 state: DB (vocab+meta), beagle_mem.npy,
    fact_vectors.npy, and a config.yaml matching the build shape."""
    from beaglemem.fingerprint import ENCODER_VERSION, tokenizer_fingerprint
    from beaglemem.corpus import WORD_RE
    from beaglemem.store import BeagleStore
    data_dir = tmp_path / "beaglemem-data"
    data_dir.mkdir()
    _make_small_corpus(str(tmp_path / "state.db"), n=5)

    mem = np.random.RandomState(0).randn(n_words, dim).astype(np.float32)
    np.save(str(data_dir / "beagle_mem.npy"), mem)
    fv = np.random.RandomState(1).randn(n_facts, dim).astype(np.float32)
    np.save(str(data_dir / "fact_vectors.npy"), fv)

    store = BeagleStore(str(data_dir / "beaglemem.db"), create=True)
    for i in range(n_facts):
        store.add(f"fact content number {i} with enough words here")
    current_fp = tokenizer_fingerprint(regex=WORD_RE.pattern, stemmer=None,
                                       dim=dim, window=window)
    store.persist_model(
        ["word_a", "word_b", "word_c"],
        {"word_a": 5, "word_b": 3, "word_c": 2},
        {
            "dim": dim, "window": window, "min_count": 2,
            "tokenizer_fingerprint": fingerprint_override or current_fp,
            "regex": WORD_RE.pattern, "stemmer": None,
            "consumed_sentences": 100, "corpus_source": "state_db",
            "last_seen_id": 5,
            "encoder_version": encoder_version or ENCODER_VERSION,
        },
    )
    if raw_override:
        for k, v in raw_override.items():
            store.set_meta(k, v)
    store.close()
    (tmp_path / "config.yaml").write_text(
        f"plugins:\n  beaglemem:\n    dim: {dim}\n    window: {window}\n"
    )
    return str(data_dir)


def test_save_fact_cache_atomic_no_json(tmp_path):
    """_save_fact_cache writes ONLY fact_vectors.npy, atomically. No
    fact_ids.json / encoder manifest — ids come from the DB, encoder_version
    lives in DB meta (v0.3 zero-JSON layout)."""
    data_dir = tmp_path / "beaglemem-data"
    data_dir.mkdir()
    p = BeagleMemoryProvider()
    p._data_dir = str(data_dir)
    p._fact_vectors = (np.zeros((2, 64), dtype=np.float32), [1, 2])
    p._save_fact_cache()
    assert (data_dir / "fact_vectors.npy").exists()
    assert not (data_dir / "fact_vectors.npy.tmp").exists()  # no orphan tmp
    assert not (data_dir / "fact_ids.json").exists()
    assert not (data_dir / "fact_cache_meta.json").exists()
    assert np.load(str(data_dir / "fact_vectors.npy")).shape == (2, 64)


def test_matching_encoder_version_no_false_reencode(tmp_path):
    """A fact cache whose encoder_version matches must load WITHOUT firing
    the 'encoder changed' notice. (Regression: the missing-meta false alarm.)"""
    from beaglemem.fingerprint import ENCODER_VERSION
    _make_built_state(tmp_path, encoder_version=ENCODER_VERSION)
    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmp_path))
    assert p._fact_vectors is not None
    assert p._pending_notice is None


def test_encoder_version_mismatch_triggers_reencode(tmp_path):
    """A genuine encoder version bump re-encodes the fact cache at load and
    surfaces a notice — the guard is real, now driven by DB meta."""
    _make_built_state(tmp_path, encoder_version="idf-v0-OLD")
    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmp_path))
    assert p._pending_notice is not None
    assert "encoder" in p._pending_notice.lower()
    assert p._fact_vectors is not None          # re-encoded at load
    assert p._fact_vectors[0].shape[0] == len(p._store.fact_ids())


def test_missing_fact_cache_no_false_encoder_notice(tmp_path):
    """A missing/deleted fact cache must NOT fire 'encoder changed'. The cache
    is simply absent, not version-mismatched."""
    data_dir = tmp_path / "beaglemem-data"
    data_dir.mkdir()
    np.save(str(data_dir / "fact_vectors.npy"),
            np.zeros((2, 64), dtype=np.float32))
    _make_small_corpus(str(tmp_path / "state.db"), n=5)
    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmp_path))
    assert p._fact_vectors is None
    assert p._pending_notice is None


def test_fingerprint_hash_corruption_self_heals(tmp_path):
    """A hand-corrupted hash (raw inputs still match) recomputes the hash and
    does NOT rebuild — vectors keep serving."""
    from beaglemem.fingerprint import tokenizer_fingerprint
    from beaglemem.corpus import WORD_RE
    _make_built_state(tmp_path, fingerprint_override="deadbeef00000000")
    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmp_path))
    assert p._model is not None
    assert p._force_rebuild is False
    expected = tokenizer_fingerprint(regex=WORD_RE.pattern, stemmer=None,
                                     dim=64, window=2)
    assert p._store.get_meta("tokenizer_fingerprint") == expected


def test_window_change_soft_stale_keeps_serving(tmp_path):
    """A genuine non-structural config change (window) keeps the old vectors
    serving and schedules a background rebuild — vectors are never dropped."""
    _make_built_state(tmp_path)  # built with window=2
    (tmp_path / "config.yaml").write_text(
        "plugins:\n  beaglemem:\n    dim: 64\n    window: 3\n"
    )
    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmp_path))
    assert p._model is not None              # KEEP SERVING
    assert p._initial_build_started is True  # background rebuild scheduled


def test_dim_change_hard_rebuild(tmp_path):
    """A genuine dim change makes the old vectors structurally unusable →
    hard rebuild (non-destructive), with a notice."""
    _make_built_state(tmp_path)  # built with dim=64
    (tmp_path / "config.yaml").write_text(
        "plugins:\n  beaglemem:\n    dim: 128\n    window: 2\n"
    )
    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmp_path))
    assert p._model is None
    assert p._force_rebuild is True
    assert p._pending_notice is not None


# --- Phase 8 (v0.4): watermark-based stub/reset detection ---
#
# The stub check used COUNT(*) on state.db — a full scan over the content
# column. Hermes session pruning (DELETE FROM messages) collapses COUNT, and
# with it the `_msg_count > 100` gate that arms the check. MAX(id) is the
# AUTOINCREMENT high-water mark: it survives DELETE, so it is pruning-immune.
# last_seen_id (the ingest watermark) lives in beaglemem.db, outside Hermes'
# prune radius — so comparing last_seen_id against MAX(id) detects stubs and
# corpus resets without a slow COUNT(*).

def _make_watermark_state(tmp_path, n_msgs=150, last_seen_id=None,
                          dim=64, window=2):
    """A valid v0.3 state where the model watermark (last_seen_id) can differ
    from the corpus MAX(id), to exercise stub/reset detection. Corpus uses
    AUTOINCREMENT so MAX(id) survives DELETE (the prune case)."""
    from beaglemem.fingerprint import ENCODER_VERSION, tokenizer_fingerprint
    from beaglemem.corpus import WORD_RE
    from beaglemem.store import BeagleStore
    data_dir = tmp_path / "beaglemem-data"
    data_dir.mkdir()
    _make_big_corpus(str(tmp_path / "state.db"), n=n_msgs)

    n_words = 5
    mem = np.random.RandomState(0).randn(n_words, dim).astype(np.float32)
    np.save(str(data_dir / "beagle_mem.npy"), mem)
    fv = np.random.RandomState(1).randn(2, dim).astype(np.float32)
    np.save(str(data_dir / "fact_vectors.npy"), fv)

    store = BeagleStore(str(data_dir / "beaglemem.db"), create=True)
    for i in range(2):
        store.add(f"fact content number {i} with enough words here")
    current_fp = tokenizer_fingerprint(regex=WORD_RE.pattern, stemmer=None,
                                       dim=dim, window=window)
    if last_seen_id is None:
        last_seen_id = n_msgs  # healthy: watermark == corpus extent
    store.persist_model(
        ["word_a", "word_b", "word_c", "word_d", "word_e"],
        {"word_a": 5, "word_b": 3, "word_c": 2, "word_d": 2, "word_e": 2},
        {
            "dim": dim, "window": window, "min_count": 2,
            "tokenizer_fingerprint": current_fp,
            "regex": WORD_RE.pattern, "stemmer": None,
            "consumed_sentences": 100, "corpus_source": "state_db",
            "last_seen_id": last_seen_id,
            "encoder_version": ENCODER_VERSION,
        },
    )
    store.close()
    (tmp_path / "config.yaml").write_text(
        f"plugins:\n  beaglemem:\n    dim: {dim}\n    window: {window}\n"
    )
    return str(data_dir)


def test_watermark_stub_detected(tmp_path):
    """A model whose watermark (last_seen_id) covers <10% of the corpus id
    extent is a stub — cleared + rebuild notice, detected via MAX(id)."""
    _make_watermark_state(tmp_path, n_msgs=150, last_seen_id=5)
    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmp_path))
    assert p._model is None
    assert p._force_rebuild is True
    assert p._pending_notice is not None


def test_watermark_prune_does_not_flag(tmp_path):
    """Pruning old (low-id) messages must NOT flag a healthy model as a stub.
    MAX(id) survives DELETE (AUTOINCREMENT), so a model built over the full
    corpus (last_seen == MAX(id)) keeps serving even after pruning leaves few
    live rows — the COUNT(*)-based check would false-negative here."""
    _make_watermark_state(tmp_path, n_msgs=150, last_seen_id=150)
    # prune: delete the 140 oldest messages, leaving 10 live rows
    conn = sqlite3.connect(str(tmp_path / "state.db"))
    conn.execute("DELETE FROM messages WHERE id <= 140")
    conn.commit()
    conn.close()
    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmp_path))
    assert p._model is not None            # KEEP SERVING
    assert p._force_rebuild is False
    assert p._pending_notice is None


def test_watermark_reset_detected(tmp_path):
    """A corpus whose id space collapsed below the model watermark (DROP/
    recreate renumbered ids) is detected via the bidirectional guard — the
    corpus shrank 10× since the build, so the model is stale and rebuilds."""
    _make_watermark_state(tmp_path, n_msgs=5, last_seen_id=500)
    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmp_path))
    assert p._model is None
    assert p._force_rebuild is True
    assert p._pending_notice is not None
