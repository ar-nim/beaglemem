import sqlite3
from beaglemem.store import MemoryStore, BeagleStore
from beaglemem.fusion import rrf


def test_memory_store_roundtrip():
    docs = [{"id": 1, "text": "hello world test"}]
    assert MemoryStore(docs).documents() == docs


def test_beagle_store_create_add_search(tmp_path):
    """BeagleStore creates its own schema, writes facts, searches via FTS5.
    add() returns autoincrement ID (no explicit fact_id)."""
    path = str(tmp_path / "beaglemem.db")
    store = BeagleStore(path, create=True)
    id1 = store.add("severance letter signed by hr", 0.6)
    id2 = store.add("weather forecast sunny weekend", 0.5)
    assert id1 != id2  # autoincrement gives distinct IDs
    assert store.fts_search("severance") == [id1]
    assert store.fts_search("retrenched") == []
    assert len(store.documents()) == 2
    store.close()


def test_beagle_store_remove(tmp_path):
    path = str(tmp_path / "beaglemem.db")
    store = BeagleStore(path, create=True)
    id1 = store.add("severance letter signed by hr")
    store.remove(id1)
    assert store.fts_search("severance") == []
    assert len(store.documents()) == 0
    store.close()


def test_beagle_store_shared_connection(tmp_path):
    """Two instances of the same DB share ONE connection (thread-safety pattern)."""
    path = str(tmp_path / "beaglemem.db")
    a = BeagleStore(path, create=True)
    b = BeagleStore(path)
    assert a._conn is b._conn  # same underlying connection
    a.close()
    b.close()
    # After both close, registry is empty
    assert path not in BeagleStore._shared


def test_rrf_single_list_order_preserved():
    assert [fid for fid, _ in rrf([[1, 2, 3]])] == [1, 2, 3]


def test_rrf_two_lists_boost_shared():
    assert rrf([[1, 2, 3], [2, 4, 5]])[0][0] == 2


def test_rrf_score_math():
    fused = rrf([[7]], k=60)
    assert abs(fused[0][1] - (1.0 / 61.0)) < 1e-9


# --- Phase 1 (v0.3): vocab + meta tables, deterministic ordering ---

def test_documents_ordered_by_fact_id(tmp_path):
    """documents() MUST return facts in fact_id order (v0.3 fact-row mapping
    contract: row N ↔ Nth fact by id). Removal must not disturb the order."""
    path = str(tmp_path / "beaglemem.db")
    store = BeagleStore(path, create=True)
    a = store.add("first fact")
    b = store.add("second fact")
    c = store.add("third fact")
    store.remove(b)  # middle removed → remaining must still be [a, c] by id
    assert [d["id"] for d in store.documents()] == [a, c]
    store.close()


def test_meta_round_trip(tmp_path):
    """meta key-value table round-trips JSON-serializable values."""
    path = str(tmp_path / "beaglemem.db")
    store = BeagleStore(path, create=True)
    store.set_meta("dim", 2048)
    store.set_meta("tokenizer_fingerprint", "abc123")
    store.set_meta("regex", r"[a-z0-9]+")
    store.set_meta("encoder_version", "idf-v1")
    assert store.get_meta("dim") == 2048
    assert store.get_meta("missing") is None
    assert store.all_meta() == {
        "dim": 2048,
        "tokenizer_fingerprint": "abc123",
        "regex": r"[a-z0-9]+",
        "encoder_version": "idf-v1",
    }
    store.close()


def test_vocab_replace_and_read(tmp_path):
    """vocab table: idx is the beagle_mem.npy row index; replace is atomic."""
    path = str(tmp_path / "beaglemem.db")
    store = BeagleStore(path, create=True)
    store.replace_vocab(["hello", "world"], {"hello": 5, "world": 3})
    assert store.vocab_words() == ["hello", "world"]
    assert store.vocab_rows() == [(0, "hello", 5), (1, "world", 3)]
    store.replace_vocab(["a", "b", "c"], {"a": 1, "b": 2, "c": 3})
    assert store.vocab_words() == ["a", "b", "c"]
    assert store.vocab_rows() == [(0, "a", 1), (1, "b", 2), (2, "c", 3)]
    store.close()


def test_fact_ids_ordered(tmp_path):
    """fact_ids() returns ids in fact_id order (in-memory row↔fact map)."""
    path = str(tmp_path / "beaglemem.db")
    store = BeagleStore(path, create=True)
    a = store.add("x")
    b = store.add("y")
    store.remove(a)
    c = store.add("z")
    assert store.fact_ids() == [b, c]
    store.close()
