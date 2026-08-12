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
