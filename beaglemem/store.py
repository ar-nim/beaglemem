"""Document stores. MemoryStore for demos/tests; BeagleStore is
self-owned SQLite with FTS5 (create + read + write)."""
import sqlite3
import threading
from pathlib import Path
from typing import Protocol


class DocumentStore(Protocol):
    def documents(self) -> list[dict]: ...


class MemoryStore:
    """In-memory store for demos and tests."""

    def __init__(self, docs: list[dict]):
        self._docs = docs

    def documents(self) -> list[dict]:
        return self._docs


class BeagleStore:
    """Self-owned SQLite store with FTS5. beaglemem owns this DB entirely.

    THREAD SAFETY: copied from holographic's MemoryStore (store.py:101-164).
    SQLite allows one writer at a time. All plugin instances + subagents in
    one process share ONE connection guarded by ONE RLock — this was a real
    bug in holographic (providers raced as WAL writers; "database is locked"
    for the full busy timeout). We inherit the fix, not the bug.

    Connection properties:
    - check_same_thread=False (cross-thread use is the norm here)
    - timeout=10.0 (WAL busy tolerance)
    - isolation_level=None (autocommit — no dangling write transactions)
    """

    _shared: dict = {}
    _shared_guard = threading.Lock()

    def __init__(self, db_path: str, create: bool = False):
        self.db_path = Path(db_path).expanduser()
        if create:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._key = str(self.db_path.resolve())
        except OSError:
            self._key = str(self.db_path)
        with BeagleStore._shared_guard:
            entry = BeagleStore._shared.get(self._key)
            if entry is None:
                conn = sqlite3.connect(
                    self._key,
                    check_same_thread=False,
                    timeout=10.0,
                    isolation_level=None,
                )
                conn.row_factory = sqlite3.Row
                entry = {"conn": conn, "lock": threading.RLock(), "refs": 0, "ready": False}
                BeagleStore._shared[self._key] = entry
            entry["refs"] += 1
            self._entry = entry
            self._conn = entry["conn"]
            self._lock = entry["lock"]

        with self._lock:
            if not entry["ready"] and create:
                self._init_db()
                entry["ready"] = True

    def _init_db(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS facts (
                fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                trust_score REAL DEFAULT 0.5,
                created_at TEXT,
                updated_at TEXT
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
                USING fts5(content, content='facts', content_rowid='fact_id');
        """)

    def add(self, content: str, trust: float = 0.5) -> int:
        """Insert a fact; returns the new autoincrement fact_id (no explicit ID)."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO facts (content, trust_score, created_at, updated_at) "
                "VALUES (?, ?, datetime('now'), datetime('now'))",
                (content, trust),
            )
            fact_id = cur.lastrowid
            self._conn.execute(
                "INSERT INTO facts_fts (rowid, content) VALUES (?, ?)",
                (fact_id, content),
            )
            return fact_id

    def remove(self, fact_id: int) -> None:
        with self._lock:
            # FTS FIRST, content second. With external-content FTS5
            # (content='facts'), the FTS delete needs the content row still
            # present to compute which terms to drop; deleting content first
            # silently no-ops the FTS delete and leaves stale index entries.
            self._conn.execute("DELETE FROM facts_fts WHERE rowid = ?", (fact_id,))
            self._conn.execute("DELETE FROM facts WHERE fact_id = ?", (fact_id,))

    def set_trust(self, fact_id: int, trust: float) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE facts SET trust_score = ?, updated_at = datetime('now') "
                "WHERE fact_id = ?",
                (trust, fact_id),
            )

    def documents(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT fact_id, content FROM facts WHERE content IS NOT NULL"
            ).fetchall()
        return [{"id": r["fact_id"], "text": r["content"]} for r in rows]

    def fts_search(self, query: str, limit: int = 20) -> list[int]:
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT rowid FROM facts_fts WHERE facts_fts MATCH ? "
                    "ORDER BY bm25(facts_fts) LIMIT ?",
                    (query, limit),
                ).fetchall()
            return [r[0] for r in rows]
        except Exception:
            return []

    def trust_of(self, fact_id: int) -> float:
        with self._lock:
            row = self._conn.execute(
                "SELECT trust_score FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
        return float(row[0]) if row and row[0] is not None else 0.5

    def close(self) -> None:
        with BeagleStore._shared_guard:
            entry = self._entry
            entry["refs"] -= 1
            if entry["refs"] <= 0:
                entry["conn"].close()
                BeagleStore._shared.pop(self._key, None)
            self._conn = None
