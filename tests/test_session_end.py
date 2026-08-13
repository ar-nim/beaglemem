"""on_session_end: incremental state.db read + no-stub-manufacture guard."""
import json
import os
import sqlite3

from beaglemem import BeagleMemoryProvider
from beaglemem.vectors import BeagleModel


def _make_state_db(path, n=10, start=1):
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            active INTEGER DEFAULT 1,
            compacted INTEGER DEFAULT 0
        );
    """)
    for i in range(start, start + n):
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES ('s', 'user', ?)",
            (f"message number {i} with enough words here",),
        )
    conn.commit()
    conn.close()


def test_on_session_end_without_model_does_not_create_stub(tmp_path):
    """The stub bug: when _model is None, on_session_end must NOT build a
    mini-model from the tail. It must leave _model None so the full auto-build
    path in initialize() runs instead."""
    data_dir = tmp_path / "beaglemem-data"
    data_dir.mkdir()
    # Simulate a leftover archive tail (the 2026-08-13 damage): old code
    # would build a stub model from this; new code must not.
    with open(str(data_dir / "corpus_archive.txt"), "w", encoding="utf-8") as fh:
        fh.write("restart gateway and start new session. delete the vector files.\n")
    state_db = str(tmp_path / "state.db")
    _make_state_db(state_db, n=10)

    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmp_path))
    assert p._model is None
    p.on_session_end()
    # No stub manufactured
    assert p._model is None
    # No model files written
    assert not os.path.exists(str(data_dir / "beagle_vocab.json"))


def test_on_session_end_increments_by_watermark(tmp_path):
    """A built model absorbs only NEW messages (id > last_seen_id), then
    advances the watermark."""
    data_dir = tmp_path / "beaglemem-data"
    data_dir.mkdir()
    state_db = str(tmp_path / "state.db")
    _make_state_db(state_db, n=10, start=1)

    from beaglemem.adapters.state_db import iter_sentences_since
    model = BeagleModel(dim=64, window=2, min_count=1)
    for words in iter_sentences_since(state_db, 0):
        model.add_sentence(words)
    model.save(str(data_dir))
    with open(str(data_dir / "last_update.json"), "w") as fh:
        json.dump({"last_seen_id": 5}, fh)

    p = BeagleMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmp_path))
    before = p._model.consumed_sentences
    p.on_session_end()
    after = p._model.consumed_sentences
    assert after > before  # absorbed the 5 new messages (ids 6..10)
    with open(str(data_dir / "last_update.json")) as fh:
        stamp = json.load(fh)
    assert stamp["last_seen_id"] == 10
