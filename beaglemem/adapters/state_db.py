"""Hermes session store adapter: reads conversation history from state.db.

state.db is Hermes' canonical session store (SQLite + FTS5), schema
verified against hermes_state.py (source of truth).

Filter rules (verified against live state.db — 172,596 messages):
- role ∈ {user, assistant}; tool rows excluded (147MB of JSON dumps)
- content non-empty. This ALSO drops assistant tool-call rows whose
  content is empty — while keeping tool-call rows that carry real prose
  ("Yes! I can check that for you:"). Do NOT filter on tool_calls
  presence; it would lose genuine assistant speech.
- (active = 1 OR compacted = 1): compaction soft-archives old rows
  (active=0, compacted=1) and inserts a lossy summary as a NEW active=1
  row. Reading compacted=1 rows recovers the raw pre-compaction text
  that would otherwise be lost. Rows with active=0 AND compacted=0 are
  rewind/undo ("user took it back") and stay excluded.
- content NOT LIKE '[SYSTEM:%': drops cron/system boilerplate.
- content NOT LIKE '[CONTEXT COMPACTION%': drops compaction SUMMARY rows
  (the only reliable identifier is their content prefix — compacted=1
  marks the originals, not the summary). Their constant words would be
  universal co-occurrence hubs polluting every vector.

SELECT is read-only. Never writes to state.db.
"""
import sqlite3

from ..corpus import MIN_SENTENCE_WORDS, split_sentences, tokenize

KEEP_ROLES = ("user", "assistant")

_MESSAGES_QUERY = """
    SELECT content FROM messages
    WHERE role IN (?, ?)
      AND content IS NOT NULL
      AND content != ''
      AND (active = 1 OR compacted = 1)
      AND content NOT LIKE '[SYSTEM:%'
      AND content NOT LIKE '[CONTEXT COMPACTION%'
"""


def iter_sentences(db_path: str):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        for (content,) in conn.execute(_MESSAGES_QUERY, KEEP_ROLES):
            for raw in split_sentences(content):
                words = tokenize(raw)
                if len(words) >= MIN_SENTENCE_WORDS:
                    yield words
    finally:
        conn.close()


def max_message_id(db_path: str) -> int:
    """Highest messages.id in state.db, or 0 if empty. Cheap (PK-indexed)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT MAX(id) FROM messages").fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    finally:
        conn.close()


def iter_sentences_since(db_path: str, min_id: int):
    """Yield tokenized sentences from messages with id > min_id (incremental)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        for (content,) in conn.execute(
            _MESSAGES_QUERY + " AND id > ?", KEEP_ROLES + (min_id,)
        ):
            for raw in split_sentences(content):
                words = tokenize(raw)
                if len(words) >= MIN_SENTENCE_WORDS:
                    yield words
    finally:
        conn.close()
