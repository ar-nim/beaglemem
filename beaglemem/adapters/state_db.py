"""Hermes session store adapter: reads conversation history from state.db.

state.db is Hermes' canonical session store (SQLite + FTS5), schema
verified against hermes_state_common.py (source of truth).

Filter rules (verified against 163K live messages):
- role ∈ {user, assistant}; tool rows excluded (172M chars of JSON dumps)
- content non-empty. This ALSO drops 43,847 assistant tool-call rows
  whose content is empty — while keeping 13,202 tool-call rows that
  carry real prose ("Yes! I can message you right now:"). Do NOT filter
  on tool_calls presence; it would lose genuine assistant speech.
- active = 1: compaction soft-archives old rows (active=0) and inserts
  the compaction summary as a new active row. active=1 gives the current
  view: recent messages + compaction summaries (condensed semantics).
- content NOT LIKE '[SYSTEM:%': drops cron/system boilerplate (349 rows,
  1.8% of user messages).

SELECT is read-only. Never writes to state.db.
"""
import sqlite3

from beaglemem.corpus import MIN_SENTENCE_WORDS, split_sentences, tokenize

KEEP_ROLES = ("user", "assistant")

_MESSAGES_QUERY = """
    SELECT content FROM messages
    WHERE role IN (?, ?)
      AND content IS NOT NULL
      AND content != ''
      AND active = 1
      AND content NOT LIKE '[SYSTEM:%'
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
