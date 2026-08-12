import pytest
from beaglemem.adapters.plain_text import iter_sentences as plain_sentences
from beaglemem.corpus import iter_sentences


def test_plain_adapter_reads_txt_and_md(tmp_path):
    (tmp_path / "a.txt").write_text("the severance letter arrived today. signed by hr.")
    (tmp_path / "b.md").write_text("sunny weekend forecast for the picnic.")
    sentences = list(plain_sentences(str(tmp_path)))
    assert ["the", "severance", "letter", "arrived", "today"] in sentences
    assert ["sunny", "weekend", "forecast", "for", "the", "picnic"] in sentences


def test_plain_adapter_drops_short_sentences(tmp_path):
    (tmp_path / "a.txt").write_text("too short. this one is long enough to keep.")
    sentences = list(plain_sentences(str(tmp_path)))
    assert all(len(s) >= 3 for s in sentences)


def test_corpus_dispatch_and_unknown_format(tmp_path):
    (tmp_path / "a.txt").write_text("the severance letter arrived today.")
    via_dispatch = list(iter_sentences(str(tmp_path), format="plain"))
    via_adapter = list(plain_sentences(str(tmp_path)))
    assert via_dispatch == via_adapter
    with pytest.raises(ValueError):
        list(iter_sentences(str(tmp_path), format="bogus"))


from beaglemem.adapters.chat_jsonl import iter_sentences as chat_sentences
from tests.fixtures import make_chat_jsonl_lines


def test_chat_adapter_filters_roles_and_blocks(tmp_path):
    (tmp_path / "s.jsonl").write_text("\n".join(make_chat_jsonl_lines()))
    sentences = list(chat_sentences(str(tmp_path)))
    joined = " | ".join(" ".join(s) for s in sentences)
    assert "let go from beta division" in joined      # user string kept
    assert "restructuring is confirmed" in joined      # assistant text block kept
    assert "tool output" not in joined                 # role=tool dropped
    assert "tool_use" not in joined                    # non-text blocks dropped
    assert all(len(s) >= 3 for s in sentences)


def test_chat_adapter_single_file(tmp_path):
    (tmp_path / "s.jsonl").write_text("\n".join(make_chat_jsonl_lines()))
    from beaglemem.corpus import iter_sentences
    assert list(iter_sentences(str(tmp_path), format="chat-jsonl")) == list(chat_sentences(str(tmp_path)))


def test_state_db_adapter(tmp_path):
    """Reads user+assistant text from a Hermes-shaped state.db. Tool rows excluded."""
    import sqlite3
    db = str(tmp_path / "state.db")
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_calls TEXT,
            active INTEGER DEFAULT 1
        );
        INSERT INTO messages (session_id, role, content, tool_calls, active) VALUES
            ('s1', 'user', 'I got laid off from Megacorp. Severance is confirmed.', NULL, 1),
            ('s1', 'assistant', 'Noted. The termination agreement is signed.', NULL, 1),
            ('s1', 'assistant', 'Yes! I can check that for you:', '[{"type":"function"}]', 1),
            ('s1', 'assistant', NULL, '[{"type":"function"}]', 1),
            ('s1', 'tool', '{"huge": "json dump that should be ignored"}', NULL, 1),
            ('s2', 'user', '', NULL, 1),
            ('s2', 'assistant', NULL, NULL, 1),
            ('s3', 'user', '[SYSTEM: You are running as a scheduled cron job. DELIVERY: ...]', NULL, 1),
            ('s4', 'user', 'Archived session text should be excluded', NULL, 0);
    """)
    conn.commit()
    conn.close()

    from beaglemem.adapters.state_db import iter_sentences as state_sentences
    sentences = list(state_sentences(db))
    joined = " | ".join(" ".join(s) for s in sentences)
    assert "laid off from megacorp" in joined        # user kept
    assert "termination agreement is signed" in joined  # assistant kept
    assert "check that for you" in joined              # tool_call WITH prose kept
    assert "json dump" not in joined                   # tool row excluded
    assert "scheduled cron job" not in joined          # [SYSTEM: boilerplate excluded
    assert "archived session text" not in joined       # active=0 excluded
    assert all(len(s) >= 3 for s in sentences)
