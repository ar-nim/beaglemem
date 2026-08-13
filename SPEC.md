# SPEC — beagle v0.1

## Corpus contract
- Adapters yield tokenized sentences: list[str], lowercase, [a-z0-9][a-z0-9'-]*
- Sentences with fewer than 3 tokens are dropped
- chat-jsonl adapter: keep rows where role ∈ {user, assistant}; content may be
  str or list of {text: str} blocks; all other rows/blocks dropped
- plain adapter: handles BOTH single files (*.txt, *.md) AND directories;
  if path is a file, read it directly; if dir, glob *.txt + *.md; split on [.!?\n]+
- state_db adapter (Hermes session store):
  - path = the state.db file (default: <hermes_home>/state.db)
  - schema VERIFIED against hermes_state.py (source of truth),
    not assumed from live DB dumps
  - SELECT content FROM messages WHERE role IN ('user','assistant')
    AND content IS NOT NULL AND content != '' AND (active = 1 OR compacted = 1)
    AND content NOT LIKE '[SYSTEM:%' AND content NOT LIKE '[CONTEXT COMPACTION%'
  - tool rows excluded (role='tool' is output noise, 147MB)
  - do NOT filter on tool_calls presence: empty-content tool-call
    rows are dropped by content != '', but tool-call rows carrying real
    prose must be KEPT
  - (active = 1 OR compacted = 1): live rows + soft-archived RAW rows.
    Compaction soft-archives originals (active=0, compacted=1) and inserts
    a lossy summary as a new active=1 row. Reading compacted=1 recovers
    pre-compaction raw text; the summary row is dropped by the
    [CONTEXT COMPACTION filter. active=0 AND compacted=0 (rewind/undo,
    "user took it back") stays excluded.
- Unknown format string → ValueError

## Vector contract
- env_vector(word): deterministic, unit-norm, identical across processes
- mem init = env; accumulate neighbor env vectors within ±window
- After each sentence: renormalize every touched mem vector to length 1
- Same corpus → same vectors (within float32 tolerance)
- Incremental ingest == batch build (tested)

## Probe contract
- encode_text: sum of mem vectors of known non-stopword tokens, normalized
- probe returns top_k (doc_id, cosine) sorted desc; unknown-only query → []
- Semantic contract: a document containing word X ranks above unrelated docs
  when the query word shares corpus neighbors with X
- FACT VECTOR CACHING: fact vectors are computed once, cached to disk
  (fact_vectors.npy + fact_ids.json), and loaded by the plugin at init.
  Re-encode only changed facts on memory write. Never re-encode all facts
  on every probe/prefetch call.

## Fusion contract
- rrf(lists, k=60): score = Σ 1/(k + rank + 1) per list

## Cold-start contract (fresh install)
- FTS5 + trust work from day one, zero vectors required.
  `prefetch()` runs FTS5-only when `_model is None`. Same design intent
  as holographic (vector path is additive, never required).
- AUTO-BUILD: on first activation, `initialize()` detects state.db with
  >100 messages and kicks off `_initial_build()` in a daemon thread.
  User chats normally (FTS5-only) while vectors build in background
  (~30-60s). Next session: model loads, full FTS5+BEAGLE active.
  No manual `hermes beaglemem build` needed for the initial experience.
- `hermes beaglemem build` is for REBUILDS only (algorithm change,
  filter update, adding a custom corpus).
- `sync_turn()` is a no-op — state.db is the canonical source of every
  turn (Hermes core persists it per-turn). No mirror file is written.
- `on_session_end()` runs incremental BEAGLE update from state.db by
  `id` watermark (`WHERE id > last_seen_id`), stamping the watermark in
  last_update.json. No double-counting, no re-reading history.
- STUB GUARD: `on_session_end()` returns immediately when `_model is None`
  — it never builds a fresh model from a partial tail. Model creation is
  owned by the full auto-build path.
- STUB DETECTION: the model meta persists `consumed_sentences` +
  `corpus_source`. On load, a model that (a) was built from a non-canonical
  source, (b) consumed <10% of the corpus's messages, or (c) has <100
  words against >10K messages is treated as damaged: cleared, a rebuild
  scheduled, and a user-visible notice set via `_pending_notice`.

## Migration contract (from holographic)
- beaglemem's CLI (register_cli in cli.py) is only visible when beaglemem is
  the ACTIVE provider — that is how Hermes scans plugin CLIs
  (plugins/memory/__init__.py:373).
- Order matters: activate provider FIRST (memory_store.db stays on disk),
  then `hermes beaglemem migrate --source ~/.hermes/memory_store.db`
  (read-only on source), then `hermes beaglemem build` (vectors from
  state.db). Nothing is lost by switching providers — holographic's DB is
  not deleted.
- Migration is opt-in. The AGENT offers it when the user asks to switch
  providers; it never auto-runs. Mechanism (CLI) vs policy (agent).

## Store contract
- MemoryStore: in-memory list for demos/tests
- BeagleStore: self-owned SQLite with FTS5. Creates own schema on first use.
  Supports add(fact_id, content), remove(fact_id), documents(), fts_search().
  beaglemem does NOT read from any other provider's DB.
- All paths configurable via get_config_schema(): db_path, vector_dir,
  corpus_dir, format. Defaults under hermes_home/beaglemem-data/.

## Ingestion contract (on_memory_write)
- on_memory_write(action, target, content): when the built-in memory tool
  writes a fact, mirror it into beaglemem's own store + vector cache:
  - action="add": store.add(fact_id, content) → encode → append to fact_vectors
  - action="replace": store.add(fact_id, content) → re-encode → update cached vector
  - action="remove": store.remove(fact_id) → drop vector from cache
- This hook is REQUIRED, not optional. Without it, new facts written after
  the initial BEAGLE build are invisible to probe.

## Demo acceptance (CI-safe, always runs)
- examples/make_demo_corpus.py (seed=42) → build → verify with
  examples/verify.demo.json → ALL PASS
- The demo's planted semantic bridges must form; the probe test must surface
  the target document. This is the open-source proof.

## Real-data acceptance (deployment-private)
- verify.local.json (gitignored) holds deployment-specific pairs and probe
  tests. FAIL = report numbers verbatim, never tune thresholds.

## Multilingual contract (v0.2)

- NFKC normalization applied before tokenization (fullwidth → halfwidth)
- Unicode word regex `[^\W_]` — superset of v0.1 ASCII regex
- CJK runs extracted as char-bigrams (sliding 2-char window)
- Script-aware noise filter: no-vowel hash check only applies to ASCII tokens
- IDF weighting (`beaglemem/idf.py`) — language-agnostic stopword replacement
- CJK single-char function words (的, は) dropped at tokenization (never enter vocab)
- CJK multi-char particles (から, まで) downweighted by IDF automatically
- Cross-lingual bridges form from shared co-occurrence, NOT translation
- Deferred: abugida scripts (Hindi/Bengali/Telugu) — v0.3
