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
  - schema VERIFIED against hermes_state_common.py (source of truth),
    not assumed from live DB dumps
  - SELECT content FROM messages WHERE role IN ('user','assistant')
    AND content IS NOT NULL AND content != '' AND active = 1
    AND content NOT LIKE '[SYSTEM:%'
  - tool rows excluded (role='tool' is output noise, 172M chars)
  - do NOT filter on tool_calls presence: 43,847 empty-content tool-call
    rows are dropped by content != '', but 13,202 carry real prose that
    must be KEPT
  - active=1 means: current view (compaction summaries included as
    active rows; soft-archived originals excluded)
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
- `sync_turn()` appends every turn to corpus_archive.txt.
- `on_session_end()` runs incremental BEAGLE update from the appended
  tail (byte-offset tracked). No double-counting, no re-reading history.

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
