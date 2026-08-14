# SPEC — beagle v0.1.0

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
  `MAX(id) > 100` and kicks off `_initial_build()` in a daemon thread.
  User chats normally (FTS5-only) while vectors build in background
  (~30-60s). Next session: model loads, full FTS5+BEAGLE active.
  No manual `hermes beaglemem build` needed for the initial experience.
- The build gate uses `MAX(id)` — the AUTOINCREMENT high-water mark
  (~2ms, PK-indexed, pruning-immune) — NOT `COUNT(*)` (a ~1s cold full
  scan that also collapses under Hermes pruning and would false-negative
  the `>100` gate).
- Cold start is OPPORTUNISTIC, not forced: a fresh install does NOT set
  `_force_rebuild`; the corpus-size gate decides. Only a forced rebuild
  (structural damage, fingerprint/config change, stub, reset) sets
  `_force_rebuild=True`, which rebuilds even on a tiny corpus.
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
- STUB DETECTION (watermark-based, v0.4): the model meta persists
  `last_seen_id` (ingest watermark) + `corpus_source`. On load, `MAX(id)`
  from state.db — the AUTOINCREMENT high-water mark — is compared against
  the watermark, NOT a `COUNT(*)`. A model is treated as damaged (cleared,
  rebuild scheduled, notice via `_pending_notice`) when:
  (a) built from a non-canonical source (`corpus_source != state_db`);
  (b) the watermark covers <10% of the corpus id extent (stub — ingested
  a sliver); or
  (c) the corpus id space shrank 10× since the build (reset — DROP/
  recreate renumbered ids from 1).
- A small vocab against a big id-extent is NOT a failure mode. `MAX(id)`
  is a high-water mark, not a live-row count: after Hermes pruning a
  corpus can have a huge id-extent but few live messages, and a full-ingest
  model rebuilt over those few messages legitimately yields few words.
  Flagging it (the old `vocab < 100 and max > 10000` check) causes a
  prune→rebuild→flag→rebuild loop. (a)+(b)+(c) already cover every real
  failure mode, so the vocab-stub check is gone.
- WHY `MAX(id)` NOT `COUNT(*)`: `COUNT(*)` is a full scan over the content
  column AND collapses under Hermes session pruning (`DELETE FROM messages`
  in `prune_sessions`), which disables the `>100 messages` gate exactly when
  a stub is most dangerous. `MAX(id)` survives DELETE (AUTOINCREMENT never
  reuses ids — `sqlite_sequence` persists through `VACUUM`), and
  `last_seen_id` lives in beaglemem.db outside Hermes' prune radius, so both
  sides of the comparison are pruning-immune. Reset (id collapse) is a
  separate failure mode caught by the bidirectional guard (c).

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

## Memory-architecture contract (v0.3)

### On-disk layout — zero JSON, all metadata in the DB
- Exactly THREE artifacts under `beaglemem-data/`:
  - `beagle_mem.npy` — corpus word vectors (spec-compliant `.npy`, no custom header keys)
  - `fact_vectors.npy` — fact doc vectors (spec-compliant `.npy`)
  - `beaglemem.db` — SQLite: `facts` + `facts_fts` + `vocab` + `meta` tables
- `beagle_vocab.json`, `fact_ids.json`, `idf.json`, `last_update.json`,
  `fact_cache_meta.json` — ALL JSON is GONE. Every non-vector datum lives
  in `beaglemem.db`.

### Why not a `.npy` header (verified, do not regress)
- The `.npy` spec defines the header dict as EXACTLY three keys:
  `descr`, `fortran_order`, `shape`.
- numpy >= 2.0 enforces it strictly: `EXPECTED_KEYS != d.keys()` → `ValueError`.
- Confirmed empirically on numpy 2.4.4: `np.load` rejects a header with
  custom keys. Markers therefore CANNOT live in the `.npy` header; they
  live in the DB.

### DB schema (replaces every sidecar)
- `facts` + `facts_fts` — unchanged (source facts, FTS5).
- `vocab(idx INTEGER PRIMARY KEY, word TEXT UNIQUE, count INTEGER)` — the
  corpus vocabulary; `idx` is the row index into `beagle_mem.npy`.
- `meta(key TEXT PRIMARY KEY, value TEXT)` — key-value rows:
  `dim`, `window`, `min_count`, `tokenizer_fingerprint`, `regex`,
  `stemmer`, `encoder_version`, `consumed_sentences`, `corpus_source`,
  `last_seen_id`.
- The RAW inputs (`regex`, `stemmer`, `dim`, `window`) are stored BESIDE the
  fingerprint hash in `meta` — this is what makes the fingerprint
  self-healing (see below).
- NO `fact_ids` on disk: the row→fact mapping is re-derived at load from
  `SELECT fact_id FROM facts ORDER BY fact_id` and held in memory only.
- NO `idf` on disk: IDF weights are recomputed on load from the fact store
  (~ms for thousands of facts); only the `encoder_version` marker persists.

### Fact-row mapping contract (why fact_ids is redundant)
- `BeagleStore.documents()` MUST be `SELECT ... FROM facts ORDER BY fact_id`
  (deterministic order — the row↔fact mapping depends on it).
- Encoding NEVER skips a fact: a fact with no vocab overlap gets a ZERO
  vector (already filtered by `sims > 0.01` at probe time, so it never
  surfaces — behaviorally identical to skipping).
- Invariant: `fact_vectors.npy` rows == `COUNT(*) FROM facts`; row N ↔ the
  Nth fact by `fact_id` order. The load-time row-count check IS the mapping
  guarantee.

### Fingerprint in the DB (decision)
- The fingerprint is a BUILD STAMP: the transaction that writes `vocab` +
  `meta` (fingerprint, watermark) is the commit of the build. Storing the
  stamp with the thing it stamps is the coherent choice; the `.npy` header
  (dim/shape) remains the independent structural anchor.

### Config contract — config takes top priority
- `config.yaml` → `plugins.beaglemem` adds: `dim` (2048), `window` (3).
  Existing keys (db_path, vector_dir, corpus_dir, format) unchanged.
- `dim`/`window` are user config; `regex`, `stemmer`, `encoder_version`
  are code constants. `tokenizer_fingerprint = sha256(json.dumps({regex,
  stemmer, dim, window}, sort_keys=True))[:16]` mixes BOTH, so a change to
  either side is detected.
- Config/code is the source of truth. The DB is evidence, not authority:
  when they disagree, config/code wins.

### Fingerprint semantics — staleness hint, NOT a destroy trigger
- Two independent signals decide what to do. Only one is hand-editable:
  - INTRINSIC (un-tamperable): `dim` + `shape` from the `.npy` binary
    header (read via `np.lib.format.read_array_header_1_0/2_0` without
    loading the matrix); fact count from `SELECT COUNT(*) FROM facts`.
  - FINGERPRINT (tamperable): the stored hash + raw inputs in the DB `meta` table.

  | State | Verdict | Action |
  |---|---|---|
  | dim/count mismatch | structurally broken | hard rebuild (non-destructive) |
  | dim/count match, fingerprint mismatch | structurally fine, probably stale | KEEP SERVING + background rebuild |

- WHY: an accidental hand-edit to the hash (or a config typo that does not
  change dim) must never destroy working vectors. "Stale but structurally
  valid" vectors are better than none, so they keep serving while a
  rebuild happens in the background.

### Self-healing fingerprint (raw inputs beside the hash)
- On load, recompute `tokenizer_fingerprint` from current config/code:
  - hash MATCH → valid, done (fast path).
  - hash MISMATCH → compare stored RAW inputs vs current config/code:
    - all raw inputs match → the hash field was hand-corrupted → recompute
      the correct hash, write it back to the `meta` table, NO rebuild.
    - any raw input differs → genuine change → rebuild (see decision table).
- `encoder_version` (fact guard) is a plain string constant; the same
  self-heal logic applies: on mismatch, re-encode facts only (never the corpus).

### Non-destructive rebuild — stage → verify → atomic swap
- A rebuild NEVER overwrites or deletes existing artifacts in place:
  1. Build the new model fully in memory (do not touch `self._model` yet).
  2. Write matrices to temp files: `beagle_mem.npy.tmp`, `fact_vectors.npy.tmp`.
  3. `fsync` each temp file.
  4. VERIFY before swap: `beagle_mem.npy` `shape[0] == len(vocab)`;
     vocab non-empty; `fact_vectors.npy` `shape[1] == dim`; fact rows == fact count.
  5. Commit the DB (vocab + meta in ONE transaction), then
     `os.replace(tmp, final)` for each `.npy` (atomic on POSIX).
  6. Only then swap `self._model = new_model`.
- On failure at ANY step (kill, exception, verify fails): the temp files are
  orphaned and the previous artifacts + in-memory model remain intact and
  serving. Vectors are never "gone" — they are replaced only after the
  replacement is proven.
- Atomicity: SQLite gives ACID for ALL metadata (facts + vocab + meta) in
  one transaction; the `.npy` files commit via `os.replace`. The two commit
  domains (DB vs file) are reconciled by the intrinsic row-count/dim checks,
  which fire on load and catch any cross-domain mismatch (closes the crash
  path where `matrix @ q` shape-errors on a dim mismatch).
- During a SOFT (stale) rebuild, `self._model` keeps serving the old
  vectors. During a HARD (structural) rebuild the old vectors are already
  unusable, so FTS5-only until the swap — but the on-disk files still
  survive a failed rebuild.

### Migration (v0.2 → v0.3)
- First init after upgrade folds legacy JSON files (`beagle_vocab.json`,
  `fact_ids.json`, `idf.json`, `last_update.json`) into the `vocab`/`meta`
  tables once, then removes them. Legacy bare-list `fact_ids.json`
  re-encodes silently (no notice). Cold start iff no DB tables and no
  legacy files.
