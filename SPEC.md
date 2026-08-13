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

## Memory-architecture contract (v0.3)

### On-disk layout — ONE sidecar
- Exactly four artifacts under `beaglemem-data/`:
  - `beagle_mem.npy` — corpus word vectors (spec-compliant `.npy`, no custom header keys)
  - `fact_vectors.npy` — fact doc vectors (spec-compliant `.npy`)
  - `beaglemem.db` — fact store (SQLite FTS5, source facts)
  - `meta.json` — the single sidecar: ALL non-vector metadata
- `beagle_vocab.json`, `fact_ids.json`, `idf.json`, `last_update.json`,
  `fact_cache_meta.json` are GONE — folded into `meta.json`.

### Why a sidecar, not a `.npy` header (verified, do not regress)
- The `.npy` spec defines the header dict as EXACTLY three keys:
  `descr`, `fortran_order`, `shape`.
- numpy >= 2.0 enforces it strictly: `EXPECTED_KEYS != d.keys()` → `ValueError`.
- Confirmed empirically on numpy 2.4.4: `np.load` rejects a header with
  custom keys. Markers therefore CANNOT live in the `.npy` header; they
  live in the sidecar. (numpy 1.x tolerated extras; 2.x does not.)

### meta.json schema
- Top-level `config_fingerprint` + two granular guards:
  - `corpus`: dim, window, min_count, tokenizer_fingerprint, regex,
    stemmer, vocab, counts, consumed_sentences, corpus_source, last_seen_id
  - `facts`: encoder_version, ids, idf
- The RAW inputs (`regex`, `stemmer`, `dim`, `window`) are stored BESIDE the
  hash — this is what makes the fingerprint self-healing (see below).
- Load-critical for BOTH `.npy` files: vocab order maps `beagle_mem`
  rows; ids map `fact_vectors` rows. Deleting `meta.json` invalidates
  both caches coherently (one full rebuild, no torn state).

### Config contract — config takes top priority
- `config.yaml` → `plugins.beaglemem` adds: `dim` (2048), `window` (3).
  Existing keys (db_path, vector_dir, corpus_dir, format) unchanged.
- `dim`/`window` are user config; `regex`, `stemmer`, `encoder_version`
  are code constants. `tokenizer_fingerprint = sha256(json.dumps({regex,
  stemmer, dim, window}, sort_keys=True))[:16]` mixes BOTH, so a change to
  either side is detected.
- Config/code is the source of truth. The on-disk sidecar is evidence,
  not authority: when they disagree, config/code wins.

### Fingerprint semantics — staleness hint, NOT a destroy trigger
- Two independent signals decide what to do. Only one is hand-editable:
  - INTRINSIC (un-tamperable): `dim` + `shape` from the `.npy` binary
    header (read via `np.lib.format.read_array_header_1_0/2_0` without
    loading the matrix); fact count from `SELECT COUNT(*) FROM facts`.
  - FINGERPRINT (tamperable): the stored hash + raw inputs in `meta.json`.

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
      the correct hash, write it back to `meta.json`, NO rebuild.
    - any raw input differs → genuine change → rebuild (see decision table).
- `encoder_version` (fact guard) is a plain string constant; the same
  self-heal logic applies: on mismatch, re-encode facts only (never the corpus).

### Non-destructive rebuild — stage → verify → atomic swap
- A rebuild NEVER overwrites or deletes existing artifacts in place:
  1. Build the new model fully in memory (do not touch `self._model` yet).
  2. Write to temp files: `beagle_mem.npy.tmp`, `meta.json.tmp`, etc.
  3. `fsync` each temp file.
  4. VERIFY before swap: `beagle_mem.npy` `shape[0] == len(vocab)`;
     vocab non-empty; `fact_vectors.npy` `shape[1] == dim`; fact rows == fact count.
  5. `os.replace(tmp, final)` for each artifact (atomic on POSIX).
  6. Only then swap `self._model = new_model`.
- On failure at ANY step (kill, exception, verify fails): the temp files are
  orphaned and the previous artifacts + in-memory model remain intact and
  serving. Vectors are never "gone" — they are replaced only after the
  replacement is proven.
- The single sidecar makes the metadata commit atomic as ONE unit (vocab +
  counts + ids + idf + watermark together), closing BOTH the torn-pair
  window (`.npy` vs `.json`) and the double-count window (`model.save`
  vs watermark stamp) that existed pre-v0.3.
- During a SOFT (stale) rebuild, `self._model` keeps serving the old
  vectors. During a HARD (structural) rebuild the old vectors are already
  unusable, so FTS5-only until the swap — but the on-disk files still
  survive a failed rebuild.
- Intrinsic cross-checks close the crash path where `matrix @ q`
  shape-errors on a dim mismatch: they fire BEFORE any vector is used.

### Migration (v0.2 → v0.3)
- First init after upgrade folds legacy files into `meta.json` once, then
  removes them. Legacy bare-list `fact_ids.json` re-encodes silently (no
  notice). Cold start iff no `meta.json` and no legacy files.
