"""Hermes memory provider: deterministic semantic recall via BEAGLE.

Self-contained plugin package. All library code (corpus, vectors, probe,
fusion, store) lives inside `beaglemem/` — no external library install needed.

Hooks:
- prefetch(query): BEAGLE semantic probe → top-K facts for system prompt
- sync_turn(user, assistant): append turn to compounding corpus archive
- on_memory_write(action, target, content): update fact vector cache
- on_session_end(): incremental BEAGLE update from new corpus material
"""
import hashlib
import json
import os
import threading

import numpy as np

# RELATIVE imports — required by the Hermes plugin loader, which registers
# this package as plugins.memory.beaglemem (or _hermes_user_memory.beaglemem),
# NOT as top-level "beaglemem". Absolute imports would break on install.
# The try/except fallback keeps standalone usage (scripts, tests) working.
try:
    from .vectors import BeagleModel
    from .store import BeagleStore, MemoryStore
    from .probe import encode_text, probe, build_doc_vectors
    from .fusion import rrf
    from .corpus import iter_sentences, split_sentences, tokenize, MIN_SENTENCE_WORDS
except ImportError:  # standalone (repo checkout, no package context)
    from beaglemem.vectors import BeagleModel
    from beaglemem.store import BeagleStore, MemoryStore
    from beaglemem.probe import encode_text, probe, build_doc_vectors
    from beaglemem.fusion import rrf
    from beaglemem.corpus import iter_sentences, split_sentences, tokenize, MIN_SENTENCE_WORDS

try:
    from agent.memory_provider import MemoryProvider
    _HAS_HERMES = True
except ImportError:
    MemoryProvider = object  # standalone fallback
    _HAS_HERMES = False


def _render_progress(done: int, total: int, label: str = "Building memory vectors") -> str:
    """Render an ASCII progress bar. Clamps to 100%, empty if total is 0."""
    if total <= 0:
        return ""
    pct = int(min(100, done / total * 100))
    filled = pct // 10
    bar = "█" * filled + "░" * (10 - filled)
    return f"{label}: [{bar}] {pct}% ({done}/{total})"


class BeagleMemoryProvider(MemoryProvider):
    """Self-contained memory provider with BEAGLE semantic retrieval."""

    @property
    def name(self) -> str:
        return "beaglemem"

    def __init__(self):
        self._data_dir = None
        self._model = None
        self._store = None
        self._fact_vectors = None  # cached (matrix, ids) from disk
        self._stored_encoder_version = None  # encoder version read from the manifest
        self._idf = None  # cached IDF weights (idf.json)
        self._initialized = False
        self._sync_thread = None
        self._pending_notice = None  # user-visible warning (rebuild events)
        self._build_progress = None  # (done, total) while a full build runs

    # -- required lifecycle -------------------------------------------------

    def is_available(self) -> bool:
        """No network, no API key. numpy is the only dependency."""
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home") or os.path.expanduser("~/.hermes")
        config = self._load_config(hermes_home)

        # All paths are configurable. Defaults live under hermes_home/beaglemem-data/.
        default_data = os.path.join(hermes_home, "beaglemem-data")
        self._data_dir = config.get("vector_dir") or default_data
        os.makedirs(self._data_dir, exist_ok=True)

        db_path = config.get("db_path") or os.path.join(default_data, "beaglemem.db")
        self._db_path = db_path

        # Resolve corpus source (state.db by default)
        corpus_db = config.get("corpus_dir") or os.path.join(hermes_home, "state.db")
        self._corpus_db = corpus_db
        fmt = config.get("format", "state_db")

        # Load BEAGLE model if built
        vocab_path = os.path.join(self._data_dir, "beagle_vocab.json")
        if os.path.exists(vocab_path):
            self._model = BeagleModel.load(self._data_dir)

        # Load cached fact vectors if built. The fact cache is a PAIR:
        # fact_vectors.npy (the matrix) + fact_ids.json (a manifest carrying
        # both the row→fact-id mapping AND the encoder_version that produced
        # the vectors). The version lives INSIDE the manifest, co-located with
        # the data it guards — never in a separate flag file a user could
        # delete independently (2026-08-13 bug: a missing fact_cache_meta.json
        # fired "encoder changed" on every session).
        fv_path = os.path.join(self._data_dir, "fact_vectors.npy")
        ids_path = os.path.join(self._data_dir, "fact_ids.json")
        self._stored_encoder_version = None
        if os.path.exists(fv_path) and os.path.exists(ids_path):
            try:
                manifest = json.load(open(ids_path))
            except Exception:
                manifest = None
            if isinstance(manifest, dict):
                ids = manifest.get("ids", [])
                self._stored_encoder_version = manifest.get("encoder_version")
            else:
                # Legacy bare-list format (pre-versioning): treat as unknown
                # version so it re-encodes once, then persists the manifest.
                ids = manifest or []
            self._fact_vectors = (np.load(fv_path), ids)

        # --- Tokenizer fingerprint check (guards the corpus model) ---
        # A mismatch means the on-disk model was built with a DIFFERENT
        # tokenizer (e.g. v0.1 → v0.2 upgrade). Its vectors are stale and must
        # be rebuilt from the raw corpus archive.
        from .fingerprint import tokenizer_fingerprint
        from .corpus import WORD_RE
        _TOKENIZER_REGEX = WORD_RE.pattern  # single source of truth (v0.2)
        if self._model is not None:
            current_fp = tokenizer_fingerprint(
                regex=_TOKENIZER_REGEX, stemmer=None,
                dim=self._model.dim, window=self._model.window,
            )
            if self._model.tokenizer_fingerprint not in (None, current_fp):
                import logging
                logging.getLogger("beaglemem").warning(
                    "beaglemem: tokenizer changed — rebuilding from state.db"
                )
                self._pending_notice = (
                    "⚠️ beaglemem: tokenizer changed — memory vectors are being "
                    "rebuilt from session history. Search stays FTS5-only until "
                    "the rebuild completes (a few seconds). This happens after "
                    "an upgrade that changes how text is tokenized."
                )
                # Clear BOTH model and fact cache. The stale fact_vectors.npy
                # was built against the OLD model — serving it is wrong.
                # Degrade to FTS5-only until the rebuild completes.
                self._model = None
                self._fact_vectors = None
                stamp = os.path.join(self._data_dir, "last_update.json")
                if os.path.exists(stamp):
                    os.remove(stamp)
                self._force_rebuild = True

        # --- Stub detection (history-depth guard) ---
        # A model that was recreated from the archive (or is implausibly small
        # vs the corpus) has lost its history. Detect it and force a rebuild,
        # exactly like a tokenizer fingerprint mismatch. Ratios are
        # corpus-agnostic (same-corpus quantities), not tuned to any one store.
        if self._model is not None:
            import sqlite3 as _sq
            try:
                _conn = _sq.connect(f"file:{corpus_db}?mode=ro", uri=True)
                _msg_count = _conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE role IN ('user','assistant') "
                    "AND content IS NOT NULL AND content != ''"
                ).fetchone()[0]
                _conn.close()
            except Exception:
                _msg_count = 0
            _stale = False
            _reason = ""
            src = getattr(self._model, "corpus_source", None)
            consumed = getattr(self._model, "consumed_sentences", 0)
            if src is not None and src != "state_db" and _msg_count > 100:
                _stale = True
                _reason = f"built from non-canonical source '{src}'"
            elif consumed and _msg_count > 100 and consumed < _msg_count * 0.1:
                _stale = True
                _reason = f"consumed only {consumed} sentences vs {_msg_count} messages"
            elif self._model.size < 100 and _msg_count > 10000:
                _stale = True
                _reason = f"vocab {self._model.size} words vs {_msg_count} messages"
            if _stale:
                import logging
                logging.getLogger("beaglemem").warning(
                    f"beaglemem: suspicious model ({_reason}) — rebuilding from state.db"
                )
                self._pending_notice = (
                    "⚠️ beaglemem: memory vectors appear damaged or incomplete "
                    f"({_reason}). Rebuilding from session history — search stays "
                    "FTS5-only until done."
                )
                self._model = None
                self._fact_vectors = None
                _stamp = os.path.join(self._data_dir, "last_update.json")
                if os.path.exists(_stamp):
                    os.remove(_stamp)
                self._force_rebuild = True

        # --- Encoder version check (guards the fact cache only) ---
        # A mismatch means the fact vectors were encoded with a DIFFERENT
        # encoder (IDF formula). Only the fact cache is stale — re-encode.
        # The version is read from the loaded manifest (co-located with the
        # cache), NOT a separate flag file. A cache that was never loaded
        # (missing/deleted files) is simply absent — not "changed" — so it
        # must NOT fire this notice.
        from .fingerprint import ENCODER_VERSION
        if self._fact_vectors is not None and \
                self._stored_encoder_version != ENCODER_VERSION:
            # Legacy (no recorded version) re-encodes silently — the encoder
            # didn't change, we just never stamped it. Only a genuine version
            # bump (stored version present but different) is user-visible.
            if self._stored_encoder_version is not None:
                import logging
                logging.getLogger("beaglemem").warning(
                    "beaglemem: encoder changed — re-encoding fact cache"
                )
                self._pending_notice = (
                    "⚠️ beaglemem: encoder changed — fact vectors are being "
                    "re-encoded with the new version. Retrieval quality is "
                    "unaffected, but the cache rebuild takes a moment."
                )
            self._fact_vectors = None  # forces _rebuild_fact_cache() on next use

        # Connect to the self-owned SQLite store (creates if missing)
        if os.path.exists(db_path):
            try:
                self._store = BeagleStore(db_path)
            except Exception:
                pass  # defer; may need build first
        else:
            # First run — create the store
            try:
                self._store = BeagleStore(db_path, create=True)
            except Exception:
                pass

        self._initial_build_started = False

        # AUTO-BUILD: if no model exists but state.db has history, build
        # vectors in a daemon thread. This is the ONLY way to guarantee users
        # get semantic recall without manual steps. Without this, a user who
        # activates beaglemem and skips `hermes beaglemem build` gets FTS5-only
        # — identical to holographic. The whole point of switching is lost.
        #
        # Conditions:
        #   1. _model is None (no beagle_vocab.json found)
        #   2. state.db exists with content (not a truly fresh install)
        #   3. not already started (idempotent — never re-trigger)
        if self._model is None and not self._initial_build_started:
            try:
                import sqlite3 as _sq
                test_conn = _sq.connect(f"file:{corpus_db}?mode=ro", uri=True)
                msg_count = test_conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE role IN ('user','assistant') AND content IS NOT NULL AND content != ''"
                ).fetchone()[0]
                test_conn.close()
                # A tokenizer change (self._force_rebuild) is MANDATORY — it
                # must rebuild even on a small corpus. Only the OPPORTUNISTIC
                # first-run auto-build is gated by msg_count > 100.
                if getattr(self, "_force_rebuild", False) or msg_count > 100:
                    import logging
                    logging.getLogger("beaglemem").info(
                        f"beaglemem: auto-building vectors from {corpus_db} ({msg_count} messages) in background"
                    )
                    self._initial_build_started = True
                    t = threading.Thread(
                        target=self._initial_build,
                        args=(corpus_db, fmt, self._data_dir, self._db_path),
                        daemon=True,
                    )
                    t.start()
            except Exception:
                pass  # never let auto-build failure block initialization

        self._initialized = True

    def _initial_build(self, corpus_db: str, fmt: str, data_dir: str, db_path: str):
        """Background daemon-thread build. Runs once on first activation when
        state.db has history. Builds vocabulary + fact vectors. Non-blocking:
        user chats normally (FTS5-only) while this runs. On completion, the
        model is saved to disk; the NEXT session's initialize() loads it.

        Failures are logged but swallowed — the watermark in last_update.json
        prevents re-triggering within the same session. A restart will retry.
        """
        import logging
        logger = logging.getLogger("beaglemem")
        try:
            from .vectors import BeagleModel
            from .corpus import iter_sentences
            from .probe import build_doc_vectors
            import time
            import numpy as np

            t0 = time.time()
            model = BeagleModel(dim=2048, window=3)

            # Count total sentences first so we can show a progress bar.
            # iter_sentences is a generator — count via a cheap first pass.
            total_sentences = 0
            for _ in iter_sentences(corpus_db, format=fmt):
                total_sentences += 1

            n = 0
            self._build_progress = (0, total_sentences) if total_sentences else None
            for words in iter_sentences(corpus_db, format=fmt):
                model.add_sentence(words)
                n += 1
                # Update progress ~every 1% (cheap, avoids lock contention)
                if total_sentences and n % max(1, total_sentences // 100) == 0:
                    self._build_progress = (n, total_sentences)
            self._build_progress = None  # done
            model.corpus_source = "state_db"
            model.save(data_dir)
            # Stamp the incremental watermark so the next on_session_end does
            # NOT re-read the whole corpus and double-count co-occurrence into
            # this freshly built model (id-watermark semantics).
            from .adapters.state_db import max_message_id
            with open(os.path.join(data_dir, "last_update.json"), "w") as fh:
                json.dump({"last_seen_id": max_message_id(corpus_db)}, fh)
            logger.info(f"beaglemem: auto-build complete — {model.size} words from {n} sentences in {time.time()-t0:.0f}s")

            # Build fact cache if store has facts
            if os.path.exists(db_path):
                store = BeagleStore(db_path)
                docs = store.documents()
                if docs:
                    from .idf import build_idf
                    idf = build_idf(docs)
                    with open(os.path.join(data_dir, "idf.json"), "w") as fh:
                        json.dump(idf, fh)
                    matrix, ids = build_doc_vectors(model, docs, idf)
                    np.save(os.path.join(data_dir, "fact_vectors.npy"), matrix)
                    with open(os.path.join(data_dir, "fact_ids.json"), "w") as fh:
                        json.dump(ids, fh)
                    logger.info(f"beaglemem: fact vectors cached — {len(ids)} facts (idf v1)")
                store.close()
        except Exception as e:
            self._build_progress = None
            logger.warning(f"beaglemem: auto-build failed (will retry on restart): {e}")

    def get_config_schema(self):
        return [
            {
                "key": "db_path",
                "description": "Path to beaglemem's own SQLite store",
                "default": "",
                "secret": False,
            },
            {
                "key": "vector_dir",
                "description": "Directory for BEAGLE vector files (.npy, corpus archive)",
                "default": "",
                "secret": False,
            },
            {
                "key": "corpus_dir",
                "description": "Path to conversation corpus (state.db path, or dir for plain/chat-jsonl)",
                "default": "",
                "secret": False,
            },
            {
                "key": "format",
                "description": "Corpus format",
                "default": "state_db",
                "choices": ["plain", "chat-jsonl", "state_db"],
            },
        ]

    def save_config(self, values: dict, hermes_home: str) -> None:
        """Write config to config.yaml under plugins.beaglemem.

        Matches holographic's convention (holographic/__init__.py:129-144):
        round-trip read-modify-write on config.yaml, not a standalone file."""
        from pathlib import Path
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            try:
                from hermes_cli.config import read_user_config_raw
                existing = read_user_config_raw(config_path)
            except ImportError:
                # Standalone (no Hermes) — read directly
                with open(config_path, "r", encoding="utf-8") as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["beaglemem"] = values
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception:
            pass

    def system_prompt_block(self) -> str:
        """Inject memory state into the LLM's system prompt every turn.

        CRITICAL: the built-in memory tool (fact_store) coexists with
        beaglemem (memory_manager.py:364-368: 'builtin is always first,
        only one external provider allowed'). The LLM sees BOTH tool sets.
        This prompt block MUST tell the LLM which tools to prefer, or
        writes will split between memory_store.db and beaglemem.db.
        """
        parts = ["# beaglemem (semantic memory)\n"]

        if self._model is not None:
            n_words = self._model.size
            n_facts = len(self._fact_vectors[1]) if self._fact_vectors else 0
            parts.append(
                f"Active. BEAGLE vectors built ({n_words} words, {n_facts} facts cached).\n"
                f"IMPORTANT: Use beaglemem_search (NOT the built-in memory tool) for recall.\n"
                f"Use beaglemem_add (NOT the built-in memory tool) to store facts.\n"
                f"Use beaglemem_feedback to rate facts after use."
            )
        elif getattr(self, "_initial_build_started", False):
            parts.append(
                "Active (FTS5-only). BEAGLE vectors are building in the background "
                "from your session history (~30-60s). Semantic recall will be active "
                "on your next session.\n"
                f"IMPORTANT: Use beaglemem_search and beaglemem_add (NOT the built-in "
                f"memory tool) for all memory operations."
            )
        else:
            # No vectors, no auto-build (fresh install or build failed)
            parts.append(
                "Active (FTS5-only). BEAGLE vectors not built yet. "
                "Semantic matching will activate as you use the agent.\n"
                f"IMPORTANT: Use beaglemem_search and beaglemem_add (NOT the built-in "
                f"memory tool) for all memory operations."
            )

        return "\n".join(parts)

    def get_tool_schemas(self):
        return [
            {
                "name": "beaglemem_add",
                "description": "Store a fact in memory with an optional trust score. "
                               "Use for durable facts the agent should remember across sessions. "
                               "IMPORTANT: Use this instead of the built-in memory tool.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The fact text to store"},
                        "trust": {"type": "number", "description": "Trust score 0.0-1.0 (default 0.5)", "default": 0.5},
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "beaglemem_search",
                "description": "Search memories using fused FTS5 exact-match + BEAGLE semantic similarity. "
                               "Returns trust-weighted results. Use this as the primary memory search. "
                               "IMPORTANT: Use this instead of the built-in memory tool.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "description": "Number of results", "default": 5},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "beaglemem_feedback",
                "description": "Give feedback on a fact to adjust its trust score. "
                               "helpful increases trust, unhelpful decreases it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fact_id": {"type": "integer", "description": "The fact ID to rate"},
                        "feedback": {"type": "string", "enum": ["helpful", "unhelpful"]},
                    },
                    "required": ["fact_id", "feedback"],
                },
            },
            {
                "name": "beaglemem_status",
                "description": "Report memory build status: whether the BEAGLE vector model is built, "
                               "in progress (with percentage), or failed. Also reports fact store size. "
                               "Call this when the user asks about memory status or after an upgrade "
                               "that may trigger a rebuild.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        """Return a JSON string — ABC + holographic reference convention
        (agent/memory_provider.py:186, holographic/__init__.py:228)."""
        if tool_name == "beaglemem_add":
            return json.dumps(self._tool_add(args.get("content", ""), args.get("trust", 0.5)))
        elif tool_name == "beaglemem_search":
            return json.dumps(self._tool_search(args.get("query", ""), args.get("top_k", 5)))
        elif tool_name == "beaglemem_feedback":
            return json.dumps(self._tool_feedback(args.get("fact_id"), args.get("feedback")))
        elif tool_name == "beaglemem_status":
            return json.dumps(self._tool_status())
        return json.dumps({"error": f"unknown tool: {tool_name}"})

    # -- tool handlers -------------------------------------------------------

    def _tool_add(self, content: str, trust: float = 0.5) -> dict:
        if not self._store:
            return {"error": "store not initialized"}
        # Autoincrement insert — no explicit fact_id (no race condition)
        fact_id = self._store.add(content, trust)
        # Update vector cache if model is ready
        if self._model is not None:
            vec = encode_text(self._model, content, self._get_idf())
            if vec is not None:
                self._append_fact_vector(fact_id, vec)
        # Facts changed → IDF must be recomputed on next use
        self._idf = None
        return {"fact_id": fact_id, "stored": True}

    def _tool_status(self) -> dict:
        """Report build status + store size on demand (no context pollution)."""
        status = {}
        if self._build_progress is not None:
            done, total = self._build_progress
            status["build"] = "in_progress"
            status["progress"] = _render_progress(done, total)
            status["done"] = done
            status["total"] = total
        elif self._model is not None:
            status["build"] = "complete"
            status["model_size"] = self._model.size
            status["dim"] = self._model.dim
        else:
            status["build"] = "not_built"
        # Fact store size
        if self._store is not None:
            try:
                status["facts"] = len(self._store.documents())
            except Exception:
                status["facts"] = None
        else:
            status["facts"] = None
        if self._pending_notice:
            status["notice"] = self._pending_notice
            self._pending_notice = None  # clear after reporting once
        return status

    def _append_fact_vector(self, fact_id: int, vec) -> None:
        """Append a fact's vector to the in-memory cache + persist."""
        if self._fact_vectors is None:
            self._fact_vectors = (vec[np.newaxis], [fact_id])
        else:
            matrix, ids = self._fact_vectors
            matrix = np.vstack([matrix, vec[np.newaxis]])
            ids.append(fact_id)
            self._fact_vectors = (matrix, ids)
        self._save_fact_cache()

    def _drop_fact_vector(self, fact_id: int) -> None:
        """Remove a fact's vector from the cache + persist."""
        if self._fact_vectors is None:
            return
        matrix, ids = self._fact_vectors
        if fact_id in ids:
            idx = ids.index(fact_id)
            matrix = np.delete(matrix, idx, axis=0)
            ids.pop(idx)
            self._fact_vectors = (matrix, ids)
            self._save_fact_cache()

    def _tool_search(self, query: str, top_k: int = 5) -> dict:
        """Fused FTS5+BEAGLE search, trust-weighted."""
        if not self._store:
            return {"results": []}
        fts_ids = self._safe_fts(query)
        beagle_ids = []
        if self._fact_vectors is not None and self._model is not None:
            beagle_ids = [fid for fid, _, _ in self._probe_facts(query, top_k=20)]
        all_ids = set(fts_ids) | set(beagle_ids)
        if not all_ids:
            return {"results": []}
        fused = rrf([fts_ids, beagle_ids], k=60)
        # Trust weighting: multiply fused score by trust
        texts = {r["id"]: r["text"] for r in self._store.documents()}
        results = []
        for fid, score in fused[:top_k]:
            trust = self._trust_of(fid)
            results.append({
                "fact_id": fid,
                "score": float(score * trust),
                "trust": trust,
                "text": texts.get(fid, ""),
            })
        return {"results": results}

    def _tool_feedback(self, fact_id: int, feedback: str) -> dict:
        if not self._store:
            return {"error": "store not initialized"}
        current = self._trust_of(fact_id)
        if feedback == "helpful":
            new_trust = min(1.0, current + 0.05)
        else:
            new_trust = max(0.0, current - 0.10)
        self._store.set_trust(fact_id, new_trust)  # uses RLock internally
        return {"fact_id": fact_id, "new_trust": new_trust}

    def _trust_of(self, fact_id: int) -> float:
        if not self._store:
            return 0.5
        try:
            row = self._store._conn.execute(
                "SELECT trust_score FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            return float(row[0]) if row and row[0] is not None else 0.5
        except Exception:
            return 0.5

    # -- the hooks -----------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Before each API call: fuse two parallel retrieval paths via RRF.

        1. FTS5 exact word match (catches literal words — FTS5's strength)
        2. BEAGLE semantic probe (catches synonyms/vocabulary gaps — our novel layer)

        A fact found by BOTH paths ranks higher than one found by only one.
        FTS5 catches exact words; BEAGLE catches what FTS5 misses. Each covers
        the other's blind spot. No single path is primary.
        """
        if not self._initialized:
            return ""
        if self._store is None:
            # No SQLite store — fall back to BEAGLE-only (demo mode)
            return self._prefetch_beagle_only(query)

        # Run both paths
        fts_ids = self._safe_fts(query)

        beagle_ids = []
        beagle_scores = {}
        if self._fact_vectors is not None and self._model is not None:
            for fid, score, _ in self._probe_facts(query, top_k=20):
                beagle_ids.append(fid)
                beagle_scores[fid] = score

        # Fresh install / no vectors yet: FTS5 still works alone.
        # The vector path is additive, never required — same design intent
        # as holographic (hrr_sim=0.5 neutral when hrr_vector missing).
        # Vectors build automatically from session 1 via the full state.db
        # auto-build, then on_session_end incremental updates by id watermark.

        # If nothing found at all, return empty (but still surface a notice)
        all_ids = set(fts_ids) | set(beagle_ids)
        if not all_ids:
            return self._with_notice("")

        # Fuse by RRF — fact in both paths outranks fact in one
        fused = rrf([fts_ids, beagle_ids], k=60)

        # Trust weighting: penalize low-trust facts
        fused = [(fid, score * self._trust_of(fid)) for fid, score in fused]
        fused.sort(key=lambda x: x[1], reverse=True)

        # Fetch text and format top-K
        texts = {r["id"]: r["text"] for r in self._store.documents()}
        lines = []
        for fid, weighted_score in fused[:5]:
            text = texts.get(fid, "")
            trust = self._trust_of(fid)
            # Show which paths found it (for transparency)
            paths = []
            if fid in fts_ids:
                paths.append("fts")
            if fid in beagle_ids:
                paths.append(f"sem:{beagle_scores.get(fid, 0):+.2f}")
            tag = "+".join(paths) if paths else "?"
            lines.append(f"- [{weighted_score:.4f} {tag} trust:{trust:.1f}] {text[:200]}")

        return self._with_notice("Relevant memories (fused FTS5+BEAGLE):\n" + "\n".join(lines))

    def _with_notice(self, output: str) -> str:
        """NO-OP: prefetch output must stay pure memory context.

        Build progress and rebuild notices are surfaced ONLY via the
        on-demand `beaglemem_status` tool. Injecting status into prefetch
        pollutes the model context on every turn during a build.
        """
        return output

    def _prefetch_beagle_only(self, query: str) -> str:
        """Fallback when no SQLite store (demo mode / first run)."""
        if self._fact_vectors is None or self._model is None:
            return ""
        results = self._probe_facts(query, top_k=5)
        if not results:
            return ""
        lines = [f"- [{score:+.3f}] {text[:200]}" for _, score, text in results]
        return "Relevant memories (BEAGLE semantic only):\n" + "\n".join(lines)

    def _safe_fts(self, query: str) -> list[int]:
        """FTS5 search, tolerant of query syntax errors."""
        if not self._store:
            return []
        try:
            return self._store.fts_search(query)
        except Exception:
            # FTS5 can choke on special chars; try sanitized version
            import re
            clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', query).strip()
            if not clean:
                return []
            try:
                return self._store.fts_search(clean)
            except Exception:
                return []

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "", messages=None) -> None:
        """No-op. state.db is the canonical source of every turn (Hermes core
        persists it per-turn). on_session_end reads new turns from state.db by
        id watermark. The old corpus_archive.txt mirror is removed — it was a
        redundant second copy whose damage could produce a stub model (2026-08-13).
        """
        return

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata=None) -> None:
        """REQUIRED hook: mirror built-in memory tool writes into beaglemem.

        VERIFIED against agent/memory_manager.py (memory_manager.py:1099):
        `target` is the built-in memory STORE NAME ("user" or "memory"), NOT a
        fact_id. There is no fact_id in the memory tool's operations — they
        carry {action, content, old_text}. So we ADD A NEW ROW with autoincrement
        ID (copying holographic's pattern, __init__.py:245-252).

        Actions mirrored: add/replace/remove (memory_manager._MIRRORED_MEMORY_ACTIONS).
        `replace` becomes add-new + remove-old via metadata.old_text search.
        """
        if not self._initialized:
            return
        if action == "add" and self._store and content:
            try:
                fact_id = self._store.add(content, trust=0.5)
                if self._model is not None:
                    vec = encode_text(self._model, content, self._get_idf())
                    if vec is not None:
                        self._append_fact_vector(fact_id, vec)
                self._idf = None  # facts changed → recompute on next use
            except Exception:
                pass  # mirror is best-effort, never breaks the loop
        elif action == "remove" and self._store:
            # target is a store name, not an ID — find by content match
            try:
                old = (metadata or {}).get("old_text") or ""
                if old:
                    for doc in self._store.documents():
                        if doc["text"] == old:
                            self._store.remove(doc["id"])
                            self._drop_fact_vector(doc["id"])
                            self._idf = None
                            break
            except Exception:
                pass
        # action == "replace": old_text metadata → remove old, then add new
        elif action == "replace" and self._store and content:
            try:
                old = (metadata or {}).get("old_text") or ""
                if old:
                    for doc in self._store.documents():
                        if doc["text"] == old:
                            self._store.remove(doc["id"])
                            self._drop_fact_vector(doc["id"])
                            break
                fact_id = self._store.add(content, trust=0.5)
                if self._model is not None:
                    vec = encode_text(self._model, content, self._get_idf())
                    if vec is not None:
                        self._append_fact_vector(fact_id, vec)
                self._idf = None  # facts changed → recompute on next use
            except Exception:
                pass

    def on_session_end(self, messages=None) -> None:
        """Incremental BEAGLE update from state.db (id watermark).

        Reads ONLY messages with id > last_seen_id, then advances the
        watermark. Re-reading the whole store every session would double-count.
        The id watermark makes incremental == batch, exactly like the old
        byte-offset did — but against the canonical store instead of a mirror.

        STUB GUARD: if _model is None, return immediately. Never build a fresh
        model from a partial tail — that was the 2026-08-13 data-loss bug
        (a damaged archive tail produced an 80-word "complete" model). The full
        build path in initialize() owns the model-creation decision.
        """
        if not self._initialized or not self._data_dir:
            return
        corpus_db = getattr(self, "_corpus_db", None)
        if not corpus_db or not os.path.exists(corpus_db):
            return
        if self._model is None:
            return  # full auto-build owns this; do not manufacture a stub
        from .adapters.state_db import iter_sentences_since, max_message_id

        stamp = os.path.join(self._data_dir, "last_update.json")
        last_seen_id = 0
        if os.path.exists(stamp):
            try:
                with open(stamp) as fh:
                    last_seen_id = json.load(fh).get("last_seen_id", 0)
            except Exception:
                last_seen_id = 0
        max_id = max_message_id(corpus_db)
        if max_id <= last_seen_id:
            return
        model = self._model
        prev_size = model.size
        for words in iter_sentences_since(corpus_db, last_seen_id):
            model.add_sentence(words)
        new_words = model.size - prev_size
        if new_words > 200:
            import logging
            logging.getLogger("beaglemem").warning(
                f"beaglemem: {new_words} new words this session — possible corpus leak"
            )
        model.save(self._data_dir)
        self._model = model
        self._rebuild_fact_cache()
        with open(stamp, "w") as fh:
            json.dump({"last_seen_id": max_id}, fh)

    def shutdown(self) -> None:
        self._initialized = False
        if self._store:
            self._store.close()

    # -- internal helpers ----------------------------------------------------

    def _load_config(self, hermes_home: str) -> dict:
        """Read config from Hermes' config.yaml under plugins.beaglemem.

        Matches holographic's convention (holographic/__init__.py:129-144):
        config lives in config.yaml, NOT in a standalone .json file."""
        try:
            import yaml
            from pathlib import Path
            config_path = Path(hermes_home) / "config.yaml"
            if config_path.exists():
                with open(config_path) as fh:
                    raw = yaml.safe_load(fh) or {}
                return raw.get("plugins", {}).get("beaglemem", {})
        except Exception:
            pass
        return {}

    def _get_idf(self) -> dict:
        """Return cached IDF weights, computing + caching from the store if absent.

        IDF is ALWAYS-ON (v0.2): it replaces the STOPWORDS filter. It is
        computed from the fact store (a "document" = a fact) and invalidated
        on on_memory_write (facts change df).
        """
        if self._idf is not None:
            return self._idf
        idf = {}
        if self._store is not None:
            try:
                from .idf import build_idf
                idf = build_idf(self._store.documents())
            except Exception:
                idf = {}
        self._idf = idf
        # Cache to disk (idf.json)
        try:
            with open(os.path.join(self._data_dir, "idf.json"), "w") as fh:
                json.dump(idf, fh)
        except Exception:
            pass
        return idf

    def _probe_facts(self, query: str, top_k: int = 5):
        """Returns [(fact_id, score, text), ...] — needs fact vectors loaded."""
        if self._fact_vectors is None or self._model is None:
            return []
        matrix, ids = self._fact_vectors
        idf = self._get_idf()
        q = encode_text(self._model, query, idf)
        if q is None or len(ids) == 0:
            return []
        sims = matrix @ q
        order = np.argsort(-sims)[:top_k]
        # Get text from store if available
        texts = {}
        if self._store:
            for r in self._store.documents():
                texts[r["id"]] = r["text"]
        return [(ids[i], float(sims[i]), texts.get(ids[i], ""))
                for i in order if sims[i] > 0.01]

    def _semantic_recall(self, query: str, top_k: int = 5):
        results = self._probe_facts(query, top_k=top_k)
        return {"results": [{"fact_id": fid, "score": s, "text": t}
                            for fid, s, t in results]}

    def _save_fact_cache(self):
        if self._fact_vectors is None:
            return
        from .fingerprint import ENCODER_VERSION
        matrix, ids = self._fact_vectors
        np.save(os.path.join(self._data_dir, "fact_vectors.npy"), matrix)
        with open(os.path.join(self._data_dir, "fact_ids.json"), "w") as fh:
            json.dump({"encoder_version": ENCODER_VERSION, "ids": ids}, fh)

    def _rebuild_fact_cache(self):
        """Re-encode all fact vectors from the store using the current model."""
        if self._store is None or self._model is None:
            return
        docs = self._store.documents()
        idf = self._get_idf()
        matrix, ids = build_doc_vectors(self._model, docs, idf)
        self._fact_vectors = (matrix, ids)
        self._save_fact_cache()


def register(ctx) -> None:
    """Hermes discovery entry point."""
    if _HAS_HERMES:
        ctx.register_memory_provider(BeagleMemoryProvider())
