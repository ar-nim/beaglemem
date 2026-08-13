"""hermes beaglemem <status|config|build|migrate> — CLI for the provider."""
import json
import os


def _hermes_home() -> str:
    """Profile-aware Hermes home. Hermes sets HERMES_HOME to the active
    profile's directory (~/.hermes/profiles/<name> for named profiles,
    ~/.hermes for default); the runtime provider receives it via
    memory_manager.initialize_all(). The CLI must resolve the SAME home or
    `hermes -p <profile> beaglemem <cmd>` operates on the wrong profile's
    data. Never hardcode ~/.hermes."""
    return os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")


def _status(args):
    home = _hermes_home()
    db_path = os.path.join(home, "beaglemem-data", "beaglemem.db")
    if os.path.exists(db_path):
        try:
            from .store import BeagleStore
            store = BeagleStore(db_path)
            n_words = len(store.vocab_words())
            meta = store.all_meta()
            n_facts = len(store.fact_ids())
            dim = meta.get("dim", "?")
            print(f"Provider: beaglemem — model built ({n_words} words, dim={dim}, {n_facts} facts cached)")
            store.close()
        except Exception:
            print("Provider: beaglemem — model NOT built yet. Run: hermes beaglemem build")
    else:
        print("Provider: beaglemem — model NOT built yet. Run: hermes beaglemem build")


def _config(args):
    home = _hermes_home()
    cfg = os.path.join(home, "config.yaml")
    if os.path.exists(cfg):
        try:
            import yaml
            raw = yaml.safe_load(open(cfg)) or {}
            bm_cfg = raw.get("plugins", {}).get("beaglemem", {})
            if bm_cfg:
                print(yaml.dump({"plugins": {"beaglemem": bm_cfg}}, default_flow_style=False))
            else:
                print("No beaglemem config in config.yaml — run hermes memory setup to configure.")
        except Exception:
            print("Could not parse config.yaml — run hermes memory setup to configure.")
    else:
        print("No config.yaml yet — run hermes setup first.")


def _migrate(args):
    """One-time migration from holographic memory_store.db.

    Reads facts from the existing memory_store.db (created by the holographic
    provider) and copies them into beaglemem's own store. Read-only on the
    source; writes only to the destination. Opt-in — beaglemem never assumes
    a holographic store exists."""
    from .store import BeagleStore

    home = _hermes_home()
    src = args.source or os.path.join(home, "memory_store.db")
    if not os.path.exists(src):
        print(f"No source store at {src} — nothing to migrate.")
        return

    data_dir = os.path.join(home, "beaglemem-data")
    os.makedirs(data_dir, exist_ok=True)
    dst = BeagleStore(os.path.join(data_dir, "beaglemem.db"), create=True)

    # Read-only on source
    import sqlite3
    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT content, trust_score FROM facts WHERE content IS NOT NULL"
        ).fetchall()
    except Exception as e:
        print(f"Source schema not compatible: {e}")
        conn.close()
        return
    conn.close()

    migrated = 0
    for row in rows:
        dst.add(row["content"], float(row["trust_score"] or 0.5))
        migrated += 1
    print(f"Migrated {migrated} facts from {src} into beaglemem.db")
    print("Next: run 'hermes beaglemem build' to build vectors, then activate the provider.")


def _build(args):
    """Run the initial full BEAGLE build from configured corpus (v0.3: DB)."""
    from .vectors import BeagleModel
    from .corpus import iter_sentences
    from .probe import build_doc_vectors
    from .idf import build_idf
    from .store import BeagleStore
    from .fingerprint import ENCODER_VERSION, tokenizer_fingerprint
    from .corpus import WORD_RE
    from .adapters.state_db import max_message_id
    import time

    home = _hermes_home()
    data_dir = os.path.join(home, "beaglemem-data")
    os.makedirs(data_dir, exist_ok=True)
    db_path = os.path.join(data_dir, "beaglemem.db")
    # Read config from config.yaml (same convention as holographic)
    try:
        import yaml
        raw = yaml.safe_load(open(os.path.join(home, "config.yaml"))) or {}
        cfg = raw.get("plugins", {}).get("beaglemem", {})
    except Exception:
        cfg = {}
    corpus_dir = cfg.get("corpus_dir", os.path.join(home, "state.db"))
    fmt = cfg.get("format", "state_db")
    dim = int(cfg.get("dim", 2048))
    window = int(cfg.get("window", 3))

    store = BeagleStore(db_path, create=True)

    t0 = time.time()
    model = BeagleModel(dim=dim, window=window)
    n = 0
    for words in iter_sentences(corpus_dir, format=fmt):
        model.add_sentence(words)
        n += 1
        if n % 10000 == 0:
            print(f"  {n} sentences, {model.size} words, {time.time()-t0:.0f}s", flush=True)
    model.corpus_source = "state_db"

    # STAGE → VERIFY → SWAP (non-destructive)
    import numpy as np
    mem_path = os.path.join(data_dir, "beagle_mem.npy")
    tmp_path = mem_path + ".tmp"
    model.save_matrix(tmp_path)
    if model.size == 0:
        raise SystemExit("Empty vocab — refusing to persist. Is the corpus empty?")
    mem = np.load(tmp_path)
    if mem.shape[0] != model.size or mem.shape[1] != model.dim:
        raise SystemExit("Matrix verify failed")
    del mem

    current_fp = tokenizer_fingerprint(regex=WORD_RE.pattern, stemmer=None,
                                       dim=model.dim, window=model.window)
    store.persist_model(model.vocab, model._counts, {
        "dim": model.dim, "window": model.window,
        "min_count": model.min_count,
        "tokenizer_fingerprint": current_fp,
        "regex": WORD_RE.pattern, "stemmer": None,
        "consumed_sentences": model.consumed_sentences,
        "corpus_source": "state_db",
        "last_seen_id": max_message_id(corpus_dir),
        "encoder_version": ENCODER_VERSION,
    })
    os.replace(tmp_path, mem_path)
    print(f"Model: {model.size} words from {n} sentences in {time.time()-t0:.0f}s")

    # Build fact cache from the store (never-skip)
    docs = store.documents()
    if not docs:
        print("\n⚠ beaglemem.db has 0 facts. Vector cache will be empty.")
        builtin = os.path.join(home, "memory_store.db")
        if os.path.exists(builtin):
            print(f"  Found existing memory at {builtin}")
            print(f"  Run: hermes beaglemem migrate --source {builtin}")
            print(f"  Then re-run: hermes beaglemem build")
        elif os.path.exists(os.path.join(home, "state.db")):
            print(f"  Fresh install — facts will accumulate as you use the agent.")
    else:
        matrix, ids = build_doc_vectors(model, docs, build_idf(docs))
        fv_path = os.path.join(data_dir, "fact_vectors.npy")
        with open(fv_path + ".tmp", "wb") as fh:
            np.save(fh, matrix)
            os.fsync(fh.fileno())
        os.replace(fv_path + ".tmp", fv_path)
        store.set_meta("encoder_version", ENCODER_VERSION)
        print(f"Fact vectors: {len(ids)} facts cached")
    store.close()


def register_cli(subparser) -> None:
    """Hermes scans the ACTIVE provider's cli.py for this exact function
    (plugins/memory/__init__.py:373). Subcommands appear as
    `hermes beaglemem <status|config|build|migrate>`.

    Migration timing: beaglemem must be active for this CLI to exist, but
    holographic's memory_store.db stays on disk after switching — the
    migrate command reads it read-only, so nothing is lost."""
    subs = subparser.add_subparsers(dest="beaglemem_cmd")
    subs.add_parser("status", help="Show provider status")
    subs.add_parser("config", help="Show provider config")
    subs.add_parser("build", help="Build BEAGLE vectors from corpus")
    migrate_p = subs.add_parser("migrate", help="Migrate facts from holographic memory_store.db")
    migrate_p.add_argument("--source", default=None, help="Path to holographic memory_store.db")
    subparser.set_defaults(func=lambda args: {
        "status": _status, "config": _config, "build": _build, "migrate": _migrate,
    }.get(args.beaglemem_cmd, _status)(args))  # default to status
