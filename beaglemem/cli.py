"""hermes beaglemem <status|config|build|migrate> — CLI for the provider."""
import json
import os


def _status(args):
    home = os.path.expanduser("~/.hermes")
    data_dir = os.path.join(home, "beaglemem-data")
    vocab = os.path.join(data_dir, "beagle_vocab.json")
    if os.path.exists(vocab):
        with open(vocab) as fh:
            meta = json.load(fh)
        n_facts = 0
        fv = os.path.join(data_dir, "fact_ids.json")
        if os.path.exists(fv):
            n_facts = len(json.load(open(fv)))
        print(f"Provider: beaglemem — model built ({len(meta['vocab'])} words, dim={meta['dim']}, {n_facts} facts cached)")
    else:
        print("Provider: beaglemem — model NOT built yet. Run: hermes beaglemem build")


def _config(args):
    home = os.path.expanduser("~/.hermes")
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
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from beaglemem.store import BeagleStore

    home = os.path.expanduser("~/.hermes")
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
    """Run the initial full BEAGLE build from configured corpus."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from beaglemem.vectors import BeagleModel
    from beaglemem.corpus import iter_sentences
    from beaglemem.probe import build_doc_vectors
    from beaglemem.store import BeagleStore
    import time

    home = os.path.expanduser("~/.hermes")
    data_dir = os.path.join(home, "beaglemem-data")
    os.makedirs(data_dir, exist_ok=True)
    # Read config from config.yaml (same convention as holographic)
    try:
        import yaml
        raw = yaml.safe_load(open(os.path.join(home, "config.yaml"))) or {}
        cfg = raw.get("plugins", {}).get("beaglemem", {})
    except Exception:
        cfg = {}
    corpus_dir = cfg.get("corpus_dir", os.path.join(home, "state.db"))
    fmt = cfg.get("format", "state_db")
    db_path = cfg.get("db_path", os.path.join(home, "beaglemem-data", "beaglemem.db"))

    t0 = time.time()
    model = BeagleModel(dim=2048, window=3)
    n = 0
    for words in iter_sentences(corpus_dir, format=fmt):
        model.add_sentence(words)
        n += 1
        if n % 10000 == 0:
            print(f"  {n} sentences, {model.size} words, {time.time()-t0:.0f}s", flush=True)
    model.save(data_dir)
    print(f"Model: {model.size} words from {n} sentences in {time.time()-t0:.0f}s")

    # Build fact cache from the store
    if os.path.exists(db_path):
        store = BeagleStore(db_path)
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
            store.close()
        else:
            matrix, ids = build_doc_vectors(model, docs)
            import numpy as np
            np.save(os.path.join(data_dir, "fact_vectors.npy"), matrix)
            with open(os.path.join(data_dir, "fact_ids.json"), "w") as fh:
                json.dump(ids, fh)
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
