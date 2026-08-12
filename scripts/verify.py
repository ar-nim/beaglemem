"""Acceptance gate. Usage:
  python3 scripts/verify.py --data data --config examples/verify.demo.json
  python3 scripts/verify.py --data data --config verify.local.json   # private

FAIL = report numbers verbatim, never tune thresholds.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from beaglemem.probe import probe
from beaglemem.store import MemoryStore
from beaglemem.vectors import BeagleModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    model = BeagleModel.load(args.data)
    with open(args.config) as fh:
        cfg = json.load(fh)

    failures = []
    print("=== Semantic pairs ===")
    for a, b, floor in cfg.get("semantic_pairs", []):
        sim = model.word_cosine(a, b)
        ok = sim > floor
        print(f"  {'PASS' if ok else 'FAIL'}  {a} ≈ {b}  {sim:+.4f} (need > {floor})")
        if not ok:
            failures.append((a, b, sim))

    print("=== Noise controls ===")
    for a, b, ceil in cfg.get("noise_pairs", []):
        sim = model.word_cosine(a, b)
        ok = sim < ceil
        print(f"  {'PASS' if ok else 'FAIL'}  {a} ≉ {b}  {sim:+.4f} (need < {ceil})")
        if not ok:
            failures.append((a, b, sim))

    print("=== Probe tests ===")
    docs = cfg.get("documents", [])
    store = MemoryStore(docs)
    for t in cfg.get("probe_tests", []):
        t0 = time.time()
        results = probe(model, t["query"], store, top_k=t.get("top_k", 20))
        found = any(did == t["must_find"] for did, _ in results)
        rank = next((i + 1 for i, (did, _) in enumerate(results) if did == t["must_find"]), None)
        print(f"  {'PASS' if found else 'FAIL'}  probe('{t['query']}') → doc {t['must_find']} "
              f"{f'at rank {rank}' if found else 'NOT FOUND'} ({(time.time()-t0)*1000:.1f}ms)")
        if not found:
            failures.append((t["query"], t["must_find"]))

    print()
    if failures:
        print(f"ACCEPTANCE: FAIL ({len(failures)}). Report numbers; do not tune.")
        sys.exit(1)
    print("ACCEPTANCE: PASS.")


if __name__ == "__main__":
    main()
