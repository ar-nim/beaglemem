"""CLI probe. Usage:
  python3 scripts/probe.py --data data --query "let go" --docs docs.json [--top-k 10]
docs.json: [{"id": ..., "text": "..."}, ...]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from beaglemem.probe import probe
from beaglemem.store import MemoryStore
from beaglemem.vectors import BeagleModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    model = BeagleModel.load(args.data)
    with open(args.docs) as fh:
        docs = json.load(fh)
    for did, score in probe(model, args.query, MemoryStore(docs), top_k=args.top_k):
        print(f"{score:+.4f}  {did}")


if __name__ == "__main__":
    main()
