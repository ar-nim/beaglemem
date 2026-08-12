"""Incremental ingest: corpus files newer than the last update stamp.

Usage: python3 scripts/update.py --corpus <dir> --format chat-jsonl --data data
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from beaglemem.corpus import iter_sentences
from beaglemem.vectors import BeagleModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--format", default="plain", choices=["plain", "chat-jsonl", "state_db"])
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    model = BeagleModel.load(args.data)
    stamp_path = os.path.join(args.data, "last_update.json")
    last = 0.0
    if os.path.exists(stamp_path):
        with open(stamp_path) as fh:
            last = json.load(fh).get("mtime", 0.0)

    if args.format == "chat-jsonl":
        pattern = "*.jsonl"
    else:
        pattern = "*.txt"
    files = sorted(
        p for p in glob.glob(os.path.join(args.corpus, pattern))
        if os.path.getmtime(p) > last
    )
    if args.format == "plain":
        files += sorted(
            p for p in glob.glob(os.path.join(args.corpus, "*.md"))
            if os.path.getmtime(p) > last
        )
    if not files:
        print("No new corpus files. Nothing to do.")
        return

    t0 = time.time()
    n = 0
    for path in files:
        for words in iter_sentences(path, format=args.format):
            model.add_sentence(words)
            n += 1
    model.save(args.data)
    with open(stamp_path, "w") as fh:
        json.dump({"mtime": max(os.path.getmtime(p) for p in files)}, fh)
    print(f"Ingested {n} sentences from {len(files)} files in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
