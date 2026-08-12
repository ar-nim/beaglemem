"""Build BEAGLE vectors from any corpus directory.

Usage: python3 scripts/build.py --corpus <path> --format plain|chat-jsonl|state_db --out <data_dir>
"""
import argparse
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
    ap.add_argument("--out", default="data")
    ap.add_argument("--dim", type=int, default=2048)
    ap.add_argument("--window", type=int, default=3)
    args = ap.parse_args()

    t0 = time.time()
    model = BeagleModel(dim=args.dim, window=args.window)
    n = 0
    for words in iter_sentences(args.corpus, format=args.format):
        model.add_sentence(words)
        n += 1
        if n % 10000 == 0:
            print(f"  {n} sentences, {model.size} words, {time.time()-t0:.0f}s", flush=True)
    model.save(args.out)
    print(f"Done: {model.size} words from {n} sentences in {time.time()-t0:.0f}s → {args.out}/")


if __name__ == "__main__":
    main()
