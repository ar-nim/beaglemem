"""Deterministic demo corpus with planted semantic bridges.

The corpus teaches: "let go" keeps company with {hr, letter, paperwork,
signed, division} — the same company "severance" keeps. No document contains
"let go". If probe("let go") surfaces the severance document, the semantic
bridge is real. Generated with a fixed seed → CI-safe, always reproducible.
"""
import json
import os
import random

WORK_NEIGHBORS = ["hr", "letter", "paperwork", "signed", "division"]
BRIDGE_TERMS = [["severance"], ["termination"], ["restructuring"], ["let", "go"], ["fired"]]
FILLER_CLUSTERS = [
    ["sunny", "forecast", "rain", "weekend", "picnic"],
    ["recipe", "oven", "bake", "flour", "cake"],
    ["kernel", "compile", "driver", "module", "boot"],
]

DOCS = [
    {"id": 1, "text": "Sunny forecast for the weekend picnic in the park"},
    {"id": 2, "text": "Recipe for cake: bake flour in the oven slowly"},
    {"id": 3, "text": "Kernel driver module fails to compile on boot"},
    {"id": 4, "text": "Weekend rain may cancel the outdoor picnic"},
    {"id": 5, "text": "Oven temperature guide for the bread recipe"},
    {"id": 6, "text": "Boot sequence hangs at the kernel module load"},
    {"id": 7, "text": "Severance letter signed by hr for the beta division"},
    {"id": 8, "text": "Picnic blanket and forecast check before saturday"},
]


def make_demo_corpus(out_dir: str, seed: int = 42, repeats: int = 40):
    rng = random.Random(seed)
    corpus_dir = os.path.join(out_dir, "mini_corpus")
    os.makedirs(corpus_dir, exist_ok=True)

    lines = []
    for _ in range(repeats):
        for bridge in BRIDGE_TERMS:
            neighbors = rng.sample(WORK_NEIGHBORS, 3)
            sentence = " ".join(bridge + neighbors)
            lines.append({"role": "user", "content": f"We discussed the {sentence} today."})
        for cluster in FILLER_CLUSTERS:
            words = rng.sample(cluster, 3)
            lines.append({"role": "user", "content": " ".join(words) + " again."})

    rng.shuffle(lines)
    with open(os.path.join(corpus_dir, "demo_session.jsonl"), "w") as fh:
        for row in lines:
            fh.write(json.dumps(row) + "\n")

    return corpus_dir, DOCS
