# beaglemem

Semantic memory that learns your vocabulary from your own text. Deterministic. Zero dependencies beyond numpy. Elasticsearch needs a cluster, BERT needs a download — this needs a folder.

## How it works

BEAGLE (Bound Encoding of the Aggregate Language Environment, Jones & Mewhort 2007) learns word meaning from co-occurrence statistics in your corpus. Words that keep the same company have similar semantic vectors:

```
"let go" ≈ "severance"  (shared neighbors: hr, letter, signed, division)
"6x"     ≈ "bus"        (shared neighbors: Kingsford, commute, Metroline)
```

Retrieval fuses two deterministic paths by Reciprocal Rank Fusion:
1. **FTS5** — exact word match (what the text literally says)
2. **BEAGLE** — semantic similarity (what the text means)

No model downloads. No API keys. No GPU. Same corpus → same vectors → same answers, every run, on every LLM.

## Quickstart (demo)

```bash
python3 examples/make_demo_corpus.py --out /tmp/demo
python3 scripts/build.py --corpus /tmp/demo/mini_corpus --format chat-jsonl --out /tmp/demo/data
python3 scripts/verify.py --data /tmp/demo/data --config examples/verify.demo.json
# → ACCEPTANCE: PASS — probe("let go") surfaces the severance doc
```

## Use on your own corpus

```bash
# Plain text files (.txt, .md)
python3 scripts/build.py --corpus ~/notes --format plain --out ~/beagle-data

# Chat logs (role/content JSONL — Hermes sessions, OpenAI-style exports)
python3 scripts/build.py --corpus ~/.hermes/sessions --format chat-jsonl --out ~/beagle-data

# Probe
python3 scripts/probe.py --data ~/beagle-data --query "retrenched" --docs docs.json

# Incremental update
python3 scripts/update.py --corpus ~/.hermes/sessions --format chat-jsonl --data ~/beagle-data
```

## What this is NOT

- **Not a database** — SQLite stays the store. BEAGLE is the index.
- **Not a replacement for exact-match FTS5** — it fuses WITH it.
- **Not an LLM** — deterministic, no generation, no reasoning. It ranks; the LLM reads the ranks.

## Corpus lifecycle

If your source rotates or cleans itself (e.g., agent session cleanup), snapshot new material into a compounding archive before cleanup. Vectors compound forever — the archive is rebuild insurance. Never reset vectors when the source prunes.

## Limitations

- Corpus sparsity: needs enough co-occurrence (~1M+ words recommended; less works with weaker signal)
- English-biased stopword list
- No order encoding (v0.2 candidate)
- No recency decay (equal weights, paper-faithful)

## Roadmap

- Order encoding
- Recency decay
- Console scripts (`pip install`)
- More adapters

## License

MIT. See NOTICE for paper attribution.
