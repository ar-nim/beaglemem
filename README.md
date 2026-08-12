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

Fused results are trust-weighted (score × trust), and trust is asymmetric:
helpful feedback +0.05, unhelpful −0.10 — trust is hard to build, easy to lose.

**Honest positioning:** FTS5 already catches most vocabulary variation in a
session-sized corpus (termination → severance, retrenched → laid off). BEAGLE's
edge is the pure-semantic gap — no shared words at all, but shared neighbors.
That gap is small but real, and BEAGLE is the layer that fills it.

No model downloads. No API keys. No GPU. Same corpus → same vectors → same answers, every run, on every LLM.

## Corpus cleaning

Built for dirty chat-log input — the BEAGLE papers use clean corpora (TASA,
Wikipedia); we feed raw conversation history. Four layers keep machine noise
out of the co-occurrence statistics:

1. **URL stripping** — full URLs removed before tokenizing (URL syntax tokens
   like http/com/org are universal co-occurrence hubs that would pollute every
   vector)
2. **Token filters** — base64 padding, hashes, no-vowel runs, single chars,
   and >25-char strings are killed; short alphanumerics (6x, k3s, pq9) are
   kept as domain vocabulary; pure numbers become `<NUM>`
3. **Line-length cap** — archive lines >1000 chars dropped (pasted code/logs)
4. **min_count pruning** — words appearing <2 times are invisible to probe
   (frequency-based, Gensim/Word2Vec standard)

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

## Build artifacts (data layout)

A build writes into the `--out` directory:

| File | What it is |
|---|---|
| `beagle_mem.npy` | BEAGLE memory vectors (float32 matrix, rows = vocab words) |
| `beagle_vocab.json` | vocab list + frequency counts + model meta (dim/window/min_count) |
| `fact_vectors.npy` | cached document vectors (rows = facts, cols = dim) |
| `fact_ids.json` | fact IDs aligned with `fact_vectors.npy` rows |
| `corpus_archive.txt` | compounding corpus (plugin writes every turn here) |
| `last_update.json` | incremental offset/mtime stamp (update.py / plugin) |

The fact-vector cache is why probe is fast: documents are encoded once at
build, not on every query. `data/` and `verify.local.json` are gitignored —
they hold generated vectors and your private acceptance config.

## Hermes memory provider

beaglemem is also a Hermes Agent memory provider plugin — the same package,
two audiences. Install under `~/.hermes/plugins/beaglemem/`, activate as the
memory provider, and:

- `hermes beaglemem status` — model/vocab/fact-cache status
- `hermes beaglemem config` — show config.yaml plugin section
- `hermes beaglemem build` — rebuild vectors from the configured corpus
- `hermes beaglemem migrate --source ~/.hermes/memory_store.db` — one-time
  read-only copy of facts from holographic's store

Cold start works FTS5-only; vectors auto-build in the background from session
history on first activation, and incremental updates run at session end.

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
- Dotted compounds (802.1q, v1.2) are split at the sentence/token level —
  deliberately NOT fixed: a dot-aware token rule was considered and rejected
  as too brittle for the co-occurrence gain
- Language-agnostic co-occurrence; cross-lingual bridges form when languages
  mix in the corpus (no explicit multilingual support)

## Roadmap

- Order encoding
- Recency decay
- Console scripts (`pip install`)
- More adapters

## License

MIT. See NOTICE for paper attribution.
