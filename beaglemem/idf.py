"""Inverse Document Frequency weighting — language-agnostic stopword replacement.

IDF downweights words that appear in many documents (universal hubs: "the",
"yang", "から") and boosts rare, specific words. No per-language lists.
A "document" in beaglemem is a fact (the retrieval unit), so IDF is computed
from the fact store, not the corpus.

idf(w) = log(N / df_w). Unknown words get the floor (0.0).

Formula (2026-08-12): log(N/df), NOT log(N/(1+df)). The +1 smoothing made
universal hubs (df=N) go NEGATIVE — log(N/(N+1)) < 0 — flipping their vector
contribution direction on small corpora. log(N/df) floors at exactly 0 for
df=N (word in every fact contributes nothing) and peaks at log(N) for df=1.
df >= 1 for every counted word, so no division-by-zero guard needed.
"""
import math

from .corpus import tokenize

IDF_FLOOR = 1.0   # unknown words: NEUTRAL weight (full contribution, no boost
                  # or penalty). MUST NOT be 0.0 — the semantic bridge depends
                  # on query words absent from the fact store (demo: "let go"
                  # → severance). 0.0 would zero their contribution and kill
                  # the bridge. 1.0 preserves it while hubs (in every doc)
                  # still floor to 0 via log(N/df).


def build_idf(docs: list[dict]) -> dict[str, float]:
    """Compute IDF weights from a list of {'id', 'text'} documents.

    docs: the fact store's documents() output. Tokens are extracted with the
    SAME tokenizer used everywhere else — this is what keeps build, query,
    and update consistent.

    idf(w) = log(N / df_w), where df_w >= 1 for every counted word, so the
    result is always >= 0. A word in every doc (df=N) floors at exactly 0.
    """
    N = len(docs)
    df: dict[str, int] = {}
    for d in docs:
        seen = set()
        for w in tokenize(d["text"]):
            if w not in seen:
                seen.add(w)
                df[w] = df.get(w, 0) + 1

    idf: dict[str, float] = {}
    for w, c in df.items():
        idf[w] = math.log(N / c)
    return idf


def idf_weight(idf: dict, word: str) -> float:
    """Weight for a word; floor for unknown words."""
    return idf.get(word, IDF_FLOOR)
