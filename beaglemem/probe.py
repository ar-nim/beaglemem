"""Semantic probe: encode text as a sum of memory vectors, rank documents by
cosine similarity.

v0.2: IDF weighting replaces STOPWORDS filtering in the encode path. STOPWORDS
remains as a constant for reference, but encode_text no longer filters it —
IDF (log(N/df)) downweights universal hubs automatically, language-agnostic.
"""
import numpy as np

from .corpus import tokenize
from .idf import build_idf, idf_weight

# Function words co-occur with everything, so their mem vectors carry a shared
# centroid direction that inflates all similarities.
# v0.2: kept as a documented reference constant; the encode path now uses IDF
# (idf_weight) instead of filtering these. IDF is language-agnostic and
# self-tuning — no curated list needed.
STOPWORDS = frozenset(
    "the a an and or of to in on for with is are was were be been it its this "
    "that these those i you he she we they me him her us them my your his their "
    "our as at by from not no yes do does did have has had will would can could "
    "should shall may might must am if then else when while so but about into "
    "over under again further once here there all any both each few more most "
    "other some such only own same than too very just im youre dont didnt arent "
    "isnt wont werent its ive id ill well "
    "<num> <NUM>".split()
)


def encode_text(model, text: str, idf: dict):
    """Encode text as a length-1 vector of summed memory vectors, each scaled
    by its IDF weight. IDF is REQUIRED (v0.2 breaking change) — it replaces
    the STOPWORDS filter.

    Words absent from the idf map get IDF_FLOOR (1.0) — neutral, so
    out-of-fact query vocabulary still contributes (semantic bridge).
    """
    words = tokenize(text)
    acc = np.zeros(model.dim, dtype=np.float32)
    known = 0
    for w in words:
        m = model.mem_of(w)
        if m is not None:
            acc += m * idf_weight(idf, w)
            known += 1
    if known == 0:
        return None
    norm = np.linalg.norm(acc)
    return acc / norm if norm > 0 else None


def build_doc_vectors(model, docs: list[dict], idf: dict = None):
    """docs: [{'id': ..., 'text': str}] → (matrix (n, dim) float32, ids).

    idf: optional. When None, computed from the docs themselves (standalone
    use); the plugin passes a cached idf for consistency across builds.

    v0.3 never-skip contract: EVERY fact gets a row. A fact with no vocab
    overlap gets a ZERO vector — probe filters sims > 0.01, so it never
    surfaces (behaviorally identical to skipping). Keeping rows == len(docs)
    == fact count is what makes the row↔fact mapping derivable from the DB
    alone, so fact_ids does not need to be persisted on disk.
    """
    if idf is None:
        idf = build_idf(docs)
    rows, ids = [], []
    zero = None
    for d in docs:
        v = encode_text(model, d["text"], idf)
        if v is not None:
            rows.append(v)
        else:
            if zero is None:
                zero = np.zeros(model.dim, dtype=np.float32)
            rows.append(zero)
        ids.append(d["id"])
    if not rows:
        return np.zeros((0, model.dim), dtype=np.float32), []
    return np.stack(rows), ids


def probe(model, query: str, store, top_k: int = 10, idf: dict = None):
    """store: any DocumentStore (documents() → [{'id','text'}])."""
    docs = store.documents()
    if idf is None:
        idf = build_idf(docs)
    matrix, ids = build_doc_vectors(model, docs, idf)
    q = encode_text(model, query, idf)
    if q is None or len(ids) == 0:
        return []
    sims = matrix @ q
    order = np.argsort(-sims)[:top_k]
    return [(ids[i], float(sims[i])) for i in order]
