"""Semantic probe: encode text as a sum of memory vectors, rank documents by
cosine similarity."""
import numpy as np

from beaglemem.corpus import tokenize

# Function words co-occur with everything, so their mem vectors carry a shared
# centroid direction that inflates all similarities. Excluded from QUERY and
# DOCUMENT encoding (NOT from corpus learning). Standard distributional-IR
# practice, not ad-hoc filtering.
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


def encode_text(model, text: str):
    words = [w for w in tokenize(text) if w not in STOPWORDS]
    acc = np.zeros(model.dim, dtype=np.float32)
    known = 0
    for w in words:
        m = model.mem_of(w)
        if m is not None:
            acc += m
            known += 1
    if known == 0:
        return None
    norm = np.linalg.norm(acc)
    return acc / norm if norm > 0 else None


def build_doc_vectors(model, docs: list[dict]):
    """docs: [{'id': ..., 'text': str}] → (matrix (n, dim) float32, ids)."""
    rows, ids = [], []
    for d in docs:
        v = encode_text(model, d["text"])
        if v is not None:
            rows.append(v)
            ids.append(d["id"])
    if not rows:
        return np.zeros((0, model.dim), dtype=np.float32), []
    return np.stack(rows), ids


def probe(model, query: str, store, top_k: int = 10):
    """store: any DocumentStore (documents() → [{'id','text'}])."""
    docs = store.documents()
    matrix, ids = build_doc_vectors(model, docs)
    q = encode_text(model, query)
    if q is None or len(ids) == 0:
        return []
    sims = matrix @ q
    order = np.argsort(-sims)[:top_k]
    return [(ids[i], float(sims[i])) for i in order]
