import numpy as np
from beaglemem.probe import encode_text, build_doc_vectors, probe, STOPWORDS
from beaglemem.store import MemoryStore
from beaglemem.vectors import BeagleModel
from tests.fixtures import SYNTHETIC_SENTENCES


def _trained_model(dim=512):
    m = BeagleModel(dim=dim, window=2)
    for s in SYNTHETIC_SENTENCES * 20:
        m.add_sentence(s)
    return m


def test_stopwords_present():
    assert "the" in STOPWORDS and "termination" not in STOPWORDS


def test_encode_text_known_words():
    v = encode_text(_trained_model(), "termination contract", {})
    assert v is not None and abs(np.linalg.norm(v) - 1.0) < 1e-4


def test_encode_text_unknown_or_stopword_only():
    model = _trained_model()
    # Unknown words: not in vocab → mem_of None → known=0 → None
    assert encode_text(model, "zzzqqq unknownword", {}) is None
    # Stopword-only: not in the synthetic corpus vocab → still None
    assert encode_text(model, "the and of", {}) is None


def test_probe_surfaces_synonym_doc():
    model = _trained_model()
    store = MemoryStore([
        {"id": 1, "text": "Beta division termination agreement signed with hr"},
        {"id": 2, "text": "Weather forecast sunny weekend ahead"},
    ])
    results = probe(model, "retrenched", store)
    assert results[0][0] == 1
    assert results[0][1] > results[1][1]


def test_probe_exact_match_still_works():
    model = _trained_model()
    store = MemoryStore([
        {"id": 1, "text": "Beta division termination agreement signed with hr"},
        {"id": 2, "text": "Weather forecast sunny weekend ahead"},
    ])
    assert probe(model, "termination", store)[0][0] == 1


def test_probe_empty_query():
    store = MemoryStore([{"id": 1, "text": "anything at all here"}])
    assert probe(_trained_model(), "", store) == []


def test_build_doc_vectors_never_skips():
    """v0.3 never-skip contract: EVERY fact gets a row. Unencodable facts
    (no vocab overlap) get a ZERO vector — rows must equal len(docs) so the
    row↔fact mapping is derivable from the DB alone (fact_ids not stored)."""
    model = BeagleModel(dim=64, window=2, min_count=1)
    model.add_sentence(["alpha", "beta", "gamma"])

    docs = [
        {"id": 1, "text": "alpha beta gamma"},   # encodable
        {"id": 2, "text": "zzzqqq"},              # no vocab overlap → unencodable
        {"id": 3, "text": "alpha zzzqqq"},        # partial overlap → encodable
    ]
    from beaglemem.idf import build_idf
    matrix, ids = build_doc_vectors(model, docs, build_idf(docs))
    assert ids == [1, 2, 3]           # never skips
    assert matrix.shape == (3, 64)    # rows == len(docs)
    assert (matrix[1] == 0).all()     # unencodable → zero vector
    assert not (matrix[0] == 0).all()
