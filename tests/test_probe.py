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
