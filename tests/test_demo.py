from examples.make_demo_corpus import make_demo_corpus
from beaglemem.corpus import iter_sentences
from beaglemem.store import MemoryStore
from beaglemem.probe import probe
from beaglemem.vectors import BeagleModel


def test_demo_bridges_form(tmp_path):
    corpus_dir, docs = make_demo_corpus(str(tmp_path))
    model = BeagleModel(dim=2048, window=3)
    for words in iter_sentences(corpus_dir, format="chat-jsonl"):
        model.add_sentence(words)

    # Planted bridge: 'let go' never appears in any document, but co-occurs
    # in the corpus with the same neighbors as 'severance'
    assert model.word_cosine("let", "severance") > 0.2
    assert model.word_cosine("let", "recipe") < 0.08

    # The money shot: probe with a word absent from the target document
    store = MemoryStore(docs)
    results = probe(model, "let go", store, top_k=3)
    assert results[0][0] == 7  # the severance document
