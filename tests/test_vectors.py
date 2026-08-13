import numpy as np
from beaglemem.vectors import env_vector, cosine


def test_env_vector_deterministic():
    assert np.array_equal(env_vector("megacorp", 256), env_vector("megacorp", 256))


def test_env_vector_unit_norm():
    assert abs(np.linalg.norm(env_vector("megacorp", 256)) - 1.0) < 1e-5


def test_env_vector_orthogonal_words():
    a = env_vector("megacorp", 256)
    b = env_vector("weather", 256)
    assert abs(cosine(a, b)) < 0.15


from beaglemem.vectors import BeagleModel
from tests.fixtures import SYNTHETIC_SENTENCES


def test_semantic_bridge_via_shared_neighbors():
    """'termination' and 'retrenched' never co-occur directly, but share
    neighbors (contract, signed, hr). Memory vectors must converge."""
    model = BeagleModel(dim=512, window=2)
    for sentence in SYNTHETIC_SENTENCES * 20:
        model.add_sentence(sentence)
    assert model.word_cosine("termination", "retrenched") > 0.3
    assert model.word_cosine("termination", "weather") < 0.3
    assert model.word_cosine("termination", "termination") > 0.99


def test_mem_renormalized_after_sentence():
    # min_count=1: this test checks normalization, not pruning. Default
    # min_count=2 would make mem_of("alpha") return None after one sentence
    # (alpha appears once) and the norm assertion would hit NoneType.
    model = BeagleModel(dim=256, window=2, min_count=1)
    model.add_sentence(["alpha", "beta", "gamma"])
    assert abs(np.linalg.norm(model.mem_of("alpha")) - 1.0) < 1e-4


def test_save_load_roundtrip(tmp_path):
    model = BeagleModel(dim=256, window=2)
    for s in SYNTHETIC_SENTENCES:
        model.add_sentence(s)
    model.save(str(tmp_path))
    loaded = BeagleModel.load(str(tmp_path))
    assert loaded.vocab == model.vocab
    assert np.allclose(loaded._mem[: loaded.size], model._mem[: model.size])


def test_deterministic_rebuild():
    m1, m2 = BeagleModel(dim=256, window=2), BeagleModel(dim=256, window=2)
    for s in SYNTHETIC_SENTENCES:
        m1.add_sentence(s)
        m2.add_sentence(s)
    assert np.allclose(m1._mem[: m1.size], m2._mem[: m2.size])


def test_min_count_prunes_rare_words():
    """Words appearing fewer than min_count times should return None from mem_of."""
    model = BeagleModel(dim=256, window=2, min_count=3)
    model.add_sentence(["common", "common", "common"])
    model.add_sentence(["rare", "common", "common"])
    assert model.mem_of("common") is not None  # appears 6x
    assert model.mem_of("rare") is None         # appears 1x, below min_count


def test_incremental_equals_batch():
    sentences = SYNTHETIC_SENTENCES * 5
    batch = BeagleModel(dim=256, window=2)
    for s in sentences:
        batch.add_sentence(s)
    inc = BeagleModel(dim=256, window=2)
    half = len(sentences) // 2
    for s in sentences[:half]:
        inc.add_sentence(s)
    for s in sentences[half:]:
        inc.add_sentence(s)
    assert np.allclose(batch._mem[: batch.size], inc._mem[: inc.size])


# --- Phase 0.5: storage-time min_count pruning (v0.2 grill Q4 fix) ---

def test_min_count_prunes_at_storage_time():
    """A word below min_count must NOT occupy a vocab slot or mem vector."""
    m = BeagleModel(dim=64, window=2, min_count=2)
    m.add_sentence(["rare", "common", "common", "common"])
    m.add_sentence(["common", "common", "common", "common"])
    # "rare" appeared once (< min_count=2) → must not be allocated
    assert "rare" not in m.index
    assert m.size == 1  # only "common"
    # "common" is retrievable
    assert m.mem_of("common") is not None


def test_min_count_word_allocated_on_threshold_cross():
    """A word at count 1 is pending; crossing to 2 allocates it."""
    m = BeagleModel(dim=64, window=2, min_count=2)
    m.add_sentence(["pending", "common", "common", "common"])
    assert "pending" not in m.index  # count 1, not yet allocated
    m.add_sentence(["pending", "common", "common", "common"])
    assert "pending" in m.index      # count 2, now allocated
    assert m.mem_of("pending") is not None
