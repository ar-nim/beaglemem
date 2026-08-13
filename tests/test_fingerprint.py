"""Fingerprint tests — Phase 0 of v0.2 multilingual plan.

Two fingerprints with different rebuild scopes:
- TOKENIZER_FINGERPRINT (regex, stemmer, dim, window) → guards corpus model
- ENCODER_VERSION (code constant) → guards fact cache

IDF is NOT in the tokenizer fingerprint (it's an encoder concern) and there is
no use_idf flag anywhere.
"""
import json

import numpy as np

from beaglemem.fingerprint import (
    ENCODER_VERSION,
    set_tokenizer_fingerprint,
    tokenizer_fingerprint,
)
from beaglemem.vectors import BeagleModel


def test_fingerprint_detects_tokenizer_change():
    """Different tokenizer regex → different fingerprint."""
    fp_ascii = tokenizer_fingerprint(regex=r"[a-z0-9][a-z0-9'\-=]*", stemmer=None)
    fp_unicode = tokenizer_fingerprint(regex=r"[^\W_][^\W_'\-=]*", stemmer=None)
    assert fp_ascii != fp_unicode


def test_fingerprint_ignores_idf():
    """IDF is NOT in the tokenizer fingerprint — it's an encoder concern."""
    # There is no idf parameter at all — the signature can't include it.
    assert "idf" not in tokenizer_fingerprint.__code__.co_varnames


def test_fingerprint_persisted_in_vocab():
    """The fingerprint is written to beagle_vocab.json on save."""
    model = BeagleModel(dim=64, window=2, min_count=2)
    model.add_sentence(["hello", "world"])
    set_tokenizer_fingerprint(model, regex=r"[a-z0-9]+", stemmer=None)
    model.save("/tmp/beagle_fp_test")
    with open("/tmp/beagle_fp_test/beagle_vocab.json") as fh:
        meta = json.load(fh)
    assert "tokenizer_fingerprint" in meta
    # set_tokenizer_fingerprint embeds the model's dim/window (64/2), so the
    # expected value must use the same model shape, not the function defaults.
    assert meta["tokenizer_fingerprint"] == tokenizer_fingerprint(
        regex=r"[a-z0-9]+", stemmer=None, dim=model.dim, window=model.window
    )


def test_fingerprint_mismatch_detected_on_load():
    """Loading a model with a different fingerprint raises a clear signal."""
    model = BeagleModel(dim=64, window=2, min_count=2)
    model.add_sentence(["hello", "world"])
    set_tokenizer_fingerprint(model, regex="OLD", stemmer=None)
    model.save("/tmp/beagle_fp_test2")
    loaded = BeagleModel.load("/tmp/beagle_fp_test2")
    assert loaded.tokenizer_fingerprint != tokenizer_fingerprint(
        regex="NEW", stemmer=None
    )


def test_encoder_version_is_code_constant():
    """ENCODER_VERSION is a string constant, not user config."""
    assert isinstance(ENCODER_VERSION, str)
    assert ENCODER_VERSION == "idf-v1"
