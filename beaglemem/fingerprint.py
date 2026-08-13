"""Two fingerprints with different rebuild scopes.

TOKENIZER_FINGERPRINT (guards corpus model beagle_mem.npy): hash of regex,
stemmer identity, dim, window. Change → full corpus rebuild.

ENCODER_VERSION (guards fact cache fact_vectors.npy): code constant. Change
→ re-encode facts only. IDF is always-on, so this changes only when the IDF
formula changes in a point release, never via user config.
"""
import hashlib
import json

ENCODER_VERSION = "idf-v1"


def tokenizer_fingerprint(regex: str, stemmer=None, dim: int = 2048,
                          window: int = 3) -> str:
    """Deterministic hash of the tokenizer + model-shape configuration.

    A change to ANY of these changes how the same raw text tokenizes (regex,
    stemmer) or how co-occurrence accumulates (dim, window), so all of them
    invalidate the corpus model.

    IDF is deliberately NOT in this hash: it is an encoder concern (weights at
    encode time), not a tokenizer concern. Changing IDF only invalidates the
    fact cache, tracked separately by ENCODER_VERSION.
    """
    stemmer_id = getattr(stemmer, "__name__", str(stemmer))
    payload = json.dumps(
        {"regex": regex, "stemmer": stemmer_id, "dim": int(dim),
         "window": int(window)},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def set_tokenizer_fingerprint(model, regex: str, stemmer=None) -> None:
    """Attach the fingerprint to a BeagleModel before save()."""
    model.tokenizer_fingerprint = tokenizer_fingerprint(
        regex, stemmer, dim=model.dim, window=model.window
    )
