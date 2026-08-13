"""BEAGLE vectors: environmental (static identity) + memory (accumulated semantics).

Reference: Jones & Mewhort (2007); Rutledge-Taylor et al. (2014).
Co-occurrence only (no order encoding) — plan decision #3.
"""
import hashlib
import json
import os
import struct

import numpy as np


def env_vector(word: str, dim: int) -> np.ndarray:
    """Deterministic unit vector via SHA-256 counter blocks. Same word → same
    vector on any machine, any run, any process. No RNG state — this is what
    makes incremental ingest and multi-machine builds safe."""
    vals: list[int] = []
    i = 0
    while len(vals) < dim:
        digest = hashlib.sha256(f"{word}:{i}".encode()).digest()
        vals.extend(struct.unpack("<16H", digest))
        i += 1
    v = np.array(vals[:dim], dtype=np.float64) / 32768.0 - 1.0  # uniform [-1, 1)
    v = v / np.linalg.norm(v)
    return v.astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class BeagleModel:
    """Co-occurrence accumulator. mem[word] starts as env[word] (paper), then
    absorbs env vectors of every neighbor within ±window. After each sentence,
    touched memory vectors are renormalized to length 1 (paper: memory vectors
    have Euclidean length 1).

    The backing matrix grows in power-of-two chunks — do NOT reallocate per
    word (that is O(vocab²·dim) and would be unusable at 50K words)."""

    def __init__(self, dim: int = 2048, window: int = 3, capacity: int = 1024,
                 min_count: int = 2):
        self.dim = dim
        self.window = window
        self.min_count = min_count  # drop words appearing < min_count times
        self.vocab: list[str] = []
        self.index: dict[str, int] = {}
        self._env_cache: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {}  # frequency tracking for min_count
        self.size = 0
        self._capacity = capacity
        self._mem = np.zeros((capacity, dim), dtype=np.float32)

    def env_of(self, word: str) -> np.ndarray:
        v = self._env_cache.get(word)
        if v is None:
            v = env_vector(word, self.dim)
            self._env_cache[word] = v
        return v

    def _ensure_word(self, word: str) -> int:
        idx = self.index.get(word)
        if idx is None:
            if self.size == self._capacity:
                new_cap = self._capacity * 2
                grown = np.zeros((new_cap, self.dim), dtype=np.float32)
                grown[: self.size] = self._mem[: self.size]
                self._mem = grown
                self._capacity = new_cap
            idx = self.size
            self.vocab.append(word)
            self.index[word] = idx
            self._mem[idx] = self.env_of(word)  # mem init = env (paper)
            self.size += 1
        return idx

    def add_sentence(self, words: list[str]) -> None:
        # Track frequency for min_count pruning FIRST. Words below min_count
        # are never allocated — they get a count but no vocab slot or vector.
        # Storage-time pruning (v0.2): a word at count 1 today becomes count 2
        # tomorrow, crossing the threshold in a later add_sentence and getting
        # allocated then. Incremental update still equals batch because a
        # sub-threshold word contributes zero co-occurrence by definition.
        for w in words:
            self._counts[w] = self._counts.get(w, 0) + 1

        # Only words that have REACHED min_count get allocated + co-occur.
        eligible = [w for w in words if self._counts[w] >= self.min_count]
        if not eligible:
            return
        idxs = [self._ensure_word(w) for w in eligible]
        n = len(idxs)
        touched: set[int] = set()
        for i, wi in enumerate(idxs):
            lo = max(0, i - self.window)
            hi = min(n, i + self.window + 1)
            acc = np.zeros(self.dim, dtype=np.float32)
            for j in range(lo, hi):
                if j != i:
                    acc += self.env_of(eligible[j])
            self._mem[wi] += acc
            touched.add(wi)
        for wi in touched:
            norm = np.linalg.norm(self._mem[wi])
            if norm > 0:
                self._mem[wi] /= norm

    def mem_of(self, word: str):
        """Returns the memory vector for word, or None if:
        - word is not in vocabulary, OR
        - word appears fewer than min_count times (frequency pruning)
        """
        idx = self.index.get(word)
        if idx is None:
            return None
        if self._counts.get(word, 0) < self.min_count:
            return None
        return self._mem[idx]

    def word_cosine(self, a: str, b: str) -> float:
        ma, mb = self.mem_of(a), self.mem_of(b)
        if ma is None or mb is None:
            return 0.0
        return cosine(ma, mb)

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "beagle_mem.npy"), self._mem[: self.size])
        with open(os.path.join(out_dir, "beagle_vocab.json"), "w") as fh:
            json.dump({
                "dim": self.dim,
                "window": self.window,
                "min_count": self.min_count,
                "vocab": self.vocab,
                "counts": self._counts,
                "tokenizer_fingerprint": getattr(self, "tokenizer_fingerprint", None),
            }, fh)

    @classmethod
    def load(cls, in_dir: str) -> "BeagleModel":
        with open(os.path.join(in_dir, "beagle_vocab.json")) as fh:
            meta = json.load(fh)
        model = cls(
            dim=meta["dim"], window=meta["window"],
            min_count=meta.get("min_count", 2),
            capacity=max(1024, len(meta["vocab"])),
        )
        model.vocab = meta["vocab"]
        model.index = {w: i for i, w in enumerate(model.vocab)}
        model._counts = meta.get("counts", {})
        model.tokenizer_fingerprint = meta.get("tokenizer_fingerprint")
        stored = np.load(os.path.join(in_dir, "beagle_mem.npy"))
        model._mem = np.zeros((model._capacity, model.dim), dtype=np.float32)
        model._mem[: stored.shape[0]] = stored
        model.size = stored.shape[0]
        return model
