"""Tokenization, sentence splitting, and adapter dispatch."""

import re

# "=" is part of the token charset so is_valid_token's base64-padding check
# (w.endswith("=")) is actually reachable. Without it, WORD_RE drops the "="
# before the check runs and base64 strings like "dGVzdGluZw==" survive as
# "dgvzdgluzw". (Plan Task 1.2 bug — found by test_tokenize_kills_base64_and_hashes.)
WORD_RE = re.compile(r"[a-z0-9][a-z0-9'\-=]*")
SENT_SPLIT_RE = re.compile(r"[.!?\n]+")
URL_RE = re.compile(r"https?://\S+")  # strip full URLs before tokenizing
NUM_RE = re.compile(r"^[0-9]+$")  # pure numeric tokens (digits only, no letters)
MIN_SENTENCE_WORDS = 3
MAX_TOKEN_LEN = 25  # real words cap at ~20; longer = encoded data
MIN_TOKEN_LEN = 2   # single chars are fragments with no standalone meaning


def is_valid_token(w: str) -> bool:
    """Filter pipeline for machine-generated noise. Keeps short alphanumeric
    tokens (6x, 250mg, k3s, pq9) because those are domain vocabulary.
    Kills base64, hex dumps, hashes, and single-char fragments."""
    # 1. Length: real words are 2-25 chars (1 char = noise fragment)
    if not (MIN_TOKEN_LEN <= len(w) <= MAX_TOKEN_LEN):
        return False
    # 2. Base64 padding: never in real words
    if w.endswith("=") or w.endswith("=="):
        return False
    # 3. Base64 symbols attached to words
    if any(c in w for c in "+/"):
        return False
    # 4. Long no-vowel strings (20+ chars, no vowels = hash/encoded)
    if len(w) >= 20 and not any(c in "aeiou" for c in w):
        return False
    return True


def tokenize(sentence: str) -> list[str]:
    """Strip URLs, tokenize, replace pure numbers with <NUM>, filter noise.
    - Pure numbers ("2026", "250") → "<NUM>" placeholder (SO standard)
    - Compounds ("250mg", "6x", "k3s") → kept intact (domain vocabulary)
    - Single chars ("d", "s", "3") → killed by min-length-2
    - URLs → stripped entirely before tokenizing
    """
    clean = URL_RE.sub(" ", sentence.lower())
    raw = [w for w in WORD_RE.findall(clean) if is_valid_token(w)]
    return ["<NUM>" if NUM_RE.match(w) else w for w in raw]


def split_sentences(text: str) -> list[str]:
    return SENT_SPLIT_RE.split(text)


def iter_sentences(path: str, format: str = "plain"):
    """Dispatch to the adapter for `format`. Yields tokenized sentences."""
    if format == "plain":
        from beaglemem.adapters.plain_text import iter_sentences as impl
    elif format == "chat-jsonl":
        from beaglemem.adapters.chat_jsonl import iter_sentences as impl
    elif format == "state_db":
        from beaglemem.adapters.state_db import iter_sentences as impl
    else:
        raise ValueError(f"unknown corpus format: {format!r}")
    yield from impl(path)
