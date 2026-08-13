"""Tokenization, sentence splitting, and adapter dispatch.

v0.2 multilingual tokenizer:
- Unicode word regex [^\\W_][^\\W_'\\-=]* — superset of v0.1 [a-z0-9]
- NFKC normalization before tokenizing (fullwidth → halfwidth)
- CJK char-bigrams for Chinese/Japanese (unspaced scripts)
- Korean (Hangul) is space-separated → WORD_RE path, never bigrammed
- Bigram-script chars (Hanzi/Kana) are STRIPPED before WORD_RE so the word
  regex cannot swallow an unspaced CJK phrase as a single token
"""

import re
import unicodedata

# "=" is part of the token charset so is_valid_token's base64-padding check
# (w.endswith("=")) is actually reachable. Without it, WORD_RE drops the "="
# before the check runs and base64 strings like "dGVzdGluZw==" survive as
# "dgvzdgluzw". (Plan Task 1.2 bug — found by test_tokenize_kills_base64_and_hashes.)
# v0.2: Unicode word regex — matches any Unicode letter/digit (not underscore).
# Superset of v0.1 [a-z0-9]: all ASCII tokens survive; diacritics, Cyrillic,
# Vietnamese tone marks, Hangul now tokenize as whole words.
#
# First char: [^\W_] = Unicode letter/digit, not underscore.
# Continuation: [\w'\-=] = Unicode word char (incl. underscore) + quote +
# hyphen + equals. POSITIVE class — v0.1's [a-z0-9'\-=] included - and =;
# a negated class ([^\W_'\-=]) would EXCLUDE them, splitting "twice-daily"
# and stripping base64 padding (which the endswith("=") check needs to see).
WORD_RE = re.compile(r"[^\W_][\w'\-=]*")
SENT_SPLIT_RE = re.compile(r"[.!?\n]+")
URL_RE = re.compile(r"https?://\S+")  # strip full URLs before tokenizing
NUM_RE = re.compile(r"^[0-9]+$")  # pure numeric tokens (digits only, no letters)
MIN_SENTENCE_WORDS = 3
MAX_TOKEN_LEN = 25  # real words cap at ~20; longer = encoded data
MIN_TOKEN_LEN = 2   # single chars are fragments with no standalone meaning


def is_valid_token(w: str) -> bool:
    """Filter pipeline for machine-generated noise. Keeps short alphanumeric
    tokens (6x, 250mg, k3s, pq9) because those are domain vocabulary.
    Kills base64, hex dumps, hashes, and single-char fragments.

    v0.2: Script-aware. The no-vowel hash detector only applies to ASCII
    tokens — hashes/base64/hex are always ASCII. Non-ASCII tokens (CJK,
    Cyrillic, accented Latin) are real vocabulary and skip the hash check.
    """
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
    #    ONLY for ASCII tokens — CJK/Cyrillic/accented words are never hashes.
    if w.isascii() and len(w) >= 20 and not any(c in "aeiou" for c in w):
        return False
    return True


def _extract_cjk_bigrams(text: str) -> list[str]:
    """Extract character bigrams from UNSPACED CJK runs in text.

    Chinese (Hanzi) and Japanese (Hiragana/Katakana) have no spaces between
    words, so word-boundary regex can't segment them. Character bigrams
    (sliding 2-char window) are the standard zero-dependency IR technique.
    雅加达 → [雅加, 加达]. Single trailing chars are dropped (need bigram = 2).

    CRITICAL (2026-08-12 real-data finding): Hangul (Korean) is NOT in these
    ranges. Korean IS space-separated (unlike CN/JP), so it is tokenized by
    WORD_RE like any alphabetic script. Bigramming Hangul shreds real words
    into meaningless syllable pairs (안녕하세요 → 녕하/하세/세요).

    Bigram ranges (UNSPACED scripts only):
    CJK Unified Ideographs (4E00-9FFF), CJK Extension A (3400-4DBF),
    Hiragana (3040-309F), Katakana (30A0-30FF), CJK Compatibility
    Ideographs (F900-FAFF). NO Hangul (AC00-D7AF).
    """
    BIGRAM_RANGES = [
        (0x3400, 0x4DBF),   # CJK Extension A (Hanzi)
        (0x4E00, 0x9FFF),   # CJK Unified Ideographs (Hanzi)
        (0xF900, 0xFAFF),   # CJK Compatibility Ideographs (Hanzi)
        (0x3040, 0x309F),   # Hiragana (Japanese)
        (0x30A0, 0x30FF),   # Katakana (Japanese)
    ]

    def is_bigram_script(ch: str) -> bool:
        cp = ord(ch)
        return any(lo <= cp <= hi for lo, hi in BIGRAM_RANGES)

    bigrams: list[str] = []
    # Scan text for bigram-script runs, extract bigrams from each run
    run: list[str] = []
    for ch in text:
        if is_bigram_script(ch):
            run.append(ch)
        else:
            if len(run) >= 2:
                for i in range(len(run) - 1):
                    bigrams.append(run[i] + run[i + 1])
            run = []
    # Handle trailing run
    if len(run) >= 2:
        for i in range(len(run) - 1):
            bigrams.append(run[i] + run[i + 1])
    return bigrams


# Regex matching any bigram-script char (Hanzi + Kana). Used to STRIP these
# chars before WORD_RE so the Unicode word regex does not swallow an entire
# unspaced CJK phrase as a single "word" (real-data bug 2026-08-12).
BIGRAM_SCRIPT_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u309f\u30a0-\u30ff]"
)


def tokenize(sentence: str) -> list[str]:
    """NFKC-normalize, strip URLs, tokenize, replace pure numbers with <NUM>,
    filter noise, extract CJK bigrams.

    Pipeline:
    1. NFKC normalization (fullwidth → halfwidth, compatibility decomposition)
    2. URL stripping (full URLs removed before tokenizing)
    3. Lowercase
    4. Strip bigram-script chars (Hanzi/Kana) → so WORD_RE sees only
       space-separated scripts (Latin, Cyrillic, Hangul, etc.)
    5. CJK bigram extraction from the UNSTRIPPED text (Hanzi/Kana only)
    6. WORD_RE tokenization (Unicode letters/digits — space-separated scripts)
    7. Noise filtering (is_valid_token)
    8. Pure number → <NUM> replacement
    9. Append bigrams

    - Diacritics (café, niño, coração) → preserved (Unicode regex)
    - Cyrillic (митинг, завтра) → preserved (Unicode regex)
    - Chinese/Japanese (会议, 日本語) → char-bigrams ONLY (never whole phrases)
    - Korean (안녕하세요) → WORD_RE (space-separated, never bigrammed)
    - Compounds (250mg, 6x) → kept intact
    - Single chars → killed by min-length-2
    - URLs → stripped entirely before tokenizing
    """
    # 1. NFKC: Japanese fullwidth ABC → ASCII, compatibility forms normalized
    normalized = unicodedata.normalize("NFKC", sentence)
    # 2. Strip URLs
    clean = normalized.lower()
    clean = URL_RE.sub(" ", clean)
    # 3. Extract bigrams from the ORIGINAL (Hanzi/Kana still present)
    bigrams = _extract_cjk_bigrams(clean)
    # 4. Strip bigram-script chars so WORD_RE can't swallow whole CJK phrases.
    #    This is the fix for the real-data bug: [^\W_] matches Hanzi/Kana with
    #    no space boundary, so it grabbed '来週予定している面接' as ONE token.
    no_cjk = BIGRAM_SCRIPT_RE.sub(" ", clean)
    # 5. WORD_RE tokenization on space-separated scripts only
    raw = [w for w in WORD_RE.findall(no_cjk) if is_valid_token(w)]
    # 6. Pure number replacement
    tokens = ["<NUM>" if NUM_RE.match(w) else w for w in raw]
    # 7. Append CJK bigrams
    tokens.extend(bigrams)
    return tokens


def split_sentences(text: str) -> list[str]:
    return SENT_SPLIT_RE.split(text)


def iter_sentences(path: str, format: str = "plain"):
    """Dispatch to the adapter for `format`. Yields tokenized sentences."""
    if format == "plain":
        from .adapters.plain_text import iter_sentences as impl
    elif format == "chat-jsonl":
        from .adapters.chat_jsonl import iter_sentences as impl
    elif format == "state_db":
        from .adapters.state_db import iter_sentences as impl
    else:
        raise ValueError(f"unknown corpus format: {format!r}")
    yield from impl(path)
