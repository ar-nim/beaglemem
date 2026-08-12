from beaglemem.corpus import tokenize, split_sentences


def test_tokenize_lowercase_and_charset():
    assert tokenize("Placebo 250mg, twice-daily!") == ["placebo", "250mg", "twice-daily"]


def test_tokenize_strips_punctuation():
    assert tokenize("  Hello, WORLD... ") == ["hello", "world"]


def test_tokenize_strips_urls():
    # SPEC: URL_RE (https?://\S+) removes the entire URL before tokenizing.
    # The "jnx" path segment does NOT survive — the test name says strips_urls.
    assert tokenize("See https://en.wikipedia.org/wiki/JNX for details") == ["see", "for", "details"]


def test_tokenize_keeps_short_alphanumeric():
    # "802" is a pure number → <NUM> per SPEC (same rule as test_tokenize_pure_numbers_become_placeholder).
    # "1q" survives as a short alphanumeric compound; the token charset has no "." so 802.1q splits.
    assert tokenize("6X bus k3s 802.1q pq9") == ["6x", "bus", "k3s", "<NUM>", "1q", "pq9"]


def test_tokenize_kills_base64_and_hashes():
    assert tokenize("key dGVzdGluZw== abc123def456ghi789jkl012mno") == ["key"]


def test_tokenize_length_cap():
    long_token = "a" * 30
    assert tokenize(long_token) == []


def test_tokenize_pure_numbers_become_placeholder():
    assert tokenize("dose 250 mg twice daily year 2026") == [
        "dose", "<NUM>", "mg", "twice", "daily", "year", "<NUM>"
    ]


def test_tokenize_compounds_not_scrubbed():
    assert tokenize("250mg 6x 3L k3s") == ["250mg", "6x", "3l", "k3s"]


def test_tokenize_single_chars_killed():
    assert tokenize("I saw a d and s") == ["saw", "and"]


def test_split_sentences():
    assert split_sentences("one two three. four five six!\nseven eight nine") == [
        "one two three", " four five six", "seven eight nine"
    ]
