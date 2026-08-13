"""Multilingual tokenizer tests — Tier 1 (Latin+diacritics + Cyrillic).

These verify the Unicode regex change. The v0.1 ASCII regex would mangle
or drop every token in this file.
"""
from beaglemem.corpus import tokenize


# --- Spanish (487M speakers) ---

def test_tokenize_spanish_diacritics():
    """Spanish accents (á, é, í, ó, ú, ñ, ü) must stay attached to words."""
    assert tokenize("El café está en la fábrica, ¿verdad?") == [
        "el", "café", "está", "en", "la", "fábrica", "verdad"
    ]


def test_tokenize_spanish_ñ():
    """ñ is a distinct letter in Spanish, not n + tilde."""
    assert tokenize("El niño pequeño quiere un piñata") == [
        "el", "niño", "pequeño", "quiere", "un", "piñata"
    ]


# --- Portuguese (252M speakers) ---

def test_tokenize_portuguese_diacritics():
    """Portuguese nasal vowels (ã, õ) and accents (á, ç) preserved."""
    assert tokenize("A reunião foi cancelada, não há correlação") == [
        "reunião", "foi", "cancelada", "não", "há", "correlação"
    ]


def test_tokenize_portuguese_cedilla():
    assert tokenize("O coração da cidade é antigo") == [
        "coração", "da", "cidade", "antigo"
    ]


# --- Russian (133M speakers) ---

def test_tokenize_russian_cyrillic():
    """Russian words in Cyrillic script must tokenize as whole words."""
    assert tokenize("Митинг был перенесён на завтра") == [
        "митинг", "был", "перенесён", "на", "завтра"
    ]


def test_tokenize_mixed_russian_english():
    """Code-switching: Russian + English in one sentence."""
    result = tokenize("Нам нужен deploy на staging environment")
    assert "deploy" in result
    assert "staging" in result
    assert "environment" in result
    assert "нам" in result
    assert "нужен" in result


# --- Vietnamese (86M speakers) ---

def test_tokenize_vietnamese_tone_marks():
    """Vietnamese has 6 tones + extensive diacritics. All must survive."""
    assert tokenize("Họp nhóm dự án vào ngày mai nhé") == [
        "họp", "nhóm", "dự", "án", "vào", "ngày", "mai", "nhé"
    ]


# --- Indonesian (deployment-critical) ---

def test_tokenize_indonesian():
    """Indonesian is ASCII-Latin — already worked in v0.1, verify regression."""
    assert tokenize("Saya pergi ke kantor dengan naik bus") == [
        "saya", "pergi", "ke", "kantor", "dengan", "naik", "bus"
    ]


# --- NFKC normalization ---

def test_tokenize_nfkc_fullwidth_to_halfwidth():
    """NFKC normalizes fullwidth forms (ＡＢＣ) to ASCII (abc)."""
    result = tokenize("ＡＰＩサーバー")
    # NFKC: ＡＰＩ → api
    assert "api" in result


def test_tokenize_ascii_still_works():
    """The v0.1 ASCII behavior must be unchanged (superset regression)."""
    assert tokenize("The quick brown fox jumps over the lazy dog") == [
        "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"
    ]
    assert tokenize("Vitamin C 500 mg twice daily") == [
        "vitamin", "<NUM>", "mg", "twice", "daily"
    ]
