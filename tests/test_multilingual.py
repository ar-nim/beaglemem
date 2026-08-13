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


# --- CJK bigram tests ---

def test_tokenize_mandarin_bigrams():
    """Mandarin Chinese: char-bigrams. 雅加达 → [雅加, 加达]."""
    result = tokenize("我在雅加达坐公交车去上班")
    assert "雅加" in result  # bigram from 雅加达
    assert "加达" in result
    assert "我在" in result  # bigram from start
    assert "上班" in result  # bigram at end


def test_tokenize_japanese_mixed_kanji_kana():
    """Japanese: Kanji + Hiragana + Katakana all produce bigrams."""
    result = tokenize("明日の会議は重要です")
    assert "明日" in result   # kanji bigram
    assert "の会" in result   # hiragana-kanji boundary bigram


def test_tokenize_cantonese():
    """Cantonese (Yue Chinese): same CJK bigram extraction."""
    result = tokenize("我哋聽日去飲茶")
    assert "我哋" in result
    assert "聽日" in result
    assert "飲茶" in result


def test_tokenize_mixed_chinese_english():
    """Code-switching: Chinese + English in one sentence."""
    result = tokenize("我要 deploy 这个 project 到 production")
    assert "deploy" in result
    assert "project" in result
    assert "production" in result
    assert "我要" in result  # CJK bigram
    assert "这个" in result


def test_tokenize_cjk_single_char_dropped():
    """A single CJK char alone produces no bigram (min 2 chars needed)."""
    result = tokenize("好")
    assert result == []  # no bigram possible from single char


def test_tokenize_cjk_two_chars_one_bigram():
    """Two CJK chars produce exactly one bigram."""
    result = tokenize("会议")
    assert result == ["会议"]


# --- Real-data regression tests (bugs found 2026-08-12 against real CJK data) ---

def test_tokenize_no_whole_cjk_phrase_as_word():
    """BUG 1 regression: WORD_RE must NOT swallow an unspaced CJK phrase.

    Before the fix, the word regex had no CJK-space awareness, so it grabbed
    '来週予定している面接' (10 chars) as ONE garbage token, AND the bigram
    extractor produced the correct bigrams — double-counting.
    """
    result = tokenize("来週予定している面接")
    # The whole 10-char phrase must NOT be a single token:
    assert "来週予定している面接" not in result
    # But the bigrams ARE present:
    assert "来週" in result
    assert "面接" in result


def test_tokenize_korean_not_bigrammed():
    """BUG 2 regression: Hangul (Korean) is space-separated, never bigrammed.

    Korean words are separated by spaces (unlike CN/JP), so they must tokenize
    via WORD_RE as whole words. Bigramming shreds them: 안녕하세요 → 녕하/하세/세요.
    """
    result = tokenize("안녕하세요 저는 학생입니다")
    assert "안녕하세요" in result      # whole word via WORD_RE
    assert "저는" in result
    assert "학생입니다" in result
    # No Hangul syllable fragments:
    assert "녕하" not in result
    assert "하세" not in result
    assert "세요" not in result


def test_tokenize_korean_chinese_mixed():
    """Korean (space-sep) + Chinese (unspaced) in one sentence handled correctly."""
    result = tokenize("저는 Jakarta에 살아요 我在上班")
    assert "저는" in result       # Korean via WORD_RE
    assert "我在" in result       # Chinese via bigram
    assert "上班" in result       # Chinese via bigram
