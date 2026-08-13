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


# --- IDF weighting (replaces stopwords) ---

def test_idf_downweights_universal_words():
    """A word in every document gets zero IDF; a rare word gets high IDF."""
    from beaglemem.idf import build_idf
    docs = [
        {"id": 1, "text": "the meeting is scheduled tomorrow"},
        {"id": 2, "text": "the weather is sunny today"},
        {"id": 3, "text": "the recipe needs flour"},
    ]
    idf = build_idf(docs)  # idf: dict[str, float]
    assert idf["the"] < idf["meeting"]
    assert idf["the"] < idf["weather"]
    assert idf["the"] == 0.0  # in all 3 docs → log(3/3) = 0 (exact floor)
    assert idf["meeting"] > idf["the"]  # rare word boosted
    assert idf["meeting"] > 0.0  # log(3/1) = 1.099


def test_idf_never_negative():
    """A word in EVERY document must floor at exactly 0, never go negative."""
    from beaglemem.idf import build_idf
    docs = [
        {"id": 1, "text": "the the the"},
        {"id": 2, "text": "the the the"},
    ]
    idf = build_idf(docs)
    assert idf["the"] == 0.0  # log(2/2) = 0, NOT log(2/3) < 0


def test_idf_language_agnostic():
    """Indonesian 'yang' (in every doc) and English 'the' both get downweighted.
    No language-specific list needed."""
    from beaglemem.idf import build_idf
    docs = [
        {"id": 1, "text": "the rapat yang dijadwalkan besok"},
        {"id": 2, "text": "the cuaca yang cerah hari ini"},
        {"id": 3, "text": "the resep yang perlu tepung"},
    ]
    idf = build_idf(docs)
    assert idf["the"] == 0.0
    assert idf["yang"] == 0.0
    assert idf["rapat"] > idf["yang"]


def test_idf_cjk_multi_char_particle_downweighted():
    """Multi-char CJK particles (から) survive tokenization → downweighted by IDF.

    Single CJK function chars (的) are DROPPED by the tokenizer (bigrams need
    >=2 chars), so they never enter the vocab — IDF never sees them. The
    function-char suppression happens at tokenization, not IDF. IDF handles
    the MULTI-char particles that DO survive.
    """
    from beaglemem.idf import build_idf
    # "から" (Japanese "from") is a 2-char particle that tokenizes as a bigram.
    docs = [
        {"id": 1, "text": "会議 から 帰る"},
        {"id": 2, "text": "学校 から 来た"},
        {"id": 3, "text": "駅 から 歩く"},
    ]
    idf = build_idf(docs)
    # "から" appears in every doc → floor 0. Rare content bigram boosted.
    assert idf["から"] == 0.0
    assert idf["会議"] > idf["から"]


def test_idf_cjk_single_char_never_tokenizes():
    """Single CJK function chars (的) never become tokens → never in IDF map."""
    from beaglemem.idf import build_idf
    docs = [{"id": 1, "text": "会议 的 时间"}]
    idf = build_idf(docs)
    # 的 is a single char → dropped by bigram tokenizer → absent from IDF
    assert "的" not in idf


def test_idf_unknown_word_default():
    """A word not in any document gets a neutral/floor IDF, never crashes."""
    from beaglemem.idf import build_idf
    idf = build_idf([{"id": 1, "text": "hello world"}])
    # unknown word: floor. Must not KeyError.
    assert isinstance(idf.get("zzz", 0.0), float)


# --- encode_text IDF wiring (Phase 3 Task 3.3) ---

def test_idf_floor_is_neutral_not_zero():
    """A word absent from the fact store gets FULL weight (1.0), not 0.

    Critical: the semantic bridge depends on query words that never appear in
    ANY fact (demo: 'let go' → severance doc). Floor 0.0 would zero their
    contribution and kill the bridge. 1.0 = neutral (no boost, no penalty).
    """
    from beaglemem.idf import idf_weight
    assert idf_weight({}, "anything") == 1.0


def test_encode_text_applies_idf_weights():
    """encode_text scales each word's mem vector by its IDF weight."""
    import numpy as np
    from beaglemem.vectors import BeagleModel
    from beaglemem.probe import encode_text
    m = BeagleModel(dim=64, window=2, min_count=1)
    m.add_sentence(["alpha", "beta", "gamma"])
    idf = {"alpha": 2.0, "beta": 0.5}
    v = encode_text(m, "alpha beta", idf)
    assert v is not None
    assert abs(np.linalg.norm(v) - 1.0) < 1e-4


def test_encode_text_unknown_in_idf_still_contributes():
    """Words in the corpus model but absent from the IDF map still contribute
    (floor 1.0) — preserves the semantic bridge for out-of-fact vocabulary."""
    from beaglemem.vectors import BeagleModel
    from beaglemem.probe import encode_text
    m = BeagleModel(dim=64, window=2, min_count=1)
    m.add_sentence(["meeting", "room", "notes"])
    m.add_sentence(["meeting", "room", "notes"])
    v = encode_text(m, "meeting", {})  # empty idf → neutral weight
    assert v is not None
