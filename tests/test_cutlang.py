import json

import cutlang
import store
from conftest import run_cli

HE_VOCAB = set("אבגדהוזחטיכלמנסעפצקרשתךםןףץ ")


def test_hebrew_digit_to_word():
    assert cutlang.norm_for_align("4", "he", HE_VOCAB) == "ארבע"


def test_niqqud_stripped():
    assert cutlang.norm_for_align("שָׁלוֹם", "he", HE_VOCAB) == "שלום"
    assert cutlang.norm_for_compare("שָׁלוֹם,", "he") == "שלום"


def test_builtin_latin_tech_word():
    assert cutlang.norm_for_align("AI", "he", HE_VOCAB) == "איי איי"
    assert cutlang.norm_for_compare("AI", "he") == "איי איי"


def test_english_accent_and_punctuation():
    assert cutlang.norm_for_align("Café,", "en") == "cafe"
    assert cutlang.norm_for_compare("Don't", "en") == "don't"


def test_english_digits_to_words():
    assert cutlang.norm_for_align("42", "en") == "forty two"
    assert cutlang.norm_for_align("0", "en") == "zero"
    assert cutlang.norm_for_align("1999", "en") == "one thousand nine hundred ninety nine"


def test_user_lexicon_overrides_builtin(home):
    store.path("lexicon").mkdir(parents=True, exist_ok=True)
    store.path("lexicon", "he.json").write_text(
        json.dumps({"AI": "אי איי", "Foo": "פו"}), encoding="utf-8")
    assert cutlang.norm_for_align("ai", "he", HE_VOCAB) == "אי איי"
    assert cutlang.norm_for_align("foo", "he", HE_VOCAB) == "פו"
    assert cutlang.lexicon("he")["foo"] == "פו"


def test_unknown_latin_with_vocab_is_empty(home):
    assert cutlang.norm_for_align("zzqx", "he", HE_VOCAB) == ""


def test_vocab_filters_letters():
    assert cutlang.norm_for_align("שלוםx", "he", set("של")) == "של"


def test_add_list_round_trip(home):
    cutlang.add_word("he", "Zed", "זד")
    assert cutlang.lexicon("he")["zed"] == "זד"
    assert cutlang.norm_for_align("zed", "he", HE_VOCAB) == "זד"
    r = run_cli("cutlang.py", "lexicon", "add", "--lang", "he", "--word", "Qux", "--spelling", "קוקס")
    assert r.returncode == 0, r.stderr
    r = run_cli("cutlang.py", "lexicon", "list", "--lang", "he")
    assert r.returncode == 0, r.stderr
    assert "qux" in r.stdout and "קוקס" in r.stdout


def test_latin_folding():
    assert cutlang.norm_for_align("Straße", "en") == "strasse"
    assert cutlang.norm_for_align("København", "en") == "kobenhavn"
    assert cutlang.norm_for_align("Łódź", "en") == "lodz"


def test_hebrew_hundreds_conjunction():
    h = cutlang.he_number
    assert h(101) == "מאה ואחת"
    assert h(120) == "מאה ועשרים"
    assert h(125) == "מאה עשרים וחמש"
    assert h(300) == "שלוש מאות"
    assert h(999) == "תשע מאות תשעים ותשע"
    assert h(1) == "אחת" and h(11) == "אחת עשרה" and h(21) == "עשרים ואחת"


def test_big_numbers_and_non_en_digits():
    assert cutlang.norm_for_align("1234", "he") == "אחת שתיים שלוש ארבע"
    assert cutlang.norm_for_align("2026", "en") == "two thousand twenty six"
    assert cutlang.norm_for_align("42", "fr") == ""
    assert cutlang.norm_for_align("(AI)", "he") == "איי איי"


def test_lexicon_lookup_strips_edge_punctuation(home):
    cutlang.add_word("en", "foo", "bar")
    assert cutlang.norm_for_align('"Foo,"', "en") == "bar"


def test_cli_rejects_bad_lang(home):
    r = run_cli("cutlang.py", "lexicon", "list", "--lang", "../x")
    assert r.returncode == 1


def test_fold_latin_public():
    import cutlang
    assert cutlang.fold_latin("Straße Łódź Café").lower() == "strasse lodz cafe"
