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
