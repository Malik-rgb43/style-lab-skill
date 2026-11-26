"""Word normalisation for forced alignment and transcript comparison, plus the spoken-form lexicon.

    cutlang.py lexicon add --lang he --word X --spelling Y
    cutlang.py lexicon list --lang he

Library:
    norm_for_align(word, lang, vocab=None) -> str   text the aligner model can score ('' if nothing left)
    norm_for_compare(word, lang) -> str             same mapping without a vocabulary (check pass)
    fold_latin(text) -> str                          special Latin letters folded, accents dropped
    lexicon(lang) -> dict                            built-in merged with the user file (user wins;
                                                     cached per process, reread when the file changes)
    add_word(lang, word, spelling)

User lexicon: <store home>/lexicon/<lang>.json = {"latin word": "how it is spoken"}: how a Latin
word is spoken, written in the language's letters, for alignment. Keys are single Latin words,
compared lowercase; only Latin words in the transcript are looked up, so an entry whose key is
written in the language's own letters, or holds several words, has no effect.
Hebrew: niqqud and punctuation dropped, digits spoken as Hebrew words, Latin words replaced by
their spoken form from the lexicon (a short built-in list of common tech words; brands belong in
the user lexicon).
Latin languages: NFKD, combining marks dropped, lowercase a-z and apostrophe; digits become English
words for `en`. Other scripts (neither Latin nor Hebrew) return ''.
"""
import argparse
import json
import re
import string
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402

NIQQUD = re.compile(r"[֑-ׇ]")

HE_UNITS = ["אפס", "אחת", "שתיים", "שלוש", "ארבע", "חמש", "שש", "שבע", "שמונה", "תשע", "עשר"]
HE_TEENS = {11: "אחת עשרה", 12: "שתים עשרה", 13: "שלוש עשרה", 14: "ארבע עשרה", 15: "חמש עשרה",
            16: "שש עשרה", 17: "שבע עשרה", 18: "שמונה עשרה", 19: "תשע עשרה"}
HE_TENS = {20: "עשרים", 30: "שלושים", 40: "ארבעים", 50: "חמישים", 60: "שישים", 70: "שבעים",
           80: "שמונים", 90: "תשעים"}
HE_HUNDREDS = {100: "מאה", 200: "מאתיים"}

BUILTIN = {
    "he": {
        "ai": "איי איי", "ok": "אוקיי", "okay": "אוקיי", "code": "קוד", "prompt": "פרומפט",
        "skill": "סקיל", "agent": "אייג'נט", "app": "אפ", "api": "איי פי איי", "mcp": "אם סי פי",
        "web": "ווב", "online": "אונליין", "file": "פייל", "data": "דאטה", "video": "וידאו",
        "email": "אימייל", "link": "לינק", "chat": "צאט", "bot": "בוט", "pdf": "פי די אף",
        "url": "יו אר אל", "gpt": "גיפיטי", "b": "בי", "roll": "רול", "m": "אם",
    },
}

EN_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
           "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
           "eighteen", "nineteen"]
EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _en_below_100(n):
    if n < 20:
        return EN_ONES[n]
    t, o = divmod(n, 10)
    return EN_TENS[t] + (" " + EN_ONES[o] if o else "")


def en_number(n: int) -> str:
    """0-9999 in English words; larger numbers are read digit by digit."""
    if n < 0 or n > 9999:
        return " ".join(EN_ONES[int(d)] for d in str(abs(n)))
    parts = []
    th, rest = divmod(n, 1000)
    if th:
        parts.append(EN_ONES[th] + " thousand")
    h, rest = divmod(rest, 100)
    if h:
        parts.append(EN_ONES[h] + " hundred")
    if rest or not parts:
        parts.append(_en_below_100(rest))
    return " ".join(parts)


def _he_below_100(n):
    if n <= 10:
        return HE_UNITS[n]
    if n < 20:
        return HE_TEENS[n]
    t, o = divmod(n, 10)
    return HE_TENS[t * 10] + (" ו" + HE_UNITS[o] if o else "")


def he_number(n: int) -> str:
    """0-999 in Hebrew words (feminine, as counted); larger numbers digit by digit."""
    if n < 0 or n > 999:
        return " ".join(HE_UNITS[int(d)] for d in str(abs(n)))
    if n < 100:
        return _he_below_100(n)
    h, rest = divmod(n, 100)
    if h <= 2:
        base = HE_HUNDREDS[h * 100]
    else:
        base = HE_UNITS[h] + " מאות"
    if not rest:
        return base
    tail = _he_below_100(rest)
    # "ו" goes before the last element only: 101 מאה ואחת, 120 מאה ועשרים, 125 מאה עשרים וחמש
    return base + (" ו" if rest < 20 or rest % 10 == 0 else " ") + tail


def _user_path(lang):
    return store.path("lexicon", f"{lang}.json")


_LEX_CACHE = {}  # lang -> (user file path, its (mtime_ns, size) or None, merged dict)


def _stamp(p):
    try:
        st = p.stat()
        return st.st_mtime_ns, st.st_size
    except OSError:
        return None


def lexicon(lang: str) -> dict:
    """Built-in entries merged with the user's file; keys lowercase, user wins.

    Cached per process, keyed by the user file's path and modification time, so the file is read
    once per run and again only after it changes."""
    p = _user_path(lang)
    stamp = _stamp(p)
    hit = _LEX_CACHE.get(lang)
    if hit and hit[0] == str(p) and hit[1] == stamp:
        return dict(hit[2])
    out = dict(BUILTIN.get(lang, {}))
    user = store.read_json(p, {}) if stamp is not None else {}
    if isinstance(user, dict):
        for k, v in user.items():
            if isinstance(k, str) and isinstance(v, str):
                out[k.lower()] = v
    _LEX_CACHE[lang] = (str(p), stamp, out)
    return dict(out)


def add_word(lang: str, word: str, spelling: str) -> None:
    key = word.strip().lower()
    if not key or not spelling.strip():
        raise ValueError("word and spelling must be non-empty")

    def fn(d):
        d[key] = spelling.strip()

    store.update_json(_user_path(lang), {}, fn)


_FOLD = str.maketrans({"ß": "ss", "ẞ": "ss", "ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae", "œ": "oe",
                       "Œ": "oe", "ł": "l", "Ł": "l", "đ": "d", "Đ": "d", "þ": "th", "Þ": "th",
                       "ı": "i"})
_EDGE = string.whitespace + string.punctuation + "‘’“”…«»"


def _strip_marks(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def fold_latin(text: str) -> str:
    """Fold special Latin letters (ß, ø, æ, ł, ...) and drop combining marks (NFKD); callers lowercase."""
    return _strip_marks(text.translate(_FOLD))


def _he(word, vocab):
    lex = lexicon("he")
    s = NIQQUD.sub("", word).lower()
    out = []
    for p in re.findall(r"[a-z]+|\d+|[א-ת]+", s):
        if p.isdigit():
            out.append(he_number(int(p)))
        elif p[0].isascii():
            out.append(lex.get(p, ""))
        else:
            out.append(p)
    txt = " ".join(x for x in out if x)
    if vocab is None:
        txt = "".join(c for c in txt if "א" <= c <= "ת" or c == " ")
    else:
        txt = "".join(c for c in txt if c in vocab or c == " ")
    return re.sub(r"\s+", " ", txt).strip()


def _latin(word, lang):
    lex = lexicon(lang)
    key = word.strip(_EDGE).lower()
    if key in lex:
        word = lex[key]
    s = fold_latin(word).lower().replace("\u2019", "'")
    out = []
    for p in re.findall(r"[a-z']+|\d+", s):
        if p.isdigit():
            if lang == "en":
                out.append(en_number(int(p)))
        else:
            p = p.strip("'")
            if p:
                out.append(p)
    return re.sub(r"\s+", " ", " ".join(out)).strip()


def _norm(word, lang, vocab):
    if lang == "he":
        return _he(word, vocab)
    return _latin(word, lang)


def norm_for_align(word: str, lang: str, vocab=None) -> str:
    return _norm(word, lang, vocab)


def norm_for_compare(word: str, lang: str) -> str:
    return _norm(word, lang, None)


def main(argv=None):
    return store.exit_on_lock_timeout(_main, argv)


def _main(argv=None):
    store.print_utf8()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    lx = sub.add_parser("lexicon").add_subparsers(dest="action", required=True)
    a = lx.add_parser("add")
    a.add_argument("--lang", required=True)
    a.add_argument("--word", required=True, help="a single Latin word as the transcript writes it")
    a.add_argument("--spelling", required=True,
                   help="how the word is spoken, written in the language's letters, for alignment")
    a = lx.add_parser("list")
    a.add_argument("--lang", required=True)
    args = p.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z-]{2,8}", args.lang):
        print("error: --lang must match [A-Za-z-]{2,8}", file=sys.stderr)
        return 1
    if args.action == "add":
        try:
            add_word(args.lang, args.word, args.spelling)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"added {args.word.strip().lower()} -> {args.spelling.strip()} ({args.lang})")
        return 0
    for k, v in sorted(lexicon(args.lang).items()):
        print(f"{k}\t{v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
