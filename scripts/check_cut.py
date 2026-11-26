#!/usr/bin/env python3
"""check_cut - check a rendered cut by listening to the WHOLE file again.

1. Transcribes the full rendered cut (faster-whisper, no VAD, no previous-text conditioning, so
   repeated takes are not merged), or reuses a transcript given with --heard.
2. Word-diffs it against the words the cut should contain (<stem>.words.json written by cut.py),
   after normalisation (niqqud/punctuation stripped, digits and Latin words spelled out).
3. Lists EXTRA / MISSING / DIFF words, pauses longer than --max-gap-ms (the ASR gap must be
   confirmed by a quiet run in the audio), any heard word longer than --max-word-ms (hidden repeat
   suspect), joins that land inside speech (energy over the local room-tone floor) and a left/right
   channel imbalance.

    check_cut.py <stem> [--lang he|en|<code>] [--model <id or dir>] [--heard <words.json>]
                 [--explain <explain.json>] [--max-gap-ms 450] [--max-word-ms 1200]
                 [--join-db 12] [--max-balance-db 1.0] [--allow-download]

Reads <stem>.mp4, <stem>.words.json and <stem>.cuts.json. Writes <stem>.check.words.json (unless
--heard) and <stem>.check.txt. --explain takes {"items": [{"match": "text in the item", "why":
"reason", "t": seconds (optional)}]}; explained items are listed but no longer fail the check.

ASR model (local cache only; nothing is downloaded without --allow-download):
    Hebrew  ivrit-ai/whisper-large-v3-turbo-ct2 (about 1.6 GB)
    others  mobiuslabsgmbh/faster-whisper-large-v3-turbo (about 1.6 GB)
    --model overrides either (a Hugging Face id or a local folder).

Exit: 0 clean (or every item explained), 1 items left or bad input, 3 missing package or model.
"""
import argparse
import difflib
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import align  # noqa: E402
import cutlang  # noqa: E402
import store  # noqa: E402

HE_ASR_MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"
OTHER_ASR_MODEL = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
ASR_SIZE = "about 1.6 GB"
ASR_PACKAGE = ("faster_whisper", "pip install faster-whisper", "about 1 MB plus ctranslate2 about 20 MB")
SLICE_MS = 10
SIMILAR_RATIO = 0.6


# ---------------------------------------------------------------- words and diff

def _tokens(words, lang):
    """(tokens, owner index) over a list of word strings."""
    toks, owner = [], []
    for k, w in enumerate(words):
        for t in cutlang.norm_for_compare(w, lang).split():
            toks.append(t)
            owner.append(k)
    return toks, owner


def diff_words(expected, heard, lang="he", expected_t=None):
    """Compare the words the cut should contain (list of str) with the heard words (dicts with
    w/start_ms/end_ms). Returns items {"kind": EXTRA|MISSING|DIFF, "t": seconds, "expected",
    "heard"}. A replace whose two sides are spelled alike (ratio >= 0.6) is the ASR spelling the
    same words differently and is not an item. expected_t: optional seconds per expected word,
    used to time MISSING items."""
    e, eo = _tokens(expected, lang)
    g, go = _tokens([h["w"] for h in heard], lang)
    sm = difflib.SequenceMatcher(None, e, g, autojunk=False)
    items = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        ew, gw = " ".join(e[i1:i2]), " ".join(g[j1:j2])
        if op == "replace" and difflib.SequenceMatcher(
                None, "".join(e[i1:i2]), "".join(g[j1:j2])).ratio() >= SIMILAR_RATIO:
            continue
        if j1 < len(g):
            t = heard[go[j1]]["start_ms"] / 1000
        else:
            t = heard[go[-1]]["end_ms"] / 1000 if go else 0.0
        kind = {"insert": "EXTRA", "delete": "MISSING", "replace": "DIFF"}[op]
        if kind == "MISSING" and expected_t is not None and i1 < len(eo):
            t = float(expected_t[eo[i1]])
        items.append({"kind": kind, "t": t, "expected": ew, "heard": gw})
    return items


# ---------------------------------------------------------------- audio levels

def _floor(audio_db, t_s):
    k = int(t_s * 1000 / SLICE_MS)
    return float(np.percentile(audio_db[max(0, k - 300): k + 300], 10))


def _silent_run(audio_db, t0_ms, t1_ms):
    """Longest run (ms) of quiet slices in [t0, t1]: ASR word times are not trusted alone."""
    thr = _floor(audio_db, (t0_ms + t1_ms) / 2000) + 10
    best = run = 0
    for v in audio_db[max(0, int(t0_ms // SLICE_MS)): int(t1_ms // SLICE_MS)]:
        run = run + SLICE_MS if v < thr else 0
        best = max(best, run)
    return best


def long_pauses(heard, audio_db, max_gap_ms):
    """Gaps between heard words over max_gap_ms that are also quiet in the audio for that long."""
    out = []
    for p, n in zip(heard, heard[1:]):
        if n["start_ms"] - p["end_ms"] > max_gap_ms and len(audio_db):
            sr = _silent_run(audio_db, p["end_ms"] - 150, n["start_ms"] + 150)
            if sr > max_gap_ms:
                out.append({"kind": "GAP", "t": p["end_ms"] / 1000, "gap_ms": sr,
                            "before": p["w"], "after": n["w"]})
    return out


def long_words(heard, max_word_ms):
    """Heard words longer than max_word_ms (a hidden repeat folded into one word)."""
    return [{"kind": "LONG WORD", "t": w["start_ms"] / 1000, "word": w["w"],
             "dur_ms": w["end_ms"] - w["start_ms"]}
            for w in heard if w["end_ms"] - w["start_ms"] > max_word_ms]


def join_energy(cuts, audio_db, join_db):
    """Joins whose cut point sits inside speech: loudest 10 ms slice within +-30 ms of the join
    is more than join_db over the local room-tone floor."""
    out = []
    if not len(audio_db):
        return out
    for j in cuts.get("joins", []):
        k = int(j["dst"] * 1000 / SLICE_MS)
        win = audio_db[max(0, k - 3): k + 3]
        if not len(win):
            continue
        over = float(win.max() - _floor(audio_db, j["dst"]))
        if over > join_db:
            out.append({"kind": "JOIN", "t": float(j["dst"]), "n": j.get("n"), "over_db": over,
                        "context": f"[{str(j.get('before', ''))[-30:]}|{str(j.get('after', ''))[:30]}]",
                        "src": f"{j.get('src_out', 0):.2f}->{j.get('src_in', 0):.2f}"})
    return out


def audio_db_of(source):
    """Mono level per 10 ms slice (dBFS) of the file's audio."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(source), "-vn", "-ac", "1",
                        "-ar", "48000", "-f", "s16le", "-"], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg could not read " + str(source))
    x = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768
    n = len(x) // 480
    if n == 0:
        return np.zeros(0)
    x = x[: n * 480].reshape(n, 480)
    return 20 * np.log10(np.sqrt((x * x).mean(axis=1)) + 1e-9)


def channel_balance(source):
    """RMS level of the left channel minus the right channel, in dB (0.0 for mono or silence)."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(source), "-vn", "-ac", "2",
                        "-ar", "16000", "-f", "s16le", "-"], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg could not read " + str(source))
    x = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float64) / 32768
    x = x[: len(x) // 2 * 2].reshape(-1, 2)
    if not len(x):
        return 0.0
    rms = np.sqrt((x * x).mean(axis=0))
    if rms.max() < 1e-6:
        return 0.0
    return float(20 * np.log10((rms[0] + 1e-9) / (rms[1] + 1e-9)))


# ---------------------------------------------------------------- ASR

def resolve_model(lang, model=None):
    return model or (HE_ASR_MODEL if lang == "he" else OTHER_ASR_MODEL)


def asr_preflight(model, allow_download):
    """Raise align.Blocked when faster-whisper or the model is missing. No model import."""
    if importlib.util.find_spec(ASR_PACKAGE[0]) is None:
        raise align.Blocked(f"missing package: faster-whisper (install: {ASR_PACKAGE[1]}, "
                            f"download {ASR_PACKAGE[2]})")
    if not allow_download and not os.path.isdir(model) and not align.hf_model_cached(model):
        raise align.Blocked(f"model not in the local cache: {model} ({ASR_SIZE}); "
                            f"rerun with --allow-download to fetch it")


def transcribe(mp4, lang, model, allow_download=False):
    """Whole-file transcript without VAD. Returns word dicts {i, w, start_ms, end_ms, p}."""
    asr_preflight(model, allow_download)
    from faster_whisper import WhisperModel
    kw = {"device": "cpu", "compute_type": "int8", "local_files_only": not allow_download}
    if not os.path.isdir(model):
        kw["download_root"] = str(align.hf_cache_dir())
    m = WhisperModel(model, **kw)
    audio = align.load_audio(mp4)
    segs, _ = m.transcribe(audio, language=None if lang == "auto" else lang, word_timestamps=True,
                           vad_filter=False, beam_size=5, condition_on_previous_text=False)
    words = []
    for s in segs:
        for w in (s.words or []):
            words.append({"i": len(words), "w": w.word.strip(), "start_ms": int(round(w.start * 1000)),
                          "end_ms": int(round(w.end * 1000)), "p": round(float(w.probability), 3)})
    return words


# ---------------------------------------------------------------- report

def describe(it):
    k = it["kind"]
    if k in ("EXTRA", "MISSING", "DIFF"):
        return f"{k}: expected [{it['expected']}] heard [{it['heard']}]"
    if k == "GAP":
        return f"GAP {it['gap_ms'] / 1000:.2f}s silence between [{it['before']}] and [{it['after']}]"
    if k == "LONG WORD":
        return f"LONG WORD {it['dur_ms'] / 1000:.2f}s [{it['word']}] (hidden repeat suspect)"
    if k == "JOIN":
        return (f"JOIN {it['n']} in speech: {it['over_db']:.0f} dB over floor "
                f"(src {it['src']}) {it['context']}")
    if k == "BALANCE":
        return f"BALANCE left/right differ by {it['db']:+.1f} dB (L minus R)"
    return str(it)


def split_explained(items, explain):
    left, done = [], []
    for it in items:
        s = describe(it)
        why = next((x.get("why", "") or "explained" for x in explain
                    if x.get("match") and x["match"] in s and abs(x.get("t", it["t"]) - it["t"]) < 3), None)
        (done if why is not None else left).append((it, s, why))
    return left, done


def check(stem, lang=None, model=None, heard_file=None, explain_file=None, max_gap_ms=450,
          max_word_ms=1200, join_db=12.0, max_balance_db=1.0, allow_download=False):
    """Run the whole check; returns (exit code, report text). Raises align.Blocked for exit 3."""
    mp4 = stem + ".mp4"
    exp_doc = json.load(open(stem + ".words.json", encoding="utf-8"))
    exp = exp_doc["words"]
    cuts = json.load(open(stem + ".cuts.json", encoding="utf-8"))
    lang = (lang if lang and lang != "auto" else exp_doc.get("language") or "he").lower()
    if heard_file:
        heard = json.load(open(heard_file, encoding="utf-8"))["words"]
    else:
        heard = transcribe(mp4, lang, resolve_model(lang, model), allow_download)
        with open(stem + ".check.words.json", "w", encoding="utf-8") as f:
            json.dump({"source": os.path.basename(mp4), "words": heard}, f, ensure_ascii=False, indent=1)
    items = diff_words([w["w"] for w in exp], heard, lang, [w["start_ms"] / 1000 for w in exp])
    db = audio_db_of(mp4)
    items += long_pauses(heard, db, max_gap_ms)
    items += long_words(heard, max_word_ms)
    items += join_energy(cuts, db, join_db)
    bal = channel_balance(mp4)
    if abs(bal) > max_balance_db:
        items.append({"kind": "BALANCE", "t": 0.0, "db": bal})
    items.sort(key=lambda i: i["t"])
    expl = []
    if explain_file and os.path.exists(explain_file):
        expl = json.load(open(explain_file, encoding="utf-8")).get("items", [])
    left, done = split_explained(items, expl)
    e, _ = _tokens([w["w"] for w in exp], lang)
    g, _ = _tokens([h["w"] for h in heard], lang)
    ratio = difflib.SequenceMatcher(None, e, g, autojunk=False).ratio()
    lines = [f"match {ratio:.3f}, {len(e)} expected / {len(g)} heard tokens, "
             f"{len(cuts.get('joins', []))} joins, {len(left)} item(s) left, {len(done)} explained"]
    lines += [f"  {it['t']:7.2f}s {s}" for it, s, _ in left]
    lines += [f"  {it['t']:7.2f}s EXPLAINED {s} -- {why}" for it, s, why in done]
    txt = "\n".join(lines)
    with open(stem + ".check.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    return (1 if left else 0), txt


def main(argv=None):
    store.print_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stem", help="cut stem: <stem>.mp4, .words.json, .cuts.json")
    ap.add_argument("--lang", default="auto", help="he, en, another code, or auto (words file language, else he)")
    ap.add_argument("--model", help="ASR model id or local folder (default by language)")
    ap.add_argument("--heard", help="reuse a transcript (words JSON) instead of transcribing")
    ap.add_argument("--explain", help="explained items JSON")
    ap.add_argument("--max-gap-ms", type=int, default=450)
    ap.add_argument("--max-word-ms", type=int, default=1200)
    ap.add_argument("--join-db", type=float, default=12.0,
                    help="a join fails when it is this many dB over the local room-tone floor")
    ap.add_argument("--max-balance-db", type=float, default=1.0,
                    help="left/right level difference that is reported")
    ap.add_argument("--allow-download", action="store_true",
                    help="permit fetching a missing ASR model (about 1.6 GB)")
    a = ap.parse_args(argv)
    stem = a.stem[:-4] if a.stem.endswith(".mp4") else a.stem
    for ext in (".mp4", ".words.json", ".cuts.json"):
        if not os.path.isfile(stem + ext):
            print(f"error: not found: {stem + ext}", file=sys.stderr)
            return 1
    if a.heard and not os.path.isfile(a.heard):
        print(f"error: not found: {a.heard}", file=sys.stderr)
        return 1
    try:
        code, txt = check(stem, a.lang, a.model, a.heard, a.explain, a.max_gap_ms, a.max_word_ms,
                          a.join_db, a.max_balance_db, a.allow_download)
    except align.Blocked as e:
        print(str(e), file=sys.stderr)
        return 3
    except (RuntimeError, OSError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(txt)
    return code


if __name__ == "__main__":
    sys.exit(main())
