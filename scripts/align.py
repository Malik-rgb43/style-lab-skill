#!/usr/bin/env python3
"""Forced alignment of a known transcript to its audio, then energy-refined word edges.

ASR word times (Whisper and friends) are often 100-300 ms off, which clips syllables when a cut
is made on them. This tool keeps the ASR WORDS but re-times them with a CTC model
(torchaudio.functional.forced_align), then moves every word edge onto the real audio energy
(10 ms slices). Every keep decision of the precise cut should use these times.

    align.py <source> <words.json> --out <words.aligned.json> [--lang he|en|auto|<code>]
             [--chunk-s 30] [--margin-s 0.6] [--from-word I --to-word J] [--threads N]
             [--allow-download]

Models (CPU is fine):
    Hebrew  imvladikon/wav2vec2-large-xlsr-53-hebrew (Hugging Face cache, ~1.3 GB)
    Latin-script languages  torchaudio.pipelines.MMS_FA on romanised text (torch hub cache, ~1.2 GB)
    Other scripts  no model: words keep their ASR times with "flag": "unaligned_script"
Models load from the local cache only; a missing model stops with exit 3 unless
--allow-download is given.

Long files are processed in windows of about --chunk-s seconds, split at the largest pause between
words and padded with --margin-s of audio on each side, so a word is never cut at a window edge.

Output: same shape as words.json; each word gets start_ms/end_ms (refined), asr_start_ms/
asr_end_ms (original), ctc_start_ms/ctc_end_ms (raw CTC), score (0..1, mean token probability)
and an optional flag (unaligned, unaligned_script, stretched, low_score). An "align_stats" block
reports how far the edges moved. --from-word/--to-word keep only that range in the output.
A low-score word (score < 0.15) still gets its refined CTC times, flagged "low_score"; its
asr_start_ms/asr_end_ms stay in the output so the cut step can weigh both spans.
Unaligned words with Latin letters in a Hebrew file get a one-line hint to add their spelling
with `cutlang.py lexicon add`.

Exit: 0 done, 1 error (bad input, ffmpeg failed), 2 usage, 3 missing package or model.
"""
import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cutlang  # noqa: E402
import store  # noqa: E402

HE_MODEL_ID = "imvladikon/wav2vec2-large-xlsr-53-hebrew"
HE_MODEL_SIZE = "about 1.3 GB"
MMS_ID = "torchaudio.pipelines.MMS_FA"
MMS_URL = "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt"
MMS_SIZE = "about 1.2 GB"
SR = 16000
SLICE = 160  # 10 ms at 16 kHz

PACKAGES = {  # import name -> (install command, approx download)
    "torch": ("pip install torch", "about 200 MB CPU build, 2.5 GB with CUDA"),
    "torchaudio": ("pip install torchaudio", "about 5 MB"),
    "transformers": ("pip install transformers", "about 12 MB"),
}

# refinement reach (as the course tool): onset up to 120 ms before the CTC start,
# offset up to 300 ms after the CTC end, local floor from +-600 ms around the word
BACK_SLICES = 12
FWD_SLICES = 30
FLOOR_SLICES = 60
FLOOR_OFFSET_DB = 8.0
MIN_WORD_MS = 40
LOW_SCORE = 0.15


class Blocked(Exception):
    """A missing package or model: the CLI prints the message on one line and exits 3."""


# ---------------------------------------------------------------- caches (no torch import)

def hf_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "huggingface" / "hub"


def hf_model_cached(model_id: str) -> bool:
    snaps = hf_cache_dir() / ("models--" + model_id.replace("/", "--")) / "snapshots"
    return any(p.is_file() for p in snaps.glob("*/config.json"))


def torch_hub_dir() -> Path:
    if os.environ.get("TORCH_HOME"):
        return Path(os.environ["TORCH_HOME"]) / "hub"
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "torch" / "hub"


def mms_cached() -> bool:
    return (torch_hub_dir() / "checkpoints" / os.path.basename(MMS_URL)).is_file()


def require_packages(names):
    for n in names:
        if importlib.util.find_spec(n) is None:
            cmd, size = PACKAGES[n]
            raise Blocked(f"missing package: {n} (install: {cmd}, download {size})")


def _model_block(model, size):
    return Blocked(f"model not in the local cache: {model} ({size}); "
                   f"rerun with --allow-download to fetch it")


# ---------------------------------------------------------------- audio

def load_audio(src) -> np.ndarray:
    """Mono 16 kHz float32 in [-1, 1] through ffmpeg."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(src), "-vn", "-ac", "1",
                        "-ar", str(SR), "-f", "s16le", "-"], capture_output=True)
    if r.returncode != 0:
        msg = r.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError("ffmpeg could not read the source: " + (msg[-1] if msg else "unknown error"))
    return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def slice_db(audio: np.ndarray) -> np.ndarray:
    n = len(audio) // SLICE
    x = audio[: n * SLICE].reshape(n, SLICE)
    rms = np.sqrt((x * x).mean(axis=1)) + 1e-9
    return 20 * np.log10(rms)


# ---------------------------------------------------------------- text

def _non_latin_letter(c, lang):
    if not c.isalpha():
        return False
    if lang == "he" and "א" <= c <= "ת":
        return False
    base = cutlang.fold_latin(c).lower()
    return not all("a" <= b <= "z" for b in base)


def word_norm(word: str, lang: str, vocab=None):
    """(model text, flag): text '' means the word keeps its ASR times.

    Flag "unaligned_script" when the word has letters the aligner for `lang` cannot read
    (non-Latin, non-Hebrew script), "unaligned" for anything else left empty (symbols,
    unknown Latin words in Hebrew)."""
    txt = cutlang.norm_for_align(word, lang, vocab)
    if lang != "he" and vocab is not None:
        txt = re.sub(r"\s+", " ", "".join(c for c in txt if c in vocab or c == " ")).strip()
    if txt:
        return txt, None
    w = unicodedata.normalize("NFC", word)
    return "", ("unaligned_script" if any(_non_latin_letter(c, lang) for c in w) else "unaligned")


# ---------------------------------------------------------------- windows

def make_chunks(words, chunk_s, margin_s, dur_ms):
    """Group word indices into windows of about chunk_s; split at the largest pause in the
    window's last half. Returns (first, last, audio_start_ms, audio_end_ms) per window; the
    audio span adds margin_s on each side (clamped to 0..dur_ms)."""
    chunks, i, n = [], 0, len(words)
    while i < n:
        t0 = words[i]["start_ms"]
        j = i
        while j + 1 < n and words[j + 1]["end_ms"] - t0 < chunk_s * 1000:
            j += 1
        if j + 1 < n and j > i:
            lo = i + (j - i) // 2
            j = max(range(lo, j + 1), key=lambda k: words[k + 1]["start_ms"] - words[k]["end_ms"])
        a = max(0, words[i]["start_ms"] - int(margin_s * 1000))
        b = min(dur_ms, words[j]["end_ms"] + int(margin_s * 1000))
        chunks.append((i, j, a, b))
        i = j + 1
    return chunks


# ---------------------------------------------------------------- CTC core (shared)

def _align_windows(words, audio, chunks, emit, vocab, blank, space, log):
    """Fill ctc_start_ms/ctc_end_ms/score for every word with norm text. `emit(seg)` returns
    log-probabilities shaped (1, T, V)."""
    import torch
    import torchaudio.functional as AF

    for ci, (i, j, s_ms, e_ms) in enumerate(chunks):
        idx = [k for k in range(i, j + 1) if words[k]["norm"]]
        if not idx:
            continue
        seg = audio[s_ms * SR // 1000: e_ms * SR // 1000]
        with torch.inference_mode():
            lp = torch.log_softmax(emit(seg).float(), dim=-1)
        T = lp.shape[1]
        if T == 0:
            for k in idx:
                words[k]["flag"] = "unaligned"
            continue
        fdur = (len(seg) / SR) / T  # seconds per CTC frame (~0.02)
        tokens, owner = [], []
        for n_, k in enumerate(idx):
            for c in words[k]["norm"]:
                if c == " ":
                    if space is not None:
                        tokens.append(space)
                        owner.append(-1)
                    continue
                tokens.append(vocab[c])
                owner.append(k)
            if space is not None and n_ < len(idx) - 1:
                tokens.append(space)
                owner.append(-1)
        if not tokens or len(tokens) > T:
            for k in idx:
                words[k]["flag"] = "unaligned"
            continue
        try:
            ali, scores = AF.forced_align(lp, torch.tensor([tokens], dtype=torch.int32), blank=blank)
        except Exception as ex:  # too little audio for the text
            log(f"  window {ci + 1}: align failed ({ex}); ASR times kept")
            for k in idx:
                words[k]["flag"] = "unaligned"
            continue
        ali, scores = ali[0].tolist(), scores[0].exp().tolist()
        # walk frames, map each non-blank emission to its token position
        pos, prev, span = -1, blank, {}
        for f, (tok, sc) in enumerate(zip(ali, scores)):
            if tok != blank and (tok != prev or prev == blank):
                pos += 1
            if tok != blank and 0 <= pos < len(owner) and owner[pos] >= 0:
                span.setdefault(owner[pos], []).append((f, sc, pos))
            prev = tok
        gapf = max(1, int(0.3 / fdur))
        for k in idx:
            if k not in span:
                words[k]["flag"] = "unaligned"
                continue
            # a word's letters can be stretched over unscripted speech or a long silence (one
            # letter parked seconds away): split its frames into clusters at gaps > 0.3 s and keep
            # the cluster with most distinct letters, ties -> closest to the ASR time
            fr = span[k]
            cl = [[fr[0]]]
            for x in fr[1:]:
                if x[0] - cl[-1][-1][0] <= gapf:
                    cl[-1].append(x)
                else:
                    cl.append([x])
            asr_mid = (words[k]["asr_start_ms"] + words[k]["asr_end_ms"]) / 2 - s_ms
            best = max(cl, key=lambda c: (len({x[2] for x in c}),
                                          -abs((c[0][0] + c[-1][0]) / 2 * fdur * 1000 - asr_mid)))
            if len(cl) > 1:
                words[k]["flag"] = "stretched"
            f0, f1 = best[0][0], best[-1][0]
            words[k]["ctc_start_ms"] = s_ms + int(round(f0 * fdur * 1000))
            words[k]["ctc_end_ms"] = s_ms + int(round((f1 + 1) * fdur * 1000))
            words[k]["score"] = round(float(np.mean([x[1] for x in best])), 3)
        log(f"  window {ci + 1}/{len(chunks)} {s_ms / 1000:7.1f}-{e_ms / 1000:7.1f}s "
            f"words {words[i]['i']}-{words[j]['i']}")


def _align_he(words, audio, chunks, allow_download, log):
    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    kw = {"cache_dir": str(hf_cache_dir()), "local_files_only": not allow_download}
    try:
        proc = Wav2Vec2FeatureExtractor.from_pretrained(HE_MODEL_ID, **kw)
        model = Wav2Vec2ForCTC.from_pretrained(HE_MODEL_ID, **kw).eval()
        with open(hf_hub_download(HE_MODEL_ID, "vocab.json", **kw), encoding="utf-8") as f:
            vocab = json.load(f)
    except OSError:
        if allow_download:
            raise
        raise _model_block(HE_MODEL_ID, HE_MODEL_SIZE)
    blank = vocab["[PAD]"]  # CTC blank (the tokenizer config names "<pad>", the vocab has "[PAD]")
    letters = {c for c in vocab if len(c) == 1 and "א" <= c <= "ת"}
    for w in words:
        if w["norm"]:
            w["norm"], flag = word_norm(w["w"], "he", letters)
            if flag:
                w["flag"] = flag

    def emit(seg):
        inp = proc(seg, sampling_rate=SR, return_tensors="pt")
        return model(inp.input_values).logits

    _align_windows(words, audio, chunks, emit, vocab, blank, vocab.get("|"), log)
    return HE_MODEL_ID


def _align_mms(words, audio, chunks, lang, allow_download, log):
    import torch
    from torchaudio.pipelines import MMS_FA

    try:
        model = MMS_FA.get_model(with_star=False,
                                 dl_kwargs={"model_dir": str(torch_hub_dir() / "checkpoints")})
    except Exception:
        if allow_download or mms_cached():
            raise
        raise _model_block(MMS_ID, MMS_SIZE)
    model.eval()
    vocab = MMS_FA.get_dict(star=None)
    for w in words:
        if w["norm"]:
            w["norm"], flag = word_norm(w["w"], lang, vocab)
            if flag:
                w["flag"] = flag

    def emit(seg):
        out, _ = model(torch.from_numpy(np.ascontiguousarray(seg)).unsqueeze(0))
        return out

    _align_windows(words, audio, chunks, emit, vocab, vocab["-"], None, log)
    return MMS_ID


# ---------------------------------------------------------------- energy refinement

def refine_edges(words, db_slices: np.ndarray, floor_db: float = None) -> None:
    """Move each aligned word's edges onto the audio energy (10 ms slices of `db_slices`).

    start = energy onset walking back from the CTC start (up to 120 ms), end = where the energy
    falls to the floor walking forward from the CTC end (up to 300 ms). Neither edge passes the
    midpoint of the gap to the neighbour word, so refined words never overlap. `floor_db` None
    = local floor (10th percentile of +-600 ms around the word + 8 dB). Words without CTC times
    keep their ASR times."""
    db = db_slices
    last = len(db) - 1
    dur_ms = len(db) * 10

    def ctc_or_asr(w, edge):
        return w.get(f"ctc_{edge}_ms", w[f"asr_{edge}_ms"])

    for n_, w in enumerate(words):
        if "ctc_start_ms" not in w:
            w["start_ms"], w["end_ms"] = w["asr_start_ms"], w["asr_end_ms"]
            continue
        cs, ce = w["ctc_start_ms"], w["ctc_end_ms"]
        prev_end = ctc_or_asr(words[n_ - 1], "end") if n_ else 0
        nxt_start = ctc_or_asr(words[n_ + 1], "start") if n_ + 1 < len(words) else dur_ms
        lo_ms = cs if prev_end > cs else (prev_end + cs) // 2 if n_ else 0
        hi_ms = ce if nxt_start < ce else (ce + nxt_start) // 2 if n_ + 1 < len(words) else dur_ms
        c0, c1 = min(cs // 10, last), min(ce // 10, last)
        if floor_db is None:
            loc = db[max(0, c0 - FLOOR_SLICES): min(last, c1 + FLOOR_SLICES) + 1]
            floor = float(np.percentile(loc, 10)) + FLOOR_OFFSET_DB if len(loc) else -60.0
        else:
            floor = floor_db
        # onset: walk back from c0 while above the floor
        k = c0
        lim = max(-(-lo_ms // 10), c0 - BACK_SLICES, 0)
        while k - 1 >= lim and db[k - 1] > floor:
            k -= 1
        start = max(min(k * 10, cs), lo_ms)  # slice floor must not cross the gap midpoint
        # offset: walk forward from c1 while above the floor
        k = c1
        lim = min(hi_ms // 10, c1 + FWD_SLICES, last + 1)
        while k < lim and db[k] > floor:
            k += 1
        end = k * 10
        if end < start + MIN_WORD_MS:  # very short word: give it 40 ms, still inside the gap
            end = max(min(start + MIN_WORD_MS, hi_ms), start + 10)
        w["start_ms"], w["end_ms"] = int(start), int(end)


# ---------------------------------------------------------------- driver

def _stats(words):
    al = [w for w in words if "ctc_start_ms" in w]
    ds = np.array([abs(w["start_ms"] - w["asr_start_ms"]) for w in al]) if al else np.zeros(1)
    de = np.array([abs(w["end_ms"] - w["asr_end_ms"]) for w in al]) if al else np.zeros(1)
    both = np.concatenate([ds, de])
    return {"aligned": len(al), "unaligned": len(words) - len(al),
            "start_shift_ms": {"median": float(np.median(ds)), "p90": float(np.percentile(ds, 90))},
            "end_shift_ms": {"median": float(np.median(de)), "p90": float(np.percentile(de, 90))},
            "edge_shift_ms": {"median": float(np.median(both)), "p90": float(np.percentile(both, 90))},
            "long_words_over_1200ms": [w["i"] for w in words if w["end_ms"] - w["start_ms"] > 1200]}


def preflight(lang, words, allow_download):
    """Raise Blocked when a needed package or model is missing. No torch import."""
    if not any(w["norm"] for w in words):
        return
    if lang == "he":
        require_packages(["torch", "torchaudio", "transformers"])
        if not allow_download and not hf_model_cached(HE_MODEL_ID):
            raise _model_block(HE_MODEL_ID, HE_MODEL_SIZE)
    else:
        require_packages(["torch", "torchaudio"])
        if not allow_download and not mms_cached():
            raise _model_block(MMS_ID, MMS_SIZE)


def align(source, words_doc, lang, chunk_s=30, margin_s=0.6, allow_download=False,
          from_word=0, to_word=None, threads=None, log=None):
    """Return the aligned doc (words re-timed, align_stats block). Raises Blocked when a package
    or model is missing (nothing is downloaded unless allow_download)."""
    log = log or (lambda msg: None)
    t_start = time.time()
    allw = words_doc["words"]
    end = len(allw) if to_word is None else min(len(allw), to_word + 1)
    words = [dict(allw[k]) for k in range(max(0, from_word), end)]
    for w in words:
        w["asr_start_ms"], w["asr_end_ms"] = w["start_ms"], w["end_ms"]
        w["norm"], flag = word_norm(w["w"], lang)
        if flag:
            w["flag"] = flag
    preflight(lang, words, allow_download)

    aligned_with = None
    db = np.zeros(0)
    if any(w["norm"] for w in words):
        audio = load_audio(source)
        dur_ms = int(len(audio) / SR * 1000)
        db = slice_db(audio)
        if threads:
            import torch
            torch.set_num_threads(threads)
        chunks = make_chunks(words, chunk_s, margin_s, dur_ms)
        log(f"{len(words)} words, {len(chunks)} windows, audio {dur_ms / 1000:.1f}s")
        if lang == "he":
            aligned_with = _align_he(words, audio, chunks, allow_download, log)
        else:
            aligned_with = _align_mms(words, audio, chunks, lang, allow_download, log)
        for w in words:
            if w["norm"] and "ctc_start_ms" not in w and not w.get("flag"):
                w["flag"] = "unaligned"
    if lang == "he":
        latin = sum(1 for w in words if "ctc_start_ms" not in w and re.search(r"[A-Za-z]", w["w"]))
        if latin:
            log(f"hint: {latin} word(s) with Latin letters kept ASR times; add their Hebrew spelling "
                f"with: cutlang.py lexicon add --lang he --word <word> --spelling <spelling>")
    refine_edges(words, db)
    for w in words:
        if "ctc_start_ms" in w and w.get("score", 1) < LOW_SCORE:
            w["flag"] = w.get("flag") or "low_score"
        w.pop("norm", None)

    out = dict(words_doc)
    out.update({"words": words, "aligned_with": aligned_with, "align_stats": _stats(words),
                "align_elapsed_s": round(time.time() - t_start, 1)})
    out.pop("segments", None)
    return out


def main(argv=None):
    store.print_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="audio or video file")
    ap.add_argument("words", help="words.json from the transcribe step")
    ap.add_argument("--out", required=True, help="aligned words JSON to write")
    ap.add_argument("--lang", default="auto", help="he, en, another language code, or auto (default: "
                    "the words file's language field)")
    ap.add_argument("--chunk-s", type=float, default=30.0)
    ap.add_argument("--margin-s", type=float, default=0.6)
    ap.add_argument("--from-word", type=int, default=0)
    ap.add_argument("--to-word", type=int, default=None)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--allow-download", action="store_true",
                    help="permit fetching a missing model (about 1.2-1.3 GB)")
    a = ap.parse_args(argv)

    try:
        with open(a.words, encoding="utf-8") as f:
            doc = json.load(f)
        if not isinstance(doc, dict) or not isinstance(doc.get("words"), list):
            raise ValueError("no 'words' list")
    except (OSError, ValueError) as e:
        print(f"error: cannot read {a.words}: {e}", file=sys.stderr)
        return 1
    lang = a.lang
    if lang == "auto":
        lang = doc.get("language")
        if not isinstance(lang, str) or not lang:
            print("error: the words file has no 'language' field; pass --lang", file=sys.stderr)
            return 2
    lang = lang.lower()
    if not re.fullmatch(r"[a-z-]{2,8}", lang):
        print("error: --lang must be a language code like he or en", file=sys.stderr)
        return 2
    if not os.path.isfile(a.source):
        print(f"error: source not found: {a.source}", file=sys.stderr)
        return 1

    try:
        out = align(a.source, doc, lang, a.chunk_s, a.margin_s, a.allow_download,
                    from_word=a.from_word, to_word=a.to_word, threads=a.threads,
                    log=lambda m: print(m, flush=True))
    except Blocked as e:
        print(str(e), file=sys.stderr)
        return 3
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    tmp = a.out + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        os.replace(tmp, a.out)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    print(json.dumps(out["align_stats"], ensure_ascii=False))
    print(f"done -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
