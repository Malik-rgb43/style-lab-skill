#!/usr/bin/env python3
"""Language profiles (he, en), timing scaling, RTL mirror plan and font mapping.

Built-in profiles live in assets/lang/<code>.json; a profile in <store>/lang/<code>.json overrides it.

Commands:
    lang.py show <code>                       print the profile (store version wins)
    lang.py add <code> --from <json>          validate and save a profile to <store>/lang/<code>.json
    lang.py adapt <pack> --to <code> [--from-lang <code>] [--source-wps X] [--target-wps Y] [--font-dir DIR ...]
                                              write <pack>/adapt-<code>.json

Exit codes: 0 ok, 2 bad input.
"""
import argparse
import copy
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "lang"
ROLES = ("headline", "body", "caption", "numbers", "mono", "handwritten")
REQUIRED = ("code", "script", "dir", "fonts", "caption", "speech_wps_typical", "align_model", "asr_hint",
            "number_format", "date_format", "triggers")
FONT_EXT = (".ttf", ".otf", ".ttc")
CODE_RE = r"[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})?"
DEFAULT_ELEMENTS = ("reading_order", "progress_bar", "arrow", "slide_in", "list_order", "clock", "chart_axes",
                    "logo", "code", "numbers", "timecode")

# (keywords, why). Matched as substrings of the normalised element name; KEEP rows win.
KEEP_TABLE = [
    (("clock", "watch"), "a clock turns the same way in every language"),
    (("axis", "axes", "chart", "graph", "plot"), "chart axes keep their direction (time and values are not text)"),
    (("logo", "brand_mark", "wordmark"), "a logo is a fixed artwork"),
    (("code", "terminal", "console", "snippet"), "code is left to right in every language"),
    (("digit", "number", "numeral", "counter", "price"), "digits stay left to right"),
    (("timecode", "timestamp", "playhead", "scrub"), "video timecode runs left to right"),
]
MIRROR_TABLE = [
    (("reading_order", "text_flow", "headline", "caption", "paragraph", "title"),
     "text block follows the reading direction"),
    (("progress",), "a progress bar fills along the reading direction"),
    (("arrow", "chevron", "pointer"), "an arrow that means forward or back follows the reading direction"),
    (("slide_in", "slide", "wipe", "swipe", "enter_from"), "motion enters from the side the reader starts on"),
    (("list", "stepper", "steps", "menu"), "list order follows the reading direction"),
]
SIDE_FLIP = {"left": "right", "right": "left"}


# ---------------------------------------------------------------- profiles
def _validate(p):
    if not isinstance(p, dict):
        raise ValueError("profile must be a JSON object")
    missing = [k for k in REQUIRED if k not in p]
    if missing:
        raise ValueError("profile is missing: " + ", ".join(missing))
    if p["dir"] not in ("rtl", "ltr"):
        raise ValueError('dir must be "rtl" or "ltr"')
    if not isinstance(p["fonts"], dict) or not all(isinstance(v, list) for v in p["fonts"].values()):
        raise ValueError("fonts must map a role to a list of families")
    w = p["speech_wps_typical"]
    if isinstance(w, bool) or not isinstance(w, (int, float)) or w <= 0:
        raise ValueError("speech_wps_typical must be a positive number")
    tr = p["triggers"]
    if not isinstance(tr, dict) or not all(k in tr for k in ("number_words", "months", "question_marks")):
        raise ValueError("triggers needs number_words, months, question_marks")
    if not isinstance(p["caption"], dict):
        raise ValueError("caption must be an object")
    if not re.fullmatch(CODE_RE, str(p["code"])):
        raise ValueError("code must look like he, en or pt-BR")
    return p


def _store_profile_path(code):
    import store
    return store.path("lang", f"{code}.json")


def load_profile(code):
    """Profile for `code`; the store version overrides the built-in one. Raises KeyError if neither exists."""
    import store
    if not re.fullmatch(CODE_RE, str(code)):
        raise KeyError(f"bad language code: {code}")
    data = store.read_json(_store_profile_path(code), None)
    if data is None:
        bp = ASSETS / f"{code}.json"
        if not bp.exists():
            raise KeyError(f"no profile for language '{code}' (add one with: lang.py add {code} --from <json>)")
        data = store.read_json(bp, None)
    return _validate(data)


# ---------------------------------------------------------------- timing
def _scale_num(v, ratio):
    return round(v * ratio, 4) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


def scale_timing(laws, ratio):
    """Copy of `laws` with word-anchored lags multiplied by `ratio` (source_wps / target_wps).

    Only laws with anchor "word" change. Visual-anchored laws, laws with no anchor and frame-based motion
    durations are left alone."""
    out = copy.deepcopy(laws)
    for law in out:
        if not isinstance(law, dict) or law.get("anchor") != "word":
            continue
        lag = law.get("lag")
        if isinstance(lag, dict):
            for k in ("median", "p10", "p90"):
                if k in lag:
                    lag[k] = _scale_num(lag[k], ratio)
        elif isinstance(lag, (int, float)) and not isinstance(lag, bool):
            law["lag"] = _scale_num(lag, ratio)
        else:
            continue
        law["lag_scaled_by"] = round(ratio, 4)
    return out


# ---------------------------------------------------------------- mirror
def _norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def mirror_plan(elements, target_dir="rtl", source_dir="ltr"):
    """[{element, mirror, why}] per element (a name, or a dict with element/kind/name/type, optional side).

    Elements are mirrored only when the source and target reading directions differ."""
    plan = []
    for el in elements:
        if isinstance(el, dict):
            name = el.get("element") or el.get("kind") or el.get("name") or el.get("type") or ""
            blob = " ".join(str(el.get(k, "")) for k in ("element", "kind", "name", "type"))
        else:
            name = blob = str(el)
        n = _norm(blob)
        res = None
        for kws, why in KEEP_TABLE:
            if any(k in n for k in kws):
                res = (False, why)
                break
        if res is None:
            for kws, why in MIRROR_TABLE:
                if any(k in n for k in kws):
                    res = (True, why)
                    break
        if res is None:
            res = (False, "not in the mirror table; kept as is")
        mirror, why = res
        if mirror and source_dir == target_dir:
            mirror, why = False, "source and target read the same direction"
        row = {"element": name, "mirror": mirror, "why": why}
        if mirror and isinstance(el, dict) and el.get("side") in SIDE_FLIP:
            row["side"] = SIDE_FLIP[el["side"]]
        plan.append(row)
    return plan


def emphasis_swaps(source_emphasis, profile):
    """[{"italic": "weight"}] when the source uses italics and the target cannot; else []."""
    cap = profile.get("caption", {})
    if not cap.get("no_italics"):
        return []
    repl = next((e for e in cap.get("emphasis", []) if e != "italic"), "weight")
    return [{"italic": repl}] if "italic" in (source_emphasis or []) else []


# ---------------------------------------------------------------- fonts
class FontList(list):
    """A list of candidates that also carries `not_run` (what could not be measured)."""

    def __init__(self, items=(), not_run=()):
        super().__init__(items)
        self.not_run = list(not_run)


def _fonttools():
    try:
        from fontTools import ttLib
        return ttLib
    except Exception:
        return None


def _family(font):
    for nid in (16, 1):
        rec = font["name"].getName(nid, 3, 1, 0x409) or font["name"].getName(nid, 1, 0, 0)
        if rec:
            return str(rec)
    return ""


def font_metrics(file):
    """{family, x_height_ratio, avg_advance, weight, italic_angle}, or None when it cannot be read."""
    ttLib = _fonttools()
    if ttLib is None:
        return None
    try:
        f = ttLib.TTFont(str(file), fontNumber=0, lazy=True)
        upm = f["head"].unitsPerEm or 1000
        os2 = f["OS/2"] if "OS/2" in f else None
        xh = getattr(os2, "sxHeight", 0) or 0
        ch = getattr(os2, "sCapHeight", 0) or 0
        adv = [a for a, _ in f["hmtx"].metrics.values() if a > 0]
        return {"family": _family(f),
                "x_height_ratio": round(xh / ch, 4) if xh and ch else None,
                "avg_advance": round(sum(adv) / len(adv) / upm, 4) if adv else None,
                "weight": getattr(os2, "usWeightClass", None),
                "italic_angle": float(f["post"].italicAngle) if "post" in f else 0.0}
    except Exception:
        return None


def _scan(font_dirs):
    files = []
    for d in font_dirs or []:
        for root, _, names in os.walk(str(d)):
            files += [os.path.join(root, n) for n in sorted(names) if n.lower().endswith(FONT_EXT)]
    return files


def _closeness(a, b, span):
    if a is None or b is None:
        return None
    return max(0.0, 1.0 - abs(a - b) / span)


def font_candidates(role, src_font_metrics, profile, font_dirs=()):
    """Ranked [{family, file, score, why}] for `role`, closest to the source font first.

    Without fontTools the profile order is returned and `.not_run` contains "font_metrics"."""
    wanted = profile.get("fonts", {}).get(role, [])
    n = max(len(wanted), 1)
    have_ft = _fonttools() is not None
    if not have_ft or not src_font_metrics:
        why = "profile order" if have_ft is False else "profile order (no source font metrics)"
        items = [{"family": f, "file": None, "score": round(1.0 - i / n, 3), "why": why}
                 for i, f in enumerate(wanted)]
        return FontList(items, [] if have_ft else ["font_metrics"])
    s = src_font_metrics
    by_fam = {}
    for path in _scan(font_dirs):
        m = font_metrics(path)
        if not m:
            continue
        key = _norm(m["family"])
        for i, fam in enumerate(wanted):
            if _norm(fam) != key:
                continue
            parts = [(_closeness(s.get("x_height_ratio"), m["x_height_ratio"], 0.4), 0.4, "x-height ratio"),
                     (_closeness(s.get("weight"), m["weight"], 800.0), 0.3, "weight"),
                     (_closeness(s.get("avg_advance"), m["avg_advance"], 0.5), 0.2, "advance width"),
                     (_closeness(abs(s.get("italic_angle") or 0), abs(m["italic_angle"] or 0), 20.0), 0.1, "slant")]
            got = [(c, w, nm) for c, w, nm in parts if c is not None]
            tw = sum(w for _, w, _ in got)
            score = round((sum(c * w for c, w, _ in got) / tw if tw else 0.0) - i * 0.001, 4)
            why = "closeness on " + ", ".join(nm for _, _, nm in got) if got else "no comparable metrics"
            cand = {"family": fam, "file": path, "score": score, "why": why}
            if fam not in by_fam or score > by_fam[fam]["score"]:
                by_fam[fam] = cand
            break
    ranked = sorted(by_fam.values(), key=lambda c: -c["score"])
    for f in wanted:  # profile families not found on disk follow, in profile order
        if f not in by_fam:
            ranked.append({"family": f, "file": None, "score": 0.0, "why": "not installed in the font folders"})
    return FontList(ranked, [])


# ---------------------------------------------------------------- adapt
def adapt(pack, to_code, from_code=None, source_wps=None, target_wps=None, font_dirs=()):
    import store
    pack = Path(pack)
    if not pack.is_dir():
        raise ValueError(f"pack folder not found: {pack}")
    fp = store.read_json(pack / "fingerprint.json", {}) or {}
    from_code = from_code or fp.get("lang") or "en"
    src, dst = load_profile(from_code), load_profile(to_code)
    raw = store.read_json(pack / "laws.json", [])
    laws = raw.get("laws", []) if isinstance(raw, dict) else raw
    sw = float(source_wps) if source_wps else float(src["speech_wps_typical"])
    tw = float(target_wps) if target_wps else float(dst["speech_wps_typical"])
    ratio = round(sw / tw, 4)
    scaled = scale_timing(laws, ratio)
    n_scaled = sum(1 for a, b in zip(laws, scaled) if a != b)
    elements = fp.get("elements") or list(DEFAULT_ELEMENTS)
    src_metrics = fp.get("font_metrics") if isinstance(fp.get("font_metrics"), dict) else {}
    fonts, not_run = {}, []
    for role in ROLES:
        c = font_candidates(role, src_metrics.get(role), dst, font_dirs)
        fonts[role] = list(c)[:3]
        not_run += [x for x in c.not_run if x not in not_run]
    src_emph = (fp.get("caption") or {}).get("emphasis") or src["caption"].get("emphasis", [])
    out = {"from_lang": from_code, "to_lang": to_code, "timing_ratio": ratio, "laws_scaled": n_scaled,
           "laws": scaled, "mirror": mirror_plan(elements, dst["dir"], src["dir"]), "fonts": fonts,
           "caption_rules": dst["caption"], "emphasis_swaps": emphasis_swaps(src_emph, dst),
           "not_run": not_run, "recreation_needed": True}
    store.write_json(pack / f"adapt-{to_code}.json", out)
    return out


# ---------------------------------------------------------------- CLI
def main(argv=None):
    import store
    return store.exit_on_lock_timeout(_main, argv)


def _main(argv=None):
    import json
    import store
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show")
    s.add_argument("code")
    a = sub.add_parser("add")
    a.add_argument("code")
    a.add_argument("--from", dest="src", required=True)
    d = sub.add_parser("adapt")
    d.add_argument("pack")
    d.add_argument("--to", required=True)
    d.add_argument("--from-lang")
    d.add_argument("--source-wps", type=float)
    d.add_argument("--target-wps", type=float)
    d.add_argument("--font-dir", action="append", default=[])
    args = p.parse_args(argv)
    store.print_utf8()
    try:
        if args.cmd == "show":
            print(json.dumps(load_profile(args.code), ensure_ascii=False, indent=2))
        elif args.cmd == "add":
            try:
                data = json.loads(Path(args.src).read_bytes().decode("utf-8-sig"))
            except (OSError, ValueError) as e:
                print(f"cannot read {args.src}: {e}", file=sys.stderr)
                return 2
            _validate(data)
            if data["code"] != args.code:
                print(f"profile code '{data['code']}' does not match '{args.code}'", file=sys.stderr)
                return 2
            sp = _store_profile_path(args.code)
            with store.lock(sp):
                store.write_json(sp, data)
            print(f"saved {sp}")
        else:
            for w in (args.source_wps, args.target_wps):
                if w is not None and w <= 0:
                    print("wps must be positive", file=sys.stderr)
                    return 2
            out = adapt(args.pack, args.to, args.from_lang, args.source_wps, args.target_wps, args.font_dir)
            print(f"wrote {Path(args.pack) / ('adapt-' + args.to + '.json')} (timing ratio {out['timing_ratio']})")
    except (KeyError, ValueError) as e:
        print(str(e.args[0]) if isinstance(e, KeyError) and e.args else str(e), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
