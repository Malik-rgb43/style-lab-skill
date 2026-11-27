#!/usr/bin/env python3
"""B4: motion variants. One question for the whole video ("3 motion variants per beat, or 1 and I decide"),
then per beat 2-4 s draft clips of the picked frame that differ ONLY in motion: entrance, ease, timing and
camera, all taken from the style's measured motion (fingerprint) or, without one, from the craft defaults.

    motion.py mode <project> three|one [--beat N]            the one question (a beat override with --beat)
    motion.py variants <project> --beat N [--more] [--owner-ok-more --owner-words "..."]
    motion.py check <project> --beat N --set K
    motion.py render <project> --beat N --set K
    motion.py list <project> [--json]

Each variant folder motion/B03/s<k>/<a|b|c>/ is a copy of the picked frame variant plus motion-params.js
(window.SL_MOTION = {entrance, ease, timing, camera}); the beat's HTML reads it, so the design stays identical.
Exit codes: 0 ok, 1 a check failed or the budget is used up, 2 bad input, 3 HyperFrames missing.
"""
import argparse
import hashlib
import itertools
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402

STEP = "motion"
LETTERS = ("a", "b", "c")
MIN_S, MAX_S, DUR_TOL = 2.0, 4.0, 0.1
EASE_RMSE_MAX = 0.05
DEFAULT_FPS = 30.0
ENTRANCES = ("rise", "slide-up", "scale-up", "mask-wipe", "slide-left", "slide-right")
CRAFT_DEFAULTS = {
    "eases": [{"name": "expo.out", "bezier": [0.16, 1, 0.3, 1], "share": 0.5},
              {"name": "power4.out", "bezier": [0.25, 1, 0.5, 1], "share": 0.3},
              {"name": "back.out", "bezier": [0.34, 1.56, 0.64, 1], "share": 0.2}],
    "entrance_dur_s": {"median": 0.75, "p10": 0.6, "p90": 0.9},
    "exit_dur_s": {"median": 0.5, "p10": 0.45, "p90": 0.6, "ease": {"name": "power2.in", "bezier": [0.55, 0, 1, 0.45]}},
    "stagger_s": {"median": 0.1, "p10": 0.08, "p90": 0.12},
    "camera_moves": [{"prop": "scale", "from": 1.0, "to": 1.1, "ease": "sine.inOut", "dur_s": 3.0}],
}
LINEAR = (0.0, 0.0, 1.0, 1.0)


class BadInput(ValueError):
    pass


class CheckFailed(Exception):
    pass


# ---------------------------------------------------------------- palette

def _stat(v):
    if isinstance(v, dict) and isinstance(v.get("median"), (int, float)):
        med = float(v["median"])
        return {"median": med, "p10": float(v.get("p10", med) or med), "p90": float(v.get("p90", med) or med)}
    return None


def custom_ease(bez):
    x1, y1, x2, y2 = (round(float(b), 4) for b in bez)
    return f"M0,0 C{x1:g},{y1:g} {x2:g},{y2:g} 1,1"


def _is_linear(bez):
    return all(abs(a - b) < 0.02 for a, b in zip(bez, LINEAR))


def motion_palette(fp):
    m = (((fp or {}).get("layers") or {}).get("motion") or {}) if isinstance(fp, dict) else {}
    eases = []
    for e in m.get("eases") or []:
        bez = e.get("bezier") if isinstance(e, dict) else None
        if not (isinstance(bez, (list, tuple)) and len(bez) == 4):
            continue
        share = float(e.get("share") or 0)
        if (_is_linear(bez) or str(e.get("name")).lower() in ("linear", "none")) and share < 0.30:
            continue
        eases.append({"name": e.get("name") or "custom", "bezier": [float(b) for b in bez],
                      "custom_ease": e.get("custom_ease") or custom_ease(bez), "share": share})
    eases.sort(key=lambda e: -e["share"])
    if not eases:
        pal = json.loads(json.dumps(CRAFT_DEFAULTS))
        for e in pal["eases"]:
            e["custom_ease"] = custom_ease(e["bezier"])
        pal["curve_source"] = "craft-defaults"
        return pal
    metrics = m.get("metrics") or {}
    fps = DEFAULT_FPS
    st = ((((fp.get("layers") or {}).get("structure") or {}).get("metrics") or {}).get("unique_fps") or {})
    if isinstance(st, dict) and isinstance(st.get("median"), (int, float)) and st["median"] > 1:
        fps = float(st["median"])
    ent = _stat(metrics.get("entrance_dur_s"))
    if ent is None and _stat(metrics.get("dur_f")):
        ent = {k: round(v / fps, 4) for k, v in _stat(metrics["dur_f"]).items()}
    cams = []
    for c in m.get("camera_moves") or []:
        if isinstance(c, dict) and c.get("prop") in ("scale", "x", "y"):
            cams.append({"prop": c["prop"], "from": c.get("from", 1.0 if c["prop"] == "scale" else 0),
                         "to": c.get("to", 1.08 if c["prop"] == "scale" else -40),
                         "ease": c.get("ease") or eases[0]["name"], "dur_s": (_stat(c.get("dur_s")) or {}).get("median", 3.0)})
    if not cams and _stat(metrics.get("camera_move_dur_f")):
        cams = [{"prop": "scale", "from": 1.0, "to": 1.08, "ease": eases[0]["name"],
                 "dur_s": round(_stat(metrics["camera_move_dur_f"])["median"] / fps, 3)}]
    return {"eases": eases, "entrance_dur_s": ent or dict(CRAFT_DEFAULTS["entrance_dur_s"]),
            "exit_dur_s": _stat(metrics.get("exit_dur_s")) or dict(CRAFT_DEFAULTS["exit_dur_s"]),
            "stagger_s": _stat(metrics.get("stagger_s")) or dict(CRAFT_DEFAULTS["stagger_s"]),
            "camera_moves": cams or json.loads(json.dumps(CRAFT_DEFAULTS["camera_moves"])),
            "curve_source": "fingerprint"}


def _bez_rmse(a, b):
    return (sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) / 4) ** 0.5


# ---------------------------------------------------------------- window and variants

def window(beat_start, beat_end, clip_len):
    """(t0, t1, warning): 2.0-4.0 s around the beat, forward first, back at the clip end, inside the clip."""
    if clip_len < MIN_S:
        return 0.0, round(float(clip_len), 3), "the clip is shorter than 2 s: the whole clip is the window"
    length = min(MAX_S, max(MIN_S, float(beat_end) - float(beat_start)))
    t0 = max(0.0, float(beat_start))
    t1 = t0 + length
    if t1 > clip_len:
        t1 = float(clip_len)
        t0 = max(0.0, t1 - length)
    return round(t0, 3), round(t1, 3), None


def _cams(pal):
    base = [{"prop": "none"}]
    for c in pal["camera_moves"]:
        base.append(dict(c))
    have = {c["prop"] for c in base}
    for prop, fr, to in (("scale", 1.0, 1.08), ("y", 0, -40), ("x", 0, 40)):
        if prop not in have:
            base.append({"prop": prop, "from": fr, "to": to, "ease": pal["eases"][0]["name"], "dur_s": 3.0})
    return base


def _q05(x):
    return round(round(float(x) / 0.05) * 0.05, 2)


def mv_sig(v):
    import project
    return project.sig(STEP, v["entrance"], v["ease"]["name"], _q05(v["timing"]["dur_s"]), v["camera"]["prop"])


def _aspects(v):
    return (v["entrance"], v["ease"]["name"], _q05(v["timing"]["dur_s"]), v["camera"]["prop"])


def _differ(a, b):
    return sum(x != y for x, y in zip(_aspects(a), _aspects(b))) >= 2


def _variant(entrance, ease, dur, stagger, cam, delay, wild, src):
    return {"entrance": entrance, "ease": {"name": ease["name"], "bezier": ease["bezier"],
                                            "custom_ease": ease["custom_ease"]},
            "timing": {"dur_s": round(dur, 3), "delay_s": round(delay, 3), "stagger_s": round(stagger, 3)},
            "camera": cam, "wild": wild, "curve_source": src}


def _wild_ease(pal):
    over = [e for e in pal["eases"] if max(e["bezier"][1], e["bezier"][3]) > 1.0]
    if over:
        return over[0]
    first = pal["eases"][0]
    return max(pal["eases"], key=lambda e: _bez_rmse(e["bezier"], first["bezier"]))


def make_variants(pal, set_no, blocked, entrance_hint="rise", delay_p90=0.0):
    """Three motion variants (a = the style median, b = tighter, c = the wild card); later sets rotate."""
    eases, d, st = pal["eases"], pal["entrance_dur_s"], pal["stagger_s"]
    cams = _cams(pal)
    hint = entrance_hint if entrance_hint in ENTRANCES else ENTRANCES[0]
    ents = [hint] + [e for e in ENTRANCES if e != hint]
    src = pal["curve_source"]
    e2 = eases[1] if len(eases) > 1 else eases[0]
    plan = [_variant(ents[0], eases[0], d["median"], st["median"], cams[min(1, len(cams) - 1)] if
                     len(cams) > 1 and src == "fingerprint" else cams[0], 0.0, False, src),
            _variant(ents[1], e2, max(d["p10"], 0.25), st["p10"], None, 0.0, False, src),
            _variant(ents[2], _wild_ease(pal), d["p90"], st["p90"], None, delay_p90, True, src)]
    plan[1]["camera"] = next(c for c in cams if c["prop"] != plan[0]["camera"]["prop"])
    plan[2]["camera"] = next((c for c in cams if c["prop"] not in (plan[0]["camera"]["prop"], plan[1]["camera"]["prop"])),
                             cams[-1])
    pool = []
    if set_no == 1:
        pool.extend(plan)
    durs = [d["median"], max(d["p10"], 0.25), d["p90"]]
    combos = itertools.product(ents, eases, durs, cams)
    rot = list(combos)
    k = (set_no - 1) * 7 % max(1, len(rot))
    for ent, ease, dur, cam in rot[k:] + rot[:k]:
        pool.append(_variant(ent, ease, dur, st["median"], cam, 0.0, False, src))
    chosen = []
    for v in pool:
        if mv_sig(v) in blocked or any(mv_sig(v) == mv_sig(c) for c in chosen):
            continue
        if all(_differ(v, c) for c in chosen):
            chosen.append(json.loads(json.dumps(v)))
        if len(chosen) == 3:
            break
    for i, v in enumerate(chosen):
        v["wild"] = i == 2
        v["sig"] = mv_sig(v)
    return chosen


# ---------------------------------------------------------------- project

def _ctx(d):
    import project
    import styles
    try:
        entry = project.load(d)
    except KeyError as e:
        raise BadInput(str(e.args[0] if e.args else e))
    st = styles.find(styles.load_registry(), entry.get("style") or "") or {}
    pack = Path(st["pack"]) if str(st.get("pack") or "").strip() and Path(st["pack"]).is_dir() else None
    fp = store.read_json(pack / "fingerprint.json", None) if pack else None
    return {"entry": entry, "pack": pack, "fp": fp if isinstance(fp, dict) else None}


def _mv_path(d):
    return Path(d) / "motion-variants.json"


def _empty():
    return {"version": 1, "mode": None, "overrides": {}, "beats": {}}


def set_mode(d, mode, beat=None):
    import picks
    if mode not in ("three", "one"):
        raise BadInput("mode must be three or one")

    def go(doc):
        for k, v in _empty().items():
            doc.setdefault(k, v)
        if beat is None:
            doc["mode"] = mode
        else:
            doc["overrides"][str(int(beat))] = mode
    store.update_json(_mv_path(d), _empty(), go)
    if beat is None:
        picks.set_motion_mode(d, mode)
    else:
        picks.set_motion_mode(d, None, {str(int(beat)): mode})


def _mode_for(d, beat):
    doc = store.read_json(_mv_path(d), _empty()) or _empty()
    return (doc.get("overrides") or {}).get(str(int(beat))) or doc.get("mode")


def _frame_pick(d, beat):
    import picks
    e = picks.load(d)["frames"].get(str(int(beat))) or {}
    if not e.get("picked"):
        return None
    for _, v in picks.options(d, "frames", beat):
        if v.get("id") == e["picked"]:
            return v
    return None


def _law_lag(d, cx, beat):
    import plan as planmod
    try:
        b = next(x for x in store.read_json(Path(d) / "plan.json", {}).get("beats", []) if int(x["n"]) == int(beat))
        trig = planmod.beat_triggers(planmod._words_in(d, b), "he", b.get("beat_types") or [])
        laws = planmod.fired_laws(planmod.load_laws(cx["pack"]), trig)
        return float(((laws[0].get("lag") or {}).get("p90")) or 0.0) if laws else 0.0
    except Exception:
        return 0.0


def _design_hash(vd):
    parts = []
    for n in ("index.html", "tokens.css"):
        p = vd / n
        parts.append(p.read_text(encoding="utf-8").replace("\r\n", "\n") if p.is_file() else "")
    return hashlib.sha1("\n--\n".join(parts).encode("utf-8")).hexdigest()


def _params_js(v):
    keep = {k: v[k] for k in ("entrance", "ease", "timing", "camera")}
    return "window.SL_MOTION = " + json.dumps(keep, ensure_ascii=False) + ";\n"


def _read_params(vd):
    p = vd / "motion-params.js"
    if not p.is_file():
        return None
    m = re.search(r"window\.SL_MOTION\s*=\s*(\{.*\})\s*;?\s*$", p.read_text(encoding="utf-8"), re.S)
    try:
        return json.loads(m.group(1)) if m else None
    except ValueError:
        return None


def _set_dir(d, beat, n):
    return Path(d) / "motion" / f"B{int(beat):02d}" / f"s{int(n)}"


def variants(d, beat, more=False, owner_ok_more=False, owner_words=None):
    import picks
    import project
    cx = _ctx(d)
    fv = _frame_pick(d, beat)
    if fv is None:
        raise CheckFailed(f"B{beat}: no frame pick yet; motion comes after the frames pick")
    plan = store.read_json(Path(d) / "plan.json", {}) or {}
    b = next((x for x in plan.get("beats", []) if int(x["n"]) == int(beat)), None)
    if b is None:
        raise BadInput(f"no beat {beat} in plan.json")
    clip_len = float((plan.get("clip") or {}).get("length_s") or plan["beats"][-1]["end"])
    t0, t1, warn = window(b["start"], b["end"], clip_len)
    pal = motion_palette(cx["fp"])
    mode = _mode_for(d, beat) or "three"
    n = project.next_set(d, STEP, int(beat), owner_ok_more, owner_words)
    blocked = set(project.blocked(d, STEP, int(beat)))
    vs = make_variants(pal, n, blocked, (fv.get("entrance") or "rise"), _law_lag(d, cx, beat))
    if mode == "one":
        vs = vs[:1]
    src = Path(d) / fv["dir"]
    sd = _set_dir(d, beat, n)
    out = []
    for x, v in zip(LETTERS, vs):
        vd = sd / x
        if vd.exists():
            shutil.rmtree(vd)
        shutil.copytree(src, vd, ignore=shutil.ignore_patterns(".hyperframes", "*.png", "*.mp4"))
        html = (vd / "index.html").read_text(encoding="utf-8")
        html = re.sub(r'(<[^>]*data-composition-id="[^"]*"[^>]*?data-duration=")[^"]*(")',
                      lambda m: m.group(1) + f"{t1 - t0:g}" + m.group(2), html, count=1)
        (vd / "index.html").write_text(html, encoding="utf-8")
        (vd / "motion-params.js").write_text(_params_js(v), encoding="utf-8")
        v.update(id=f"B{int(beat):02d}-m{n}-{x}", design_hash=_design_hash(vd), clip=None,
                 dir=vd.relative_to(d).as_posix(), frame=fv["id"])
        out.append(vd)

    def go(doc):
        for k, val in _empty().items():
            doc.setdefault(k, val)
        bt = doc["beats"].setdefault(str(int(beat)), {"window": [t0, t1], "sets": []})
        bt["window"] = [t0, t1]
        bt["sets"] = [s for s in bt["sets"] if int(s["set"]) != n] + [{"set": n, "variants": vs}]
    store.update_json(_mv_path(d), _empty(), go)
    if mode == "one":
        picks.pick(d, "motion", beat, vs[0]["id"], by="claude", source="chat")
    return n, out, pal["curve_source"], warn


def _set(d, beat, n):
    doc = store.read_json(_mv_path(d), _empty()) or _empty()
    for s in (doc.get("beats", {}).get(str(int(beat))) or {}).get("sets", []):
        if int(s["set"]) == int(n):
            return s
    raise BadInput(f"no motion set {n} for beat {beat}")


def check_variants(d, beat, set_no):
    import project
    cx = _ctx(d)
    pal = motion_palette(cx["fp"])
    s = _set(d, beat, set_no)
    probs = []
    vds = [Path(d) / v["dir"] for v in s["variants"]]
    hashes = {_design_hash(vd) for vd in vds}
    if len(hashes) > 1:
        probs.append(f"B{beat} s{set_no}: the variants' design differs (index.html or tokens.css changed); "
                     "only motion may differ")
    shown = store.read_json(Path(d) / "shown.json", {}).get("items", []) or []
    own = {it.get("sig") for it in shown if it.get("step") == STEP and int(it.get("beat", -1)) == int(beat)
           and int(it.get("set", 0)) == int(set_no)}
    blocked = set(project.blocked(d, STEP, int(beat))) - own
    live = []
    for v, vd in zip(s["variants"], vds):
        tag = f"B{beat} s{set_no} {v['id'][-1]}"
        html = (vd / "index.html").read_text(encoding="utf-8") if (vd / "index.html").is_file() else ""
        if "motion-params.js" not in html:
            probs.append(f"{tag}: index.html does not read motion-params.js")
        prm = _read_params(vd)
        if prm is None:
            probs.append(f"{tag}: motion-params.js is missing or not readable")
            continue
        bez = (prm.get("ease") or {}).get("bezier")
        name = str((prm.get("ease") or {}).get("name") or "")
        if not (isinstance(bez, list) and len(bez) == 4) or name.lower() in ("linear", "none") or \
                min(_bez_rmse(bez, e["bezier"]) for e in pal["eases"]) > EASE_RMSE_MAX:
            probs.append(f"{tag}: ease {name or bez} is not one of the style's eases")
        cur = {"entrance": prm.get("entrance"), "ease": prm.get("ease") or {}, "timing": prm.get("timing") or {},
               "camera": prm.get("camera") or {"prop": "none"}}
        try:
            sg = mv_sig(cur)
        except (KeyError, TypeError):
            probs.append(f"{tag}: motion-params.js lacks entrance, ease, timing or camera")
            continue
        if sg in blocked:
            probs.append(f"{tag}: repeats a motion already shown or rejected ({sg})")
        live.append((tag, cur))
    for (ta, a), (tb, b) in itertools.combinations(live, 2):
        if not _differ(a, b):
            probs.append(f"{ta}/{tb[-1]}: differ in fewer than 2 of entrance, ease, duration, camera")
    return probs


def render(d, beat, set_no):
    import hf
    import imgmetrics
    import project
    probs = check_variants(d, beat, set_no)
    if probs:
        raise CheckFailed("\n".join(probs))
    s = _set(d, beat, set_no)
    sd = _set_dir(d, beat, set_no)
    clips, bad = [], []
    for v in s["variants"]:
        vd = Path(d) / v["dir"]
        out = sd / f"{v['id'][-1]}.mp4"
        hf.render_draft(vd, out)
        dur = imgmetrics._duration(out)
        if not (MIN_S - DUR_TOL <= dur <= MAX_S + DUR_TOL):
            bad.append(f"{v['id']}: the draft is {dur:.2f} s, outside 2.0-4.0 s")
        v["clip"] = out.relative_to(d).as_posix()
        clips.append(out)
    if bad:
        raise CheckFailed("\n".join(bad))

    def go(doc):
        for st in doc["beats"][str(int(beat))]["sets"]:
            if int(st["set"]) == int(set_no):
                st["variants"] = s["variants"]
    store.update_json(_mv_path(d), _empty(), go)
    project.record_shown(d, STEP, int(beat), int(set_no), s["variants"])
    return clips


# ---------------------------------------------------------------- CLI

def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("mode"); a.add_argument("project"); a.add_argument("mode", choices=("three", "one"))
    a.add_argument("--beat", type=int)
    a = sub.add_parser("variants"); a.add_argument("project"); a.add_argument("--beat", type=int, required=True)
    a.add_argument("--more", action="store_true"); a.add_argument("--owner-ok-more", action="store_true")
    a.add_argument("--owner-words")
    for name in ("check", "render"):
        a = sub.add_parser(name); a.add_argument("project"); a.add_argument("--beat", type=int, required=True)
        a.add_argument("--set", type=int, required=True)
    a = sub.add_parser("list"); a.add_argument("project"); a.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    store.print_utf8()
    import hf
    import project
    try:
        if not Path(args.project).is_dir():
            raise BadInput(f"project folder not found: {args.project}")
        if args.cmd == "mode":
            set_mode(args.project, args.mode, args.beat)
            print("נשמר: " + ("3 וריאציות תנועה" if args.mode == "three" else "וריאציה אחת, Claude מחליט")
                  + (f" לביט {args.beat}" if args.beat else " לכל הסרטון"))
            return 0
        if args.cmd == "variants":
            n, dirs, src, warn = variants(args.project, args.beat, args.more, args.owner_ok_more, args.owner_words)
            if src == "craft-defaults":
                print("לסגנון אין מדידת תנועה: העקומות מברירות המחדל של המלאכה (לא ליניארי)")
            if warn:
                print(warn)
            print(f"סט {n} לביט {args.beat}: " + ", ".join(p.relative_to(args.project).as_posix() for p in dirs))
            return 0
        if args.cmd == "check":
            probs = check_variants(args.project, args.beat, args.set)
            for p in probs:
                print(p)
            return 1 if probs else 0
        if args.cmd == "render":
            clips = render(args.project, args.beat, args.set)
            print("\n".join(p.relative_to(args.project).as_posix() for p in clips))
            return 0
        doc = store.read_json(_mv_path(args.project), _empty()) or _empty()
        if args.json:
            print(json.dumps(doc, ensure_ascii=False, indent=2))
        else:
            print(f"mode: {doc.get('mode')}  overrides: {doc.get('overrides')}")
            for b, bt in sorted(doc.get("beats", {}).items(), key=lambda kv: int(kv[0])):
                for s in bt.get("sets", []):
                    print(f"B{b} s{s['set']} {bt['window']}: " + ", ".join(v["id"] for v in s["variants"]))
        return 0
    except hf.HFMissing as e:
        print(str(e), file=sys.stderr)
        return 3
    except project.BudgetExceeded as e:
        print(str(e), file=sys.stderr)
        return 1
    except CheckFailed as e:
        print(str(e), file=sys.stderr)
        return 1
    except (BadInput, project._Bad) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def main(argv=None):
    return store.exit_on_lock_timeout(_main, argv)


if __name__ == "__main__":
    sys.exit(main())
