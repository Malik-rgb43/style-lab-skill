#!/usr/bin/env python3
"""Proof of a trained style: hold-out scoring, recreation scoring, motion QA, graph-editor view.

The `ready` gate in styles.py reads <pack>/proof.json written here:
    {holdout: {ids, scores, passed}, recreation: {moments: [{id,t0,t1,scores,owner_ok}], passed}}

Commands:
    prove.py holdout-pick <pack> [--n 3] [--seed 7] [--ids a,b]
        pick hold-out video ids (never ones the fingerprint trained on) and record them
    prove.py holdout <pack> --predicted <dir>
        <dir>/<id>.events.json per hold-out video (written by Claude from the transcript and the laws),
        scored against <pack>/analysis/<id>/events.json, or, without it, the events measured in
        analysis/<id>/ (fingerprint.events, word triggers left out)
    prove.py recreation <pack> --moment <id>:<t0>-<t1> --build <video or motion.json> [--owner-ok]
        compare the build's moves in the window with the reference's. The script never sets owner_ok
        by itself: --owner-ok is passed only after the owner said yes.
    prove.py qa <pack> --build <video or motion.json> [--out <json>]
        flag build moves outside the style's spread
    prove.py graph <motion.json> [--compare <motion.json>] --out <html>
        value + speed graphs, reference and build overlaid (self-contained, light/dark, RTL-safe)

Exit: 0 ok, 1 check failed, 2 bad input, 3 missing measurement/tool. Inputs are read-only.
"""
import argparse
import html
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402
from styles import trained_hash  # noqa: E402

HOLDOUT_MIN_F1 = 0.6
RECREATION_MAX_RMSE = 0.05
QA_EASE_MAX_RMSE = 0.05
DEFAULT_TOL_S = 0.3
DEFAULT_DUR_RATIO = (0.8, 1.25)  # build/ref duration allowed when no style spread is known
PEAK_AT_TOL = 0.15
SAMPLES = 41


class Bad(Exception):
    def __init__(self, msg, rc=2):
        super().__init__(msg)
        self.rc = rc


# ---------------------------------------------------------------- pure functions

def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def score_placement(pred, actual, tol_s=DEFAULT_TOL_S):
    """Match predicted to actual events of the same type within tol_s (nearest first, one-to-one).
    Returns {per_type: {type: {precision, recall, f1, tp, n_pred, n_actual}}, timing_mae_s, overall_f1,
    tp, n_pred, n_actual, abs_err_sum}. overall_f1 pools all types; timing_mae_s is None with no match."""
    pred = [e for e in (pred or []) if isinstance(e, dict) and _num(e.get("t"))]
    actual = [e for e in (actual or []) if isinstance(e, dict) and _num(e.get("t"))]
    types = sorted({str(e.get("type")) for e in pred} | {str(e.get("type")) for e in actual})
    per, tp_all, errs = {}, 0, []
    for ty in types:
        p = [e["t"] for e in pred if str(e.get("type")) == ty]
        a = [e["t"] for e in actual if str(e.get("type")) == ty]
        pairs = sorted((abs(pt - at), i, j) for i, pt in enumerate(p) for j, at in enumerate(a)
                       if abs(pt - at) <= tol_s)
        used_p, used_a, tp = set(), set(), 0
        for d, i, j in pairs:
            if i in used_p or j in used_a:
                continue
            used_p.add(i)
            used_a.add(j)
            tp += 1
            errs.append(d)
        prec = tp / len(p) if p else 0.0
        rec = tp / len(a) if a else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[ty] = {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "n_pred": len(p), "n_actual": len(a)}
        tp_all += tp
    return _overall(per, tp_all, len(pred), len(actual), errs)


def _overall(per, tp, n_pred, n_actual, errs):
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_actual if n_actual else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"per_type": per, "timing_mae_s": (sum(errs) / len(errs)) if errs else None, "overall_f1": f1,
            "tp": tp, "n_pred": n_pred, "n_actual": n_actual, "abs_err_sum": sum(errs)}


def pool_scores(scores):
    """Pool per-video score_placement results into one overall result (micro average)."""
    tp = sum(s.get("tp", 0) for s in scores)
    npred = sum(s.get("n_pred", 0) for s in scores)
    nact = sum(s.get("n_actual", 0) for s in scores)
    err_sum = sum(s.get("abs_err_sum", 0.0) for s in scores)
    out = _overall({}, tp, npred, nact, [err_sum / tp] * tp if tp else [])
    out.pop("per_type")
    return out


def bezier_y(bez, x):
    """y of the cubic bezier (0,0)(x1,y1)(x2,y2)(1,1) at time x in 0..1."""
    x1, y1, x2, y2 = bez
    x = min(1.0, max(0.0, x))
    lo, hi = 0.0, 1.0
    for _ in range(40):
        t = (lo + hi) / 2
        bx = 3 * (1 - t) ** 2 * t * x1 + 3 * (1 - t) * t * t * x2 + t ** 3
        if bx < x:
            lo = t
        else:
            hi = t
    t = (lo + hi) / 2
    return 3 * (1 - t) ** 2 * t * y1 + 3 * (1 - t) * t * t * y2 + t ** 3


def _bez(m):
    b = (m or {}).get("bezier")
    return list(b) if isinstance(b, (list, tuple)) and len(b) == 4 and all(_num(v) for v in b) else None


def curve_rmse(b1, b2):
    xs = [i / (SAMPLES - 1) for i in range(SAMPLES)]
    return math.sqrt(sum((bezier_y(b1, x) - bezier_y(b2, x)) ** 2 for x in xs) / len(xs))


def _metric(spread, name):
    """(p10, p90) of a motion metric from a fingerprint (or a bare metrics dict), else None."""
    if not isinstance(spread, dict):
        return None
    metrics = ((spread.get("layers") or {}).get("motion") or {}).get("metrics") if "layers" in spread else spread
    m = (metrics or {}).get(name)
    if not isinstance(m, dict) or m.get("n") == 0 or not (_num(m.get("p10")) and _num(m.get("p90"))):
        return None
    return m["p10"], m["p90"]


def compare_moves(ref, build, spread=None):
    """{curve_rmse, dur_ratio, within_spread}. spread: a fingerprint (or its motion metrics dict); when
    given, the build duration must sit inside the style's dur_f p10-p90, else within 0.8x-1.25x of ref."""
    rb, bb = _bez(ref), _bez(build)
    rmse = curve_rmse(rb, bb) if rb and bb else None
    rd, bd = (ref or {}).get("dur_f"), (build or {}).get("dur_f")
    ratio = (bd / rd) if _num(rd) and _num(bd) and rd > 0 else None
    ok = ratio is not None
    band = _metric(spread, "dur_f")
    if ok and band:
        ok = band[0] <= bd <= band[1]
    elif ok:
        ok = DEFAULT_DUR_RATIO[0] <= ratio <= DEFAULT_DUR_RATIO[1]
    rp, bp = (ref or {}).get("peak_at"), (build or {}).get("peak_at")
    if ok and _num(rp) and _num(bp):
        ok = abs(rp - bp) <= PEAK_AT_TOL
    return {"curve_rmse": rmse, "dur_ratio": ratio, "within_spread": bool(ok and rmse is not None)}


def all_moves(motion):
    """Flatten motion.json: camera_moves then elements[].moves, ordered by t0."""
    out = []
    for m in (motion or {}).get("camera_moves") or []:
        if isinstance(m, dict):
            out.append({**m, "prop": m.get("prop", "camera")})
    for el in (motion or {}).get("elements") or []:
        if not isinstance(el, dict):
            continue
        for m in el.get("moves") or []:
            if isinstance(m, dict):
                out.append({**m, "prop": m.get("prop") or el.get("name") or el.get("id") or "element"})
    return sorted(out, key=lambda m: m["t0"] if _num(m.get("t0")) else 0.0)


def _style_eases(fp):
    mo = ((fp or {}).get("layers") or {}).get("motion") or {}
    found = []

    def take(v):
        if isinstance(v, list) and len(v) == 4 and all(_num(x) for x in v):
            found.append(list(v))
        elif isinstance(v, list):
            for x in v:
                take(x)
        elif isinstance(v, dict):
            if "bezier" in v:
                take(v["bezier"])
            else:
                for x in v.values():
                    take(x)
    for key in ("eases", "ease", "ease_set"):
        if key in mo:
            take(mo[key])
    return found


def qa_moves(build_motion, fingerprint):
    """Flag build moves outside the style spread: dur_f or peak_at outside p10-p90, or an ease further
    than QA_EASE_MAX_RMSE from every style ease. Returns [{t, prop, issue}]."""
    issues = []
    eases = _style_eases(fingerprint)
    for m in all_moves(build_motion):
        t, prop = m.get("t0"), m.get("prop")
        for name in ("dur_f", "peak_at"):
            band = _metric(fingerprint, name)
            v = m.get(name)
            if band and _num(v) and not band[0] <= v <= band[1]:
                issues.append({"t": t, "prop": prop,
                               "issue": f"{name} {v:g} outside the style spread {band[0]:g}-{band[1]:g}"})
        b = _bez(m)
        if eases and b:
            best = min(curve_rmse(b, e) for e in eases)
            if best > QA_EASE_MAX_RMSE:
                issues.append({"t": t, "prop": prop,
                               "issue": f"ease differs from the nearest style ease by rmse {best:.3f} "
                                        f"(limit {QA_EASE_MAX_RMSE})"})
    return issues


# ---------------------------------------------------------------- graph view

def _curve(bez, kind):
    xs = [i / (SAMPLES - 1) for i in range(SAMPLES)]
    ys = [bezier_y(bez, x) for x in xs]
    if kind == "speed":
        ys = [(ys[min(i + 1, len(ys) - 1)] - ys[max(i - 1, 0)]) /
              (xs[min(i + 1, len(xs) - 1)] - xs[max(i - 1, 0)]) for i in range(len(xs))]
    return xs, ys


def _svg(label, series, kind):
    """series: [(css class, bezier or None)]. One inline <svg>; text labels live outside it (dir=auto)."""
    w, h, pad = 320, 160, 8
    curves = [(c, _curve(b, kind)) for c, b in series if b]
    allys = [y for _, (_, ys) in curves for y in ys] or [0.0, 1.0]
    lo, hi = min(0.0, min(allys)), max(1.0, max(allys))
    span = (hi - lo) or 1.0

    def px(x):
        return pad + x * (w - 2 * pad)

    def py(y):
        return h - pad - (y - lo) / span * (h - 2 * pad)

    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(label)}" class="plot">',
             f'<line class="grid" x1="{pad}" y1="{py(0):.1f}" x2="{w - pad}" y2="{py(0):.1f}"/>']
    if kind == "value":
        parts.append(f'<line class="grid" x1="{pad}" y1="{py(1):.1f}" x2="{w - pad}" y2="{py(1):.1f}"/>')
    for cls, (xs, ys) in curves:
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
        parts.append(f'<polyline class="{cls}" points="{pts}"/>')
    parts.append("</svg>")
    return "".join(parts)


CSS = """
:root{--bg:#fff;--fg:#1a1a1f;--muted:#5d5d68;--grid:#d4d4dc;--ref:#1f5fd1;--build:#c2410c;--card:#f5f5f8}
@media (prefers-color-scheme:dark){:root{--bg:#14141a;--fg:#ececf2;--muted:#a3a3b2;--grid:#3a3a46;--ref:#7fb0ff;--build:#ffa66b;--card:#1d1d26}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:16px}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px;color:var(--muted)}
.key{display:inline-block;width:28px;height:0;border-top:3px solid;vertical-align:middle;margin-inline-end:6px}
.key.ref{border-color:var(--ref)}.key.build{border-color:var(--build);border-top-style:dashed}
.move{background:var(--card);border-radius:8px;padding:12px;margin:0 0 12px}
.move h2{font-size:15px;margin:0 0 8px;unicode-bidi:plaintext}
.row{display:flex;gap:12px;flex-wrap:wrap}.cell{flex:1 1 280px;direction:ltr}.cell small{color:var(--muted);unicode-bidi:plaintext}
.plot{width:100%;height:auto;display:block}
.grid{stroke:var(--grid);stroke-width:1}
polyline{fill:none;stroke-width:2.5;stroke-linejoin:round}
polyline.ref{stroke:var(--ref)}polyline.build{stroke:var(--build);stroke-dasharray:6 4}
"""


def pair_moves(moves, other):
    """Align `other` to `moves`: each move gets the nearest-t0 unused move with the same prop, or None."""
    used, out = set(), []
    for m in moves:
        best, bj = None, None
        for j, o in enumerate(other):
            if j in used or o.get("prop") != m.get("prop"):
                continue
            d = abs((o.get("t0") or 0.0) - (m.get("t0") or 0.0))
            if best is None or d < best:
                best, bj = d, j
        if bj is not None:
            used.add(bj)
        out.append(other[bj] if bj is not None else None)
    return out


def graph_html(moves, compare=None):
    """Self-contained HTML: per move, a value-graph <svg> and a speed-graph <svg> with the reference
    (solid) and the build from `compare` (dashed) overlaid. No network, light/dark, RTL-safe."""
    moves = list(moves or [])
    compare = list(compare) if compare else []
    out = ['<!doctype html><html lang="en" dir="auto"><head><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width,initial-scale=1">',
           f"<title>Motion graphs</title><style>{CSS}</style></head><body>",
           '<div class="legend"><span><i class="key ref"></i>reference</span>']
    if compare:
        out.append('<span><i class="key build"></i>build</span>')
    out.append("</div>")
    if not moves:
        out.append("<p>no moves</p>")
    for i, m in enumerate(moves):
        c = compare[i] if i < len(compare) else None
        title = f"{m.get('prop', 'move')} @ {m['t0']:g}s" if _num(m.get("t0")) else str(m.get("prop", "move"))
        info = ", ".join(f"{k} {m[k]:g}" for k in ("dur_f", "peak_at", "overshoot") if _num(m.get(k)))
        series = [("ref", _bez(m))] + ([("build", _bez(c))] if c else [])
        out.append(f'<section class="move"><h2 dir="auto">{html.escape(title)}</h2>')
        if info:
            out.append(f'<div class="cell"><small dir="auto">{html.escape(info)}</small></div>')
        out.append('<div class="row">')
        for kind in ("value", "speed"):
            out.append(f'<div class="cell"><small>{kind}</small>{_svg(f"{title} {kind}", series, kind)}</div>')
        out.append("</div></section>")
    out.append("</body></html>")
    return "".join(out)


# ---------------------------------------------------------------- files / CLI

def _read(path, what):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise Bad(f"{what} not found: {path}")
    except (OSError, ValueError) as e:
        raise Bad(f"{what} unreadable ({path}): {e}")


def _read_dict(path, what):
    d = _read(path, what)
    if not isinstance(d, dict):
        raise Bad(f"{what} must be a JSON object: {path}")
    return d


def _events(data):
    if isinstance(data, dict):
        data = data.get("events", [])
    return data if isinstance(data, list) else []


def measured_events(adir):
    """The actual events of one analysis folder from its measurements (fingerprint.events, word triggers
    left out: the prediction is made from the transcript, so words would be a free match). None without
    measure.json."""
    import fingerprint
    if not (Path(adir) / "measure.json").exists():
        return None
    a = fingerprint.load_analysis(adir)
    return [{"t": e["t"], "type": e["type"]} for e in fingerprint.events(a, words=False)]


def _fp(pack):
    p = Path(pack) / "fingerprint.json"
    if not p.exists():
        return {}
    return _read_dict(p, "fingerprint.json")


def _proof_path(pack):
    return Path(pack) / "proof.json"


def _update_proof(pack, fn):
    store.update_json(_proof_path(pack), {}, fn)


def _check_pack(pack):
    if not Path(pack).is_dir():
        raise Bad(f"pack folder not found: {pack}")


def _refuse_overlap(ids, fp):
    ov = sorted(set(ids) & set(fp.get("trained_on") or []))
    if ov:
        raise Bad("hold-out overlaps training (scoring refused): " + ", ".join(ov))


def cmd_holdout_pick(a):
    _check_pack(a.pack)
    fp = _fp(a.pack)
    no_fp = not (Path(a.pack) / "fingerprint.json").exists()
    if no_fp and not a.allow_no_fingerprint:
        raise Bad("no fingerprint.json; run fingerprint.py first, or pass --allow-no-fingerprint")
    if a.ids:
        ids = [x.strip() for x in a.ids.split(",") if x.strip()]
        _refuse_overlap(ids, fp)
    else:
        adir = Path(a.pack) / "analysis"
        trained = set(fp.get("trained_on") or [])
        cands = sorted(d.name for d in adir.iterdir() if d.is_dir()) if adir.is_dir() else []
        cands = [c for c in cands if c not in trained]
        if len(cands) < a.n:
            raise Bad(f"need {a.n} hold-out videos, only {len(cands)} analysed and not trained on")
        ids = sorted(random.Random(a.seed).sample(cands, a.n))
    if not ids:
        raise Bad("no hold-out ids")

    def fn(d):
        d["holdout"] = {"ids": ids, "scores": {}, "passed": False}  # a new pick invalidates old scores
        if no_fp:
            d["holdout"]["trained_unknown"] = True
    _update_proof(a.pack, fn)
    print(json.dumps({"holdout_ids": ids}, ensure_ascii=False))
    return 0


def _timing_threshold(pack):
    p = Path(pack) / "laws.json"
    if p.exists():
        laws = _read(p, "laws.json")
        laws = laws.get("laws", []) if isinstance(laws, dict) else laws
        vals = [x["lag"]["p90"] for x in laws or []
                if isinstance(x, dict) and isinstance(x.get("lag"), dict) and _num(x["lag"].get("p90"))]
        if vals:
            return statistics.median(vals), "laws"
    return None, "not_run"


def _list_of_dicts(data, key, what):
    """A prediction file: a list of dicts, or {key: [...]}. Anything else is bad input."""
    if isinstance(data, dict):
        data = data.get(key, [])
    if not isinstance(data, list) or not all(isinstance(x, dict) for x in data):
        raise ValueError(f"{what} must be a list of objects")
    return data


def _creator_scores(pack, ids, pdir):
    """Optional structure/retention scores for hold-out videos that have <id>.structure.json or
    <id>.retention.json predictions and analysis/<id>/creator.json. Returns ({structure?, retention?}, not_run).
    Never raises on malformed files: the id goes to not_run as creator_scores:<id>:bad_input."""
    not_run, want = [], []
    for v in ids:
        sf, rf = pdir / f"{v}.structure.json", pdir / f"{v}.retention.json"
        has_pred = sf.exists() or rf.exists()
        has_doc = (Path(pack) / "analysis" / v / "creator.json").exists()
        if has_pred and has_doc:
            want.append((v, sf, rf))
        elif has_pred:
            not_run.append(f"creator_scores:{v}:no_creator_json")
        elif has_doc:
            not_run.append(f"creator_scores:{v}:no_prediction")
    if not want:
        return {}, not_run
    try:
        import creator
    except ImportError:
        return {}, ["structure", "retention"] + not_run
    out = {"structure": {"per_video": {}}, "retention": {"per_video": {}}}
    for vid, sf, rf in want:
        try:
            doc = _read_dict(Path(pack) / "analysis" / vid / "creator.json", "creator.json")
            story = doc.get("story") or {}
            part = {}
            if sf.exists():
                pred = _list_of_dicts(_read(sf, "predicted structure"), "structure", "predicted structure")
                actual = _list_of_dicts(story.get("structure") or [], "structure", "actual structure")
                part["structure"] = creator.score_structure(pred, actual)
            if rf.exists():
                pred = _list_of_dicts(_read(rf, "predicted retention"), "retention", "predicted retention")
                actual = _list_of_dicts(doc.get("retention") or [], "retention", "actual retention")
                part["retention"] = score_placement(creator.retention_events(pred),
                                                    creator.retention_events(actual))
        except (AttributeError, TypeError, KeyError, ValueError, Bad):
            not_run.append(f"creator_scores:{vid}:bad_input")
            continue
        for k, v in part.items():
            out[k]["per_video"][vid] = v
    res = {}
    for k, v in out.items():
        if v["per_video"]:
            v.update(pool_scores(list(v["per_video"].values())))
            res[k] = v
    return res, not_run


def cmd_holdout(a):
    _check_pack(a.pack)
    proof = _read_dict(_proof_path(a.pack), "proof.json") if _proof_path(a.pack).exists() else {}
    fp = _fp(a.pack)
    ids = (proof.get("holdout") or {}).get("ids") or fp.get("held_out") or []
    if not ids:
        raise Bad("no hold-out ids: run holdout-pick first")
    _refuse_overlap(ids, fp)
    pdir = Path(a.predicted)
    scores = {}
    for vid in ids:
        pf = pdir / f"{vid}.events.json"
        af = Path(a.pack) / "analysis" / vid / "events.json"
        if not pf.exists():
            raise Bad(f"missing predicted events: {pf}")
        if af.exists():
            actual = _events(_read(af, "actual events"))
        else:
            actual = measured_events(Path(a.pack) / "analysis" / vid)
            if actual is None:
                raise Bad(f"missing actual events: no {af.name} and no measure.json in analysis/{vid}")
        scores[vid] = score_placement(_events(_read(pf, "predicted events")), actual)
    pooled = pool_scores(list(scores.values()))
    extra, not_run = _creator_scores(a.pack, ids, pdir)
    thr, src = _timing_threshold(a.pack)
    mae = pooled["timing_mae_s"]
    timing_run = thr is not None
    passed = (timing_run and pooled["overall_f1"] >= HOLDOUT_MIN_F1 and mae is not None and mae <= thr)
    result = {"overall_f1": pooled["overall_f1"], "timing_mae_s": mae,
              "timing": "checked" if timing_run else "not_run", "timing_threshold_s": thr,
              "timing_threshold_source": src, "per_video": scores, **extra}
    if not_run:
        result["not_run"] = not_run  # reported only; structure and retention do not decide pass yet
    unknown = bool((proof.get("holdout") or {}).get("trained_unknown"))

    def fn(d):
        d["holdout"] = {"ids": list(ids), "scores": result, "passed": bool(passed),
                        "trained_hash": trained_hash(fp.get("trained_on") or [])}
        if unknown:
            d["holdout"]["trained_unknown"] = True
    _update_proof(a.pack, fn)
    if not timing_run:
        print("timing not checked: no laws with lag.p90; run laws.py first")
        return 1
    shown = "none" if mae is None else f"{mae:.3f} s"
    print(f"hold-out {'PASSED' if passed else 'FAILED'}: overall_f1 {pooled['overall_f1']:.2f} "
          f"(need {HOLDOUT_MIN_F1}), timing MAE {shown} (max {thr:.3f} s, {src})")
    return 0 if passed else 1


def _load_motion(build):
    p = Path(build)
    if p.suffix.lower() == ".json":
        return _read_dict(p, "build motion")
    for cand in (p.with_suffix(".motion.json"), p.parent / "motion.json"):
        if cand.exists():
            return _read_dict(cand, "build motion")
    raise Bad(f"no motion data for {p}: run track_motion.py on it and pass the resulting motion.json "
              f"as --build", rc=3)


def _in_window(moves, t0, t1):
    return [m for m in moves if _num(m.get("t0")) and t0 - 1e-6 <= m["t0"] <= t1 + 1e-6]


def recreation_scores(ref_moves, build_moves, spread=None):
    """Pair each reference move with the nearest unused build move of the same prop; unpaired = missing."""
    scores, used = [], set()
    for rm in ref_moves:
        best, bj = None, None
        for j, bm in enumerate(build_moves):
            if j in used or (rm.get("prop") and bm.get("prop") and rm["prop"] != bm["prop"]):
                continue
            d = abs(bm["t0"] - rm["t0"])
            if best is None or d < best:
                best, bj = d, j
        if bj is None:
            scores.append({"prop": rm.get("prop"), "t": rm["t0"], "missing": True, "passed": False})
            continue
        used.add(bj)
        c = compare_moves(rm, build_moves[bj], spread)
        ok = c["within_spread"] and c["curve_rmse"] is not None and c["curve_rmse"] <= RECREATION_MAX_RMSE
        scores.append({"prop": rm.get("prop"), "t": rm["t0"], **c, "passed": bool(ok)})
    return scores


def cmd_recreation(a):
    _check_pack(a.pack)
    try:
        vid, rng = a.moment.rsplit(":", 1)
        t0s, t1s = rng.split("-", 1)
        t0, t1 = float(t0s), float(t1s)
    except ValueError:
        raise Bad("--moment must look like <id>:<t0>-<t1> (seconds)")
    if not vid or t1 <= t0:
        raise Bad("--moment needs an id and t1 > t0")
    ref = _read_dict(Path(a.pack) / "analysis" / vid / "motion.json", "reference motion.json")
    build = _load_motion(a.build)
    fp = _fp(a.pack)
    scores = recreation_scores(_in_window(all_moves(ref), t0, t1), _in_window(all_moves(build), t0, t1),
                               fp or None)
    moment_ok = bool(scores) and all(s["passed"] for s in scores)  # no reference move = nothing proven
    entry = {"id": vid, "t0": t0, "t1": t1, "scores": scores, "scores_passed": moment_ok,
             "owner_ok": bool(a.owner_ok)}
    state = {}

    def fn(d):
        rec = d.setdefault("recreation", {"moments": [], "passed": False})
        ms = [m for m in rec.get("moments", []) if (m.get("id"), m.get("t0"), m.get("t1")) != (vid, t0, t1)]
        ms.append(entry)  # a re-scored build drops the earlier owner_ok unless it is given again
        rec["moments"] = ms
        rec["passed"] = all(m.get("scores_passed") and m.get("owner_ok") is True for m in ms)
        state["passed"] = rec["passed"]
    _update_proof(a.pack, fn)
    n_ok = sum(1 for s in scores if s["passed"])
    print(f"moment {vid} {t0:g}-{t1:g}s: {n_ok}/{len(scores)} moves match; "
          f"owner_ok {entry['owner_ok']}; recreation passed {state['passed']}")
    return 0 if moment_ok else 1


def cmd_qa(a):
    _check_pack(a.pack)
    issues = qa_moves(_load_motion(a.build), _fp(a.pack))
    if a.out:
        store.write_json(Path(a.out), issues)
    print(json.dumps(issues, ensure_ascii=False, indent=2))
    return 1 if issues else 0


def cmd_graph(a):
    moves = all_moves(_read_dict(a.motion, "motion.json"))
    cmp_moves = None
    if a.compare:
        raw = all_moves(_read_dict(a.compare, "compare motion.json"))
        if len(raw) != len(moves):
            print(f"warning: reference has {len(moves)} moves, build {len(raw)}; paired by prop and time",
                  file=sys.stderr)
        cmp_moves = pair_moves(moves, raw)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(graph_html(moves, cmp_moves), encoding="utf-8")
    tmp.replace(out)
    print(f"wrote {out} ({len(moves)} moves)")
    return 0


def main(argv=None):
    return store.exit_on_lock_timeout(_main, argv)


def _main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("holdout-pick")
    a.add_argument("pack"); a.add_argument("--n", type=int, default=3)
    a.add_argument("--seed", type=int, default=7); a.add_argument("--ids")
    a.add_argument("--allow-no-fingerprint", action="store_true")
    a = sub.add_parser("holdout")
    a.add_argument("pack"); a.add_argument("--predicted", required=True)
    a = sub.add_parser("recreation")
    a.add_argument("pack"); a.add_argument("--moment", required=True)
    a.add_argument("--build", required=True); a.add_argument("--owner-ok", action="store_true")
    a = sub.add_parser("qa")
    a.add_argument("pack"); a.add_argument("--build", required=True); a.add_argument("--out")
    a = sub.add_parser("graph")
    a.add_argument("motion"); a.add_argument("--compare"); a.add_argument("--out", required=True)
    args = p.parse_args(argv)
    store.print_utf8()
    fns = {"holdout-pick": cmd_holdout_pick, "holdout": cmd_holdout, "recreation": cmd_recreation,
           "qa": cmd_qa, "graph": cmd_graph}
    try:
        return fns[args.cmd](args)
    except Bad as e:
        print(str(e), file=sys.stderr)
        return e.rc


if __name__ == "__main__":
    sys.exit(main())
