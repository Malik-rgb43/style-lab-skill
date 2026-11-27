#!/usr/bin/env python3
"""The look check of a generated asset: its measured look against the style's reference frames and the
recipe's numbers, plus Claude's own visual verdict. Both must pass.

    lookcheck.py <project> --item G001                      metrics only (exit 0 pass / 1 fail); status unchanged
    lookcheck.py <project> --item G001 --claude pass|fail --why "..." [--override "..."]

The reference frames come from the recipe's `source_evidence` ("V03@12.4" or {video, t}); the video id maps to
<pack>/refs/<id>.mp4 (or to the path in animations.json "videos"). The frames are cached as PNGs in
<project>/gen/refcache/ (local only, never published). Each metric's range is [min(refs) - m, max(refs) + m],
m = max(15 % of |median|, FLOOR[metric]); a numeric recipe look value adds a +-15 % range. The palette passes
at ΔE2000 <= 15 against the reference colours plus the client's palette roles. No references and no numbers:
the metrics are "not_run" and Claude's verdict decides. A metrics fail that Claude passes stays look_fail
unless --override "<why>" is given (logged).
Every check appends {"type": "look", ...} to gen/log.jsonl. Exit codes: 0 pass, 1 fail, 2 bad input, 3 missing tool.
"""
import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402
import imgmetrics as im  # noqa: E402

METRICS = ("mean_L", "contrast", "chroma", "grain", "vignette", "sharpness")
FLOOR = {"mean_L": 0.05, "contrast": 0.03, "chroma": 0.02, "grain": 0.004, "vignette": 0.05}
SHARP_FLOOR_SHARE = 0.2
REL = 0.15
PALETTE_MAX_DE = 15.0
METRIC_HE = {"mean_L": "בהירות", "contrast": "ניגודיות", "chroma": "רוויה", "grain": "גרעיניות",
             "vignette": "ויגנט", "sharpness": "חדות"}


class BadInput(ValueError):
    pass


def _evidence(recipe):
    out = []
    for e in recipe.get("source_evidence") or []:
        if isinstance(e, dict) and e.get("video") is not None and e.get("t") is not None:
            out.append((str(e["video"]), float(e["t"])))
        elif isinstance(e, str):
            m = re.fullmatch(r"\s*([^@\s]+)@([\d.]+)\s*", e)
            if m:
                out.append((m.group(1), float(m.group(2))))
    return out


def _video_path(pack, vid):
    anim = store.read_json(Path(pack) / "animations.json", None)
    vids = anim.get("videos") if isinstance(anim, dict) else None
    if isinstance(vids, dict) and vids.get(vid):
        p = Path(vids[vid])
        return p if p.is_absolute() else Path(pack) / p
    for ext in (".mp4", ".mov", ".webm", ".mkv"):
        p = Path(pack) / "refs" / f"{vid}{ext}"
        if p.is_file():
            return p
    return None


def ref_stats(pack, recipe, cache_dir):
    if pack is None:
        return []
    cache = Path(cache_dir)
    out = []
    for vid, t in _evidence(recipe):
        png = cache / f"{vid}@{t:.2f}.png"
        if not png.is_file():
            src = _video_path(pack, vid)
            if src is None:
                continue
            cache.mkdir(parents=True, exist_ok=True)
            subprocess.run([im._ffmpeg(), "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(src), "-frames:v", "1",
                            str(png)], capture_output=True)
            if not png.is_file():
                continue
        out.append(im.look_stats(im.decode_image(png)))
    return out


def judge(stats, refs, palette_hexes=(), look_numbers=None):
    checks = []
    for k in METRICS:
        vals = [r[k] for r in refs if r.get(k) is not None]
        if not vals:
            continue
        med = sorted(vals)[len(vals) // 2]
        floor = SHARP_FLOOR_SHARE * abs(med) if k == "sharpness" else FLOOR[k]
        m = max(REL * abs(med), floor)
        lo, hi = min(vals) - m, max(vals) + m
        checks.append({"metric": k, "value": stats[k], "lo": round(lo, 6), "hi": round(hi, 6),
                       "ok": lo <= stats[k] <= hi, "from": "references"})
    for k, v in (look_numbers or {}).items():
        if k in METRICS and isinstance(v, (int, float)) and not isinstance(v, bool):
            lo, hi = v - REL * abs(v), v + REL * abs(v)
            checks.append({"metric": k, "value": stats[k], "lo": round(lo, 6), "hi": round(hi, 6),
                           "ok": lo <= stats[k] <= hi, "from": "recipe"})
    ref_hexes = [p["hex"] for r in refs for p in r.get("palette", [])] + [h for h in palette_hexes if h]
    de = im.palette_distance(stats.get("palette"), ref_hexes) if refs else None
    if not checks and de is None:
        return {"verdict": "not_run", "checks": [], "palette_de": None}
    ok = all(c["ok"] for c in checks) and (de is None or de <= PALETTE_MAX_DE)
    return {"verdict": "pass" if ok else "fail", "checks": checks, "palette_de": de}


# ---------------------------------------------------------------- project check

def _ctx(d):
    import clients
    import ideas
    import project
    import styles
    try:
        entry = project.load(d)
    except KeyError as e:
        raise BadInput(str(e.args[0] if e.args else e))
    st = styles.find(styles.load_registry(), entry.get("style") or "") or {}
    pack = Path(st["pack"]) if str(st.get("pack") or "").strip() and Path(st["pack"]).is_dir() else None
    client = clients.load_client(entry["client"]) if entry.get("client") else None
    roles = ideas.tokens(client, "dark")[0] if client and client.get("palette") else {}
    return pack, [v for v in (roles or {}).values() if isinstance(v, str) and v.startswith("#")]


def _queue(d):
    return Path(d) / "gen" / "queue.json"


def check(d, item_id, claude=None, why="", override=None):
    import genprep
    if claude not in (None, "pass", "fail"):
        raise BadInput("--claude must be pass or fail")
    if claude is not None and not str(why or "").strip():
        raise BadInput("--why is required with --claude: what you saw")
    q = store.read_json(_queue(d), {"items": []}) or {"items": []}
    item = next((i for i in q.get("items", []) if i.get("id") == item_id), None)
    if item is None:
        raise BadInput(f"no item {item_id}")
    if not item.get("file"):
        raise BadInput(f"{item_id} has no file yet (record it with spend.py record)")
    f = Path(d) / item["file"]
    if not f.is_file():
        raise BadInput(f"file not found: {item['file']}")
    pack, client_hexes = _ctx(d)
    recipe = next((r for r in genprep.load_recipes(pack) if r["kind"] == item.get("recipe")), {})
    img = im.frames_of(f, n=1)[0] if item.get("kind") == "video" else im.decode_image(f)
    stats = im.look_stats(img)
    refs = ref_stats(pack, recipe, Path(d) / "gen" / "refcache")
    res = judge(stats, refs, client_hexes, recipe.get("look") or {})
    metrics_ok = res["verdict"] in ("pass", "not_run")
    passed = bool(claude == "pass" and (metrics_ok or bool(str(override or "").strip())))
    line = {"type": "look", "date": dt.datetime.now().isoformat(timespec="seconds"), "item": item_id,
            "verdict": res["verdict"], "checks": res["checks"], "palette_de": res["palette_de"],
            "claude": {"verdict": claude, "why": why.strip()} if claude else None, "pass": passed}
    if override and claude == "pass" and not metrics_ok:
        line["override"] = str(override).strip()
    store.append_jsonl(Path(d) / "gen" / "log.jsonl", line)
    if claude is not None:
        status = "look_pass" if passed else "look_fail"

        def fn(doc):
            for i in doc["items"]:
                if i["id"] == item_id:
                    i["status"] = status
        store.update_json(_queue(d), {"version": 1, "items": []}, fn)
        line["status"] = status
    return line


def _print(line):
    v = {"pass": "עבר", "fail": "נכשל", "not_run": "לא נמדד (אין רפרנסים או מספרים)"}[line["verdict"]]
    print(f"{line['item']}: מדדים {v}")
    for c in line["checks"]:
        mark = "✓" if c["ok"] else "✗"
        print(f"  {mark} {METRIC_HE.get(c['metric'], c['metric'])} {c['value']:.4g} (טווח {c['lo']:.4g}-{c['hi']:.4g})")
    if line["palette_de"] is not None:
        print(f"  פלטה ΔE {line['palette_de']:.1f} (עד {PALETTE_MAX_DE:g})")


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project")
    ap.add_argument("--item", required=True)
    ap.add_argument("--claude", choices=("pass", "fail"))
    ap.add_argument("--why", default="")
    ap.add_argument("--override")
    a = ap.parse_args(argv)
    store.print_utf8()
    try:
        if not Path(a.project).is_dir():
            raise BadInput(f"project folder not found: {a.project}")
        line = check(a.project, a.item, a.claude, a.why, a.override)
    except BadInput as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except im.MissingTool as e:
        print(str(e), file=sys.stderr)
        return 3
    _print(line)
    if a.claude is None:
        return 0 if line["verdict"] in ("pass", "not_run") else 1
    if not line["pass"] and line["verdict"] == "fail" and a.claude == "pass":
        print("המדדים נכשלו: אם בכל זאת זה נכון, --override \"<למה>\"", file=sys.stderr)
    return 0 if line["pass"] else 1


def main(argv=None):
    return store.exit_on_lock_timeout(_main, argv)


if __name__ == "__main__":
    sys.exit(main())
