#!/usr/bin/env python3
"""Recipes into the generation queue: one queue item per AI image, video or 3D call, with its prompt built
from the style's recipe (template, measured look, negative), the beat's words and the client's palette.

This script never makes a call. Claude runs `spend.py estimate` and `approve`, then makes each call only for
an item that `spend.py next` releases, through the tool named here, and logs it with `spend.py record`.
A generated still is checked with `lookcheck.py`; only a still that passed can be animated (`animate`), and
the camera move of an animated shot is applied in code with the recipe's exact curve, because a generator
cannot follow a bezier. A failed look is fixed with `refine` (a retry in the same batch, new seed).

Commands:
    genprep.py add <project> --beat N --stage frames|build --recipe <kind> [--variant a] [--slot k=v]...
                   [--model M]
    genprep.py animate <project> --from G001
    genprep.py refine <project> --item G001 [--add "..."]... [--drop "..."]... --why "..."
    genprep.py list <project> [--json]
Exit codes: 0 ok, 1 refused (no tool for a required capability, a still not passed), 2 bad input.
"""
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402

AI_HOWS = ("ai_image", "ai_image+i2v", "ai_video", "3d")
NEGATIVE_ALWAYS = "text, letters, words, captions, watermark, logo, signature"
SLOT_RE = r"\{([a-z_]+)\}"
STAGES = ("frames", "build")
# 16 basic colours, named for prompts ("deep green (#014d46)")
BASIC = {"black": "#000000", "white": "#ffffff", "grey": "#808080", "silver": "#c0c0c0", "red": "#d62828",
         "maroon": "#800000", "orange": "#f18044", "yellow": "#f2c300", "olive": "#808000", "green": "#2a9d3f",
         "teal": "#008080", "deep green": "#014d46", "blue": "#1f5fd6", "navy": "#0b1f4d", "purple": "#7c5cff",
         "pink": "#e85d9c"}
PALETTE_ROLES = ("brand", "brand_deep", "accent", "background")


class BadInput(ValueError):
    pass


class MissingSlot(BadInput):
    def __init__(self, name):
        self.name = name
        super().__init__(f"missing slot '{name}': pass --slot {name}=<value>")


class NoTool(Exception):
    def __init__(self, capability, fallback):
        self.capability, self.fallback = capability, fallback
        super().__init__(f"no tool for {capability}: connect one, or use the recipe's fallback: {fallback or '-'}")


class Refused(Exception):
    pass


# ---------------------------------------------------------------- recipes and slots

def load_recipes(pack):
    if pack is None:
        return []
    data = store.read_json(Path(pack) / "recipes.json", None)
    rows = data.get("recipes", []) if isinstance(data, dict) else data if isinstance(data, list) else []
    return [r for r in rows if isinstance(r, dict) and r.get("kind")]


def colour_name(hexc):
    import palette
    lab = palette._lab(hexc)
    best = min(BASIC.items(), key=lambda kv: palette._de2000(lab, palette._lab(kv[1])))
    return f"{best[0]} ({palette.parse_hex(hexc)})"


def _look_text(look):
    return ", ".join(f"{k}: {v}" for k, v in (look or {}).items() if v not in (None, "", [], {}))


def auto_slots(beat, client, recipe, aspect):
    import ideas
    roles, _ = ideas.tokens(client, "dark")
    pal = [colour_name(roles[r]) for r in PALETTE_ROLES if roles.get(r)]
    return {"words": str((beat or {}).get("script") or ""), "palette": ", ".join(pal),
            "look": _look_text(recipe.get("look")), "aspect": aspect or ""}


def fill(template, slots):
    def rep(m):
        k = m.group(1)
        v = slots.get(k)
        if v is None or str(v).strip() == "":
            raise MissingSlot(k)
        return str(v)
    return re.sub(SLOT_RE, rep, template or "").strip()


def _negative(recipe):
    own = str(recipe.get("negative") or "").strip().strip(",").strip()
    return f"{own}, {NEGATIVE_ALWAYS}" if own else NEGATIVE_ALWAYS


# ---------------------------------------------------------------- tool, units, seed

def load_caps():
    c = store.read_json(store.path("capabilities.json"), {}) or {}
    return c if isinstance(c, dict) else {}


def pick_tool(recipe, caps):
    import doctor
    tools = [t for t in recipe.get("tools") or [] if isinstance(t, dict) and t.get("capability")]
    for t in tools:
        res = doctor.resolve(t["capability"], caps)
        if res["ok"]:
            return {"capability": res["via"], "model": t.get("model"), "fallback_used": False,
                    "fallback": t.get("fallback") or None}
    req = [t for t in tools if t.get("level", "required") == "required"]
    if req:
        raise NoTool(req[0]["capability"], req[0].get("fallback") or "")
    return {"capability": tools[0]["capability"] if tools else "unknown", "model": None, "fallback_used": True,
            "fallback": (tools[0].get("fallback") if tools else None)}


def _median(v):
    if isinstance(v, dict):
        v = v.get("median")
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def seed_for(project_id, item_id):
    return int.from_bytes(hashlib.sha1(f"{project_id}|{item_id}".encode("utf-8")).digest()[:4], "big")


def _units(recipe, video):
    if recipe["how"] == "3d":
        return 1, "model"
    if video:
        d = _median((recipe.get("motion") or {}).get("dur_s"))
        return max(2, int(round(d if d is not None else 4))), "second"
    return 1, "image"


# ---------------------------------------------------------------- context

def _ctx(d):
    import clients
    import project
    import styles
    try:
        entry = project.load(d)
    except KeyError as e:
        raise BadInput(str(e.args[0] if e.args else e))
    st = styles.find(styles.load_registry(), entry.get("style") or "") or {}
    pack = Path(st["pack"]) if str(st.get("pack") or "").strip() and Path(st["pack"]).is_dir() else None
    client = clients.load_client(entry["client"]) if entry.get("client") else None
    return {"entry": entry, "pack": pack, "client": client, "aspect": st.get("aspect") or "9:16"}


def _recipe(cx, kind):
    for r in load_recipes(cx["pack"]):
        if r["kind"] == kind:
            return r
    raise BadInput(f"no recipe '{kind}' in the style's recipes.json")


def _beat(d, n):
    plan = store.read_json(Path(d) / "plan.json", {}) or {}
    for b in plan.get("beats", []):
        if int(b.get("n", -1)) == int(n):
            return b
    raise BadInput(f"no beat {n} in plan.json")


def _queue(d):
    return Path(d) / "gen" / "queue.json"


def _append(d, build):
    import spend
    box = {}

    def fn(q):
        q.setdefault("version", 1)
        items = q.setdefault("items", [])
        item = build(items, spend._next_id(items))
        items.append(item)
        box["item"] = item
    store.update_json(_queue(d), {"version": 1, "items": []}, fn)
    return box["item"]


def _item(iid, beat, stage, variant, recipe, tool, model, kind, units, unit, prompt, negative, seed, params,
          camera_curve=None, what=""):
    return {"id": iid, "beat": beat, "stage": stage, "variant": variant, "recipe": recipe["kind"],
            "how": recipe["how"], "tool": tool, "model": model or "default", "kind": kind, "units": units,
            "unit": unit, "prompt": prompt, "negative": negative, "seed": seed, "params": params,
            "camera_curve": camera_curve, "attempt": 1, "retry_of": None, "status": "queued", "batch": None,
            "file": None, "cost": None, "what": what}


# ---------------------------------------------------------------- add / animate / refine

def add_item(d, beat, stage, recipe_kind, variant=None, slots=None, model=None, units=None):
    if stage not in STAGES:
        raise BadInput(f"stage must be one of {', '.join(STAGES)}")
    cx = _ctx(d)
    r = _recipe(cx, recipe_kind)
    if r.get("how") not in AI_HOWS:
        raise BadInput(f"recipe '{recipe_kind}' is not an AI recipe (how: {r.get('how') or '-'})")
    tool = pick_tool(r, load_caps())
    b = _beat(d, beat)
    s = auto_slots(b, cx["client"], r, cx["aspect"])
    s.update({k: v for k, v in (slots or {}).items() if v is not None})
    prompt = fill(r.get("prompt_template") or "{words}", s)
    video = r["how"] == "ai_video"
    u, unit = _units(r, video)
    if units is not None:
        u = units
    kind = "3d" if r["how"] == "3d" else "video" if video else "image"
    pid = cx["entry"].get("id")
    return _append(d, lambda items, iid: _item(
        iid, int(beat), stage, variant, r, tool["capability"], model or tool.get("model"), kind, u, unit, prompt,
        _negative(r), seed_for(pid, iid), {"slots": s, "fallback": tool.get("fallback")},
        what=f"{r['kind']} for beat {beat}"))


def camera_phrase(ease):
    """'slow settle' for an out ease, 'smooth push' for an inout one, 'gentle start' for an in one."""
    name = str(ease or "")
    m = re.search(r"C\s*([-\d.]+)[ ,]+([-\d.]+)[ ,]+([-\d.]+)[ ,]+([-\d.]+)", name)
    if m:
        try:
            import track_motion
            name = track_motion.nearest_named(tuple(float(x) for x in m.groups())) or name
        except Exception:
            pass
    low = name.lower().replace("_", "").replace("-", "")
    if "inout" in low:
        return "smooth push"
    if low.endswith(".in") or low == "easein":
        return "gentle start"
    return "slow settle"


def animate(d, still_item_id):
    cx = _ctx(d)
    q = store.read_json(_queue(d), {"items": []}) or {"items": []}
    src = next((i for i in q.get("items", []) if i.get("id") == still_item_id), None)
    if src is None:
        raise BadInput(f"no item {still_item_id}")
    if src.get("status") != "look_pass":
        raise Refused(f"{still_item_id} has not passed the look check (status {src.get('status')}): "
                      "run lookcheck.py first")
    if src.get("kind") != "image":
        raise BadInput(f"{still_item_id} is not an image")
    r = _recipe(cx, src["recipe"])
    mo = r.get("motion") or {}
    dur = _median(mo.get("dur_s")) or 4.0
    ease = mo.get("camera_ease")
    prompt = ", ".join(x for x in (str(mo.get("subject_motion") or "").strip(),
                                   f"camera: {camera_phrase(ease)}" if ease else "") if x)
    tool = pick_tool({"tools": [t for t in r.get("tools") or [] if "video" in str(t.get("capability"))]
                      or [{"capability": "video_generator", "level": "required", "fallback": "code camera move"}]},
                     load_caps())
    pid = cx["entry"].get("id")
    return _append(d, lambda items, iid: _item(
        iid, src.get("beat"), "build", src.get("variant"), r, tool["capability"], tool.get("model"), "video",
        max(2, int(round(dur))), "second", prompt or "subtle motion", _negative(r), seed_for(pid, iid),
        {"source_image": src.get("file"), "from_item": still_item_id},
        camera_curve={"custom_ease": ease, "dur_s": dur}, what=f"animate {still_item_id}"))


def refine(d, item_id, add=(), drop=(), why=""):
    import spend
    if not str(why or "").strip():
        raise BadInput("--why is required: what was wrong with the result")
    cx = _ctx(d)
    new = spend.retry(d, item_id)
    pid = cx["entry"].get("id")

    def fn(q):
        for it in q["items"]:
            if it["id"] == new["id"]:
                p = it.get("prompt") or ""
                for ph in drop:
                    p = re.sub(r"\s*,?\s*" + re.escape(ph.strip()) + r"\s*", " ", p, flags=re.I)
                p = re.sub(r"\s{2,}", " ", p).strip(" ,")
                extra = ", ".join(x.strip() for x in add if x.strip())
                it["prompt"] = f"{p}, {extra}" if extra else p
                it["seed"] = seed_for(pid, it["id"])
                it.setdefault("params", {})["refine"] = {"why": why.strip(), "add": list(add), "drop": list(drop)}
                new.update(it)
                return
    store.update_json(_queue(d), {"version": 1, "items": []}, fn)
    return new


# ---------------------------------------------------------------- CLI

def _kv(pairs):
    out = {}
    for kv in pairs:
        k, sep, v = kv.partition("=")
        if not sep or not k.strip():
            raise BadInput(f"bad --slot {kv!r}: use name=value")
        out[k.strip()] = v
    return out


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add"); a.add_argument("project"); a.add_argument("--beat", type=int, required=True)
    a.add_argument("--stage", required=True, choices=STAGES); a.add_argument("--recipe", required=True)
    a.add_argument("--variant"); a.add_argument("--slot", action="append", default=[]); a.add_argument("--model")
    a = sub.add_parser("animate"); a.add_argument("project"); a.add_argument("--from", dest="src", required=True)
    a = sub.add_parser("refine"); a.add_argument("project"); a.add_argument("--item", required=True)
    a.add_argument("--add", action="append", default=[]); a.add_argument("--drop", action="append", default=[])
    a.add_argument("--why", default="")
    a = sub.add_parser("list"); a.add_argument("project"); a.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    store.print_utf8()
    try:
        if not Path(args.project).is_dir():
            raise BadInput(f"project folder not found: {args.project}")
        if args.cmd == "add":
            it = add_item(args.project, args.beat, args.stage, args.recipe, args.variant, _kv(args.slot), args.model)
        elif args.cmd == "animate":
            it = animate(args.project, args.src)
        elif args.cmd == "refine":
            it = refine(args.project, args.item, args.add, args.drop, args.why)
        else:
            items = (store.read_json(_queue(args.project), {"items": []}) or {}).get("items", [])
            if args.json:
                print(json.dumps(items, ensure_ascii=False, indent=2))
            else:
                for i in items:
                    print(f"{i['id']}\t{i['status']}\tB{i.get('beat')}\t{i.get('recipe')}\t{i.get('tool')}\t"
                          f"{i.get('units')} {i.get('unit')}")
            return 0
        print(json.dumps(it, ensure_ascii=False))
        return 0
    except NoTool as e:
        print(f"חסר כלי ל־{e.capability}. חלופה לפי המתכון: {e.fallback or '-'}", file=sys.stderr)
        print(str(e), file=sys.stderr)
        return 1
    except Refused as e:
        print(str(e), file=sys.stderr)
        return 1
    except BadInput as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # spend.BadInput from a retry of an unknown item
        if type(e).__name__ == "BadInput":
            print(f"error: {e}", file=sys.stderr)
            return 2
        raise


def main(argv=None):
    return store.exit_on_lock_timeout(_main, argv)


if __name__ == "__main__":
    sys.exit(main())
