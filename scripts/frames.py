#!/usr/bin/env python3
"""B3: three real stills per beat. For the beat's picked plan concept Claude builds 3 HyperFrames variants
(different composition or treatment, never only a different colour), with the client's tokens and the beat's
real words; an AI beat uses images that were generated and passed the look check. This script makes the
folders, checks the set, snapshots one still per variant and records what was shown.

    frames.py new-set <project> --beat N [--more] [--owner-ok-more --owner-words "..."]
    frames.py check <project> --beat N --set K
    frames.py snap <project> --beat N --set K
    frames.py list <project> [--json]

Per variant folder frames/B03/s<k>/<a|b|c>/: index.html (from assets/frame-scaffold.html), tokens.css,
words.json (the beat's words from 0 s) and variant.json {id, title, composition, treatment, wild, concept_key,
at_s, ai_items, idea_id} that Claude fills. Stills land in frames/B03/s<k>/<a|b|c>.png; frames.json lists them.
Exit codes: 0 ok, 1 a check failed or the budget is used up, 2 bad input, 3 HyperFrames missing.
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402

STEP = "frames"
LETTERS = ("a", "b", "c")
SCAFFOLD = Path(__file__).resolve().parent.parent / "assets" / "frame-scaffold.html"
WIDTH = 1080
PLACEHOLDERS = ("lorem", "טקסט לדוגמה", "sample text", "placeholder")
_HEX = re.compile(r"#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3,4})(?![0-9a-zA-Z])")
_SRC = re.compile(r"<(?:img|video|source)\b[^>]*?\bsrc\s*=\s*[\"']([^\"']+)[\"']", re.I)
ASPECTS = {"9:16": (9, 16), "16:9": (16, 9), "1:1": (1, 1), "4:5": (4, 5)}


class BadInput(ValueError):
    pass


class CheckFailed(Exception):
    pass


# ---------------------------------------------------------------- tokens

def _theme(client):
    pal = (client or {}).get("palette") or {}
    return pal.get("theme") if pal.get("theme") in ("dark", "light") else "dark"


def _labels(client, theme):
    import ideas
    pal = (client or {}).get("palette") or {}
    lab = (pal.get("labels") or {}).get(theme) if pal else None
    return lab or ideas.brand_defaults().get("labels", {}).get(theme, {})


def tokens_css(client, theme="dark"):
    import ideas
    roles, fonts = ideas.tokens(client if (client or {}).get("palette") else None, theme)
    if client and not client.get("palette"):
        fonts = {**ideas.brand_defaults()["fonts"], **{k: v for k, v in (client.get("fonts") or {}).items() if v}}
    lines = [":root {"]
    lines += [f"  --sl-{k.replace('_', '-')}: {v};" for k, v in roles.items()]
    lines += [f"  --sl-label-{k.replace('_', '-')}: {v};" for k, v in _labels(client, theme).items()]
    for k in ("heading", "body", "mono"):
        fam = str(fonts.get(k) or "")
        fam = re.sub(r"\s+\d{3}$", "", fam)  # "Karantina 700" -> "Karantina"
        lines.append(f"  --sl-font-{k}: {fam if ',' in fam else repr(fam).replace(chr(39), chr(34))};")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _expand(h):
    h = h.lstrip("#").upper()
    if len(h) in (3, 4):
        h = "".join(c * 2 for c in h[:3])
    return "#" + h[:6]


def allowed_hexes(client, theme="dark"):
    import ideas
    roles, _ = ideas.tokens(client if (client or {}).get("palette") else None, theme)
    vals = list(roles.values()) + list(_labels(client, theme).values())
    return {_expand(v) for v in vals if isinstance(v, str) and v.startswith("#")} | {"#000000", "#FFFFFF"}


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
    return {"entry": entry, "style": st, "pack": pack, "client": client, "aspect": st.get("aspect") or "9:16"}


def _beat(d, n):
    plan = store.read_json(Path(d) / "plan.json", {}) or {}
    for b in plan.get("beats", []):
        if int(b.get("n", -1)) == int(n):
            return b
    raise BadInput(f"no beat {n} in plan.json")


def _picked_option(d, beat):
    import picks
    e = picks.load(d)["plan"].get(str(int(beat))) or {}
    if not (e.get("picked") or e.get("own")):
        return None, e
    for _, o in picks.options(d, "plan", beat):
        if o.get("id") == e.get("picked"):
            return o, e
    return {"id": None, "own": e.get("own")}, e


def _words(d, b):
    doc = store.read_json(Path(d) / "edit_ready.words.json", None)
    ws = doc.get("words", []) if isinstance(doc, dict) else doc if isinstance(doc, list) else []
    t0, t1 = float(b["start"]), float(b["end"])
    out = []
    for w in ws:
        if isinstance(w, dict) and w.get("start") is not None and t0 - 1e-6 <= float(w["start"]) < t1:
            out.append({"word": w.get("word") or w.get("text") or "", "start": round(float(w["start"]) - t0, 3),
                        "end": round(min(float(w.get("end", w["start"])), t1) - t0, 3)})
    if not out:
        out = [{"word": x, "start": None, "end": None} for x in str(b.get("script") or "").split()]
    lang = (doc or {}).get("language") if isinstance(doc, dict) else None
    return out, lang or "he"


def _set_dir(d, beat, set_no):
    return Path(d) / "frames" / f"B{int(beat):02d}" / f"s{int(set_no)}"


# ---------------------------------------------------------------- new set

def new_set(d, beat, more=False, owner_ok_more=False, owner_words=None):
    import project
    cx = _ctx(d)
    b = _beat(d, beat)
    opt, _ = _picked_option(d, beat)
    if opt is None:
        raise CheckFailed(f"B{beat}: no plan pick yet; frames come after the plan pick")
    n = project.next_set(d, STEP, int(beat), owner_ok_more, owner_words)
    words, lang = _words(d, b)
    aw, ah = ASPECTS.get(cx["aspect"], (9, 16))
    w, h = WIDTH, int(round(WIDTH * ah / aw / 2) * 2)
    dur = round(float(b["end"]) - float(b["start"]), 3)
    theme = _theme(cx["client"])
    scaffold = SCAFFOLD.read_text(encoding="utf-8")
    out = []
    for x in LETTERS:
        vd = _set_dir(d, beat, n) / x
        vd.mkdir(parents=True, exist_ok=True)
        html = (scaffold.replace("{{LANG}}", lang).replace("{{DIR}}", "rtl" if lang in ("he", "ar", "fa") else "ltr")
                .replace("{{BEAT}}", f"{int(beat):02d}").replace("{{SET}}", str(n)).replace("{{VARIANT}}", x)
                .replace("{{W}}", str(w)).replace("{{H}}", str(h)).replace("{{DUR}}", f"{dur:g}"))
        if not (vd / "index.html").exists():
            (vd / "index.html").write_text(html, encoding="utf-8")
        (vd / "tokens.css").write_text(tokens_css(cx["client"], theme), encoding="utf-8")
        store.write_json(vd / "words.json", {"language": lang, "words": words})
        if not (vd / "variant.json").exists():
            store.write_json(vd / "variant.json", {"id": f"B{int(beat):02d}-f{n}-{x}", "title": "", "composition": "",
                                                   "treatment": "", "wild": False,
                                                   "concept_key": opt.get("concept_key") or "", "at_s": round(dur / 2, 3),
                                                   "ai_items": [], "idea_id": opt.get("idea_id")})
        out.append(vd)
    return n, out, sorted(project.blocked(d, STEP, int(beat)))


# ---------------------------------------------------------------- checks

def _strip_comments(html):
    t = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    return re.sub(r"/\*.*?\*/", "", t, flags=re.S)


def design_sig(html):
    t = _strip_comments(html)
    t = re.sub(r"<title>.*?</title>", "", t, flags=re.S | re.I)  # ids that name the variant itself
    t = re.sub(r'data-composition-id="[^"]*"', "", t)
    t = re.sub(r"__timelines\[\s*[\"'][^\"']*[\"']\s*\]", "__timelines[]", t)
    t = _HEX.sub("#c", t)
    t = re.sub(r"\b(?:rgba?|hsla?)\([^)]*\)", "c()", t, flags=re.I)
    t = re.sub(r"var\(\s*--sl-[a-z0-9-]+\s*\)", "var(c)", t, flags=re.I)
    t = re.sub(r"\s+", "", t)
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def variant_sig(v):
    import project
    return project.sig(STEP, v.get("concept_key") or "", v.get("composition") or "", v.get("treatment") or "")


def _inside(p, root):
    try:
        Path(p).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _ai_recipe(d, cx, opt):
    import genprep
    if not opt or not opt.get("recipe"):
        return None
    for r in genprep.load_recipes(cx["pack"]):
        if r["kind"] == opt["recipe"] and r.get("how") in genprep.AI_HOWS:
            return r
    return None


def _variants(d, beat, set_no):
    sd = _set_dir(d, beat, set_no)
    out = []
    for x in LETTERS:
        vd = sd / x
        if vd.is_dir():
            v = store.read_json(vd / "variant.json", {}) or {}
            html = (vd / "index.html").read_text(encoding="utf-8") if (vd / "index.html").is_file() else ""
            out.append((x, vd, v, html))
    return out


def check_set(d, beat, set_no):
    import project
    cx = _ctx(d)
    b = _beat(d, beat)
    opt, _ = _picked_option(d, beat)
    probs = []
    vs = _variants(d, beat, set_no)
    if len(vs) != 3:
        probs.append(f"B{beat} s{set_no}: {len(vs)} variants, needs exactly 3")
    wild = [x for x, _, v, _ in vs if v.get("wild")]
    if len(wild) != 1:
        probs.append(f"B{beat} s{set_no}: {len(wild)} wild cards, needs exactly 1")
    words, _ = _words(d, b)
    beat_words = [w["word"] for w in words if len(w["word"].strip("?!.,:;\"'")) >= 2]
    allowed = allowed_hexes(cx["client"], _theme(cx["client"]))
    blocked = set(project.blocked(d, STEP, int(beat)))
    shown = store.read_json(Path(d) / "shown.json", {}).get("items", []) or []
    own = {it.get("sig") for it in shown if it.get("step") == STEP and int(it.get("beat", -1)) == int(beat)
           and int(it.get("set", 0)) == int(set_no)}
    blocked -= own
    q = {i["id"]: i for i in (store.read_json(Path(d) / "gen" / "queue.json", {"items": []}) or {}).get("items", [])}
    by_file = {str(i.get("file") or "").replace("\\", "/"): i for i in q.values() if i.get("file")}
    recipe = _ai_recipe(d, cx, opt)
    for i, (x, vd, v, html) in enumerate(vs):
        tag = f"B{beat} s{set_no} {x}"
        for k in ("composition", "treatment"):
            if not str(v.get(k) or "").strip():
                probs.append(f"{tag}: variant.json has no {k}")
        raw, html = html, _strip_comments(html)
        if "{{" in html:
            probs.append(f"{tag}: a {{{{...}}}} placeholder is left in index.html")
        low = html.lower()
        for ph in PLACEHOLDERS:
            if ph in low:
                probs.append(f"{tag}: sample text '{ph}' in index.html; use the beat's real words")
        if beat_words and not any(w in html for w in beat_words):
            probs.append(f"{tag}: none of the beat's words appear in index.html")
        for hx in sorted({_expand(m) for m in _HEX.findall(html)}):
            if hx not in allowed:
                probs.append(f"{tag}: colour {hx} is not one of the client's tokens (use var(--sl-...))")
        sg = variant_sig(v)
        if sg in blocked:
            probs.append(f"{tag}: repeats a frame design already shown or rejected ({sg})")
        srcs = _SRC.findall(html)
        gen_srcs = []
        for s in srcs:
            if re.match(r"^[a-z]+:", s, re.I) and not s.lower().startswith("file:"):
                probs.append(f"{tag}: {s} is not a local file")
                continue
            p = (vd / s).resolve()
            if not _inside(p, d):
                probs.append(f"{tag}: {s} points outside the project")
                continue
            rel = p.relative_to(Path(d).resolve()).as_posix()
            if "refs" in [c.lower() for c in Path(rel).parts]:
                probs.append(f"{tag}: {s} points into a refs/ folder (reference videos are never used as assets)")
                continue
            if rel.startswith("gen/files/"):
                gen_srcs.append(rel)
                it = by_file.get(rel)
                if it is None:
                    probs.append(f"{tag}: {rel} is not a logged generation (spend.py record)")
                elif it.get("status") not in ("look_pass", "placed"):
                    probs.append(f"{tag}: {rel} ({it['id']}) has not passed the look check ({it.get('status')})")
        if recipe is not None and not gen_srcs:
            probs.append(f"{tag}: AI beat without a generated image (recipe {recipe['kind']})")
        for y, _, w2, html2 in vs[i + 1:]:
            if (str(v.get("composition")), str(v.get("treatment"))) == (str(w2.get("composition")),
                                                                        str(w2.get("treatment"))):
                probs.append(f"B{beat} s{set_no} {x}/{y}: same composition and treatment")
            if html and design_sig(html) == design_sig(html2):
                probs.append(f"B{beat} s{set_no} {x}/{y}: the variants differ only in colour")
    return probs


# ---------------------------------------------------------------- snap

def snap(d, beat, set_no):
    import hf
    import project
    probs = check_set(d, beat, set_no)
    if probs:
        raise CheckFailed("\n".join(probs))
    sd = _set_dir(d, beat, set_no)
    variants, stills = [], []
    for x, vd, v, _ in _variants(d, beat, set_no):
        png = hf.snapshot(vd, float(v.get("at_s") or 0))
        dst = sd / f"{x}.png"
        shutil.copyfile(png, dst)
        stills.append(dst)
        sg = variant_sig(v)
        variants.append({"id": v.get("id") or f"B{int(beat):02d}-f{set_no}-{x}", "title": v.get("title") or "",
                         "composition": v.get("composition"), "treatment": v.get("treatment"),
                         "wild": bool(v.get("wild")), "concept_key": v.get("concept_key"), "sig": sg,
                         "at_s": v.get("at_s"), "still": dst.relative_to(d).as_posix(),
                         "dir": vd.relative_to(d).as_posix(), "ai_items": list(v.get("ai_items") or []),
                         "idea_id": v.get("idea_id"), "tags": list(v.get("tags") or [])})

    def go(doc):
        doc.setdefault("version", 1)
        bt = doc.setdefault("beats", {}).setdefault(str(int(beat)), {"sets": []})
        bt["sets"] = [s for s in bt.get("sets", []) if int(s.get("set", 0)) != int(set_no)]
        bt["sets"].append({"set": int(set_no), "variants": variants})
        bt["sets"].sort(key=lambda s: s["set"])
    store.update_json(Path(d) / "frames.json", {"version": 1, "beats": {}}, go)
    project.record_shown(d, STEP, int(beat), int(set_no), variants)
    return stills


# ---------------------------------------------------------------- CLI

def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("new-set"); a.add_argument("project"); a.add_argument("--beat", type=int, required=True)
    a.add_argument("--more", action="store_true"); a.add_argument("--owner-ok-more", action="store_true")
    a.add_argument("--owner-words")
    for name in ("check", "snap"):
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
        if args.cmd == "new-set":
            n, dirs, blocked = new_set(args.project, args.beat, args.more, args.owner_ok_more, args.owner_words)
            print(f"סט {n} לביט {args.beat}: " + ", ".join(p.relative_to(args.project).as_posix() for p in dirs))
            if blocked:
                print("חסומים (כבר הוצגו או נדחו): " + ", ".join(blocked))
            return 0
        if args.cmd == "check":
            probs = check_set(args.project, args.beat, args.set)
            for p in probs:
                print(p)
            return 1 if probs else 0
        if args.cmd == "snap":
            stills = snap(args.project, args.beat, args.set)
            print("\n".join(p.relative_to(args.project).as_posix() for p in stills))
            return 0
        doc = store.read_json(Path(args.project) / "frames.json", {"beats": {}}) or {"beats": {}}
        if args.json:
            print(json.dumps(doc, ensure_ascii=False, indent=2))
        else:
            for b, bt in sorted(doc.get("beats", {}).items(), key=lambda kv: int(kv[0])):
                for s in bt.get("sets", []):
                    print(f"B{b} s{s['set']}: " + ", ".join(v["id"] for v in s["variants"]))
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
