#!/usr/bin/env python3
"""Read type, colour, motion and sound tokens from a HyperFrames project; manage per-layer sources.

A HyperFrames project is an index.html (plus optional sibling .css/.js files) holding a GSAP
timeline. No browser is run: the <style> and <script> text is read with html.parser and regexes.

Usage:
  code_tokens.py <project dir> --out <file>          write tokens.json (schema "tokens/1")
  code_tokens.py layer <pack> --set <layer>=<source>[:<ref>]
        layers: type motion colour broll structure sound; sources: refs code brand client
        writes <pack>/layers.json (default source of every layer is "refs")

Output: {schema, source:{kind:"code", path_hint:<basename>}, type:{levels, scale_ratios},
colour:{roles}, motion:{eases, durations}, sound:{files}}. Levels are clustered by font size
(sizes within 10 % share a level) and ordered from the largest to the smallest.

Exit: 0 ok, 2 bad input (missing project, bad --set), 1 other error.
"""
import argparse
import colorsys
import re
import statistics
import sys
from html.parser import HTMLParser
from pathlib import Path

import store

LAYERS = ("type", "motion", "colour", "broll", "structure", "sound")
SOURCES = ("refs", "code", "brand", "client")
CLUSTER_TOL = 0.10
MAX_FILE = 4_000_000


class ProjectError(Exception):
    pass


# ---------------------------------------------------------------- file reading

class _Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.styles, self.scripts, self.links, self.script_srcs = [], [], [], []
        self.audio = []
        self.elements = []  # (tag, id, classes)
        self.inline = []  # (selector label, style text)
        self._buf, self._in = None, None
        self._in_audio = False

    def handle_starttag(self, tag, attrs):
        a = {k: (v or "") for k, v in attrs}
        if tag == "style":
            self._in, self._buf = "style", []
        elif tag == "script":
            if a.get("src"):
                self.script_srcs.append(a["src"])
            else:
                self._in, self._buf = "script", []
        elif tag == "link" and "stylesheet" in a.get("rel", "").lower() and a.get("href"):
            self.links.append(a["href"])
        self.elements.append((tag, a.get("id", ""), a.get("class", "").split()))
        if a.get("style"):
            self.inline.append((f"{tag}#{a['id']}" if a.get("id") else tag, a["style"]))
        if tag == "audio":
            self._in_audio = True
            if a.get("src"):
                self.audio.append(a)
        elif tag == "source" and self._in_audio and a.get("src"):
            self.audio.append(a)

    def handle_endtag(self, tag):
        if tag == self._in:
            (self.styles if tag == "style" else self.scripts).append("".join(self._buf))
            self._in = self._buf = None
        if tag == "audio":
            self._in_audio = False

    def handle_data(self, data):
        if self._buf is not None:
            self._buf.append(data)

    def handle_entityref(self, name):
        if self._buf is not None:
            self._buf.append(f"&{name};")

    def handle_charref(self, name):
        if self._buf is not None:
            self._buf.append(f"&#{name};")


def _read_text(p):
    if p.stat().st_size > MAX_FILE:
        raise ProjectError(f"file too large: {p.name}")
    return p.read_bytes().decode("utf-8-sig", errors="replace")


def load_project(directory):
    d = Path(directory)
    if d.is_file():
        d = d.parent
    index = d / "index.html"
    if not index.is_file():
        raise ProjectError(f"no index.html in {d.name or d}")
    try:
        page = _Page()
        page.feed(_read_text(index))
        page.close()
    except (OSError, ProjectError) as e:
        raise ProjectError(f"cannot read index.html: {e}")
    css, js = list(page.styles), list(page.scripts)
    seen = set()
    refs = [(h, css) for h in page.links] + [(s, js) for s in page.script_srcs]
    for ext, bucket in ((".css", css), (".js", js)):
        refs += [(p.name, bucket) for p in sorted(d.glob("*" + ext))]
    for ref, bucket in refs:
        if re.match(r"^[a-z]+:|^//", ref, re.I):
            continue
        p = d / ref.split("?")[0].split("#")[0]
        if p in seen or not p.is_file() or p.suffix.lower() not in (".css", ".js"):
            continue
        seen.add(p)
        try:
            bucket.append(_read_text(p))
        except (OSError, ProjectError):
            continue
    return d, page, "\n".join(css), "\n".join(js)


# ---------------------------------------------------------------------- CSS

def _decls(text):
    out = {}
    for part in text.split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip().lower()] = v.strip()
    return out


def _parse_css(css):
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    css = re.sub(r"@font-face\s*\{[^{}]*\}", "", css, flags=re.I)
    rules = []
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        sel = m.group(1).strip()
        if sel.startswith("@"):
            continue
        rules.append(([s.strip() for s in sel.split(",") if s.strip()], _decls(m.group(2))))
    return rules


def _resolve(value, vars_, depth=0):
    if depth > 10:
        return value

    def sub(m):
        name, fallback = m.group(1), m.group(2)
        if name in vars_:
            return _resolve(vars_[name], vars_, depth + 1)
        return fallback.strip() if fallback else m.group(0)

    return re.sub(r"var\(\s*(--[\w-]+)\s*(?:,\s*([^)]*))?\)", sub, value)


def _to_hex(token):
    token = token.strip().lower()
    m = re.fullmatch(r"#([0-9a-f]{3,8})", token)
    if m:
        h = m.group(1)
        if len(h) in (3, 4):
            h = "".join(c * 2 for c in h)
        return "#" + h[:6] if len(h) in (6, 8) else None
    m = re.fullmatch(r"rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)[^)]*\)", token)
    if m:
        return "#%02x%02x%02x" % tuple(max(0, min(255, round(float(x)))) for x in m.groups())
    m = re.fullmatch(r"hsla?\(\s*([\d.]+)(?:deg)?[\s,]+([\d.]+)%[\s,]+([\d.]+)%[^)]*\)", token)
    if m:
        h, s, l = float(m.group(1)) / 360, float(m.group(2)) / 100, float(m.group(3)) / 100
        r, g, b = colorsys.hls_to_rgb(h % 1, l, s)
        return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))
    return None


_COLOUR_IN = re.compile(r"#[0-9a-fA-F]{3,8}\b|rgba?\([^)]*\)|hsla?\([^)]*\)")
_COLOUR_PROPS = ("color", "background", "background-color", "border-color", "border", "outline-color",
                 "fill", "stroke", "text-shadow", "box-shadow", "border-top", "border-bottom",
                 "border-left", "border-right", "caret-color", "text-decoration-color")


def _colours_in(value):
    out = []
    for m in _COLOUR_IN.finditer(value):
        h = _to_hex(m.group(0))
        if h:
            out.append(h)
    return out


def _px(value):
    m = re.fullmatch(r"([\d.]+)\s*(px|rem|em)?", value.strip().lower())
    if not m:
        return None
    n = float(m.group(1))
    return n * 16 if m.group(2) in ("rem", "em") else n


def _weight(value):
    v = value.strip().lower()
    if v in ("normal", ""):
        return 400
    if v == "bold":
        return 700
    m = re.match(r"\d+", v)
    return int(m.group()) if m else None


def _family(value):
    first = value.split(",")[0].strip().strip("\"'").strip()
    return first or None


def _num(x):
    f = float(x)
    return int(f) if f == int(f) else round(f, 3)


def _matches(selector, el):
    tag, eid, classes = el
    parts = selector.split()
    if not parts:
        return False
    s = re.sub(r"::?[\w-]+(\([^)]*\))?", "", parts[-1])
    m = re.fullmatch(r"([a-z0-9]*)((?:[.#][\w-]+)*)", s, re.I)
    if not m or not (m.group(1) or m.group(2)):
        return False
    if m.group(1) and m.group(1).lower() != tag:
        return False
    for q in re.findall(r"[.#][\w-]+", m.group(2)):
        if q[0] == "." and q[1:] not in classes:
            return False
        if q[0] == "#" and q[1:] != eid:
            return False
    return True


def _style_data(page, css):
    rules = _parse_css(css)
    for label, text in page.inline:
        rules.append(([label], _decls(text)))
    vars_ = {}
    for _, decls in rules:
        for k, v in decls.items():
            if k.startswith("--"):
                vars_[k] = v
    return rules, vars_


def _uses(selectors, page):
    n = sum(1 for el in page.elements if any(_matches(s, el) for s in selectors))
    return max(1, n)


# --------------------------------------------------------------------- type

def _type_tokens(rules, vars_, page):
    base = {}
    for sels, decls in rules:
        if any(s in ("body", "html", ":root", "*") for s in sels):
            for k in ("font-family", "font-weight", "color"):
                if k in decls:
                    base[k] = _resolve(decls[k], vars_)
    items = []
    for sels, decls in rules:
        if "font-size" not in decls:
            continue
        size = _px(_resolve(decls["font-size"], vars_))
        if not size:
            continue
        fam = _family(_resolve(decls.get("font-family", base.get("font-family", "")), vars_))
        wt = _weight(_resolve(decls.get("font-weight", base.get("font-weight", "400")), vars_))
        col = _colours_in(_resolve(decls.get("color", base.get("color", "")), vars_))
        lh_v = None
        if "line-height" in decls:
            lh = _resolve(decls["line-height"], vars_)
            lh_v = _num(lh) if re.fullmatch(r"[\d.]+", lh) else (_num(_px(lh)) if _px(lh) is not None else lh)
        tr = _px(_resolve(decls["letter-spacing"], vars_)) if "letter-spacing" in decls else None
        items.append({"size": size, "font": fam, "weight": wt, "colour": col[0] if col else None,
                      "line_height": lh_v, "tracking": None if tr is None else _num(tr),
                      "uses": _uses(sels, page)})
    items.sort(key=lambda i: -i["size"])
    clusters = []
    for it in items:
        if clusters and it["size"] >= clusters[-1][0]["size"] * (1 - CLUSTER_TOL):
            clusters[-1].append(it)
        else:
            clusters.append([it])
    levels = []
    for i, cl in enumerate(clusters, 1):
        top = max(cl, key=lambda c: c["uses"])
        size = max({c["size"] for c in cl}, key=lambda s: (sum(c["uses"] for c in cl if c["size"] == s), s))
        levels.append({"name": f"L{i}", "font": top["font"], "size_px": _num(size), "weight": top["weight"],
                       "colour": top["colour"], "line_height": top["line_height"], "tracking": top["tracking"],
                       "uses": sum(c["uses"] for c in cl)})
    ratios = [round(a["size_px"] / b["size_px"], 3) for a, b in zip(levels, levels[1:])]
    return {"levels": levels, "scale_ratios": ratios}


# ------------------------------------------------------------------- colour

def _colour_tokens(rules, vars_):
    roles = {}
    for sels, decls in rules:
        if all(k.startswith("--") for k in decls):
            continue
        for k, v in decls.items():
            if k not in _COLOUR_PROPS:
                continue
            for h in _colours_in(_resolve(v, vars_)):
                r = roles.setdefault(h, {"hex": h, "uses": 0, "where": []})
                r["uses"] += 1
                for s in sels:
                    if s not in r["where"]:
                        r["where"].append(s)
    return {"roles": sorted(roles.values(), key=lambda r: (-r["uses"], r["hex"]))}


# ------------------------------------------------------------------- motion

_CALL = re.compile(r"\.(to|from|fromTo|set)\s*\(")
_CUSTOM = re.compile(r"CustomEase\.create\(\s*([\"'])(.*?)\1\s*,\s*([\"'])(.*?)\3\s*\)", re.S)
_RESERVED = {"duration", "ease", "delay", "stagger", "repeat", "yoyo", "overwrite", "onComplete", "onStart",
             "onUpdate", "immediateRender", "paused", "id", "repeatDelay", "force3D", "transformOrigin",
             "lazy", "keyframes", "defaults", "data", "startAt"}


def _balanced(text, start):
    """text[start] == '(' -> index after the matching ')' (string aware), or None."""
    depth, i, q = 0, start, None
    while i < len(text):
        c = text[i]
        if q:
            if c == "\\":
                i += 1
            elif c == q:
                q = None
        elif c in "\"'`":
            q = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def _top_args(s):
    args, depth, cur, q = [], 0, [], None
    for c in s:
        if q:
            cur.append(c)
            if c == q:
                q = None
            continue
        if c in "\"'`":
            q = c
        elif c in "({[":
            depth += 1
        elif c in ")}]":
            depth -= 1
        elif c == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
            continue
        cur.append(c)
    if "".join(cur).strip():
        args.append("".join(cur).strip())
    return args


def _top_keys(obj):
    obj = obj.strip()
    if obj.startswith("{"):
        obj = obj[1:-1]
    keys = []
    for a in _top_args(obj):
        m = re.match(r"[\"']?([\w$]+)[\"']?\s*:", a)
        if m:
            keys.append(m.group(1))
    return keys


def _pct(vals, p):
    s = sorted(vals)
    if not s:
        return None
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 3)


def _motion_tokens(js):
    customs = {m.group(2): m.group(4) for m in _CUSTOM.finditer(js)}
    groups, durations = {}, []
    for m in _CALL.finditer(js):
        end = _balanced(js, m.end() - 1)
        if end is None:
            continue
        args = _top_args(js[m.end():end - 1])
        method = m.group(1)
        if method == "set" or len(args) < 2:
            continue
        obj = next((a for a in reversed(args) if a.startswith("{")), None)
        if obj is None:
            continue
        dm = re.search(r"\bduration\s*:\s*([\d.]+)", obj)
        if dm:
            dur = float(dm.group(1))
        elif len(args) > 2 and re.fullmatch(r"[\d.]+", args[1]):
            dur = float(args[1])
        else:
            dur = 0.5  # GSAP default
        em = re.search(r"\bease\s*:\s*([\"'])(.*?)\1", obj)
        ease = em.group(2) if em else "power1.out"  # GSAP default
        props = [k for k in _top_keys(obj) if k not in _RESERVED]
        if method == "fromTo" and len(args) >= 3 and args[-2].startswith("{"):
            props += [k for k in _top_keys(args[-2]) if k not in _RESERVED]
        key = ("custom", customs[ease]) if ease in customs else ("name", ease)
        g = groups.setdefault(key, {"durs": [], "props": set(), "label": ease})
        g["durs"].append(dur)
        g["props"].update(props)
        durations.append(dur)
    eases = []
    for (kind, val), g in groups.items():
        item = {"custom_ease": val, "label": g["label"]} if kind == "custom" else {"name": val}
        item.update({"duration_s": round(statistics.median(g["durs"]), 3), "uses": len(g["durs"]),
                     "props": sorted(g["props"])})
        eases.append(item)
    eases.sort(key=lambda e: (-e["uses"], e.get("name") or e.get("custom_ease")))
    return {"eases": eases,
            "durations": {"median": _pct(durations, 0.5), "p10": _pct(durations, 0.1),
                          "p90": _pct(durations, 0.9), "n": len(durations)}}


# -------------------------------------------------------------------- sound

def _sound_tokens(page):
    files = []
    for a in page.audio:
        src = a["src"].replace("\\", "/").split("?")[0]
        item = {"file": src.rsplit("/", 1)[-1]}
        for k, out in (("data-start", "start_s"), ("data-volume", "volume"), ("volume", "volume"),
                       ("data-duration", "duration_s")):
            if k in a and out not in item:
                try:
                    item[out] = float(a[k])
                except ValueError:
                    pass
        files.append(item)
    return {"files": files}


# ---------------------------------------------------------------- top level

def read_project(directory):
    d, page, css, js = load_project(directory)
    rules, vars_ = _style_data(page, css)
    return {"schema": "tokens/1",
            "source": {"kind": "code", "path_hint": d.resolve().name},
            "type": _type_tokens(rules, vars_, page),
            "colour": _colour_tokens(rules, vars_),
            "motion": _motion_tokens(js),
            "sound": _sound_tokens(page)}


def read_layers(pack):
    data = store.read_json(Path(pack) / "layers.json", {})
    out = {l: {"source": "refs", "ref": ""} for l in LAYERS}
    for l in LAYERS:
        if isinstance(data.get(l), dict):
            out[l].update({k: data[l][k] for k in ("source", "ref") if k in data[l]})
    return out


def set_layer_source(pack, layer, source, ref=""):
    if layer not in LAYERS:
        raise ValueError(f"unknown layer {layer!r}; use one of {', '.join(LAYERS)}")
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; use one of {', '.join(SOURCES)}")
    p = Path(pack) / "layers.json"
    with store.lock(p):
        data = read_layers(pack)
        data[layer] = {"source": source, "ref": ref or ""}
        store.write_json(p, data)
    return data


def main(argv=None):
    store.print_utf8()
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv and argv[0] == "layer":
            ap = argparse.ArgumentParser(prog="code_tokens.py layer")
            ap.add_argument("pack")
            ap.add_argument("--set", dest="sets", action="append", required=True, metavar="LAYER=SOURCE[:REF]")
            a = ap.parse_args(argv[1:])
            if not Path(a.pack).is_dir():
                print(f"error: pack folder not found: {Path(a.pack).name}", file=sys.stderr)
                return 2
            for s in a.sets:
                layer, eq, rest = s.partition("=")
                if not eq:
                    print(f"error: --set needs LAYER=SOURCE[:REF], got {s!r}", file=sys.stderr)
                    return 2
                source, _, ref = rest.partition(":")
                set_layer_source(a.pack, layer.strip(), source.strip(), ref)
            print(f"layers.json updated in {Path(a.pack).name}")
            return 0
        ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        ap.add_argument("project", help="HyperFrames project folder (holds index.html)")
        ap.add_argument("--out", required=True, help="tokens.json to write")
        a = ap.parse_args(argv)
        tokens = read_project(a.project)
        store.write_json(Path(a.out), tokens)
        print(f"{len(tokens['type']['levels'])} type levels, {len(tokens['colour']['roles'])} colours, "
              f"{len(tokens['motion']['eases'])} eases, {len(tokens['sound']['files'])} sound files -> {a.out}")
        return 0
    except (ProjectError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(store.exit_on_lock_timeout(main))
