#!/usr/bin/env python3
"""Rank candidate fonts against a text crop (chamfer distance on ink edges).

    font_match.py <crop.png> --text "<the words>" [--fonts-dir D ...] [--lang he]

Each candidate font that covers every character of --text (cmap check) is rendered with
Pillow at the crop's text height; both images are binarised (Otsu), cropped to the ink box,
scaled to a common height and compared by symmetric chamfer distance on the ink edges
(lower is better). For an RTL language profile (lang.py) the text is shaped right to left:
with raqm directly, otherwise by reversing the logical string (bidi: "reversed_fallback").
Prints JSON: {bidi, skipped_no_glyphs, results: [{family, file, score, other_files}]} with the
best file per family, top 5 families. Local only: no network, no upload.
Exit codes: 0 ok, 2 bad input, 3 missing tool (Pillow/numpy).
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402

FONT_EXT = {".ttf", ".otf", ".ttc"}
NORM_H = 48
BORDER = 2
_CMAPS = {}


def _system_dirs():
    dirs = []
    if os.name == "nt":
        dirs.append(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts")
    dirs += [Path("/usr/share/fonts"), Path("/Library/Fonts"), Path.home() / "Library/Fonts"]
    return [d for d in dirs if d.is_dir()]


def list_fonts(dirs):
    out = []
    for d in dirs:
        d = Path(d)
        if d.is_file() and d.suffix.lower() in FONT_EXT:
            out.append(d)
        elif d.is_dir():
            out += sorted(p for p in d.rglob("*") if p.suffix.lower() in FONT_EXT)
    return out


def family_name(path):
    try:
        from fontTools.ttLib import TTFont
        f = TTFont(str(path), fontNumber=0, lazy=True)
        for rec in f["name"].names:
            if rec.nameID == 1:
                return rec.toUnicode()
    except Exception:
        pass
    return Path(path).stem


def font_cmap(path):
    """Set of code points the font maps, or None when fontTools is missing or the file is unreadable."""
    key = str(path)
    if key not in _CMAPS:
        try:
            from fontTools.ttLib import TTFont
            _CMAPS[key] = set(TTFont(key, fontNumber=0, lazy=True).getBestCmap() or {})
        except Exception:
            _CMAPS[key] = None
    return _CMAPS[key]


def covers(path, text):
    """True when the font has a glyph for every non-space character (unknown cmap counts as covered)."""
    cm = font_cmap(path)
    if cm is None:
        return True
    return all(ord(c) in cm for c in text if not c.isspace())


def otsu_threshold(gray):
    import numpy as np
    g = np.asarray(gray).astype(np.uint8)
    hist = np.bincount(g.ravel(), minlength=256).astype(float)
    total = hist.sum()
    if total == 0:
        return 127
    omega = np.cumsum(hist) / total
    mu = np.cumsum(hist * np.arange(256)) / total
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    denom[denom == 0] = np.nan
    sigma = (mu_t * omega - mu) ** 2 / denom
    return int(np.nanargmax(sigma)) if not np.all(np.isnan(sigma)) else 127


def edges(mask):
    import numpy as np
    p = np.pad(mask, 1)
    inner = p[1:-1, 1:-1]
    all_n = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return inner & ~all_n


def binarise(gray, border=BORDER):
    """Ink mask: strip a border, Otsu threshold, ink = the side with the higher edge density."""
    import numpy as np
    g = np.asarray(gray)
    if border and min(g.shape) > 2 * border + 2:
        g = g[border:-border, border:-border]
    dark = g <= otsu_threshold(g)
    light = ~dark

    def density(m):
        n = int(m.sum())
        return edges(m).sum() / n if n else 0.0
    return dark if density(dark) >= density(light) else light


def ink_crop(mask):
    import numpy as np
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def normalise(mask, height=NORM_H):
    """Scale an ink crop to a fixed height, keeping aspect."""
    import numpy as np
    from PIL import Image
    m = ink_crop(mask)
    if m is None:
        return None
    h, w = m.shape
    nw = max(1, int(round(w * height / h)))
    im = Image.fromarray((m * 255).astype("uint8")).resize((nw, height), Image.LANCZOS)
    return np.asarray(im) > 127


def chamfer_dt(points_mask):
    """Distance (pixels) to the nearest True pixel. scipy when importable, pure loops otherwise."""
    import numpy as np
    try:
        from scipy.ndimage import distance_transform_cdt
        return distance_transform_cdt(~points_mask, metric="chessboard").astype(float) \
            if points_mask.any() else np.full(points_mask.shape, 1e6)
    except ImportError:
        return _chamfer_loops(points_mask)


def _chamfer_loops(points_mask):
    import numpy as np
    h, w = points_mask.shape
    INF = 10 ** 6
    d = np.where(points_mask, 0, INF).astype(np.int64)
    for y in range(h):
        for x in range(w):
            v = d[y, x]
            if y > 0:
                v = min(v, d[y - 1, x] + 3)
                if x > 0:
                    v = min(v, d[y - 1, x - 1] + 4)
                if x < w - 1:
                    v = min(v, d[y - 1, x + 1] + 4)
            if x > 0:
                v = min(v, d[y, x - 1] + 3)
            d[y, x] = v
    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            v = d[y, x]
            if y < h - 1:
                v = min(v, d[y + 1, x] + 3)
                if x > 0:
                    v = min(v, d[y + 1, x - 1] + 4)
                if x < w - 1:
                    v = min(v, d[y + 1, x + 1] + 4)
            if x < w - 1:
                v = min(v, d[y, x + 1] + 3)
            d[y, x] = v
    return d / 3.0


def chamfer_score(a, b):
    """Symmetric mean chamfer distance between two ink masks (edges), padded to a common size."""
    import numpy as np
    H = max(a.shape[0], b.shape[0])
    W = max(a.shape[1], b.shape[1])

    def pad(m):
        out = np.zeros((H, W), dtype=bool)
        out[:m.shape[0], :m.shape[1]] = m
        return out

    ea, eb = edges(pad(a)), edges(pad(b))
    if not ea.any() or not eb.any():
        return float("inf")
    da, db = chamfer_dt(ea), chamfer_dt(eb)
    return float((db[ea].mean() + da[eb].mean()) / 2.0)


def is_rtl(lang):
    try:
        import lang as langmod
        return langmod.load_profile(lang)["dir"] == "rtl"
    except Exception:
        return False


def has_raqm():
    try:
        from PIL import features
        return bool(features.check("raqm"))
    except Exception:
        return False


def reverse_visual(text):
    """Logical RTL string -> visual order for a renderer without bidi; Latin/digit runs stay readable."""
    rev = text[::-1]
    return re.sub(r"[A-Za-z0-9]+", lambda m: m.group(0)[::-1], rev)


def render_mask(font_path, text, height, rtl=False, raqm=False):
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    size = max(8, int(round(height)))
    kw = {}
    if rtl:
        if raqm:
            kw["direction"] = "rtl"
        else:
            text = reverse_visual(text)
    font = ImageFont.truetype(str(font_path), size)
    l, t, r, b = font.getbbox(text, **kw)
    im = Image.new("L", (max(1, r - l) + 8, max(1, b - t) + 8), 255)
    ImageDraw.Draw(im).text((4 - l, 4 - t), text, font=font, fill=0, **kw)
    return binarise(np.asarray(im), border=0)


def load_crop(path):
    from PIL import Image
    import numpy as np
    try:
        with Image.open(path) as im:
            return binarise(np.asarray(im.convert("L")))
    except Exception as e:
        raise ValueError(f"cannot read crop as an image: {path} ({type(e).__name__})")


def rank_full(crop_path, text, font_files, lang="he", top=5):
    rtl = is_rtl(lang)
    raqm = has_raqm()
    mask = load_crop(crop_path)
    crop = normalise(mask)
    if crop is None:
        raise ValueError("crop has no ink")
    ink_h = ink_crop(mask).shape[0]
    skipped, scored = [], []
    for f in font_files:
        if not covers(f, text):
            skipped.append(str(f))
            continue
        try:
            cand = normalise(render_mask(f, text, ink_h * 1.4, rtl, raqm))
            s = chamfer_score(crop, cand) if cand is not None else float("inf")
        except Exception:
            continue
        if s != float("inf"):
            scored.append({"family": family_name(f), "file": str(f), "score": round(s, 4)})
    scored.sort(key=lambda r: r["score"])
    fams, order = {}, []
    for r in scored:
        if r["family"] not in fams:
            fams[r["family"]] = dict(r, other_files=[])
            order.append(r["family"])
        else:
            fams[r["family"]]["other_files"].append(r["file"])
    bidi = "none" if not rtl else ("raqm" if raqm else "reversed_fallback")
    return {"bidi": bidi, "skipped_no_glyphs": skipped, "results": [fams[k] for k in order[:top]]}


def rank(crop_path, text, font_files, lang="he", top=5):
    return rank_full(crop_path, text, font_files, lang, top)["results"]


def main(argv=None):
    store.print_utf8()
    ap = argparse.ArgumentParser(description="rank candidate fonts against a text crop")
    ap.add_argument("crop")
    ap.add_argument("--text", required=True)
    ap.add_argument("--fonts-dir", action="append", default=[])
    ap.add_argument("--lang", default="he")
    a = ap.parse_args(argv)
    try:
        import numpy  # noqa: F401
        import PIL  # noqa: F401
    except ImportError:
        print("missing Pillow/numpy: pip install pillow numpy", file=sys.stderr)
        return 3
    if not Path(a.crop).is_file():
        print(f"crop not found: {a.crop}", file=sys.stderr)
        return 2
    fonts = list_fonts(a.fonts_dir or [str(d) for d in _system_dirs()])
    if not fonts:
        print("no fonts found: pass --fonts-dir", file=sys.stderr)
        return 2
    try:
        out = rank_full(a.crop, a.text, fonts, a.lang)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
