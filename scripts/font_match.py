#!/usr/bin/env python3
"""Rank candidate fonts against a text crop (chamfer distance on ink edges).

    font_match.py <crop.png> --text "<the words>" [--fonts-dir D ...] [--lang he]

Each candidate font is rendered with Pillow at the crop's text height, both images are
binarised, cropped to the ink box, scaled to a common height and compared by symmetric
chamfer distance on the ink edges (lower is better). Prints the top 5 as JSON:
[{family, file, score}]. Local only: no network, no upload.
Exit codes: 0 ok, 2 bad input, 3 missing tool (Pillow/numpy).
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402

FONT_EXT = {".ttf", ".otf", ".ttc"}
NORM_H = 48


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


def binarise(gray):
    """Ink mask from a grayscale array: ink is the minority side of the mean."""
    import numpy as np
    g = np.asarray(gray, dtype=float)
    thr = (g.min() + g.max()) / 2.0
    dark = g < thr
    return dark if dark.sum() <= (~dark).sum() else ~dark


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


def edges(mask):
    import numpy as np
    p = np.pad(mask, 1)
    inner = p[1:-1, 1:-1]
    all_n = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return inner & ~all_n


def chamfer_dt(points_mask):
    """3-4 chamfer distance transform (pixels) to the nearest True pixel."""
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


def render_mask(font_path, text, height):
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    size = max(8, int(round(height)))
    font = ImageFont.truetype(str(font_path), size)
    l, t, r, b = font.getbbox(text)
    im = Image.new("L", (max(1, r - l) + 8, max(1, b - t) + 8), 255)
    ImageDraw.Draw(im).text((4 - l, 4 - t), text, font=font, fill=0)
    return binarise(np.asarray(im))


def rank(crop_path, text, font_files, top=5):
    from PIL import Image
    import numpy as np
    crop = normalise(binarise(np.asarray(Image.open(crop_path).convert("L"))))
    if crop is None:
        raise ValueError("crop has no ink")
    ink_h = ink_crop(binarise(np.asarray(Image.open(crop_path).convert("L")))).shape[0]
    res = []
    for f in font_files:
        try:
            cand = normalise(render_mask(f, text, ink_h * 1.4))
            if cand is None:
                continue
            s = chamfer_score(crop, cand)
        except Exception:
            continue
        if s != float("inf"):
            res.append({"family": family_name(f), "file": str(f), "score": round(s, 4)})
    res.sort(key=lambda r: r["score"])
    return res[:top]


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
    dirs = a.fonts_dir or [str(d) for d in _system_dirs()]
    fonts = list_fonts(dirs)
    if not fonts:
        print("no fonts found: pass --fonts-dir", file=sys.stderr)
        return 2
    try:
        out = rank(a.crop, a.text, fonts)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
