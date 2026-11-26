"""Three palette proposals (monochrome, analogous, accent) from one brand colour.

Each proposal maps colours to the roles a video style uses, for a dark and a light theme, and carries
contrast (WCAG 2.x) and colour-blind (deuteranopia, Machado 2009 at severity 1.0) checks.

Colour maths: OKLab/OKLCH per Bjorn Ottosson; CIEDE2000 per Sharma, Wu and Dalal (2005).

Commands:
    palette.py <hex> [--json]                                  print the three proposals
    palette.py <hex> --save <client> --pick monochrome|analogous|accent
                                                               write one into the client profile `palette`
"""
import argparse
import datetime as dt
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

KINDS = ("monochrome", "analogous", "accent")
ROLE_KEYS = ("background", "surface", "text", "muted", "brand", "brand_deep", "tint", "accent", "highlight",
             "caption_pill", "cta")
STEPS = (50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950)
HEX_MSG = "hex colour like #008754"
MIN_TEXT = 4.5     # every text ratio in checks
MIN_CVD = 10.0     # accent vs brand, CIEDE2000 under deuteranopia
TEXT_TARGET = 7.0  # text is built for AAA; nudged in L when it misses
SHAPE_MIN = 3.0    # brand role vs background (WCAG non-text contrast)
LARGE_TEXT = 3.0   # accent as large headline words on the background
ONE_ACCENT_DE = 20.0   # one_accent: accent vs brand CIEDE2000 at least this
ONE_ACCENT_HUE = 40.0  # one_accent: non-accent roles (C >= 0.02) within this many degrees of the brand hue
NEUTRAL_BRAND_C = 0.02  # brands under this OKLCH chroma are neutral: accent hue 250, hue noise ignored
CHECK_BARS = {"text_on_background": MIN_TEXT, "text_on_surface": MIN_TEXT, "muted_on_background": MIN_TEXT,
              "brand_on_background": SHAPE_MIN, "caption_text": MIN_TEXT, "cta_text": MIN_TEXT,
              "text_on_highlight": MIN_TEXT, "accent_large_text": LARGE_TEXT, "accent_vs_brand_cvd": MIN_CVD}

# ---- Final role table (tuned on the owner's teal set; Task 7 copies it into references/palette.md).
# Ramp step -> (OKLCH L, chroma as a factor of the brand chroma). Hue = brand hue; steps darker than 500
# drift toward blue (hue 260) by up to RAMP_HUE_SHIFT degrees at 950.
RAMP_TABLE = {50: (0.97, 0.10), 100: (0.94, 0.25), 200: (0.895, 0.44), 300: (0.82, 0.62), 400: (0.73, 0.82),
              500: (0.64, 0.96), 600: (0.55, 1.00), 700: (0.465, 0.85), 800: (0.38, 0.60), 900: (0.30, 0.45),
              950: (0.22, 0.32)}
RAMP_HUE_SHIFT = 22.0
NEUTRAL_C = 0.008           # neutrals: brand hue, chroma min(NEUTRAL_C, brand C), same L steps
ACCENT_L, ACCENT_C = 0.70, 0.16
ACCENT_WARM_H, ACCENT_COOL_H = 50.0, 250.0
ANALOGOUS_STEP = 30.0
ANALOGOUS_MIN_C = 0.10      # a grey brand still gets a coloured neighbour
HIGHLIGHT_C = 0.45          # highlight = accent hue at the given L, accent chroma x this
# role -> (light theme, dark theme).  (L, C) = OKLCH at the brand hue, C capped at the brand chroma;
# "ramp:N" / "neutral:N" = that step; "brand" = brand colour (nudged to 3:1 vs background when needed);
# "accent" = proposal accent; "accent:L" = accent hue at L; "cta" = CTA fill of the proposal.
ROLE_TABLE = {
    "background":   ("neutral:50",   (0.18, 0.020)),
    "surface":      ((0.995, 0.004), (0.235, 0.025)),
    "text":         ((0.22, 0.030),  (0.97, 0.008)),
    "muted":        ((0.48, 0.020),  (0.74, 0.020)),
    "brand":        ("brand",        "brand"),
    "brand_deep":   ("ramp:800",     "ramp:900"),
    "tint":         ("ramp:200",     "ramp:800"),
    "accent":       ("accent",       "accent"),
    "highlight":    ("accent:0.88",  "accent:0.80"),
    "caption_pill": ("ramp:800",     (0.13, 0.020)),
    "cta":          ("cta",          "cta"),
}
# Per kind: accent = monochrome ramp 300 (ramp 800 for light brands) | analogous neighbour hue at brand L |
# accent rule colour; CTA fill = brand | the theme accent | the theme accent.
# Bars, held per theme by moving L (accent: L first, then hue in 10 degree steps):
#   text >= 7:1 on background and surface; muted >= 4.5 on background; brand role >= 3:1 on background;
#   theme accent vs theme brand: deuteranopia dE >= 10 and dE >= 20 (the accent fill moves only for this);
#   labels.accent_text = theme accent moved in L only until 3:1 on background (and still CVD >= 10);
#   cta, highlight and caption_pill: best label (theme text, background, white or black) >= 4.5.
# Labels are stored per theme in `labels`; checks report the worse of the two themes.


# ---------------------------------------------------------------- parsing

def parse_hex(s):
    """'#rgb' / '#rrggbb' (with or without one '#', spaces ok) -> '#RRGGBB'; ValueError otherwise."""
    h = str(s).strip()
    h = h[1:] if h.startswith("#") else h
    if re.fullmatch(r"[0-9a-fA-F]{3}", h):
        h = "".join(c * 2 for c in h)
    if not re.fullmatch(r"[0-9a-fA-F]{6}", h):
        raise ValueError(f"{s!r}: {HEX_MSG}")
    return "#" + h.upper()


# ---------------------------------------------------------------- sRGB <-> linear <-> OKLab

def _hex_rgb(h):
    h = parse_hex(h)
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))


def _rgb_hex(rgb):
    return "#" + "".join(f"{round(min(1.0, max(0.0, c)) * 255):02X}" for c in rgb)


def _lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _gam(c):
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def _lin_to_oklab(r, g, b):
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s = (math.copysign(abs(x) ** (1 / 3), x) for x in (l, m, s))
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def _oklab_to_lin(L, a, b):
    l = L + 0.3963377774 * a + 0.2158037573 * b
    m = L - 0.1055613458 * a - 0.0638541728 * b
    s = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l ** 3, m ** 3, s ** 3
    return (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)


def hex_to_oklch(h):
    L, a, b = _lin_to_oklab(*(_lin(c) for c in _hex_rgb(h)))
    return L, math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360


def oklch_to_hex(L, C, H):
    """OKLCH -> '#RRGGBB'; out-of-gamut colours keep L and H and lose chroma until they fit sRGB."""
    L = min(1.0, max(0.0, L))
    hr = math.radians(H)

    def lin(c):
        return _oklab_to_lin(L, c * math.cos(hr), c * math.sin(hr))

    eps = 1e-6
    if all(-eps <= x <= 1 + eps for x in lin(C)):
        return _rgb_hex(_gam(min(1.0, max(0.0, x))) for x in lin(C))
    lo, hi = 0.0, C
    for _ in range(30):
        mid = (lo + hi) / 2
        if all(-eps <= x <= 1 + eps for x in lin(mid)):
            lo = mid
        else:
            hi = mid
    return _rgb_hex(_gam(min(1.0, max(0.0, x))) for x in lin(lo))


# ---------------------------------------------------------------- metrics

def _luminance(h):
    r, g, b = (_lin(c) for c in _hex_rgb(h))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    """WCAG 2.x contrast ratio, 1..21."""
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _lab(h):
    """sRGB hex -> CIELAB (D65)."""
    r, g, b = (_lin(c) for c in _hex_rgb(h))
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883

    def f(t):
        return t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def delta_e2000(c1, c2):
    """CIEDE2000 colour difference between two hex colours (kL = kC = kH = 1)."""
    return _de2000(_lab(c1), _lab(c2))


def _de2000(lab1, lab2):
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1, C2 = math.hypot(a1, b1), math.hypot(a2, b2)
    Cb7 = ((C1 + C2) / 2) ** 7
    G = 0.5 * (1 - math.sqrt(Cb7 / (Cb7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360 if C1p else 0.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360 if C2p else 0.0
    dLp, dCp = L2 - L1, C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    else:
        dhp = h2p - h1p
        if dhp > 180:
            dhp -= 360
        elif dhp < -180:
            dhp += 360
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2))
    Lbp, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    if C1p * C2p == 0:
        hbp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbp = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        hbp = (h1p + h2p + 360) / 2
    else:
        hbp = (h1p + h2p - 360) / 2
    T = (1 - 0.17 * math.cos(math.radians(hbp - 30)) + 0.24 * math.cos(math.radians(2 * hbp))
         + 0.32 * math.cos(math.radians(3 * hbp + 6)) - 0.20 * math.cos(math.radians(4 * hbp - 63)))
    dtheta = 30 * math.exp(-(((hbp - 275) / 25) ** 2))
    Cbp7 = Cbp ** 7
    RC = 2 * math.sqrt(Cbp7 / (Cbp7 + 25 ** 7))
    SL = 1 + 0.015 * (Lbp - 50) ** 2 / math.sqrt(20 + (Lbp - 50) ** 2)
    SC = 1 + 0.045 * Cbp
    SH = 1 + 0.015 * Cbp * T
    RT = -math.sin(math.radians(2 * dtheta)) * RC
    return math.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2 + RT * (dCp / SC) * (dHp / SH))


# Machado, Oliveira and Fernandes (2009), deuteranomaly at severity 1.0, applied to linear RGB.
_DEUTAN = ((0.367322, 0.860646, -0.227968),
           (0.280085, 0.672501, 0.047413),
           (-0.011820, 0.042940, 0.968881))


def simulate_deuteranopia(h):
    lin = [_lin(c) for c in _hex_rgb(h)]
    return _rgb_hex(_gam(min(1.0, max(0.0, sum(m * c for m, c in zip(row, lin))))) for row in _DEUTAN)


def cvd_delta(a, b):
    """CIEDE2000 between two colours as a deuteranope sees them."""
    return delta_e2000(simulate_deuteranopia(a), simulate_deuteranopia(b))


# ---------------------------------------------------------------- ramps and hue rules

def _toward_blue(H, amount):
    d = (260.0 - H + 540.0) % 360.0 - 180.0
    return (H + math.copysign(min(abs(d), amount), d)) % 360.0


def ramp(h):
    """Tonal ramp 50..950: L 0.97 -> 0.22, chroma eased toward 0 at both ends, darks drift toward blue."""
    _, C, H = hex_to_oklch(h)
    out = {}
    for step, (L, f) in RAMP_TABLE.items():
        dark = max(0.0, (RAMP_TABLE[500][0] - L) / (RAMP_TABLE[500][0] - RAMP_TABLE[950][0]))
        out[step] = oklch_to_hex(L, C * f, _toward_blue(H, RAMP_HUE_SHIFT * dark))
    return out


def neutrals(h):
    """Near-grey ramp at the brand hue (same L steps as ramp)."""
    _, C, H = hex_to_oklch(h)
    return {step: oklch_to_hex(L, min(NEUTRAL_C, C), H) for step, (L, _f) in RAMP_TABLE.items()}


def is_cool(H):
    return 90.0 <= H < 270.0


def accent_for(h):
    """Accent rule: cool brand hue [90, 270) -> warm orange (hue 50), else blue (hue 250); L 0.70, C 0.16.
    Neutral brands (C < 0.02: greys, black, white) always get blue."""
    _, C, H = hex_to_oklch(h)
    warm = C >= NEUTRAL_BRAND_C and is_cool(H)
    return oklch_to_hex(ACCENT_L, ACCENT_C, ACCENT_WARM_H if warm else ACCENT_COOL_H)


def neighbour_hue(H):
    """Analogous neighbour: brand hue +-30, toward the warm side for cool brands, staying warm for warm ones."""
    if is_cool(H):
        return (H - ANALOGOUS_STEP) % 360 if H < 180 else (H + ANALOGOUS_STEP) % 360
    return (H + ANALOGOUS_STEP) % 360 if H >= 270 else (H - ANALOGOUS_STEP) % 360


# ---------------------------------------------------------------- adjustments and roles

def _nudge(hexc, ok, max_dl=1.0):
    """Smallest OKLCH lightness change (hue and chroma held, chroma gamut-clamped) that makes ok(hex) true.
    Returns None when no lightness works."""
    if ok(hexc):
        return hexc
    L, C, H = hex_to_oklch(hexc)
    for i in range(1, int(max_dl * 200) + 1):
        for dl in (i / 200, -i / 200):
            if 0.0 <= L + dl <= 1.0:
                c = oklch_to_hex(L + dl, C, H)
                if ok(c):
                    return c
    return None


def _nudge_hue(hexc, ok):
    """Lightness first; when no lightness works, rotate the hue in 10 degree steps (nearest first)."""
    got = _nudge(hexc, ok)
    if got:
        return got
    L, C, H = hex_to_oklch(hexc)
    C = max(C, ACCENT_C)
    for d in range(10, 181, 10):
        for s in (d, -d):
            got = _nudge(oklch_to_hex(L, C, (H + s) % 360), ok)
            if got:
                return got
    return None


def best_label(fill, roles):
    """Label colour for words on `fill`: the theme's text, background, white or black, most contrast wins."""
    cands = (roles["text"], roles["background"], "#FFFFFF", "#000000")
    return max(cands, key=lambda c: contrast(fill, c))


def _hue_gap(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _accent_ok(c, brand):
    return cvd_delta(c, brand) >= MIN_CVD and delta_e2000(c, brand) >= ONE_ACCENT_DE


def _resolve(spec, ctx):
    if isinstance(spec, tuple):
        L, c = spec
        return oklch_to_hex(L, min(c, ctx["C"]), ctx["H"])
    kind, _, arg = spec.partition(":")
    if kind == "ramp":
        return ctx["ramp"][int(arg)]
    if kind == "neutral":
        return ctx["neutrals"][int(arg)]
    if kind == "accent" and arg:
        _, aC, aH = hex_to_oklch(ctx["accent"])
        return oklch_to_hex(float(arg), aC * HIGHLIGHT_C, aH)
    return ctx[kind]  # accent, cta, brand


def _theme(i, ctx):
    """Roles and label colours of one theme (0 light, 1 dark), every role held to its bar by moving L."""
    r = {k: _resolve(ROLE_TABLE[k][i], ctx) for k in ROLE_KEYS}
    bg, surface = r["background"], r["surface"]
    r["text"] = _nudge(r["text"], lambda c: min(contrast(c, bg), contrast(c, surface)) >= TEXT_TARGET) or r["text"]
    r["muted"] = _nudge(r["muted"], lambda c: contrast(c, bg) >= MIN_TEXT) or r["muted"]
    r["brand"] = _nudge(r["brand"], lambda c: contrast(c, bg) >= SHAPE_MIN) or r["brand"]
    # the accent is checked against this theme's brand role (which may have moved), not the input colour
    # the accent fill only moves for the colour-blind / one-accent rule, never for text contrast
    r["accent"] = _nudge_hue(r["accent"], lambda c: _accent_ok(c, r["brand"])) or r["accent"]
    if ctx["cta_is_accent"]:
        r["cta"] = r["accent"]
    hl_L = ROLE_TABLE["highlight"][i]
    if isinstance(hl_L, str) and hl_L.startswith("accent:"):
        _, aC, aH = hex_to_oklch(r["accent"])
        r["highlight"] = oklch_to_hex(float(hl_L.split(":")[1]), aC * HIGHLIGHT_C, aH)
    for k in ("cta", "highlight", "caption_pill"):
        r[k] = _nudge(r[k], lambda c: contrast(c, best_label(c, r)) >= MIN_TEXT) or r[k]
    labels = {"caption": best_label(r["caption_pill"], r), "cta": best_label(r["cta"], r),
              "highlight": best_label(r["highlight"], r),
              # accent as headline words: the accent moved in L only as far as 3:1 on background needs
              "accent_text": _nudge(r["accent"], lambda c: contrast(c, bg) >= LARGE_TEXT
                                    and cvd_delta(c, r["brand"]) >= MIN_CVD) or r["accent"]}
    return r, labels


def _one_accent(r, H, brand_C):
    if delta_e2000(r["accent"], r["brand"]) < ONE_ACCENT_DE:
        return False
    if brand_C < NEUTRAL_BRAND_C:
        return True  # a grey brand has no hue to drift from; every non-accent role is grey too
    for k in ROLE_KEYS:
        if k in ("accent", "highlight", "cta"):
            continue
        _, c, h = hex_to_oklch(r[k])
        if c >= 0.02 and _hue_gap(h, H) > ONE_ACCENT_HUE:
            return False
    return True


def _check(p, H, brand_C):
    pairs = [(p["roles"][t], p["labels"][t]) for t in ("light", "dark")]

    def worst(fn):
        return min(fn(r, lab) for r, lab in pairs)

    raw = {
        "text_on_background": worst(lambda r, lab: contrast(r["text"], r["background"])),
        "text_on_surface": worst(lambda r, lab: contrast(r["text"], r["surface"])),
        "muted_on_background": worst(lambda r, lab: contrast(r["muted"], r["background"])),
        "brand_on_background": worst(lambda r, lab: contrast(r["brand"], r["background"])),
        "caption_text": worst(lambda r, lab: contrast(lab["caption"], r["caption_pill"])),
        "cta_text": worst(lambda r, lab: contrast(lab["cta"], r["cta"])),
        "text_on_highlight": worst(lambda r, lab: contrast(lab["highlight"], r["highlight"])),
        "accent_large_text": worst(lambda r, lab: contrast(lab["accent_text"], r["background"])),
        "accent_vs_brand_cvd": worst(lambda r, lab: cvd_delta(r["accent"], r["brand"])),
    }
    one = all(_one_accent(r, H, brand_C) for r, _ in pairs)
    # bars are compared on the raw values; rounding is for the stored / printed numbers only
    ok = one and all(raw[k] >= bar for k, bar in CHECK_BARS.items())
    c = {k: round(v, 2) for k, v in raw.items()}
    c["one_accent"] = one
    c["pass"] = bool(ok)
    return c


def proposals(h):
    """Exactly three proposals for one brand colour, in order monochrome, analogous, accent."""
    brand = parse_hex(h)
    L, C, H = hex_to_oklch(brand)
    rmp, neu = ramp(brand), neutrals(brand)

    def far(c):
        return cvd_delta(c, brand) >= MIN_CVD and delta_e2000(c, brand) >= ONE_ACCENT_DE

    nb = oklch_to_hex(L, max(C, ANALOGOUS_MIN_C), neighbour_hue(H))
    picks = {"monochrome": (rmp[300] if L < 0.66 else rmp[800], brand),
             "analogous": (nb, None),
             "accent": (accent_for(brand), None)}
    out = []
    for kind in KINDS:
        acc, cta = picks[kind]
        acc = _nudge_hue(acc, far) or acc
        ctx = {"H": H, "C": C, "ramp": rmp, "neutrals": neu, "accent": acc, "cta": cta or acc,
               "cta_is_accent": cta is None, "brand": brand}
        (light, ll), (dark, dl) = _theme(0, ctx), _theme(1, ctx)
        p = {"kind": kind, "brand": brand, "ramp": {str(k): v for k, v in rmp.items()},
             "neutrals": {str(k): v for k, v in neu.items()}, "accent": acc,
             "roles": {"dark": dark, "light": light}, "labels": {"dark": dl, "light": ll}}
        p["checks"] = _check(p, H, C)
        out.append(p)
    return out


# ---------------------------------------------------------------- output and saving

def format_proposals(ps):
    lines = []
    for p in ps:
        c = p["checks"]
        lines.append(f"== {p['kind']}  brand {p['brand']}  accent {p['accent']}  "
                     f"{'PASS' if c['pass'] else 'FAIL'}")
        lines.append(f"   {'role':<15} {'light':<9} {'dark':<9}")
        for k in ROLE_KEYS:
            lines.append(f"   {k:<15} {p['roles']['light'][k]:<9} {p['roles']['dark'][k]:<9}")
        for k in ("caption", "cta", "highlight", "accent_text"):
            lines.append(f"   {k + ' label':<15} {p['labels']['light'][k]:<9} {p['labels']['dark'][k]:<9}")
        lines.append("   checks: " + "  ".join(f"{k} {v}" for k, v in c.items() if k != "pass"))
        lines.append("")
    return "\n".join(lines)


def save(client_key, source_hex, kind):
    """Write proposal `kind` for source_hex into the client's profile `palette`. Returns an exit code."""
    import clients
    import store
    if kind not in KINDS:
        print(f"--pick must be one of {', '.join(KINDS)}", file=sys.stderr)
        return 1
    try:
        src = parse_hex(source_hex)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    chosen = next(p for p in proposals(src) if p["kind"] == kind)
    if not chosen["checks"]["pass"]:
        print(f"warning: the {kind} proposal fails its checks: {chosen['checks']}", file=sys.stderr)
    box = {}

    def fn(reg):
        e = clients._find_entry(reg, client_key)
        c = store.read_json(clients.profile_path(e["slug"]), None) if e else None
        if not c:
            raise clients._Abort(clients._fail(f"no client {client_key!r}"))
        c["palette"] = {**chosen, "source_hex": src, "picked": dt.date.today().isoformat()}
        clients.save_client(c)
        box["slug"] = c["slug"]

    try:
        store.update_json(clients.registry_path(), clients._empty(), fn)
    except clients._Abort as e:
        return e.rc
    print(f"saved the {kind} palette from {src} to {box['slug']}")
    return 0


def main(argv=None):
    import store
    return store.exit_on_lock_timeout(_main, argv)


def _main(argv=None):
    import store
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hex")
    p.add_argument("--json", action="store_true")
    p.add_argument("--save", metavar="CLIENT")
    p.add_argument("--pick", choices=KINDS)
    args = p.parse_args(argv)
    store.print_utf8()
    try:
        src = parse_hex(args.hex)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    if args.save:
        if not args.pick:
            print("--save needs --pick monochrome|analogous|accent", file=sys.stderr)
            return 1
        return save(args.save, src, args.pick)
    if args.pick:
        print("--pick needs --save <client>", file=sys.stderr)
        return 1
    ps = proposals(src)
    print(json.dumps(ps, indent=2) if args.json else format_proposals(ps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
