import json
import random

import pytest

import palette
from conftest import run_cli

ROLE_KEYS = {"background", "surface", "text", "muted", "brand", "brand_deep", "tint", "accent", "highlight",
             "caption_pill", "cta"}
CHECK_KEYS = {"text_on_background", "text_on_surface", "cta_text", "accent_vs_brand_cvd", "pass"}


def test_parse_hex_forms():
    assert palette.parse_hex("008754") == "#008754"
    assert palette.parse_hex("#0a5") == "#00AA55"
    assert palette.parse_hex(" #008754 ") == "#008754"
    for bad in ("rgb(0,0,0)", "##008754", "#12345", ""):
        with pytest.raises(ValueError, match="hex colour like #008754"):
            palette.parse_hex(bad)


def test_oklch_roundtrip():
    rnd = random.Random(7)
    for _ in range(20):
        h = "#" + "".join(f"{rnd.randrange(256):02X}" for _ in range(3))
        assert palette.delta_e2000(h, palette.oklch_to_hex(*palette.hex_to_oklch(h))) < 0.5, h


def test_oklch_gamut_clamp():
    out = palette.oklch_to_hex(0.7, 0.5, 145)  # far outside sRGB
    L, C, H = palette.hex_to_oklch(out)
    assert abs(L - 0.7) < 0.01 and abs(H - 145) < 3 and C < 0.5


def test_ramp_monotonic():
    for h in ("#008754", "#7C5CFF", "#F18044"):
        r = palette.ramp(h)
        assert list(r) == [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950]
        ls = [palette.hex_to_oklch(r[k])[0] for k in r]
        assert all(a > b for a, b in zip(ls, ls[1:])), (h, ls)
        assert ls[0] > 0.95 and ls[-1] < 0.25


def test_contrast_known():
    assert palette.contrast("#FFFFFF", "#000000") == pytest.approx(21, rel=1e-3)
    assert palette.contrast("#777777", "#777777") == pytest.approx(1)


def test_delta_e2000_sharma_pair():
    # Sharma, Wu and Dalal (2005) test data, pair 1
    assert palette._de2000((50, 2.6772, -79.7751), (50, 0, -82.7485)) == pytest.approx(2.0425, abs=1e-4)


def test_accent_rule():
    def hue(h):
        return palette.hex_to_oklch(h)[2]
    assert abs(hue(palette.accent_for("#008754")) - 50) < 3   # cool brand -> warm orange
    assert abs(hue(palette.accent_for("#E11D48")) - 250) < 3  # warm brand -> blue


def test_proposals_shape():
    ps = palette.proposals("#008754")
    assert [p["kind"] for p in ps] == ["monochrome", "analogous", "accent"]
    for p in ps:
        assert set(p) == {"kind", "brand", "ramp", "neutrals", "accent", "roles", "checks"}
        assert p["brand"] == "#008754"
        assert set(p["roles"]) == {"dark", "light"}
        for theme in p["roles"].values():
            assert set(theme) == ROLE_KEYS
        assert set(p["checks"]) == CHECK_KEYS


def test_teal_fixture():
    p = palette.proposals("#008754")[2]
    assert p["kind"] == "accent"
    de = palette.delta_e2000
    light = p["roles"]["light"]
    assert de(p["accent"], "#F18044") <= 10
    assert de(light["brand_deep"], "#014D46") <= 12
    assert de(light["tint"], "#BFE9CF") <= 12
    assert de(light["background"], "#F5F6F1") <= 12


@pytest.mark.parametrize("h", ["#008754", "#7C5CFF", "#E11D48", "#111111", "#FFD400"])
def test_all_proposals_pass_checks(h):
    for p in palette.proposals(h):
        c = p["checks"]
        assert c["pass"] is True, (h, p["kind"], c)
        # never reported falsely: recompute from the roles
        for theme in p["roles"].values():
            assert palette.contrast(theme["text"], theme["background"]) >= c["text_on_background"] - 0.01
            assert palette.contrast(theme["text"], theme["surface"]) >= c["text_on_surface"] - 0.01
        assert palette.cvd_delta(p["accent"], p["brand"]) >= 10


def test_cli_prints_and_json(home):
    r = run_cli("palette.py", "#008754")
    assert r.returncode == 0, r.stderr
    for kind in ("monochrome", "analogous", "accent"):
        assert kind in r.stdout
    assert "caption_pill" in r.stdout and "PASS" in r.stdout
    r = run_cli("palette.py", "008754", "--json")
    assert r.returncode == 0, r.stderr
    assert [p["kind"] for p in json.loads(r.stdout)] == ["monochrome", "analogous", "accent"]
    r = run_cli("palette.py", "rgb(0,0,0)")
    assert r.returncode == 1 and "hex colour like #008754" in r.stderr and "Traceback" not in r.stderr


def _profile(home, slug):
    return json.loads((home / "clients" / slug / "client.json").read_text(encoding="utf-8"))


def test_save_to_client(home):
    assert run_cli("clients.py", "add", "--name", "Teal Co", "--color", "#008754").returncode == 0
    r = run_cli("palette.py", "#008754", "--save", "Teal Co", "--pick", "accent")
    assert r.returncode == 0, r.stderr
    pal = _profile(home, "teal-co")["palette"]
    assert pal["kind"] == "accent" and pal["source_hex"] == "#008754"
    assert pal["picked"] and set(pal["roles"]["light"]) == ROLE_KEYS


def test_save_errors(home):
    r = run_cli("palette.py", "#008754", "--save", "ghost", "--pick", "accent")
    assert r.returncode == 1 and "no client" in r.stderr
    assert run_cli("clients.py", "add", "--name", "Teal Co").returncode == 0
    r = run_cli("palette.py", "#008754", "--save", "teal-co")
    assert r.returncode == 1 and "--pick" in r.stderr
    assert _profile(home, "teal-co")["palette"] is None


def test_clients_palette_command(home):
    assert run_cli("clients.py", "add", "--name", "Teal Co", "--color", "#008754").returncode == 0
    r = run_cli("clients.py", "palette", "teal-co")
    assert r.returncode == 0, r.stderr
    assert "accent" in r.stdout and _profile(home, "teal-co")["palette"] is None  # show only
    r = run_cli("clients.py", "palette", "teal-co", "--pick", "monochrome")
    assert r.returncode == 0, r.stderr
    pal = _profile(home, "teal-co")["palette"]
    assert pal["kind"] == "monochrome" and pal["source_hex"] == "#008754"
    assert run_cli("clients.py", "add", "--name", "No Colour").returncode == 0
    r = run_cli("clients.py", "palette", "no-colour")
    assert r.returncode == 1 and "brand colour" in r.stderr
