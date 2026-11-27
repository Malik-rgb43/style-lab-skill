import json
import re
import shutil

import code_tokens
from conftest import REPO_ROOT, run_cli

FIX = REPO_ROOT / "tests" / "fixtures" / "hf_project"


def test_type_levels_from_fixture():
    t = code_tokens.read_project(FIX)["type"]
    lv = t["levels"]
    assert [l["size_px"] for l in lv] == [120, 48, 40, 32]
    assert [l["name"] for l in lv] == ["L1", "L2", "L3", "L4"]
    assert lv[0]["font"] == "Karantina" and lv[0]["weight"] == 700
    assert lv[2]["font"] == "Heebo" and lv[2]["weight"] == 800
    assert lv[3]["font"] == "Gveret Levin"
    assert lv[0]["colour"] == "#7c3aed"  # var chain resolved
    assert lv[0]["tracking"] == -1  # negative letter-spacing (found by the A0 eval)
    assert abs(t["scale_ratios"][0] - 2.5) <= 0.01
    assert lv[1]["uses"] == 2


def test_eases_and_durations():
    m = code_tokens.read_project(FIX)["motion"]
    p3 = next(e for e in m["eases"] if e.get("name") == "power3.out")
    assert p3["uses"] == 2 and 0.5 <= p3["duration_s"] <= 0.6
    assert "y" in p3["props"] and "opacity" in p3["props"]
    custom = next(e for e in m["eases"] if "custom_ease" in e)
    assert custom["custom_ease"] == "M0,0 C0.2,0.9 0.1,1 1,1"
    assert custom["uses"] == 2
    assert m["durations"]["n"] == 4  # tl.set is not a tween
    assert m["durations"]["p10"] <= m["durations"]["median"] <= m["durations"]["p90"]


def test_colour_roles_counted():
    roles = {r["hex"]: r for r in code_tokens.read_project(FIX)["colour"]["roles"]}
    assert roles["#7c3aed"]["uses"] == 3  # headline (via 2 vars), note, card background
    assert ".headline" in roles["#7c3aed"]["where"] and ".card" in roles["#7c3aed"]["where"]
    assert set(roles["#0b0b12"]["where"]) == {"body", ".caption"}
    assert "#ffffff" in roles


def test_sound_files_basenames():
    files = code_tokens.read_project(FIX)["sound"]["files"]
    assert [f["file"] for f in files] == ["music-bed.mp3", "hit.wav"]
    assert files[0]["volume"] == 0.35 and files[1]["start_s"] == 1.2


def test_layer_source_written(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    r = run_cli("code_tokens.py", "layer", pack, "--set", f"type=code:{FIX}")
    assert r.returncode == 0, r.stderr
    d = json.loads((pack / "layers.json").read_text(encoding="utf-8"))
    assert d["type"]["source"] == "code" and d["type"]["ref"] == str(FIX)
    for l in ("motion", "colour", "broll", "structure", "sound"):
        assert d[l]["source"] == "refs"
    assert run_cli("code_tokens.py", "layer", pack, "--set", "type=bogus").returncode == 2
    assert run_cli("code_tokens.py", "layer", pack, "--set", "nolayer=code").returncode == 2


def test_cli_writes_tokens_json(tmp_path):
    out = tmp_path / "pack" / "tokens.json"
    r = run_cli("code_tokens.py", FIX, "--out", out)
    assert r.returncode == 0, r.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["schema"] == "tokens/1"


def test_path_hint_no_machine_path(tmp_path):
    proj = tmp_path / "my_project"
    shutil.copytree(FIX, proj)
    tok = code_tokens.read_project(proj)
    hint = tok["source"]["path_hint"]
    assert hint == "my_project" and tok["source"]["kind"] == "code"
    assert not re.search(r"[:\\/]", hint)
    assert str(tmp_path) not in json.dumps(tok)


def test_unreadable_project_exit2(tmp_path):
    r = run_cli("code_tokens.py", tmp_path, "--out", tmp_path / "t.json")
    assert r.returncode == 2
    assert len((r.stdout + r.stderr).strip().splitlines()) == 1
    assert not (tmp_path / "t.json").exists()


def test_empty_composition_valid(tmp_path):
    (tmp_path / "index.html").write_text("<html><body></body></html>", encoding="utf-8")
    t = code_tokens.read_project(tmp_path)
    assert t["type"]["levels"] == [] and t["type"]["scale_ratios"] == []
    assert t["motion"]["eases"] == [] and t["motion"]["durations"]["n"] == 0
    assert t["colour"]["roles"] == [] and t["sound"]["files"] == []


def _proj(tmp_path, css="", js="", body=""):
    (tmp_path / "index.html").write_text(
        f"<html><head><style>{css}</style></head><body>{body}<script>{js}</script></body></html>",
        encoding="utf-8")
    return tmp_path


def test_unresolved_and_legacy_eases(tmp_path):
    js = """
    const e = "x";
    tl.to(".a", { x: 1, duration: 0.4, ease: e });
    tl.to(".b", { x: 1, duration: 0.4, ease: Power3.easeOut });
    tl.to(".c", { x: 1, duration: dur, ease: "power2.inOut" });
    tl.to(".d", { x: 1 });
    """
    m = code_tokens.read_project(_proj(tmp_path, js=js))["motion"]
    by = {(e["name"], e.get("unresolved")): e for e in m["eases"]}
    assert by[(None, "ease")]["duration_s"] == 0.4
    assert by[("power3.out", None)]["uses"] == 1
    assert by[("power2.inOut", "duration")]["duration_s"] is None
    assert by[("power1.out", None)]["default"] is True
    assert m["durations"]["n"] == 3  # the variable duration is left out


def test_timeline_defaults_honoured(tmp_path):
    js = """
    const tl = gsap.timeline({ defaults: { ease: "expo.out", duration: 0.8 } });
    tl.to(".a", { x: 1 });
    tl.to(".b", { x: 1, duration: 0.2 });
    """
    m = code_tokens.read_project(_proj(tmp_path, js=js))["motion"]
    assert len(m["eases"]) == 1 and m["eases"][0]["name"] == "expo.out"
    assert "default" not in m["eases"][0]
    assert m["durations"]["n"] == 2 and m["durations"]["p90"] > 0.7


def test_commented_tween_not_counted(tmp_path):
    js = """
    // tl.to(".a", { duration: 1, ease: "linear" });
    /* tl.to(".b", { duration: 2, ease: "linear" }); */
    const u = "http://example.test/x"; tl.to(".c", { x: 1, duration: 0.3, ease: "sine.out" });
    """
    m = code_tokens.read_project(_proj(tmp_path, js=js))["motion"]
    assert [e["name"] for e in m["eases"]] == ["sine.out"]


def test_fromto_with_position(tmp_path):
    js = 'tl.fromTo(".a", { scale: 0.5 }, { scale: 1, duration: 0.7, ease: "back.out" }, 1.5);'
    e = code_tokens.read_project(_proj(tmp_path, js=js))["motion"]["eases"][0]
    assert e["name"] == "back.out" and e["duration_s"] == 0.7 and e["props"] == ["scale"]


def test_zero_use_level_reported(tmp_path):
    t = code_tokens.read_project(_proj(tmp_path, css=".ghost { font-size: 20px; }"))["type"]
    assert t["levels"][0]["uses"] == 0


def test_code_layer_needs_ref_and_path_hint(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    assert run_cli("code_tokens.py", "layer", pack, "--set", "type=code").returncode == 2
    assert run_cli("code_tokens.py", "layer", pack, "--set", "type=client").returncode == 2
    assert run_cli("code_tokens.py", "layer", pack, "--set", f"type=code:{FIX}").returncode == 0
    entry = json.loads((pack / "layers.json").read_text(encoding="utf-8"))["type"]
    assert entry["path_hint"] == "hf_project"
    tok = code_tokens.read_project(FIX)
    assert code_tokens.tokens_match_layer(entry, tok)
    other = dict(tok, source={"kind": "code", "path_hint": "elsewhere"})
    assert not code_tokens.tokens_match_layer(entry, other)


def test_bad_layers_json_exit2(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    for bad in ("[1, 2]", '{"type": 5}', '{"type": {"source": "bogus"}}'):
        (pack / "layers.json").write_text(bad, encoding="utf-8")
        r = run_cli("code_tokens.py", "layer", pack, "--set", f"motion=code:{FIX}")
        assert r.returncode == 2, bad
        assert len(r.stderr.strip().splitlines()) == 1 and "Traceback" not in r.stderr
