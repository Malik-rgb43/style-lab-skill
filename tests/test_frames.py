import json

import pytest

import clients
import frames
import hf
import ideas
import picks
import store
from conftest import run_cli
from fixtures import plan5 as P

PNG = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000"
                    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082")


@pytest.fixture
def fake_hf(monkeypatch, tmp_path):
    calls = []

    def snapshot(cwd, at_s):
        calls.append((str(cwd), at_s))
        p = tmp_path / "snaps" / f"{len(calls)}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(PNG)
        return p
    monkeypatch.setattr(hf, "snapshot", snapshot)
    monkeypatch.setattr(hf, "run", lambda *a, **k: pytest.fail("hf.run must not be called"))
    return calls


def _world(home, tmp_path, client="example-shop", palette=True, recipe=None):
    P.write_style(home, tmp_path)
    if client:
        P.write_client(run_cli, client, palette=palette)
    d = P.write_project(tmp_path, client=client)
    s = P.good_set(1)
    if recipe:
        s[0]["recipe"] = recipe
    P.set_options(d, {1: s, 2: P.good_set(2), 3: P.good_set(3)})
    picks.pick(d, "plan", 1, "B01-p1")
    return d


def _fill(d, n=1, layouts=("grid", "stack", "split"), colours=("var(--sl-accent)",) * 3, body=None, extra=""):
    sd = d / "frames" / "B01" / f"s{n}"
    for k, x in enumerate("abc"):
        vd = sd / x
        v = store.read_json(vd / "variant.json", {})
        v.update(title=f"t{x}", composition=layouts[k], treatment=f"treat-{k}", wild=(x == "c"))
        store.write_json(vd / "variant.json", v)
        inner = body[k] if body else f'<div class="{layouts[k]}" style="color:{colours[k]}">יש 3 דרכים</div>{extra}'
        html = (vd / "index.html").read_text(encoding="utf-8")
        html = html.replace('<div id="text" class="clip" data-start="0" data-duration="2">{{BEAT_TEXT}}</div>',
                            inner)
        (vd / "index.html").write_text(html, encoding="utf-8")
    return sd


def test_new_set_scaffold_and_tokens(home, tmp_path):
    d = _world(home, tmp_path)
    n, dirs, _ = frames.new_set(d, 1)
    assert n == 1 and [p.name for p in dirs] == ["a", "b", "c"]
    accent = clients.load_client("example-shop")["palette"]["roles"]["dark"]["accent"]
    css = (dirs[0] / "tokens.css").read_text(encoding="utf-8")
    assert f"--sl-accent: {accent};" in css and "--sl-font-heading" in css
    w = json.loads((dirs[0] / "words.json").read_text(encoding="utf-8"))
    assert w["words"][0]["start"] == 0.0 and w["words"][0]["word"] == "יש"
    html = (dirs[0] / "index.html").read_text(encoding="utf-8")
    assert 'data-width="1080" data-height="1920"' in html and 'data-duration="2"' in html and "{{BEAT_TEXT}}" in html
    r = run_cli("frames.py", "new-set", d, "--beat", "2")
    assert r.returncode == 1 and "plan pick" in r.stderr


def test_no_client_brand_defaults(home, tmp_path):
    defaults = ideas.brand_defaults()["roles"]["dark"]
    for client, palette in ((None, False), ("example-shop", False)):
        d = _world(home, tmp_path / str(client), client=client, palette=palette)
        _, dirs, _ = frames.new_set(d, 1)
        css = (dirs[0] / "tokens.css").read_text(encoding="utf-8")
        assert f"--sl-accent: {defaults['accent']};" in css
        _fill(d, colours=(defaults["accent"],) * 3)
        assert frames.check_set(d, 1, 1) == []


def test_colour_only_difference_refused(home, tmp_path):
    d = _world(home, tmp_path)
    frames.new_set(d, 1)
    _fill(d, layouts=("grid", "grid", "grid"), colours=("var(--sl-accent)", "var(--sl-brand)", "#ffffff"))
    probs = "\n".join(frames.check_set(d, 1, 1))
    assert "differ only in colour" in probs and "same composition and treatment" not in probs


def test_foreign_hex_refused(home, tmp_path):
    d = _world(home, tmp_path)
    frames.new_set(d, 1)
    _fill(d, colours=("#FF0000", "var(--sl-brand)", "var(--sl-text)"))
    probs = frames.check_set(d, 1, 1)
    assert any("#FF0000" in p for p in probs)


def test_real_words_required(home, tmp_path):
    d = _world(home, tmp_path)
    frames.new_set(d, 1)
    _fill(d, body=('<div class="x">hello world</div>', '<div class="y">{{BEAT_TEXT}}</div>',
                   '<div class="z">lorem ipsum דרכים</div>'))
    probs = "\n".join(frames.check_set(d, 1, 1))
    assert "B1 s1 a: none of the beat's words" in probs
    assert "B1 s1 b: a {{...}} placeholder" in probs
    assert "B1 s1 c: sample text 'lorem'" in probs


def _queue(d, status):
    f = d / "gen" / "files" / "a.png"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(PNG)
    store.write_json(d / "gen" / "queue.json", {"version": 1, "items": [
        {"id": "G001", "beat": 1, "status": status, "file": "gen/files/a.png", "kind": "image"}]})


def test_ai_beat_needs_generated_image(home, tmp_path):
    d = _world(home, tmp_path, recipe="paper-cutout")
    frames.new_set(d, 1)
    _fill(d)
    assert any("AI beat without a generated image" in p for p in frames.check_set(d, 1, 1))
    img = '<img src="../../../../gen/files/a.png">'
    _queue(d, "generated")
    sd = d / "frames" / "B01" / "s1"
    for x in "abc":
        h = (sd / x / "index.html").read_text(encoding="utf-8").replace("</body>", img + "</body>")
        (sd / x / "index.html").write_text(h, encoding="utf-8")
    assert any("has not passed the look check" in p for p in frames.check_set(d, 1, 1))
    _queue(d, "look_pass")
    assert frames.check_set(d, 1, 1) == []


def test_refs_path_refused(home, tmp_path):
    d = _world(home, tmp_path)
    frames.new_set(d, 1)
    (d / "refs").mkdir()
    (d / "refs" / "V01.png").write_bytes(PNG)
    _fill(d, extra='<img src="../../../../refs/V01.png">')
    assert any("refs/ folder" in p for p in frames.check_set(d, 1, 1))
    for x in "abc":
        f = d / "frames" / "B01" / "s1" / x / "index.html"
        f.write_text(f.read_text(encoding="utf-8").replace("../../../../refs/V01.png", "../../../../../outside.png"),
                     encoding="utf-8")
    assert any("outside the project" in p for p in frames.check_set(d, 1, 1))


def test_snap_writes_frames_and_shown(home, tmp_path, fake_hf):
    d = _world(home, tmp_path)
    frames.new_set(d, 1)
    _fill(d)
    stills = frames.snap(d, 1, 1)
    assert len(stills) == 3 and all(p.is_file() for p in stills) and len(fake_hf) == 3
    doc = store.read_json(d / "frames.json", {})
    vs = doc["beats"]["1"]["sets"][0]["variants"]
    assert [v["id"] for v in vs] == ["B01-f1-a", "B01-f1-b", "B01-f1-c"] and vs[0]["still"].endswith("a.png")
    shown = [i for i in store.read_json(d / "shown.json", {})["items"] if i["step"] == "frames"]
    assert len(shown) == 3
    assert frames.check_set(d, 1, 1) == []  # its own shown set is not a repeat


def test_hf_missing_exit3(home, tmp_path, monkeypatch):
    d = _world(home, tmp_path)
    frames.new_set(d, 1)
    _fill(d)

    def missing(*a, **k):
        raise hf.HFMissing(hf.HINT)
    monkeypatch.setattr(hf, "snapshot", missing)
    assert frames.main(["snap", str(d), "--beat", "1", "--set", "1"]) == 3
