import re

import pytest

import pickpage
import picks
import store
from conftest import run_cli
from fixtures import plan5 as P
from test_picks import frames_doc


@pytest.fixture
def world(home, tmp_path):
    P.write_style(home, tmp_path)
    P.write_client(run_cli)
    d = P.write_project(tmp_path)
    P.set_options(d, {1: P.good_set(1), 2: P.good_set(2), 3: P.good_set(3)})
    return d


def test_parse_tolerant():
    text = ("‏# בחירות · השקה · פריימים\r\n"
            "פרויקט： launch\r\n"
            "B1: B01-f1-b\r\n"
            "B2: עוד רעיונות\r\n"
            "B3： B03-f1-a | הערה: יותר‏ חם\r\n"
            "סתם שורה שהבעלים הוסיף\r\n"
            "B4: own | כרטיס משלי\r\n")
    r = pickpage.parse_copy_text(text)
    assert r["step"] == "frames" and r["project"] == "launch"
    assert [(x["beat"], x["picked"], x["more"]) for x in r["lines"]] == [
        (1, "B01-f1-b", False), (2, None, True), (3, "B03-f1-a", False), (4, None, False)]
    assert r["lines"][2]["note"] == "יותר חם" and r["lines"][3]["own"] == "כרטיס משלי"
    bad = "# בחירות · השקה · תוכנית\nפרויקט: launch\nB1: B01-p1\nB2: B02-p1\nB9: B09-p7\n"
    with pytest.raises(pickpage.ParseError) as e:
        pickpage.parse_copy_text(bad, known={1: {"B01-p1"}, 2: {"B02-p1"}, 9: set()})
    assert e.value.line_no == 5 and "line 5" in str(e.value)
    with pytest.raises(pickpage.ParseError):
        pickpage.parse_copy_text("B2: B03-p1\n")  # an id of another beat


def test_copy_text_golden():
    data = {"step": "motion", "project": "launch", "title": "השקה", "mode": "three", "beats": [
        {"n": 1, "recommended": "B01-m1-a", "cards": [{"id": "B01-m1-a"}]},
        {"n": 2, "recommended": "B02-m1-b", "cards": [{"id": "B02-m1-a"}]},
        {"n": 3, "recommended": "B03-m1-a", "cards": [{"id": "B03-m1-a"}]}]}
    state = {2: {"picked": "B02-m1-c", "note": "יותר מהר", "motion": "one"}, 3: {"more": True}}
    assert pickpage.copy_text(data, state) == (
        "# בחירות · השקה · תנועה\nפרויקט: launch\nמצב תנועה: 3\nB1: B01-m1-a\n"
        "B2: B02-m1-c | הערה: יותר מהר\nB2 תנועה: 1\nB3: עוד רעיונות\n")
    r = pickpage.parse_copy_text(pickpage.copy_text(data, state))
    assert r["mode"] == "three" and r["lines"][2]["motion_override"] == "one"


def test_import_roundtrip(world, tmp_path):
    frames_doc(world)
    for b in (1, 2, 3):
        picks.pick(world, "plan", b, f"B0{b}-p1")
    out = pickpage.export(world, "frames", tmp_path / "page.html")
    html = (tmp_path / "page.html").read_text(encoding="utf-8")
    import json
    data = json.loads(re.search(r'id="picks-data">(.*?)</script>', html, re.S).group(1))
    assert len(out["files"]) == 9 and all(k.startswith("media/") for k in out["files"])
    text = pickpage.copy_text(data, {2: {"picked": "B02-f1-c"}, 3: {"more": True}})
    (tmp_path / "t.txt").write_text(text, encoding="utf-8")
    r = run_cli("pickpage.py", "import", world, "--text-file", tmp_path / "t.txt")
    assert r.returncode == 0, r.stderr
    fr = picks.load(world)["frames"]
    assert fr["1"]["picked"] == "B01-f1-a" and fr["2"]["picked"] == "B02-f1-c" and fr["3"]["more"] is True
    assert fr["1"]["decided_by"] == "owner" and picks.load(world)["source"] == "page"


def test_export_refuses_refs_and_outside(world, tmp_path):
    doc = frames_doc(world)
    outside = tmp_path / "elsewhere.png"
    outside.write_bytes(b"x")
    doc["beats"]["1"]["sets"][0]["variants"][0]["still"] = str(outside)
    store.write_json(world / "frames.json", doc)
    r = run_cli("pickpage.py", "export", world, "--step", "frames", "--out", tmp_path / "p.html")
    assert r.returncode == 1 and "outside" in r.stderr
    refs = world / "refs" / "V01.png"
    refs.parent.mkdir()
    refs.write_bytes(b"x")
    doc["beats"]["1"]["sets"][0]["variants"][0]["still"] = "refs/V01.png"
    store.write_json(world / "frames.json", doc)
    r = run_cli("pickpage.py", "export", world, "--step", "frames", "--out", tmp_path / "p.html")
    assert r.returncode == 1 and "refs" in r.stderr


def test_page_self_contained(world, tmp_path):
    frames_doc(world)
    pickpage.export(world, "frames", tmp_path / "p.html")
    html = (tmp_path / "p.html").read_text(encoding="utf-8")
    assert 'id="picks-data"' in html
    urls = [l for l in html.splitlines() if re.search(r"https?://", l)]
    assert urls and all("fonts.googleapis.com" in l for l in urls)
    pickpage.export(world, "plan", tmp_path / "plan.html")
    plan_html = (tmp_path / "plan.html").read_text(encoding="utf-8")
    assert "B01-p1" in plan_html and "machineLines" in plan_html
