import json
import threading
import urllib.error
import urllib.request

import pytest

import picks
import store
import studio
from conftest import run_cli
from fixtures import plan5 as P


@pytest.fixture
def srv(home, tmp_path):
    P.write_style(home, tmp_path)
    P.write_client(run_cli)
    d = P.write_project(tmp_path)
    P.set_options(d, {1: P.good_set(1), 2: P.good_set(2), 3: P.good_set(3)})
    (d / "refs").mkdir()
    (d / "refs" / "V01.png").write_bytes(b"x")
    (d / "frames").mkdir()
    (d / "frames" / "a.png").write_bytes(b"png")
    s, url = studio.make_server(0, token="tok123")
    th = threading.Thread(target=s.serve_forever, daemon=True)
    th.start()
    base = url.split("/?")[0]
    import project
    yield {"base": base, "dir": d, "pid": project.load(d)["id"]}
    s.shutdown()
    s.server_close()


def req(url, data=None, origin=None, cookie=True, token=None):
    h = {"Content-Type": "application/json"}
    if cookie:
        h["Cookie"] = "sl_studio=tok123"
    if origin:
        h["Origin"] = origin
    if token:
        url += ("&" if "?" in url else "?") + "t=" + token
    r = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None, headers=h,
                               method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(r, timeout=20) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_token_required(srv):
    assert req(srv["base"] + "/api/overview", cookie=False)[0] == 403
    code, body = req(srv["base"] + "/", cookie=False, token="tok123")
    assert code == 200 and b"/static/app.js" in body
    code, body = req(srv["base"] + "/api/overview")
    assert code == 200
    o = json.loads(body)
    assert [s["slug"] for s in o["styles"]] == ["paper-3d"] and o["clients"][0]["slug"] == "example-shop"
    assert o["projects"][0]["id"] == srv["pid"]


def test_pick_needs_origin_and_writes_picks_json(srv):
    url = f"{srv['base']}/api/project/{srv['pid']}/pick"
    body = {"step": "plan", "beat": 1, "id": "B01-p2"}
    assert req(url, body)[0] == 403
    assert req(url, body, origin="http://evil.example")[0] == 403
    code, out = req(url, body, origin=srv["base"])
    assert code == 200, out
    pk = picks.load(srv["dir"])
    assert pk["plan"]["1"]["picked"] == "B01-p2" and pk["source"] == "studio" and pk["plan"]["1"]["decided_by"] == "owner"
    code, out = req(url, {"step": "plan", "beat": "x"}, origin=srv["base"])
    assert code == 400


def test_project_room_and_files(srv):
    code, body = req(f"{srv['base']}/api/project/{srv['pid']}")
    room = json.loads(body)
    assert code == 200 and len(room["plan"]["beats"]) == 3
    assert req(f"{srv['base']}/files/project/{srv['pid']}/frames/a.png")[0] == 200
    assert req(f"{srv['base']}/files/project/{srv['pid']}/refs/V01.png")[0] == 403
    assert req(f"{srv['base']}/files/project/{srv['pid']}/../../x")[0] in (403, 404)


def test_motion_mode(srv):
    code, _ = req(f"{srv['base']}/api/project/{srv['pid']}/motion-mode", {"mode": "one"}, origin=srv["base"])
    assert code == 200 and picks.load(srv["dir"])["motion"]["mode"] == "one"
