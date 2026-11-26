import json

from conftest import run_cli


def mkstyle(tmp_path, name="Alpha"):
    pack = tmp_path / ("pack-" + name)
    pack.mkdir(exist_ok=True)
    r = run_cli("styles.py", "add", "--name", name, "--pack", pack)
    assert r.returncode == 0, r.stderr
    return name.lower()


def profile(home, slug):
    return json.loads((home / "clients" / slug / "client.json").read_text(encoding="utf-8"))


def add_client(name="חנות הדגמה", *extra):
    r = run_cli("clients.py", "add", "--name", name, "--color", "#008754", *extra)
    assert r.returncode == 0, r.stderr
    return r


def test_add_and_show_hebrew(home):
    add_client()
    p = profile(home, "chnvt-hdgmh")
    assert p["fonts"] == {"heading": "Karantina 700", "body": "Heebo", "mono": "Consolas"}
    assert p["brand_colors"] == ["#008754"]
    assert p["palette"] is None and p["styles"] == []
    r = run_cli("clients.py", "show", "חנות הדגמה")
    assert r.returncode == 0 and "חנות הדגמה" in r.stdout


def test_link_requires_existing_style(home):
    add_client()
    r = run_cli("clients.py", "link", "chnvt-hdgmh", "ghost")
    assert r.returncode == 1 and "no style" in r.stderr
    assert profile(home, "chnvt-hdgmh")["styles"] == []


def test_link_default_is_single(home, tmp_path):
    a, b = mkstyle(tmp_path, "Alpha"), mkstyle(tmp_path, "Beta")
    add_client()
    assert run_cli("clients.py", "link", "chnvt-hdgmh", a, "--default").returncode == 0
    assert run_cli("clients.py", "link", "chnvt-hdgmh", b, "--default").returncode == 0
    st = {s["slug"]: s["default"] for s in profile(home, "chnvt-hdgmh")["styles"]}
    assert st == {a: False, b: True}


def test_linked_clients_computed(home, tmp_path):
    a = mkstyle(tmp_path)
    add_client()
    run_cli("clients.py", "link", "chnvt-hdgmh", a, "--default")
    r = run_cli("styles.py", "show", a)
    assert json.loads(r.stdout)["clients"] == [
        {"slug": "chnvt-hdgmh", "name": "חנות הדגמה", "default": True}]


def test_dangling_link_reported(home, tmp_path):
    a = mkstyle(tmp_path)
    add_client()
    run_cli("clients.py", "link", "chnvt-hdgmh", a)
    assert run_cli("styles.py", "remove", a).returncode == 0
    r = run_cli("clients.py", "show", "chnvt-hdgmh")
    assert r.returncode == 0 and "(missing style)" in r.stdout
    assert run_cli("styles.py", "list").returncode == 0


def test_remove_keeps_folder(home):
    add_client()
    r = run_cli("clients.py", "remove", "chnvt-hdgmh")
    assert r.returncode == 0 and "folder kept" in r.stdout
    assert (home / "clients" / "chnvt-hdgmh").exists()
    assert json.loads((home / "clients.json").read_text(encoding="utf-8"))["clients"] == []


def test_bad_color_rejected(home):
    r = run_cli("clients.py", "add", "--name", "X", "--color", "rgb(1,2,3)")
    assert r.returncode == 1 and "hex colour like #008754" in r.stderr
    assert not (home / "clients.json").exists()


def test_set_and_unlink(home, tmp_path):
    a = mkstyle(tmp_path)
    add_client()
    r = run_cli("clients.py", "set", "chnvt-hdgmh", "languages=he,en", "fonts.body=Rubik", "tone=warm")
    assert r.returncode == 0, r.stderr
    p = profile(home, "chnvt-hdgmh")
    assert p["languages"] == ["he", "en"] and p["fonts"]["body"] == "Rubik" and p["tone"] == "warm"
    assert run_cli("clients.py", "set", "chnvt-hdgmh", "slug=zz").returncode == 1
    run_cli("clients.py", "link", "chnvt-hdgmh", a)
    assert run_cli("clients.py", "unlink", "chnvt-hdgmh", a).returncode == 0
    assert profile(home, "chnvt-hdgmh")["styles"] == []


def test_readd_does_not_overwrite_kept_folder(home):
    add_client()
    run_cli("clients.py", "set", "chnvt-hdgmh", "notes=keep me")
    run_cli("clients.py", "remove", "chnvt-hdgmh")
    f = home / "clients" / "chnvt-hdgmh" / "client.json"
    before = f.read_text(encoding="utf-8")
    r = run_cli("clients.py", "add", "--slug", "chnvt-hdgmh", "--name", "חנות הדגמה")
    assert r.returncode == 1 and "restore" in r.stderr and "Traceback" not in r.stderr
    assert f.read_text(encoding="utf-8") == before


def test_restore_reregisters(home):
    add_client()
    run_cli("clients.py", "remove", "chnvt-hdgmh")
    assert run_cli("clients.py", "restore", "nope").returncode == 1
    r = run_cli("clients.py", "restore", "chnvt-hdgmh")
    assert r.returncode == 0, r.stderr
    assert run_cli("clients.py", "show", "chnvt-hdgmh").returncode == 0
    assert run_cli("clients.py", "restore", "chnvt-hdgmh").returncode == 1


def test_colour_inputs(home):
    for i, (raw, want) in enumerate([("008754", "#008754"), ("#0a5", "#00AA55"), (" #008754 ", "#008754")]):
        r = run_cli("clients.py", "add", "--slug", f"c{i}", "--name", f"C{i}", "--color", raw)
        assert r.returncode == 0, r.stderr
        assert profile(home, f"c{i}")["brand_colors"] == [want]
    for bad in ("##008754", "rgb(1,2,3)"):
        r = run_cli("clients.py", "add", "--slug", "bad", "--name", "Bad", "--color", bad)
        assert r.returncode == 1 and "Traceback" not in r.stderr


def test_auto_slug_skips_kept_folder(home):
    add_client()
    run_cli("clients.py", "remove", "chnvt-hdgmh")
    r = run_cli("clients.py", "add", "--name", "חנות הדגמה")
    assert r.returncode == 0, r.stderr
    assert "chnvt-hdgmh-2" in r.stdout and "restore" in r.stdout
    assert profile(home, "chnvt-hdgmh-2")["name"] == "חנות הדגמה"


def test_restore_rejects_bad_slug(home):
    outside = home / "x"  # a profile outside clients/ that ../x would reach
    outside.mkdir(parents=True)
    (outside / "client.json").write_text(json.dumps({"slug": "x", "name": "X"}), encoding="utf-8")
    for bad in ("../x", "..", "A", "x/y"):
        r = run_cli("clients.py", "restore", bad)
        assert r.returncode == 1 and "Traceback" not in r.stderr, bad
        assert "lowercase latin letters" in r.stderr, bad
    assert not (home / "clients.json").exists()


def test_clients_uses_palette_parse_hex():
    import clients
    import palette
    assert clients.parse_hex is palette.parse_hex


def _seed_clients(home, rows):
    """rows: (slug, name, aliases). Writes the registry and profiles directly."""
    reg = {"version": 1, "clients": []}
    for slug, name, aliases in rows:
        d = home / "clients" / slug
        d.mkdir(parents=True, exist_ok=True)
        (d / "client.json").write_text(json.dumps(
            {"slug": slug, "name": name, "aliases": aliases, "styles": []}), encoding="utf-8")
        reg["clients"].append({"slug": slug, "name": name, "dir": f"clients/{slug}"})
    (home / "clients.json").write_text(json.dumps(reg), encoding="utf-8")


def _reg_slugs(home):
    return [e["slug"] for e in json.loads((home / "clients.json").read_text(encoding="utf-8"))["clients"]]


def test_find_prefers_slug_then_name_over_alias(home):
    import clients
    _seed_clients(home, [("aaa", "Alpha", ["bbb", "zed"]), ("bbb", "Beta", []), ("ccc", "Zed", []),
                         ("ddd", "eee", []), ("eee", "Echo", [])])
    assert clients.load_client("bbb")["slug"] == "bbb"
    assert clients.load_client("eee")["slug"] == "eee"  # slug beats another client's name
    assert clients.load_client("zed")["slug"] == "ccc"
    r = run_cli("clients.py", "remove", "bbb")
    assert r.returncode == 0, r.stderr
    assert _reg_slugs(home) == ["aaa", "ccc", "ddd", "eee"]


def test_client_alias_collision_rejected(home):
    add_client()
    assert run_cli("clients.py", "add", "--name", "Other", "--slug", "other").returncode == 0
    run_cli("clients.py", "set", "other", "aliases=oth")
    for bad in ("other", "OTHER", "oth"):
        r = run_cli("clients.py", "set", "chnvt-hdgmh", f"aliases=demo,{bad}")
        assert r.returncode == 1 and "already" in r.stderr and "Traceback" not in r.stderr, bad
    assert profile(home, "chnvt-hdgmh")["aliases"] == []
    assert run_cli("clients.py", "set", "chnvt-hdgmh", "aliases=demo").returncode == 0
    r = run_cli("clients.py", "set", "other", "aliases=demo")
    assert r.returncode == 1 and "already" in r.stderr


def test_client_empty_name_rejected(home):
    r = run_cli("clients.py", "add", "--name", " ", "--slug", "x")
    assert r.returncode == 1 and "name" in r.stderr
    assert not (home / "clients.json").exists()
    add_client()
    r = run_cli("clients.py", "set", "chnvt-hdgmh", "name=")
    assert r.returncode == 1 and "name" in r.stderr
    assert profile(home, "chnvt-hdgmh")["name"] == "חנות הדגמה"


def test_restore_copied_folder_uses_folder_slug(home):
    import shutil
    assert run_cli("clients.py", "add", "--name", "Shop", "--slug", "shop").returncode == 0
    shutil.copytree(home / "clients" / "shop", home / "clients" / "shop-copy")
    r = run_cli("clients.py", "restore", "shop-copy")
    assert r.returncode == 0, r.stderr
    assert _reg_slugs(home) == ["shop", "shop-copy"]
    assert profile(home, "shop-copy")["slug"] == "shop-copy"
    assert profile(home, "shop")["slug"] == "shop"
    r = run_cli("clients.py", "restore", "shop-copy")
    assert r.returncode == 1 and "already registered" in r.stderr
    assert run_cli("clients.py", "set", "shop-copy", "tone=calm").returncode == 0
    assert profile(home, "shop-copy")["tone"] == "calm" and profile(home, "shop")["tone"] == ""


def test_brand_colour_change_notes_stale_palette(home):
    add_client()
    assert run_cli("clients.py", "palette", "chnvt-hdgmh", "--pick", "accent").returncode == 0
    r = run_cli("clients.py", "set", "chnvt-hdgmh", "brand_colors=#008754,#FFFFFF")
    assert r.returncode == 0 and "--pick" not in r.stdout  # first colour unchanged
    r = run_cli("clients.py", "set", "chnvt-hdgmh", "brand_colors=#123456")
    assert r.returncode == 0
    assert "clients.py palette chnvt-hdgmh --pick accent" in r.stdout
