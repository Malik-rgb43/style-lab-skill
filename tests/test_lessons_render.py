import datetime as dt
import json

from conftest import run_cli

START = "<!-- style-lab:learned-rules:start -->"
END = "<!-- style-lab:learned-rules:end -->"
HAND = (
    "# Test style\r\n"
    "\r\n"
    "Hand text, written by the owner: כתוביות במרכז.\r\n"
    "\r\n"
    "## 7. Review-round rules (added after each round)\r\n"
    "\r\n"
    "- hand rule one\r\n"
    "\r\n"
    "## 8. Other\r\n"
    "trailing hand text\r\n"
)


def reg(home):
    return json.loads((home / "lessons" / "lessons.json").read_text(encoding="utf-8"))


def lesson(home, i=0):
    return reg(home)["lessons"][i]


def add(*extra, layer="craft", text="the speed is linear", measurable="easing curve is linear",
        project="proj-a", video="v1"):
    r = run_cli("lessons.py", "add", "--text", text, "--measurable", measurable, "--layer", layer,
                "--project", project, "--video", video, *extra)
    assert r.returncode == 0, r.stderr
    return r


def match(lid, *extra, project="proj-b", video="v1"):
    return run_cli("lessons.py", "match", lid, "--project", project, "--video", video, *extra)


def mkstyle(tmp_path, body=HAND, skill_body="# Style skill\n\nskill hand text\n"):
    pack = tmp_path / "pack"
    pack.mkdir(exist_ok=True)
    (pack / "STYLE.md").write_bytes(body.encode("utf-8"))
    skill = tmp_path / "skill" / "SKILL.md"
    skill.parent.mkdir(exist_ok=True)
    skill.write_bytes(skill_body.encode("utf-8"))
    r = run_cli("styles.py", "add", "--name", "Test Style", "--pack", pack, "--skill", skill)
    assert r.returncode == 0, r.stderr
    return pack / "STYLE.md", skill


def proposed_style_lesson(text="captions sit low", measurable="caption baseline at y >= 90%"):
    add("--target", "test-style", "--scope", "captions", layer="style", text=text, measurable=measurable)
    lid = reg_last_id()
    assert match(lid).returncode == 0
    return lid


def reg_last_id():
    r = run_cli("lessons.py", "list", "--json")
    return json.loads(r.stdout)[-1]["id"]


def strip_block(data: bytes) -> bytes:
    s = data.index(START.encode())
    e = data.index(END.encode()) + len(END)
    nl = b"\r\n" if b"\r\n" in data else b"\n"
    assert data[e:e + len(nl)] == nl
    return data[:s] + data[e + len(nl):]


def test_two_projects_make_proposed(home):
    add("--kind", "quality")
    assert lesson(home)["status"] == "candidate"
    r = match("L0001", "--t", "12.5", "--note", "again")
    assert r.returncode == 0, r.stderr
    x = lesson(home)
    assert x["status"] == "proposed" and len(x["evidence"]) == 2
    assert x["evidence"][1]["t"] == "12.5" and x["evidence"][1]["note"] == "again"


def test_same_note_twice_not_counted(home):
    add("--kind", "quality", "--t", "3", "--round", "1")
    r = match("L0001", "--t", "3", "--round", "1", project="proj-a", video="v1")
    assert r.returncode == 0, r.stderr
    x = lesson(home)
    assert len(x["evidence"]) == 1 and x["status"] == "candidate"


def test_two_rounds_same_video_make_proposed(home):
    add("--kind", "quality", "--round", "1")
    r = match("L0001", "--round", "2", project="proj-a", video="v1")
    assert r.returncode == 0, r.stderr
    x = lesson(home)
    assert x["status"] == "proposed"
    assert [e["round"] for e in x["evidence"]] == [1, 2]


def test_auto_promote(home):
    r = run_cli("lessons.py", "settings", "auto_promote=true")
    assert r.returncode == 0, r.stderr
    assert json.loads((home / "settings.json").read_text(encoding="utf-8"))["auto_promote"] is True
    add("--kind", "taste")
    r = match("L0001")
    assert r.returncode == 0, r.stderr
    x = lesson(home)
    assert x["status"] == "rule" and x["rule_no"] == 1
    assert "L0001" in (home / "craft-rules.learned.md").read_text(encoding="utf-8")
    assert run_cli("lessons.py", "settings", "auto_promote=maybe").returncode == 1
    assert run_cli("lessons.py", "settings", "nope=true").returncode == 1


def test_approve_style_renders_between_markers(home, tmp_path):
    style_md, skill = mkstyle(tmp_path)
    before = style_md.read_bytes()
    lid = proposed_style_lesson()
    r = run_cli("lessons.py", "approve", lid)
    assert r.returncode == 0, r.stderr
    x = lesson(home)
    assert x["status"] == "rule" and x["rule_no"] == 1 and x["promoted"] == dt.date.today().isoformat()
    after = style_md.read_bytes()
    heading = b"## 7. Review-round rules (added after each round)\r\n"
    i = after.index(heading) + len(heading)
    assert after[i:].startswith(START.encode() + b"\r\n")
    assert strip_block(after) == before
    line = (f"1. caption baseline at y >= 90% — captions. (evidence: 2, since "
            f"{dt.date.today().isoformat()}, {lid})")
    assert line in after.decode("utf-8")
    bak = style_md.with_name(f"STYLE.md.bak-{dt.date.today():%Y%m%d}")
    assert bak.read_bytes() == before
    sk = skill.read_bytes().decode("utf-8")
    assert START in sk and line in sk and "## Review-round rules (learned)" in sk
    assert sk.startswith("# Style skill\n\nskill hand text\n")


def test_second_promotion_reuses_block(home, tmp_path):
    style_md, _ = mkstyle(tmp_path)
    before = style_md.read_bytes()
    a = proposed_style_lesson()
    assert run_cli("lessons.py", "approve", a).returncode == 0
    b = proposed_style_lesson(text="zoom is slow", measurable="push-in lasts >= 1.5 s")
    r = run_cli("lessons.py", "approve", b, "--yes-no-conflict")
    assert r.returncode == 0, r.stderr
    text = style_md.read_bytes().decode("utf-8")
    assert text.count(START) == 1 and text.count(END) == 1
    block = text[text.index(START):text.index(END)]
    assert f"1. caption baseline" in block and f"2. push-in lasts >= 1.5 s" in block
    assert block.index("1. caption") < block.index("2. push-in")
    assert strip_block(style_md.read_bytes()) == before
    assert [x["rule_no"] for x in reg(home)["lessons"]] == [1, 2]


def test_hand_edit_between_promotions_survives(home, tmp_path):
    style_md, _ = mkstyle(tmp_path)
    a = proposed_style_lesson()
    assert run_cli("lessons.py", "approve", a).returncode == 0
    edited = style_md.read_bytes().replace(b"trailing hand text", "טקסט יד חדש".encode("utf-8"))
    edited = edited.replace(b"- hand rule one", b"- hand rule one, edited")
    style_md.write_bytes(edited)
    b = proposed_style_lesson(text="zoom is slow", measurable="push-in lasts >= 1.5 s")
    assert run_cli("lessons.py", "approve", b, "--yes-no-conflict").returncode == 0
    after = style_md.read_bytes()
    assert strip_block(after) == strip_block(edited)
    assert "טקסט יד חדש" in after.decode("utf-8") and b"hand rule one, edited" in after


def test_missing_pack_keeps_status(home, tmp_path):
    style_md, skill = mkstyle(tmp_path)
    lid = proposed_style_lesson()
    style_md.unlink()
    style_md.parent.rmdir()
    skill_before = skill.read_bytes()
    r = run_cli("lessons.py", "approve", lid)
    assert r.returncode == 1
    assert str(style_md) in r.stderr
    x = lesson(home)
    assert x["status"] == "proposed" and x["rule_no"] is None
    assert not style_md.parent.exists()
    assert skill.read_bytes() == skill_before
    assert list(skill.parent.iterdir()) == [skill]


def test_missing_skill_warns_but_writes_style(home, tmp_path):
    style_md, skill = mkstyle(tmp_path)
    lid = proposed_style_lesson()
    skill.unlink()
    r = run_cli("lessons.py", "approve", lid)
    assert r.returncode == 0, r.stderr
    assert "warning" in r.stderr and str(skill) in r.stderr
    assert START in style_md.read_text(encoding="utf-8")
    assert lesson(home)["status"] == "rule"


def test_style_without_heading_appends_section(home, tmp_path):
    style_md, _ = mkstyle(tmp_path, body="# Plain\n\nno rules heading here")
    lid = proposed_style_lesson()
    assert run_cli("lessons.py", "approve", lid).returncode == 0
    text = style_md.read_text(encoding="utf-8")
    assert text.startswith("# Plain\n\nno rules heading here\n")
    assert text.index("## Review-round rules (learned)") < text.index(START)


def test_damaged_markers_refused(home, tmp_path):
    style_md, _ = mkstyle(tmp_path, body=f"# S\n\n{START}\nhalf a block\n")
    before = style_md.read_bytes()
    lid = proposed_style_lesson()
    r = run_cli("lessons.py", "approve", lid)
    assert r.returncode == 1 and "marker" in r.stderr
    assert style_md.read_bytes() == before and lesson(home)["status"] == "proposed"


def test_craft_and_client_files(home):
    add("--kind", "quality", "--scope", "all cuts")
    assert match("L0001").returncode == 0
    r = run_cli("lessons.py", "approve", "L0001")
    assert r.returncode == 0, r.stderr
    craft = (home / "craft-rules.learned.md").read_text(encoding="utf-8")
    assert "1. easing curve is linear — all cuts." in craft and "[quality]" in craft and "L0001" in craft
    assert run_cli("clients.py", "add", "--name", "Client X", "--slug", "client-x").returncode == 0
    add("--target", "client-x", layer="client", text="logo too small", measurable="logo width >= 12%")
    # a candidate (one note) can still be approved by an explicit owner yes
    r = run_cli("lessons.py", "approve", "L0002", "--one-note")
    assert r.returncode == 0, r.stderr
    cl = (home / "clients" / "client-x" / "lessons.md").read_text(encoding="utf-8")
    assert "1. logo width >= 12%" in cl and "L0002" in cl and "[quality]" not in cl
    # render refreshes on demand
    assert run_cli("lessons.py", "render", "--layer", "craft").returncode == 0
    assert run_cli("lessons.py", "render", "--layer", "client", "--target", "Client X").returncode == 0


def test_one_off_never_promotes(home):
    add(layer="video")
    r = match("L0001")
    assert r.returncode == 1 and "one-off" in r.stderr
    r = run_cli("lessons.py", "approve", "L0001")
    assert r.returncode == 1 and "one-off" in r.stderr
    assert lesson(home)["status"] == "one-off" and len(lesson(home)["evidence"]) == 1


def test_merge_moves_evidence(home):
    add("--kind", "quality")
    add("--kind", "quality", text="speed feels robotic", project="proj-b")
    r = run_cli("lessons.py", "merge", "L0002", "L0001")
    assert r.returncode == 0, r.stderr
    a, b = reg(home)["lessons"]
    assert b["status"] == "merged" and b["merged_into"] == "L0001"
    assert len(a["evidence"]) == 2 and a["status"] == "proposed"
    # a merged source cannot be merged, matched or rejected again
    assert run_cli("lessons.py", "merge", "L0002", "L0001").returncode == 1
    assert run_cli("lessons.py", "reject", "L0002").returncode == 1


def test_merge_refusals(home):
    add("--kind", "quality")
    add(layer="video", text="one-off note")
    add("--kind", "quality", text="third")
    assert run_cli("lessons.py", "merge", "L0002", "L0001").returncode == 1  # one-off source
    assert run_cli("lessons.py", "merge", "L0001", "L0002").returncode == 1  # one-off target
    assert run_cli("lessons.py", "reject", "L0003").returncode == 0
    assert run_cli("lessons.py", "merge", "L0001", "L0003").returncode == 1  # rejected target
    assert run_cli("lessons.py", "merge", "L0001", "L0001").returncode == 1
    assert run_cli("lessons.py", "approve", "L0001", "--one-note").returncode == 0
    add("--kind", "quality", text="fourth")
    assert run_cli("lessons.py", "merge", "L0001", "L0004").returncode == 1  # rule source
    assert [x["status"] for x in reg(home)["lessons"]] == ["rule", "one-off", "rejected", "candidate"]


def test_candidate_needs_one_note(home):
    add("--kind", "quality")
    r = run_cli("lessons.py", "approve", "L0001")
    assert r.returncode == 1 and "only one note so far" in r.stderr and "--one-note" in r.stderr
    assert lesson(home)["status"] == "candidate"
    assert not (home / "craft-rules.learned.md").exists()
    r = run_cli("lessons.py", "approve", "L0001", "--one-note")
    assert r.returncode == 0, r.stderr
    assert lesson(home)["status"] == "rule"


def test_proposed_approves_without_flag(home):
    add("--kind", "taste")
    assert match("L0001").returncode == 0
    r = run_cli("lessons.py", "approve", "L0001")
    assert r.returncode == 0, r.stderr
    assert "[taste]" in (home / "craft-rules.learned.md").read_text(encoding="utf-8")


def test_lock_busy_exits_2(home):
    add("--kind", "quality")
    assert match("L0001").returncode == 0
    lk = home / "lessons" / "lessons.json.lock"
    lk.write_text("")
    try:
        r = run_cli("lessons.py", "approve", "L0001")
        assert r.returncode == 2 and "lock busy" in r.stderr
        r = run_cli("lessons.py", "render", "--layer", "craft")
        assert r.returncode == 2 and "lock busy" in r.stderr
    finally:
        lk.unlink()
    assert lesson(home)["status"] == "proposed"


def test_style_registry_counts_and_version(home, tmp_path):
    mkstyle(tmp_path)

    def entry():
        data = json.loads((home / "styles.json").read_text(encoding="utf-8"))
        return next(s for s in data["styles"] if s["slug"] == "test-style")

    assert entry()["rules_count"] == 0 and entry()["version"] == 1
    a = proposed_style_lesson()
    assert run_cli("lessons.py", "approve", a).returncode == 0
    assert entry()["rules_count"] == 1 and entry()["version"] == 2
    b = proposed_style_lesson(text="zoom is slow", measurable="push-in lasts >= 1.5 s")
    assert run_cli("lessons.py", "approve", b, "--yes-no-conflict").returncode == 0
    assert entry()["rules_count"] == 2 and entry()["version"] == 3
    # a render with nothing new changes no file and no version
    assert run_cli("lessons.py", "render", "--layer", "style", "--target", "test-style").returncode == 0
    assert entry()["rules_count"] == 2 and entry()["version"] == 3


def test_registry_write_failure_says_resync(home, monkeypatch):
    import lessons
    import store
    add("--kind", "quality")
    assert match("L0001").returncode == 0
    real = store.write_json

    def failing(p, data):
        if str(p).endswith("lessons.json"):
            raise PermissionError("disk says no")
        return real(p, data)

    monkeypatch.setattr(store, "write_json", failing)
    try:
        lessons.approve("L0001")
        raise AssertionError("approve should fail")
    except OSError as e:
        assert "rule files were updated" in str(e) and "lessons.py render" in str(e)
    assert lesson(home)["status"] == "proposed"
    assert "L0001" in (home / "craft-rules.learned.md").read_text(encoding="utf-8")


def test_edit_rule_kind_refused_measurable_rerenders(home):
    add("--kind", "quality")
    assert match("L0001").returncode == 0
    assert run_cli("lessons.py", "approve", "L0001").returncode == 0
    r = run_cli("lessons.py", "edit", "L0001", "kind=taste")
    assert r.returncode == 1 and "rule" in r.stderr
    assert lesson(home)["kind"] == "quality"
    r = run_cli("lessons.py", "edit", "L0001", "measurable=cuts land in silence", "scope=every join")
    assert r.returncode == 0, r.stderr
    craft = (home / "craft-rules.learned.md").read_text(encoding="utf-8")
    assert "1. cuts land in silence — every join." in craft and "easing curve is linear" not in craft


def test_edit_rule_rerender_all_or_nothing(home, tmp_path):
    style_md, _ = mkstyle(tmp_path)
    lid = proposed_style_lesson()
    assert run_cli("lessons.py", "approve", lid).returncode == 0
    for f in style_md.parent.iterdir():
        f.unlink()
    style_md.parent.rmdir()
    r = run_cli("lessons.py", "edit", lid, "scope=intro")
    assert r.returncode == 1 and str(style_md) in r.stderr
    assert lesson(home)["scope"] == "captions"
