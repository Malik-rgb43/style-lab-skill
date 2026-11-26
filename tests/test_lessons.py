import json
import os
import subprocess
import sys

from conftest import REPO_ROOT, run_cli

HE = "המהירות ליניארית, צריך להתחיל מהר ולסיים לאט"


def reg(home):
    return json.loads((home / "lessons" / "lessons.json").read_text(encoding="utf-8"))


def add(*extra, text=HE, layer="craft"):
    args = ["add", "--text", text, "--measurable", "easing curve is linear", "--layer", layer,
            "--project", "proj-a", "--video", "v1", *extra]
    return run_cli("lessons.py", *args)


def mkstyle(tmp_path, name="סגנון בדיקה"):
    pack = tmp_path / "pack"
    pack.mkdir(exist_ok=True)
    r = run_cli("styles.py", "add", "--name", name, "--pack", pack)
    assert r.returncode == 0, r.stderr


def test_add_candidate_and_one_off(home):
    r = add("--kind", "quality")
    assert r.returncode == 0, r.stderr
    r = add(layer="video")
    assert r.returncode == 0, r.stderr
    ls = reg(home)["lessons"]
    assert [x["id"] for x in ls] == ["L0001", "L0002"]
    assert ls[0]["status"] == "candidate" and ls[0]["kind"] == "quality"
    assert ls[1]["status"] == "one-off" and ls[1]["kind"] is None and ls[1]["target"] is None
    assert reg(home)["next_id"] == 3
    assert ls[0]["evidence"][0]["project"] == "proj-a"


def test_target_validation(home):
    r = add("--target", "nope", layer="style")
    assert r.returncode == 1 and "no style" in r.stderr
    r = add("--target", "nope", layer="client")
    assert r.returncode == 1 and "no client" in r.stderr
    r = add()
    assert r.returncode == 1 and "kind" in r.stderr
    assert not (home / "lessons" / "lessons.json").exists() or reg(home)["lessons"] == []


def test_style_target_resolved_to_slug(home, tmp_path):
    mkstyle(tmp_path)
    r = add("--target", "סגנון בדיקה", layer="style")
    assert r.returncode == 0, r.stderr
    assert reg(home)["lessons"][0]["target"] == "sgnvn-bdykh"


def test_hebrew_roundtrip(home):
    assert add("--kind", "taste").returncode == 0
    assert reg(home)["lessons"][0]["text"] == HE
    r = run_cli("lessons.py", "show", "L0001")
    assert r.returncode == 0 and HE in r.stdout
    r = run_cli("lessons.py", "list")
    assert HE in r.stdout


def test_rejected_readd_warns(home):
    assert add("--kind", "taste").returncode == 0
    assert run_cli("lessons.py", "reject", "L0001", "--reason", "x").returncode == 0
    assert reg(home)["lessons"][0]["status"] == "rejected"
    r = add("--kind", "taste", text="  " + HE + " ")
    assert r.returncode == 0 and "L0001" in r.stderr
    assert len(reg(home)["lessons"]) == 2


def test_edit_layer_revalidates(home):
    assert add("--kind", "quality").returncode == 0
    r = run_cli("lessons.py", "edit", "L0001", "layer=style")
    assert r.returncode == 1
    assert reg(home)["lessons"][0]["layer"] == "craft"
    r = run_cli("lessons.py", "edit", "L0001", "kind=")
    assert r.returncode == 1
    r = run_cli("lessons.py", "edit", "L0001", "scope=intro")
    assert r.returncode == 0 and reg(home)["lessons"][0]["scope"] == "intro"


def test_list_filters(home):
    add("--kind", "quality")
    add(layer="video")
    r = run_cli("lessons.py", "list", "--status", "one-off", "--json")
    assert [x["id"] for x in json.loads(r.stdout)] == ["L0002"]


def test_parallel_adds(home):
    env = dict(os.environ)
    ps = [subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "scripts" / "lessons.py"), "add", "--text", f"t{i}",
         "--measurable", "m", "--layer", "video", "--project", "p", "--video", "v"],
        cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for i in range(6)]
    assert all(p.wait() == 0 for p in ps)
    ids = [x["id"] for x in reg(home)["lessons"]]
    assert len(ids) == 6 and len(set(ids)) == 6


def test_round_is_int(home):
    assert add("--kind", "quality", "--round", "2").returncode == 0
    assert reg(home)["lessons"][0]["evidence"][0]["round"] == 2
    r = add("--kind", "quality", "--round", "two")
    assert r.returncode == 2


def test_scope_edit_survives_removed_style(home, tmp_path):
    mkstyle(tmp_path)
    assert add("--target", "סגנון בדיקה", layer="style").returncode == 0
    assert run_cli("styles.py", "remove", "sgnvn-bdykh").returncode == 0
    r = run_cli("lessons.py", "edit", "L0001", "scope=intro")
    assert r.returncode == 0, r.stderr
    x = reg(home)["lessons"][0]
    assert x["scope"] == "intro" and x["target"] == "sgnvn-bdykh"
    r = run_cli("lessons.py", "edit", "L0001", "target=sgnvn-bdykh")
    assert r.returncode == 1 and "no style" in r.stderr


def test_edit_kind_still_validated(home):
    assert add("--kind", "quality").returncode == 0
    assert run_cli("lessons.py", "edit", "L0001", "kind=taste").returncode == 0
    assert reg(home)["lessons"][0]["kind"] == "taste"
    assert run_cli("lessons.py", "edit", "L0001", "kind=bold").returncode == 1
    assert add(layer="video").returncode == 0
    assert run_cli("lessons.py", "edit", "L0002", "kind=taste").returncode == 1


def test_edit_layer_only_candidate_or_one_off(home):
    assert add("--kind", "quality").returncode == 0
    assert run_cli("lessons.py", "match", "L0001", "--project", "proj-b", "--video", "v2").returncode == 0
    assert reg(home)["lessons"][0]["status"] == "proposed"
    r = run_cli("lessons.py", "edit", "L0001", "layer=video")
    assert r.returncode == 1 and "candidate" in r.stderr
    assert reg(home)["lessons"][0]["layer"] == "craft"
    assert add(layer="video").returncode == 0
    r = run_cli("lessons.py", "edit", "L0002", "layer=craft", "kind=taste")
    assert r.returncode == 0, r.stderr
    assert reg(home)["lessons"][1]["status"] == "candidate"


def test_reject_refuses_rule(home):
    assert add("--kind", "quality").returncode == 0
    assert run_cli("lessons.py", "approve", "L0001", "--one-note").returncode == 0
    r = run_cli("lessons.py", "reject", "L0001")
    assert r.returncode == 1 and "use a later retire command for rules" in r.stderr
    assert reg(home)["lessons"][0]["status"] == "rule"


# ---------- Task 3: effective rules, exceptions, conflicts ----------

def mkrule(layer, text, *extra, scope="", approve=True):
    r = run_cli("lessons.py", "add", "--text", text, "--measurable", f"m {text}", "--layer", layer,
                "--scope", scope, "--project", "proj-a", "--video", "v1", *extra)
    assert r.returncode == 0, r.stderr
    lid = reg_ids()[-1]
    if approve:
        r = run_cli("lessons.py", "approve", lid, "--one-note", "--yes-no-conflict")
        assert r.returncode == 0, r.stderr
    return lid


def reg_ids():
    import conftest  # noqa: F401
    from lessons import registry_path
    return [x["id"] for x in json.loads(registry_path().read_text(encoding="utf-8"))["lessons"]]


def mkclient(name="Acme"):
    r = run_cli("clients.py", "add", "--name", name)
    assert r.returncode == 0, r.stderr


def rules_json(*extra):
    r = run_cli("lessons.py", "rules", "--json", *extra)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def mkstyle_md(tmp_path, name="Test Style"):
    pack = tmp_path / "pack"
    pack.mkdir(exist_ok=True)
    (pack / "STYLE.md").write_text("# Test style\n", encoding="utf-8")
    r = run_cli("styles.py", "add", "--name", name, "--pack", pack)
    assert r.returncode == 0, r.stderr


def test_effective_order(home, tmp_path):
    mkstyle_md(tmp_path)
    mkclient()
    c1 = mkrule("craft", "c one", "--kind", "taste")
    s1 = mkrule("style", "s one", "--target", "test-style")
    k1 = mkrule("client", "k one", "--target", "acme")
    c2 = mkrule("craft", "c two", "--kind", "quality")
    rows = rules_json("--style", "test-style", "--client", "acme")
    assert [r["id"] for r in rows] == [c1, c2, s1, k1]
    assert [r["layer"] for r in rows] == ["craft", "craft", "style", "client"]
    assert [r["rule_no"] for r in rows] == [1, 2, 1, 1]
    assert [r["id"] for r in rules_json()] == [c1, c2]
    assert [r["id"] for r in rules_json("--style", "test-style")] == [c1, c2, s1]
    r = run_cli("lessons.py", "rules", "--style", "nope")
    assert r.returncode == 1 and "no style" in r.stderr


def test_rules_table_hebrew(home):
    mkrule("craft", HE, "--kind", "taste", scope="פתיחה")
    r = run_cli("lessons.py", "rules")
    assert r.returncode == 0, r.stderr
    assert "L0001" in r.stdout and "פתיחה" in r.stdout and "overridden" in r.stdout


def test_quality_exception_refused(home, tmp_path):
    mkstyle_md(tmp_path)
    q = mkrule("craft", "q one", "--kind", "quality")
    r = run_cli("lessons.py", "exception", "--style", "test-style", "--craft", q,
                "--measurable", "m", "--reason", "because")
    assert r.returncode == 1 and "quality" in r.stderr
    assert len(reg_ids()) == 1
    cand = mkrule("craft", "c cand", "--kind", "taste", approve=False)
    r = run_cli("lessons.py", "exception", "--style", "test-style", "--craft", cand,
                "--measurable", "m", "--reason", "because")
    assert r.returncode == 1 and "not a craft rule" in r.stderr


def test_taste_exception_needs_reason(home, tmp_path):
    mkstyle_md(tmp_path)
    t = mkrule("craft", "t one", "--kind", "taste")
    r = run_cli("lessons.py", "exception", "--style", "test-style", "--craft", t, "--measurable", "m")
    assert r.returncode == 1 and "reason" in r.stderr
    r = run_cli("lessons.py", "exception", "--style", "test-style", "--craft", t, "--measurable", "m",
                "--reason", "   ")
    assert r.returncode == 1
    r = run_cli("lessons.py", "exception", "--style", "test-style", "--craft", t,
                "--measurable", "slow ease allowed", "--reason", "calm lessons")
    assert r.returncode == 0, r.stderr
    ex = reg_ids()[-1]
    x = reg(home)["lessons"][-1]
    assert x["status"] == "candidate" and x["layer"] == "style" and x["exception_of"] == t
    assert x["reason"] == "calm lessons"
    assert run_cli("lessons.py", "approve", ex, "--one-note").returncode == 0
    rows = rules_json("--style", "test-style")
    assert [r["overridden_by"] for r in rows] == [ex, None]
    assert rows[1]["exception_of"] == t
    assert rules_json()[0]["overridden_by"] is None  # other styles are unaffected
    r = run_cli("lessons.py", "rules", "--style", "test-style")
    assert f"exception to {t}: calm lessons" in r.stdout
    style_md = (tmp_path / "pack" / "STYLE.md").read_text(encoding="utf-8")
    assert f"(exception to {t}: calm lessons)" in style_md


def test_conflict_blocks_approve(home):
    a = mkrule("craft", "first", "--kind", "taste", scope="opening captions")
    b = mkrule("craft", "second", "--kind", "taste", scope="captions fade", approve=False)
    r = run_cli("lessons.py", "conflicts", b)
    assert r.returncode == 0 and a in r.stdout
    r = run_cli("lessons.py", "approve", b, "--one-note")
    assert r.returncode == 1 and a in r.stdout and "--yes-no-conflict" in r.stderr
    assert reg(home)["lessons"][1]["status"] == "candidate"
    r = run_cli("lessons.py", "approve", b, "--one-note", "--yes-no-conflict")
    assert r.returncode == 0, r.stderr
    assert reg(home)["lessons"][1]["status"] == "rule"


def test_no_conflict_without_shared_scope(home, tmp_path):
    mkstyle_md(tmp_path)
    mkrule("craft", "first", "--kind", "taste", scope="opening")
    s = mkrule("style", "s", "--target", "test-style", scope="ending", approve=False)
    r = run_cli("lessons.py", "approve", s, "--one-note")
    assert r.returncode == 0, r.stderr
    # a style lesson sees craft rules (lower layer); a craft lesson never sees style rules
    c = mkrule("craft", "third", "--kind", "taste", scope="ending", approve=False)
    r = run_cli("lessons.py", "conflicts", c)
    assert r.returncode == 0 and "no possible conflicts" in r.stdout
    s2 = mkrule("style", "s2", "--target", "test-style", scope="opening ending", approve=False)
    r = run_cli("lessons.py", "conflicts", s2)
    assert "L0001" in r.stdout and s in r.stdout


def test_exception_conflicts_only_with_siblings(home, tmp_path):
    mkstyle_md(tmp_path)
    t = mkrule("craft", "t", "--kind", "taste", scope="easing")
    run_cli("lessons.py", "exception", "--style", "test-style", "--craft", t, "--measurable", "m1",
            "--reason", "r1")
    e1 = reg_ids()[-1]
    run_cli("lessons.py", "exception", "--style", "test-style", "--craft", t, "--measurable", "m2",
            "--reason", "r2")
    e2 = reg_ids()[-1]
    assert run_cli("lessons.py", "approve", e1, "--one-note", "--yes-no-conflict").returncode == 0
    r = run_cli("lessons.py", "conflicts", e2)
    assert e1 in r.stdout and t not in r.stdout
    assert run_cli("lessons.py", "approve", e2, "--one-note").returncode == 1


def mkexception(t, measurable="slow ease allowed", reason="calm lessons"):
    r = run_cli("lessons.py", "exception", "--style", "test-style", "--craft", t,
                "--measurable", measurable, "--reason", reason)
    assert r.returncode == 0, r.stderr
    return reg_ids()[-1]


def test_lone_exception_approves_without_flag(home, tmp_path):
    mkstyle_md(tmp_path)
    t = mkrule("craft", "t", "--kind", "taste", scope="easing curves")
    e = mkexception(t)
    r = run_cli("lessons.py", "conflicts", e)
    assert r.returncode == 0 and "no possible conflicts" in r.stdout and t not in r.stdout
    r = run_cli("lessons.py", "approve", e, "--one-note")
    assert r.returncode == 0, r.stderr


def test_exception_layer_target_not_editable(home, tmp_path):
    mkstyle_md(tmp_path)
    t = mkrule("craft", "t", "--kind", "taste")
    e = mkexception(t)
    for pair in ("layer=craft", "layer=video", "target=test-style"):
        r = run_cli("lessons.py", "edit", e, pair)
        assert r.returncode == 1 and "exception" in r.stderr
    x = reg(home)["lessons"][-1]
    assert x["layer"] == "style" and x["target"] == "test-style" and x["exception_of"] == t
    assert run_cli("lessons.py", "edit", e, "scope=easing").returncode == 0


def test_approve_revalidates_exception_target(home, tmp_path):
    import lessons
    mkstyle_md(tmp_path)
    t = mkrule("craft", "t", "--kind", "taste")
    e = mkexception(t)

    def corrupt(d):
        c = next(x for x in d["lessons"] if x["id"] == t)
        c["kind"] = "quality"

    lessons.store.update_json(lessons.registry_path(), lessons._empty(), corrupt)
    r = run_cli("lessons.py", "approve", e, "--one-note")
    assert r.returncode == 1 and "no longer" in r.stderr
    assert reg(home)["lessons"][-1]["status"] == "candidate"
    lessons.store.update_json(lessons.registry_path(), lessons._empty(),
                              lambda d: next(x for x in d["lessons"] if x["id"] == t).update(kind="taste", status="rejected"))
    r = run_cli("lessons.py", "approve", e, "--one-note")
    assert r.returncode == 1 and reg(home)["lessons"][-1]["status"] == "candidate"


def test_empty_scope_rule_conflicts_everywhere(home, tmp_path):
    mkstyle_md(tmp_path)
    c = mkrule("craft", "c everywhere", "--kind", "taste", scope="")
    s = mkrule("style", "s", "--target", "test-style", scope="ending", approve=False)
    assert c in run_cli("lessons.py", "conflicts", s).stdout
    k = mkrule("craft", "k", "--kind", "quality", scope="captions", approve=False)
    assert c in run_cli("lessons.py", "conflicts", k).stdout


def test_same_target_rule_conflicts_without_shared_word(home, tmp_path):
    mkstyle_md(tmp_path)
    (tmp_path / "other").mkdir()
    mkstyle_md(tmp_path / "other", name="Other Style")
    s1 = mkrule("style", "s1", "--target", "test-style", scope="opening")
    o1 = mkrule("style", "o1", "--target", "other-style", scope="closing")
    c1 = mkrule("craft", "c1", "--kind", "taste", scope="titles")
    s2 = mkrule("style", "s2", "--target", "test-style", scope="closing", approve=False)
    r = run_cli("lessons.py", "conflicts", s2)
    assert s1 in r.stdout and o1 not in r.stdout and c1 not in r.stdout
    # craft lessons have no target: craft rules still need a shared word (or an empty scope)
    c2 = mkrule("craft", "c2", "--kind", "taste", scope="endings", approve=False)
    assert "no possible conflicts" in run_cli("lessons.py", "conflicts", c2).stdout
