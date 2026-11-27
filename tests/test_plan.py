import json

import pytest

import ideas
import plan
import project
import store
import taste
from conftest import run_cli
from fixtures import plan5 as P


@pytest.fixture
def world(home, tmp_path):
    P.write_style(home, tmp_path)
    P.write_client(run_cli)
    return P.write_project(tmp_path)


def _idea(tmp_path, beat_types=("number",), title="counter"):
    sn = tmp_path / f"{title}.html"
    sn.write_text("<div>{{title}}</div>", encoding="utf-8")
    return ideas.add(title, "b-roll", list(beat_types), "9:16", sn,
                     {"title": {"type": "text", "default": "x", "client_bound": False}},
                     made_for={"style": "paper-3d", "client": None, "project": None}, tags=["warm"])


def test_triggers_number_and_type():
    words = [{"word": w} for w in ("יש", "3", "דרכים")]
    assert plan.beat_triggers(words, "he", ["list"]) == {"number", "list"}
    assert "number" in plan.beat_triggers([{"word": "שלושה"}, {"word": "דברים"}], "he")
    assert "question" in plan.beat_triggers([{"word": "מה"}, {"word": "?"}], "he")


def test_candidates_laws_and_ideas(world, tmp_path):
    i = _idea(tmp_path)
    project.record_shown(world, "plan", 1, 1, [{"id": "B01-p9", "sig": "plan:aaaaaaaaaaaa"}])
    c = plan.candidates(world)
    b1 = c["1"]
    assert b1["laws"][0]["then"] == "counter_card" and b1["laws"][0]["status"] == "rule-candidate"
    assert [x["id"] for x in b1["ideas"]] == [i["id"]]
    assert "plan:aaaaaaaaaaaa" in b1["blocked"]
    assert c["2"]["laws"][0]["then"] == "big_text"  # only an option law fires on the question
    r = run_cli("plan.py", "candidates", world, "--json")
    assert r.returncode == 0 and json.loads(r.stdout)["1"]["laws"]


def _base(world, extra=None):
    sets = {1: P.good_set(1), 2: P.good_set(2), 3: P.good_set(3)}
    sets.update(extra or {})
    P.set_options(world, sets)


def test_check_set_rules(world):
    _base(world)
    assert plan.check(world) == []
    two_wild = P.good_set(2)
    two_wild[0]["wild"] = True
    law_wild = P.good_set(3)
    law_wild[2]["source"] = "law"
    law_wild[2]["law"] = "x->y"
    no_law = [P.option(1, 1), P.option(1, 2), P.option(1, 3, wild=True)]
    _base(world, {1: no_law, 2: two_wild, 3: law_wild})
    probs = "\n".join(plan.check(world))
    assert "B2: 2 wild cards" in probs
    assert "B3: wild card B03-p3 must be an invention" in probs
    assert "B1: a style law fires here" in probs
    short = P.good_set(1)
    short[1]["design_q"]["new"] = "  tiny "
    _base(world, {1: short})

    def gap(pl):
        pl["beats"][1]["start"] = 2.2
    store.update_json(world / "plan.json", {}, gap)
    probs = "\n".join(plan.check(world))
    assert "design question 'new'" in probs and "gap of 0.20 s" in probs
    _base(world)
    store.update_json(world / "plan.json", {}, lambda pl: pl["beats"][1].update(start=2.0))
    blocked_sig = plan.option_sig(P.good_set(3)[0])
    project.record_shown(world, "plan", 3, 1, [{"id": "B03-p1", "sig": blocked_sig}])
    set2 = P.good_set(3, set_no=2, start=4)  # set 2 brings back the concept shown in set 1
    set2[0]["concept_key"] = P.good_set(3)[0]["concept_key"]
    _base(world, {3: set2})
    assert any("repeats a concept" in p for p in plan.check(world))
    r = run_cli("plan.py", "check", world)
    assert r.returncode == 1 and "repeats a concept" in r.stdout


def test_recipe_must_exist(world):
    s = P.good_set(2)
    s[0]["recipe"] = "no-such-recipe"
    s[1]["recipe"] = "paper-cutout"
    _base(world, {2: s})
    probs = plan.check(world)
    assert any("no-such-recipe" in p for p in probs) and not any("paper-cutout" in p for p in probs)


def test_recommend_taste_decides(world):
    for k in range(3):  # the client picked warm 3 times and rejected neon twice
        taste.log("example-shop", "paper-3d", "plan", f"x{k}", [f"y{k}"], f"old{k}",
                  tags={"picked": ["warm"], "rejected": ["neon"] if k < 2 else []})
    s = P.good_set(2)
    s[0].update(fit=0.70, tags=["neon"])
    s[1].update(fit=0.65, tags=["warm"])
    _base(world, {2: s})
    rows = {r["beat"]: r for r in plan.recommend(world)}
    assert rows[2]["recommended"] == "B02-p2" and rows[2]["taste_decided"] is True
    b2 = store.read_json(world / "plan.json", {})["beats"][1]
    assert b2["pick"] == "B" and all("sig" in o and "score" in o for o in b2["options"])


def test_no_repeat_across_beats(world):
    s2, s3 = P.good_set(2), P.good_set(3)
    s2[0].update(concept_key="same-idea", fit=0.9)
    s3[0].update(concept_key="same-idea", fit=0.8)
    _base(world, {2: s2, 3: s3})
    rows = {r["beat"]: r for r in plan.recommend(world)}
    assert rows[2]["recommended"] == "B02-p1"
    assert rows[3]["recommended"] != "B03-p1" and "2" in rows[3]["note"]


def test_finalize_records_shown_and_offered(world, tmp_path):
    i = _idea(tmp_path)
    _base(world, {1: P.good_set(1, idea_id=i["id"])})
    assert plan.finalize(world) == 0
    shown = store.read_json(world / "shown.json", {})
    for b in (1, 2, 3):
        assert len([x for x in shown["items"] if x["beat"] == b]) == 3
    assert ideas.get(i["id"])["offered"] == 1
    assert plan.finalize(world) == 0  # idempotent: the same set is not recorded twice
    assert len(store.read_json(world / "shown.json", {})["items"]) == 9
    _base(world, {2: P.good_set(2)[:2]})
    r = run_cli("plan.py", "finalize", world)
    assert r.returncode == 1 and "B2" in r.stderr


def test_more_budget(world):
    _base(world)
    assert plan.finalize(world) == 0
    for n in (2, 3, 4):
        r = run_cli("plan.py", "more", world, "--beat", "1")
        assert r.returncode == 0, r.stderr
        assert json.loads(r.stdout)["set"] == n
        project.record_shown(world, "plan", 1, n, [{"id": f"B01-p{10 + n}", "sig": f"plan:{n:012d}"}])
    r = run_cli("plan.py", "more", world, "--beat", "1")
    assert r.returncode == 1 and "לשאול את הבעלים" in r.stderr
    r = run_cli("plan.py", "more", world, "--beat", "1", "--owner-ok-more", "--owner-words", "כן, עוד סט")
    assert r.returncode == 0


def test_no_pack_still_works(home, tmp_path):
    P.write_style(home, tmp_path, laws=None, fingerprint=None, recipes=None)
    d = P.write_project(tmp_path, client=None)
    c = plan.candidates(d)
    assert c["1"]["laws"] == [] and "number" in c["1"]["triggers"]


def test_creator_mind_feeds_candidates(world, tmp_path):
    pack = tmp_path / "pack"
    store.write_json(pack / "creator-mind.json", {"schema": "broken"})
    def mt(pl):
        pl["beats"][0]["message_type"] = "number"
    store.update_json(world / "plan.json", {}, mt)
    c = plan.candidates(world)["1"]
    assert "creator" in c or "creator_error" in c  # never a crash
