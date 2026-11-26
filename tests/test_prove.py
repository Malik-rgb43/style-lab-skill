import json
import math
from pathlib import Path

import pytest

from conftest import run_cli
import prove

POWER3_OUT = [0.215, 0.61, 0.355, 1.0]
LINEAR = [0.0, 0.0, 1.0, 1.0]


def mv(t0=1.0, dur=15, bez=POWER3_OUT, prop="scale", peak_at=0.3, **kw):
    return {"t0": t0, "prop": prop, "dur_f": dur, "bezier": bez, "peak_at": peak_at, **kw}


def wj(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def fp(trained=("a", "b"), dur=(10, 20), peak=(0.2, 0.5), eases=None):
    motion = {"metrics": {"dur_f": {"median": 15, "p10": dur[0], "p90": dur[1], "n": 5},
                          "peak_at": {"median": 0.3, "p10": peak[0], "p90": peak[1], "n": 5}}}
    if eases is not None:
        motion["eases"] = eases
    return {"trained_on": list(trained), "held_out": [], "layers": {"motion": motion}}


# ---- scoring

def test_score_placement_known():
    pred = [{"t": 1.0, "type": "cut"}, {"t": 2.0, "type": "cut"}, {"t": 5.0, "type": "cut"},
            {"t": 3.0, "type": "sfx"}]
    actual = [{"t": 1.1, "type": "cut"}, {"t": 2.2, "type": "cut"}, {"t": 4.0, "type": "cut"},
              {"t": 3.0, "type": "sfx"}, {"t": 9.0, "type": "sfx"}]
    s = prove.score_placement(pred, actual, tol_s=0.3)
    # cut: tp 2 of 3 pred, 3 actual -> p .667 r .667 f1 .667; sfx: tp 1, p 1, r .5 f1 .667
    assert s["per_type"]["cut"]["precision"] == pytest.approx(2 / 3)
    assert s["per_type"]["cut"]["f1"] == pytest.approx(2 / 3)
    assert s["per_type"]["sfx"]["recall"] == 0.5
    # pooled: tp 3, pred 4, actual 5 -> p .75 r .6 f1 = 2*.45/1.35
    assert s["overall_f1"] == pytest.approx(2 * 0.75 * 0.6 / 1.35)
    assert s["timing_mae_s"] == pytest.approx((0.1 + 0.2 + 0.0) / 3)


def test_score_placement_one_to_one_and_empty():
    s = prove.score_placement([{"t": 1.0, "type": "a"}, {"t": 1.05, "type": "a"}], [{"t": 1.0, "type": "a"}])
    assert s["per_type"]["a"]["tp"] == 1
    e = prove.score_placement([], [])
    assert e["overall_f1"] == 0.0 and e["timing_mae_s"] is None


def test_compare_moves_same_curve():
    r = prove.compare_moves(mv(), mv())
    assert r["curve_rmse"] == pytest.approx(0, abs=1e-9)
    assert r["within_spread"] is True and r["dur_ratio"] == 1
    lin = prove.compare_moves(mv(), mv(bez=LINEAR))
    assert lin["curve_rmse"] > 0.05


def test_compare_moves_spread_and_missing_curve():
    assert prove.compare_moves(mv(dur=15), mv(dur=40), spread=fp())["within_spread"] is False
    assert prove.compare_moves(mv(dur=15), mv(dur=18), spread=fp())["within_spread"] is True
    no = prove.compare_moves({"dur_f": 10}, {"dur_f": 10})
    assert no["curve_rmse"] is None and no["within_spread"] is False


def test_qa_flags_out_of_spread():
    motion = {"camera_moves": [mv(dur=40, prop="camera")], "elements": [{"name": "title", "moves": [mv(dur=15)]}]}
    issues = prove.qa_moves(motion, fp())
    assert len(issues) == 1
    assert issues[0]["prop"] == "camera" and "dur_f" in issues[0]["issue"]
    assert prove.qa_moves({"camera_moves": [], "elements": []}, fp()) == []


def test_qa_flags_ease_and_peak_and_tolerates_missing_keys():
    f = fp(eases=[{"name": "p3o", "bezier": POWER3_OUT}])
    issues = prove.qa_moves({"camera_moves": [mv(bez=LINEAR, peak_at=0.9)]}, f)
    kinds = " ".join(i["issue"] for i in issues)
    assert "peak_at" in kinds and "ease differs" in kinds
    assert prove.qa_moves({"camera_moves": [mv(dur=99)]}, {}) == []  # no fingerprint metrics: nothing to flag
    assert prove.qa_moves({}, f) == []


# ---- graph

def test_graph_html_self_contained():
    moves = [mv(t0=0.5), mv(t0=2.0, bez=LINEAR, prop="עברית")]
    out = prove.graph_html(moves, compare=[mv(bez=LINEAR), mv()])
    assert "http" not in out
    assert out.count("<svg") == 2 * len(moves)
    assert "prefers-color-scheme:dark" in out and "dir=\"auto\"" in out
    assert out.count("<polyline") == 2 * len(moves) * 2  # ref + build, value + speed
    assert "עברית" in out and "build" in out


def test_graph_html_escapes_and_empty():
    out = prove.graph_html([mv(prop="<b>x</b>")])
    assert "<b>x</b>" not in out and "&lt;b&gt;" in out
    assert "no moves" in prove.graph_html([])


def test_graph_cli(tmp_path):
    m = tmp_path / "motion.json"
    wj(m, {"camera_moves": [mv(prop="camera")], "elements": []})
    out = tmp_path / "g" / "view.html"
    r = run_cli("prove.py", "graph", m, "--compare", m, "--out", out)
    assert r.returncode == 0, r.stderr
    assert out.read_text(encoding="utf-8").count("<svg") == 2


# ---- hold-out

def make_pack(tmp_path, trained=("a", "b"), videos=("a", "b", "c", "d", "e")):
    pack = tmp_path / "pack"
    for v in videos:
        wj(pack / "analysis" / v / "events.json", [{"t": 1.0, "type": "cut"}, {"t": 3.0, "type": "cut"}])
    wj(pack / "fingerprint.json", fp(trained=trained))
    return pack


def test_holdout_refuses_overlap(tmp_path):
    pack = make_pack(tmp_path)
    r = run_cli("prove.py", "holdout-pick", pack, "--ids", "b,c")
    assert r.returncode == 2 and "b" in r.stderr and "Traceback" not in r.stderr
    assert not (pack / "proof.json").exists()
    # proof already names an overlapping id: scoring refuses and names it
    wj(pack / "proof.json", {"holdout": {"ids": ["a", "c"], "scores": {}, "passed": False}})
    pred = tmp_path / "pred"
    wj(pred / "a.events.json", [])
    wj(pred / "c.events.json", [])
    r = run_cli("prove.py", "holdout", pack, "--predicted", pred)
    assert r.returncode == 2 and "overlaps training" in r.stderr and "a" in r.stderr


def test_holdout_pick_excludes_trained_and_is_seeded(tmp_path):
    pack = make_pack(tmp_path)
    r = run_cli("prove.py", "holdout-pick", pack, "--n", "3", "--seed", "7")
    assert r.returncode == 0, r.stderr
    ids = json.loads((pack / "proof.json").read_text(encoding="utf-8"))["holdout"]["ids"]
    assert sorted(ids) == ["c", "d", "e"]  # only 3 untrained videos exist
    r = run_cli("prove.py", "holdout-pick", pack, "--n", "4")
    assert r.returncode == 2


def test_holdout_scores_and_gate(tmp_path):
    pack = make_pack(tmp_path)
    wj(pack / "laws.json", {"laws": [{"lag": {"median": 0.1, "p90": 0.3}}, {"lag": {"p90": 0.5}}]})
    run_cli("prove.py", "holdout-pick", pack, "--ids", "c,d")
    pred = tmp_path / "pred"
    for v in "cd":
        wj(pred / f"{v}.events.json", [{"t": 1.1, "type": "cut"}, {"t": 3.1, "type": "cut"}])
    r = run_cli("prove.py", "holdout", pack, "--predicted", pred)
    assert r.returncode == 0, r.stderr
    h = json.loads((pack / "proof.json").read_text(encoding="utf-8"))["holdout"]
    assert h["passed"] is True and h["scores"]["overall_f1"] == 1.0
    assert h["scores"]["timing_threshold_s"] == pytest.approx(0.4)
    # bad predictions fail with exit 1 and passed False
    for v in "cd":
        wj(pred / f"{v}.events.json", [{"t": 20.0, "type": "cut"}])
    r = run_cli("prove.py", "holdout", pack, "--predicted", pred)
    assert r.returncode == 1
    assert json.loads((pack / "proof.json").read_text(encoding="utf-8"))["holdout"]["passed"] is False


def test_holdout_missing_predicted_file(tmp_path):
    pack = make_pack(tmp_path)
    run_cli("prove.py", "holdout-pick", pack, "--ids", "c")
    r = run_cli("prove.py", "holdout", pack, "--predicted", tmp_path / "none")
    assert r.returncode == 2 and "missing predicted" in r.stderr


# ---- recreation

def recreation_pack(tmp_path):
    pack = tmp_path / "pack"
    wj(pack / "analysis" / "v1" / "motion.json", {"camera_moves": [mv(t0=1.0, prop="camera")],
                                                    "elements": [{"name": "t", "moves": [mv(t0=2.0)]}]})
    wj(pack / "fingerprint.json", fp())
    return pack


def test_recreation_needs_owner_ok(tmp_path):
    pack = recreation_pack(tmp_path)
    build = tmp_path / "build.motion.json"
    wj(build, {"camera_moves": [mv(t0=1.05, prop="camera")], "elements": [{"name": "t", "moves": [mv(t0=2.0)]}]})
    r = run_cli("prove.py", "recreation", pack, "--moment", "v1:0-3", "--build", build)
    assert r.returncode == 0, r.stderr
    rec = json.loads((pack / "proof.json").read_text(encoding="utf-8"))["recreation"]
    assert rec["moments"][0]["owner_ok"] is False and rec["passed"] is False
    assert rec["moments"][0]["scores_passed"] is True
    r = run_cli("prove.py", "recreation", pack, "--moment", "v1:0-3", "--build", build, "--owner-ok")
    rec = json.loads((pack / "proof.json").read_text(encoding="utf-8"))["recreation"]
    assert len(rec["moments"]) == 1 and rec["passed"] is True


def test_recreation_fails_on_bad_curve_and_missing_move(tmp_path):
    pack = recreation_pack(tmp_path)
    build = tmp_path / "b.json"
    wj(build, {"camera_moves": [mv(t0=1.0, bez=LINEAR, prop="camera")], "elements": []})
    r = run_cli("prove.py", "recreation", pack, "--moment", "v1:0-3", "--build", build, "--owner-ok")
    assert r.returncode == 1
    rec = json.loads((pack / "proof.json").read_text(encoding="utf-8"))["recreation"]
    assert rec["passed"] is False
    assert any(s.get("missing") for s in rec["moments"][0]["scores"])


def test_recreation_empty_reference_window_proves_nothing(tmp_path):
    pack = recreation_pack(tmp_path)
    build = tmp_path / "b.json"
    wj(build, {"camera_moves": []})
    r = run_cli("prove.py", "recreation", pack, "--moment", "v1:10-12", "--build", build, "--owner-ok")
    assert r.returncode == 1
    assert json.loads((pack / "proof.json").read_text(encoding="utf-8"))["recreation"]["passed"] is False


def test_recreation_bad_input(tmp_path):
    pack = recreation_pack(tmp_path)
    assert run_cli("prove.py", "recreation", pack, "--moment", "nonsense", "--build", "x.json").returncode == 2
    r = run_cli("prove.py", "recreation", pack, "--moment", "v1:0-3", "--build", tmp_path / "x.mp4")
    assert r.returncode == 3 and "track_motion" in r.stderr


def test_qa_cli(tmp_path):
    pack = recreation_pack(tmp_path)
    build = tmp_path / "b.json"
    wj(build, {"camera_moves": [mv(dur=40)]})
    r = run_cli("prove.py", "qa", pack, "--build", build)
    assert r.returncode == 1 and "dur_f" in r.stdout
    wj(build, {"camera_moves": [mv(dur=15)]})
    assert run_cli("prove.py", "qa", pack, "--build", build).returncode == 0


# ---- ready gate (styles.py)

def add_style(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir(exist_ok=True)
    assert run_cli("styles.py", "add", "x1", "--name", "X One", "--pack", pack).returncode == 0
    return pack


def reg(home):
    return json.loads((home / "styles.json").read_text(encoding="utf-8"))["styles"][0]


def passed_proof(pack, trained=("a", "b")):
    from styles import trained_hash
    wj(pack / "fingerprint.json", fp(trained=trained))
    wj(pack / "proof.json", {"holdout": {"ids": ["c"], "scores": {}, "passed": True,
                                         "trained_hash": trained_hash(trained)},
                             "recreation": {"moments": [], "passed": True}})


def test_ready_gate(home, tmp_path):
    pack = add_style(tmp_path)
    r = run_cli("styles.py", "set-status", "x1", "ready")
    assert r.returncode == 1 and "cannot be ready" in r.stderr
    assert reg(home)["status"] == "training"
    # `set status=ready` is gated too
    assert run_cli("styles.py", "set", "x1", "status=ready").returncode == 1
    # one test passed is not enough
    wj(pack / "proof.json", {"holdout": {"passed": True}, "recreation": {"passed": False}})
    assert run_cli("styles.py", "set-status", "x1", "ready").returncode == 1
    # force needs a reason
    assert run_cli("styles.py", "set-status", "x1", "ready", "--force").returncode == 1
    r = run_cli("styles.py", "set-status", "x1", "ready", "--force", "--reason", "x")
    assert r.returncode == 0, r.stderr
    e = reg(home)
    assert e["status"] == "ready" and e["ready_forced"]["reason"] == "x"
    # a fully passed proof needs no force
    run_cli("styles.py", "set-status", "x1", "training")
    passed_proof(pack)
    assert run_cli("styles.py", "set-status", "x1", "ready").returncode == 0
    assert "ready_forced" not in {k for k in reg(home) if reg(home)[k] is None}
    assert run_cli("styles.py", "set-status", "x1", "archived").returncode == 0


def test_ready_blocked_by_untested_ai_recipe(home, tmp_path):
    pack = add_style(tmp_path)
    passed_proof(pack)
    wj(pack / "recipes.json", {"recipes": [
        {"kind": "bg", "how": "ai_image", "test": {"status": "failed"}},
        {"kind": "title", "how": "code_text", "test": {"status": "untested"}},
        {"kind": "ok", "how": "ai_video", "test": {"status": "passed"}}]})
    r = run_cli("styles.py", "set-status", "x1", "ready")
    assert r.returncode == 1 and "bg" in r.stderr and "title" not in r.stderr
    assert reg(home)["status"] == "training"
    # --force does not waive recipes
    assert run_cli("styles.py", "set-status", "x1", "ready", "--force", "--reason", "r").returncode == 1
    assert run_cli("styles.py", "set-status", "x1", "ready", "--waive-recipes").returncode == 1  # no reason
    r = run_cli("styles.py", "set-status", "x1", "ready", "--waive-recipes", "--reason", "will test later")
    assert r.returncode == 0, r.stderr
    e = reg(home)
    assert e["status"] == "ready" and e["recipes_waived"]["recipes"] == ["bg"]
    assert "ready_forced" not in e
    # an untested recipe with no test block at all also blocks
    run_cli("styles.py", "set-status", "x1", "training")
    wj(pack / "recipes.json", {"recipes": [{"kind": "k", "how": "ai_audio"}]})
    assert run_cli("styles.py", "set-status", "x1", "ready").returncode == 1
    wj(pack / "recipes.json", {"recipes": [{"kind": "k", "how": "ai_audio", "test": {"status": "passed"}}]})
    assert run_cli("styles.py", "set-status", "x1", "ready").returncode == 0


def test_ready_gate_corrupt_files_fail_closed(home, tmp_path):
    pack = add_style(tmp_path)
    (pack / "proof.json").write_text("{not json", encoding="utf-8")
    r = run_cli("styles.py", "set-status", "x1", "ready")
    assert r.returncode == 1 and "Traceback" not in r.stderr
    passed_proof(pack)
    (pack / "recipes.json").write_text("{not json", encoding="utf-8")
    assert run_cli("styles.py", "set-status", "x1", "ready").returncode == 1


# ---- fix round 1

def test_holdout_timing_not_run_without_laws(tmp_path):
    pack = make_pack(tmp_path)
    run_cli("prove.py", "holdout-pick", pack, "--ids", "c")
    pred = tmp_path / "pred"
    wj(pred / "c.events.json", [{"t": 1.0, "type": "cut"}, {"t": 3.0, "type": "cut"}])
    r = run_cli("prove.py", "holdout", pack, "--predicted", pred)
    assert r.returncode == 1
    assert "timing not checked: no laws with lag.p90; run laws.py first" in r.stdout
    h = json.loads((pack / "proof.json").read_text(encoding="utf-8"))["holdout"]
    assert h["passed"] is False and h["scores"]["timing"] == "not_run"


def test_ready_blocked_by_stale_proof(home, tmp_path):
    pack = add_style(tmp_path)
    passed_proof(pack)
    assert run_cli("styles.py", "set-status", "x1", "ready").returncode == 0
    run_cli("styles.py", "set-status", "x1", "training")
    wj(pack / "fingerprint.json", fp(trained=("a", "b", "z")))  # fingerprint retrained
    r = run_cli("styles.py", "set-status", "x1", "ready")
    assert r.returncode == 1 and "proof is stale: re-run prove.py holdout" in r.stderr
    wj(pack / "fingerprint.json", fp(trained=("a", "c")))  # hold-out id now in training
    r = run_cli("styles.py", "set-status", "x1", "ready")
    assert r.returncode == 1 and "overlaps training" in r.stderr and "c" in r.stderr


def test_holdout_pick_needs_fingerprint(tmp_path):
    pack = tmp_path / "pack"
    wj(pack / "analysis" / "c" / "events.json", [])
    r = run_cli("prove.py", "holdout-pick", pack, "--ids", "c")
    assert r.returncode == 2 and "no fingerprint.json; run fingerprint.py first" in r.stderr
    assert not (pack / "proof.json").exists()
    assert run_cli("prove.py", "holdout-pick", pack, "--ids", "c", "--allow-no-fingerprint").returncode == 0
    h = json.loads((pack / "proof.json").read_text(encoding="utf-8"))["holdout"]
    assert h["trained_unknown"] is True


def test_gate_refuses_trained_unknown(home, tmp_path):
    pack = add_style(tmp_path)
    passed_proof(pack)
    proof = json.loads((pack / "proof.json").read_text(encoding="utf-8"))
    proof["holdout"]["trained_unknown"] = True
    wj(pack / "proof.json", proof)
    r = run_cli("styles.py", "set-status", "x1", "ready")
    assert r.returncode == 1 and "without a fingerprint" in r.stderr


def test_gate_empty_pack_and_clears_overrides(home, tmp_path):
    pack = add_style(tmp_path)
    run_cli("styles.py", "set-status", "x1", "ready", "--force", "--reason", "x")
    assert "ready_forced" in reg(home)
    run_cli("styles.py", "set-status", "x1", "training")
    passed_proof(pack)
    assert run_cli("styles.py", "set-status", "x1", "ready").returncode == 0
    e = reg(home)
    assert "ready_forced" not in e and "recipes_waived" not in e
    run_cli("styles.py", "set-status", "x1", "training")
    assert run_cli("styles.py", "set", "x1", "pack=").returncode == 0
    r = run_cli("styles.py", "set-status", "x1", "ready")
    assert r.returncode == 1 and "no pack folder" in r.stderr


def test_non_dict_json_is_bad_input(tmp_path):
    pack = recreation_pack(tmp_path)
    build = tmp_path / "b.json"
    wj(build, [1, 2])
    r = run_cli("prove.py", "qa", pack, "--build", build)
    assert r.returncode == 2 and "Traceback" not in r.stderr
    wj(pack / "proof.json", [])
    run_cli("prove.py", "holdout-pick", pack, "--ids", "x", "--allow-no-fingerprint")
    wj(pack / "proof.json", ["x"])
    r = run_cli("prove.py", "holdout", pack, "--predicted", tmp_path)
    assert r.returncode == 2 and "Traceback" not in r.stderr


def test_graph_pairs_by_prop_and_time(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    wj(a, {"camera_moves": [mv(t0=1.0, prop="camera")], "elements": [{"name": "t", "moves": [mv(t0=2.0, prop="t")]}]})
    wj(b, {"camera_moves": [], "elements": [{"name": "t", "moves": [mv(t0=2.1, bez=LINEAR, prop="t")]}]})
    out = tmp_path / "o.html"
    r = run_cli("prove.py", "graph", a, "--compare", b, "--out", out)
    assert r.returncode == 0 and "warning" in r.stderr
    html = out.read_text(encoding="utf-8")
    # camera has no partner: 1 polyline x2 svgs; t has ref+build: 2 x2
    assert html.count("<polyline") == 2 + 4


def test_holdout_reports_structure_and_retention(tmp_path):
    pytest.importorskip("creator")
    pack = make_pack(tmp_path)
    wj(pack / "laws.json", {"laws": [{"lag": {"p90": 0.5}}]})
    wj(pack / "analysis" / "c" / "creator.json", {
        "story": {"structure": [{"part": "hook", "t0": 0.0, "t1": 3.0}, {"part": "body", "t0": 3.0, "t1": 8.0},
                            {"part": "payoff", "t0": 8.0, "t1": 12.0}]},
        "retention": [{"t": 2.0, "device": "zoom"}]})
    run_cli("prove.py", "holdout-pick", pack, "--ids", "c")
    pred = tmp_path / "pred"
    wj(pred / "c.events.json", [{"t": 1.0, "type": "cut"}, {"t": 3.0, "type": "cut"}])
    # the opening (t < 0.5 s) is not scored; body matches, payoff is far off
    wj(pred / "c.structure.json", [{"part": "hook", "t0": 0.2, "t1": 3.0}, {"part": "body", "t0": 3.4, "t1": 8.0},
                                   {"part": "payoff", "t0": 20.0, "t1": 25.0}])
    wj(pred / "c.retention.json", [{"t": 2.1, "device": "zoom"}])
    r = run_cli("prove.py", "holdout", pack, "--predicted", pred)
    assert r.returncode == 0, r.stderr
    h = json.loads((pack / "proof.json").read_text(encoding="utf-8"))["holdout"]
    assert h["passed"] is True  # structure/retention do not decide pass
    sc = h["scores"]
    assert sc["structure"]["per_video"]["c"]["tp"] == 1 and sc["structure"]["n_actual"] == 2
    assert sc["retention"]["overall_f1"] == 1.0
    assert "not_run" not in sc


def test_holdout_structure_not_run_without_creator_module(tmp_path, monkeypatch):
    pack = make_pack(tmp_path)
    wj(pack / "analysis" / "c" / "creator.json", {"story": {"structure": []}})
    pred = tmp_path / "pred"
    wj(pred / "c.structure.json", [])
    monkeypatch.setitem(__import__("sys").modules, "creator", None)  # import raises ImportError
    extra, not_run = prove._creator_scores(pack, ["c"], pred)
    assert extra == {} and not_run == ["structure", "retention"]
    extra, not_run = prove._creator_scores(pack, ["d"], pred)  # no creator.json / predictions: nothing
    assert extra == {} and not_run == []


def test_creator_scores_malformed_and_skipped_ids(tmp_path):
    pytest.importorskip("creator")
    pack = make_pack(tmp_path)
    wj(pack / "analysis" / "c" / "creator.json", {"story": {"structure": []}})
    wj(pack / "analysis" / "d" / "creator.json", {"story": {"structure": []}})  # doc, no prediction
    wj(pack / "analysis" / "e" / "creator.json", {"story": {"structure": []}})
    pred = tmp_path / "pred"
    wj(pred / "c.structure.json", ["hook", "payoff"])  # list of strings
    wj(pred / "a.structure.json", [])  # prediction, no creator.json
    wj(pred / "e.retention.json", {"retention": 5})
    extra, not_run = prove._creator_scores(pack, ["a", "c", "d", "e"], pred)
    assert extra == {}
    assert "creator_scores:c:bad_input" in not_run and "creator_scores:e:bad_input" in not_run
    assert "creator_scores:a:no_creator_json" in not_run and "creator_scores:d:no_prediction" in not_run
    (pred / "c.structure.json").write_text("{broken", encoding="utf-8")
    assert "creator_scores:c:bad_input" in prove._creator_scores(pack, ["c"], pred)[1]
