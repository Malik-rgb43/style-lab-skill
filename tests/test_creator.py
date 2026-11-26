"""creator.py: the "think like the creator" layer (story, decisions and why, retention, toolchain,
visual ideas), its synthesis into creator-mind.json, the planner checklist and the ideation brief."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

import creator
from conftest import run_cli

PARTS = ["hook", "setup", "turn", "payoff", "cta"]


def _idea(t0, line, mtype, translation, idea, why="it lands faster than a literal shot", pattern=None):
    d = {"t0": t0, "t1": t0 + 2.0, "line": line, "message": f"the viewer gets: {line}", "emotion": "surprise",
         "idea": idea, "translation": translation, "message_type": mtype, "why_this": why,
         "alternatives": ["a plain close-up of the thing", "a text card with the words"],
         "craft": {"framing": "centred, full frame", "motion": "slow push in", "text_role": "one keyword",
                   "sound_role": "low hit on the reveal"}}
    if pattern:
        d["pattern"] = pattern
    return d


def make_doc(order=PARTS, devices=(1.0, 3.0, 6.0, 8.0), ideas=None, tool=("broll", "ai_image+i2v"),
             rules=("Show, don't tell.",), principle="contrast before payoff"):
    structure, t = [], 0.0
    for p in order:
        structure.append({"part": p, "t0": t, "t1": t + 4.0, "words": f"{p} words"})
        t += 4.0
    retention, prev = [], 0.0
    for d in devices:
        retention.append({"t": d, "device": "pattern_interrupt", "gap_s_since_last": round(d - prev, 4)})
        prev = d
    return {
        "schema": "creator/1",
        "story": {"structure": structure,
                  "hook": {"type": "question", "first_words_s": 0.4, "visual_device": "object in hand"},
                  "open_loops": [{"opened_t": 0.5, "closed_t": 14.0, "text": "what is in the box"}],
                  "audience_promise": "you will learn one trick"},
        "decisions": [{"t": 4.0, "what": "cut to a wide shot", "why": "the close-up had run long",
                       "problem_solved": "attention_dip", "principle": principle,
                       "alternative_rejected": "a zoom", "evidence": "measure"}],
        "retention": retention,
        "toolchain": [{"element": tool[0], "likely_tool": tool[1], "how": "still generated, then animated",
                       "confidence": 0.7, "evidence": "morphing edges on the hands"}],
        "rules_of_thumb": list(rules),
        "visual_ideas": list(ideas) if ideas is not None else [
            _idea(1.0, "three times faster", "number", "scale_shift", "a tiny clock grows to fill the room")],
    }


def write_pack(tmp_path, docs):
    pack = tmp_path / "pack"
    for vid, doc in docs.items():
        d = pack / "analysis" / vid
        d.mkdir(parents=True)
        (d / "creator.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return pack


# ---------------------------------------------------------------- check

def test_check_flags_bad_schema(tmp_path):
    bad = make_doc()
    del bad["decisions"][0]["why"]
    pack = write_pack(tmp_path, {"V01": make_doc(), "V02": bad})
    r = run_cli("creator.py", "check", pack)
    assert r.returncode == 1, r.stdout + r.stderr
    out = r.stdout + r.stderr
    assert "V02" in out and "decisions[0].why" in out
    assert "V01" not in out.replace("V01 ok", "")

    ok = write_pack(tmp_path / "b", {"V01": make_doc()})
    r = run_cli("creator.py", "check", ok)
    assert r.returncode == 0, r.stdout + r.stderr


def test_visual_ideas_schema_checked(tmp_path):
    no_trans = make_doc()
    del no_trans["visual_ideas"][0]["translation"]
    no_msg = make_doc()
    del no_msg["visual_ideas"][0]["message"]
    wrong = make_doc()
    wrong["visual_ideas"][0]["translation"] = "teleport"
    pack = write_pack(tmp_path, {"V01": no_trans, "V02": no_msg, "V03": wrong})
    r = run_cli("creator.py", "check", pack)
    assert r.returncode == 1
    out = r.stdout + r.stderr
    assert "V01" in out and "visual_ideas[0].translation" in out
    assert "V02" in out and "visual_ideas[0].message" in out
    assert "V03" in out and "teleport" in out


def test_check_pure_function_messages():
    doc = make_doc()
    assert creator.validate(doc) == []
    doc["toolchain"][0]["evidence"] = ""
    doc["toolchain"][0]["confidence"] = 1.5
    errs = creator.validate(doc)
    assert any("toolchain[0].evidence" in e for e in errs)
    assert any("toolchain[0].confidence" in e for e in errs)


def test_check_bad_input_exit_2(tmp_path):
    r = run_cli("creator.py", "check", tmp_path / "missing")
    assert r.returncode == 2
    empty = tmp_path / "empty"
    (empty / "analysis" / "V01").mkdir(parents=True)
    r = run_cli("creator.py", "check", empty)
    assert r.returncode == 2
    assert "Traceback" not in r.stderr


# ---------------------------------------------------------------- synth

def test_synth_story_template_and_gaps():
    docs = {
        "V01": make_doc(devices=(1.0, 3.0, 6.0, 8.0)),          # max gap 3.0
        "V02": make_doc(devices=(2.0, 4.0, 5.0, 9.5)),          # 4.5
        "V03": make_doc(devices=(0.5, 1.5, 2.5, 3.5)),          # 1.0
        "V04": make_doc(order=["hook", "setup", "payoff", "cta"], devices=(1.0, 7.0, 8.0)),  # 6.0
        "V05": make_doc(devices=(1.0, 3.0, 5.0, 7.0, 9.0)),     # 2.0
    }
    mind = creator.synth(docs)
    st = mind["story_template"]
    assert st["order"] == PARTS
    assert st["seen_in"] == 4 and st["share"] == pytest.approx(0.8)
    assert st["parts"]["turn"]["duration_s"]["median"] == pytest.approx(4.0)
    assert st["parts"]["turn"]["duration_s"]["seen_in"] == 4

    maxes = [3.0, 4.5, 1.0, 6.0, 2.0]
    gap = mind["retention"]["max_gap_s"]
    assert gap["p90"] == pytest.approx(round(float(np.percentile(maxes, 90)), 4))
    assert gap["median"] == pytest.approx(3.0)
    assert gap["n"] == 5
    assert mind["retention"]["never_more_than_s"] == pytest.approx(gap["p90"])
    # V01: 4 devices in 20 s -> 2 per 10 s
    assert mind["retention"]["devices_per_10s"]["n"] == 5
    assert mind["hooks"]["types"] == {"question": 5}
    assert mind["principles"][0]["principle"] == "contrast before payoff"
    assert mind["principles"][0]["seen_in"] == 5
    assert mind["trained_on"] == ["V01", "V02", "V03", "V04", "V05"]


def test_synth_handles_empty_and_short():
    doc = make_doc(order=[], devices=(), ideas=[], rules=())
    doc["decisions"] = []
    doc["toolchain"] = []
    mind = creator.synth({"V01": doc})
    assert mind["retention"]["max_gap_s"]["n"] == 0
    assert mind["story_template"]["order"] == []
    assert mind["ideation"]["methods"] == {}
    assert mind["ideation"]["literal_share"] is None
    assert len(creator.checklist(mind)) >= 5


def test_rules_of_thumb_seen_in():
    docs = {"V01": make_doc(rules=["Show, don't tell."]),
            "V02": make_doc(rules=["  show DONT   tell "]),
            "V03": make_doc(rules=["Cut on the breath"])}
    mind = creator.synth(docs)
    rules = {r["key"]: r for r in mind["rules_of_thumb"]}
    assert rules["show dont tell"]["seen_in"] == 2
    assert rules["show dont tell"]["videos"] == ["V01", "V02"]
    assert rules["cut on the breath"]["seen_in"] == 1
    assert mind["rules_of_thumb"][0]["key"] == "show dont tell"
    assert creator.norm_text("Show, don't tell.") == creator.norm_text("show   DONT tell")


def test_toolchain_consensus():
    docs = {"V01": make_doc(tool=("broll", "ai_image+i2v")),
            "V02": make_doc(tool=("broll", "AI_image + i2v")),
            "V03": make_doc(tool=("broll", "ai_image+i2v")),
            "V04": make_doc(tool=("broll", "stock"))}
    mind = creator.synth(docs)
    b = mind["toolchain"]["broll"]
    assert b["consensus"] == "ai_image+i2v"
    assert b["confidence"] == pytest.approx(0.75)
    assert b["seen_in"] == 4
    assert {v["tool"]: v["videos"] for v in b["votes"]}["stock"] == 1


# ---------------------------------------------------------------- ideation

def _ideation_docs():
    def ideas(vid_n):
        out = [
            _idea(1.0, f"ten times bigger {vid_n}", "number", "scale_shift", "a coin grows taller than a house",
                  why="size makes the number felt", pattern="Object grows past the frame"),
            _idea(4.0, f"half the cost {vid_n}", "number", "scale_shift", "a receipt shrinks to a stamp",
                  why="size makes the number felt", pattern="object grows past the frame!"),
            _idea(8.0, f"before it was slow {vid_n}", "contrast", "before_after", "split screen, old and new",
                  pattern="split screen"),
            _idea(12.0, f"it just works {vid_n}", "claim", "literal", "the app running on a phone"),
        ]
        if vid_n % 2 == 0:
            out.append(_idea(15.0, f"forty percent {vid_n}", "number", "data_viz", "a bar fills to 40 %"))
        return out
    return {f"V0{n}": make_doc(ideas=ideas(n)) for n in range(1, 5)}


def test_ideation_playbook():
    mind = creator.synth(_ideation_docs())
    ide = mind["ideation"]
    num = ide["playbook"]["number"]
    assert num["n"] == 10
    assert num["methods"][0]["method"] == "scale_shift"
    assert num["methods"][0]["share"] == pytest.approx(0.8)
    assert num["methods"][1]["method"] == "data_viz"
    assert num["methods"][1]["share"] == pytest.approx(0.2)
    assert "number" in num["summary"] and "scale_shift 80 %" in num["summary"] and "data_viz 20 %" in num["summary"]
    assert num["methods"][0]["examples"][0]["idea"]
    assert num["methods"][0]["examples"][0]["line"]

    assert ide["methods"]["scale_shift"]["count"] == 8
    assert ide["methods"]["scale_shift"]["seen_in"] == 4
    assert ide["literal_share"] == pytest.approx(4 / 18, abs=1e-4)
    assert ide["figurative_share"] == pytest.approx(14 / 18, abs=1e-4)
    pats = {p["key"]: p for p in ide["patterns"]}
    assert pats["object grows past the frame"]["count"] == 8
    assert pats["object grows past the frame"]["seen_in"] == 4


def test_ideate_brief():
    mind = creator.synth(_ideation_docs())
    brief = creator.ideate(mind, "the new tool is five times cheaper", "number")
    tops = [m["method"] for m in brief["top_methods"]]
    assert tops[:2] == ["scale_shift", "data_viz"]
    assert brief["top_methods"][0]["examples"]
    ex = brief["top_methods"][0]["examples"][0]
    assert ex["line"] and ex["idea"]
    wild = brief["wild_card"]["method"]
    assert wild not in tops and wild not in ("literal", "other")
    assert wild in creator.TRANSLATIONS
    assert "3 ideas" in brief["rule"] and wild in brief["rule"]
    assert brief["why_patterns"] and "size makes the number felt" in brief["why_patterns"]
    assert brief["craft_notes"]["scale_shift"]["motion"]

    with pytest.raises(ValueError):
        creator.ideate(mind, "x", "banana")
    # a message type the creator never used falls back to the overall distribution
    fb = creator.ideate(mind, "click the link below", "cta")
    assert fb["fallback"] == "overall" and fb["top_methods"]


def test_ideate_cli(tmp_path):
    pack = write_pack(tmp_path, _ideation_docs())
    r = run_cli("creator.py", "synth", pack, "--all")
    assert r.returncode == 0, r.stdout + r.stderr
    mind = json.loads((pack / "creator-mind.json").read_text(encoding="utf-8"))
    assert mind["schema"] == "creator-mind/1"
    r = run_cli("creator.py", "ideate", pack, "--line", "פי שלושה מהר יותר", "--type", "number")
    assert r.returncode == 0, r.stdout + r.stderr
    brief = json.loads(r.stdout)
    assert brief["line"] == "פי שלושה מהר יותר"
    assert brief["top_methods"][0]["method"] == "scale_shift"
    r = run_cli("creator.py", "ideate", tmp_path / "nopack", "--line", "x", "--type", "number")
    assert r.returncode == 2


def test_synth_cli_refuses_bad_files(tmp_path):
    bad = make_doc()
    del bad["visual_ideas"][0]["message"]
    pack = write_pack(tmp_path, {"V01": make_doc(), "V02": bad})
    r = run_cli("creator.py", "synth", pack)
    assert r.returncode == 1
    assert not (pack / "creator-mind.json").exists()
    r = run_cli("creator.py", "synth", pack, "--train", "V01,V09")
    assert r.returncode == 2
    r = run_cli("creator.py", "synth", pack, "--train", "V01")
    assert r.returncode == 0, r.stdout + r.stderr
    mind = json.loads((pack / "creator-mind.json").read_text(encoding="utf-8"))
    assert mind["trained_on"] == ["V01"]


def test_synth_dedupes_identical_sources(tmp_path):
    pack = write_pack(tmp_path, {"V01": make_doc(), "V02": make_doc(), "V03": make_doc(rules=["other rule"])})
    for vid, sha in (("V01", "a" * 64), ("V02", "a" * 64), ("V03", "b" * 64)):
        (pack / "analysis" / vid / "measure.json").write_text(
            json.dumps({"video": {"sha256": sha}}), encoding="utf-8")
    r = run_cli("creator.py", "synth", pack)
    assert r.returncode == 0, r.stdout + r.stderr
    mind = json.loads((pack / "creator-mind.json").read_text(encoding="utf-8"))
    assert mind["trained_on"] == ["V01", "V03"]
    assert mind["dupes"] == [["V01", "V02"]]


# ---------------------------------------------------------------- structure scoring

def test_score_structure():
    actual = [{"part": "hook", "t0": 0.0, "t1": 3.0}, {"part": "setup", "t0": 3.0, "t1": 10.0},
              {"part": "turn", "t0": 10.0, "t1": 15.0}, {"part": "payoff", "t0": 15.0, "t1": 22.0}]
    pred = [{"part": "hook", "t0": 0.0, "t1": 3.5},      # hook start matches (0.0)
            {"part": "setup", "t0": 3.5, "t1": 12.0},    # 0.5 off: match
            {"part": "turn", "t0": 12.0, "t1": 16.0},    # 2.0 off: no match
            {"part": "cta", "t0": 16.0, "t1": 22.0}]     # wrong part
    s = creator.score_structure(pred, actual)
    assert s["tp"] == 2 and s["n_pred"] == 4 and s["n_actual"] == 4
    assert s["precision"] == pytest.approx(0.5) and s["recall"] == pytest.approx(0.5)
    assert s["overall_f1"] == pytest.approx(0.5)
    assert s["timing_mae_s"] == pytest.approx(0.25)
    assert s["order_match"] is False
    assert creator.score_structure(pred, actual, tol_s=2.0)["tp"] == 3
    perfect = creator.score_structure(copy.deepcopy(actual), actual)
    assert perfect["overall_f1"] == pytest.approx(1.0) and perfect["order_match"] is True
    empty = creator.score_structure([], [])
    assert empty["overall_f1"] == 0.0 and empty["timing_mae_s"] is None


def test_retention_events_for_prove():
    evs = creator.retention_events([{"t": 1.0, "device": "reveal", "gap_s_since_last": 1.0}, {"t": "x"}])
    assert evs == [{"t": 1.0, "type": "reveal"}]


# ---------------------------------------------------------------- checklist

def test_checklist_non_empty_and_generic():
    docs = _ideation_docs()
    for d in docs.values():
        d["rules_of_thumb"] = ["Zebra stripes at every corner"]
        d["toolchain"][0]["likely_tool"] = "frobnicator studio"
        d["decisions"][0]["principle"] = "quokka principle"
    mind = creator.synth(docs)
    items = creator.checklist(mind)
    assert len(items) >= 5
    assert all(isinstance(i, str) and i.strip() for i in items)
    text = " ".join(items).lower()
    for word in ("zebra", "frobnicator", "quokka", "coin", "receipt", "v01", "broll"):
        assert word not in text
    assert any("gap" in i.lower() for i in items)
    assert any("loop" in i.lower() for i in items)
    assert any("principle" in i.lower() for i in items)
    assert any("message" in i.lower() for i in items)
    assert str(mind["retention"]["never_more_than_s"]) in text
