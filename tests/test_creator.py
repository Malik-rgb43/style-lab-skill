"""creator.py: the "think like the creator" layer (story, decisions and why, retention, toolchain,
visual ideas), its synthesis into creator-mind.json, the planner checklist and the ideation brief."""
import copy
import json

import numpy as np
import pytest

import creator
from conftest import run_cli

PARTS = ["hook", "setup", "turn", "payoff", "cta"]
PIPE = ["ai_image", "ai_i2v"]


def _idea(t0, line, mtype, translation, idea, why="it lands faster than a literal shot", pattern=None,
          principle="show_dont_tell", emotion="surprise"):
    d = {"t0": t0, "t1": t0 + 2.0, "line": line, "message": f"the viewer gets: {line}", "emotion": emotion,
         "idea": idea, "translation": translation, "message_type": mtype, "principle": principle, "why_this": why,
         "alternatives": ["a plain close-up of the thing", "a text card with the words"],
         "craft": {"framing": "centred, full frame", "motion": "slow push in", "text_role": "one keyword",
                   "sound_role": "low hit on the reveal"}}
    if pattern:
        d["pattern"] = pattern
    return d


def make_doc(order=PARTS, devices=(2.0, 5.0, 9.0, 13.0, 17.0), ideas=None, tool=("broll", PIPE, 0.7),
             rules=("Show, don't tell.",), principle="contrast_for_clarity", rejected="a zoom on the punchline"):
    structure, t = [], 0.0
    for p in order:
        structure.append({"part": p, "t0": t, "t1": t + 4.0, "words": f"{p} words"})
        t += 4.0
    retention = [{"t": d, "device": "pattern_interrupt", "gap_s_since_last": 99.0} for d in devices]
    return {
        "schema": "creator/1",
        "story": {"structure": structure,
                  "hook": {"type": "question", "first_words_s": 0.4, "visual_device": "Object in hand"},
                  "open_loops": [{"opened_t": 0.5, "closed_t": 14.0, "text": "what is in the box"}],
                  "audience_promise": "You will learn one trick"},
        "decisions": [{"t": 4.0, "what": "cut to a wide shot", "why": "the close-up had run long",
                       "problem_solved": "attention_dip", "principle": principle,
                       "alternative_rejected": rejected, "evidence": "measure"}],
        "retention": retention,
        "toolchain": [{"element": tool[0], "likely_tool": tool[1], "how": "still generated, then animated",
                       "confidence": tool[2], "evidence": "morphing edges on the hands"}],
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


# ---------------------------------------------------------------- check + vocab

def test_check_flags_bad_schema(tmp_path):
    bad = make_doc()
    del bad["decisions"][0]["why"]
    pack = write_pack(tmp_path, {"V01": make_doc(), "V02": bad})
    r = run_cli("creator.py", "check", pack)
    assert r.returncode == 1, r.stdout + r.stderr
    out = r.stdout + r.stderr
    assert "V02/creator.json: decisions[0].why" in out
    assert "V01/" not in out

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
    assert "V01/creator.json: visual_ideas[0].translation" in out
    assert "V02/creator.json: visual_ideas[0].message" in out
    assert "V03" in out and "teleport" in out


def test_vocab_enums_checked():
    doc = make_doc()
    assert creator.validate(doc) == []
    assert "typographic" in creator.TRANSLATIONS
    assert len(creator.PRINCIPLES) >= 20
    doc["toolchain"][0]["element"] = "brol"
    doc["toolchain"][0]["likely_tool"] = ["ai_image", "magic_box"]
    doc["toolchain"][0]["evidence"] = ""
    doc["toolchain"][0]["confidence"] = 1.5
    doc["decisions"][0]["principle"] = "my own wording"
    doc["visual_ideas"][0]["principle"] = "nope"
    errs = creator.validate(doc)
    for field in ("toolchain[0].element", "toolchain[0].likely_tool", "toolchain[0].evidence",
                  "toolchain[0].confidence", "decisions[0].principle", "visual_ideas[0].principle"):
        assert any(field in e for e in errs), field
    ok = make_doc()
    ok["decisions"][0]["principle"] = "other"
    ok["decisions"][0]["principle_note"] = "free text is still allowed here"
    ok["toolchain"][0]["likely_tool"] = "stock"
    assert creator.validate(ok) == []


def test_visual_only_line():
    doc = make_doc()
    doc["visual_ideas"][0]["line"] = ""
    assert any("visual_ideas[0].line" in e for e in creator.validate(doc))
    doc["visual_ideas"][0]["visual_only"] = True
    assert creator.validate(doc) == []


def test_check_bad_input_exit_2(tmp_path):
    r = run_cli("creator.py", "check", tmp_path / "missing")
    assert r.returncode == 2
    empty = tmp_path / "empty"
    (empty / "analysis" / "V01").mkdir(parents=True)
    r = run_cli("creator.py", "check", empty)
    assert r.returncode == 2
    assert "Traceback" not in r.stderr


# ---------------------------------------------------------------- merge-rules

def test_merge_rules_similar_and_apply(tmp_path):
    docs = {"V01": make_doc(rules=["Show the real screen to prove a claim"]),
            "V02": make_doc(rules=["show the real screen to prove claims", "Cut on the breath"]),
            "V03": make_doc(rules=["Show the real screen to prove a claim", "never hold a shot past four seconds"])}
    clusters = creator.similar_rules(docs)
    assert len(clusters) == 1
    c = clusters[0]
    assert c["suggested"] == "Show the real screen to prove a claim"
    assert {r["key"] for r in c["rules"]} == {"show the real screen to prove a claim",
                                              "show the real screen to prove claims"}

    pack = write_pack(tmp_path, docs)
    r = run_cli("creator.py", "merge-rules", pack)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(r.stdout)["clusters"][0]["suggested"] == "Show the real screen to prove a claim"
    m = tmp_path / "map.json"
    m.write_text(json.dumps({"show the real screen to prove claims": "Show the real screen to prove a claim"}),
                 encoding="utf-8")
    r = run_cli("creator.py", "merge-rules", pack, "--apply", m)
    assert r.returncode == 0, r.stdout + r.stderr
    v2 = json.loads((pack / "analysis" / "V02" / "creator.json").read_text(encoding="utf-8"))
    assert v2["rules_of_thumb"] == ["Show the real screen to prove a claim", "Cut on the breath"]
    r = run_cli("creator.py", "merge-rules", pack)
    assert json.loads(r.stdout)["clusters"] == []
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    assert run_cli("creator.py", "merge-rules", pack, "--apply", bad).returncode == 2


# ---------------------------------------------------------------- candidates

def test_candidates_from_measure_and_sound(tmp_path):
    measure = {"video": {"duration_s": 12.0, "sha256": "c" * 64},
               "shots": [{"t0": 0, "t1": 2, "dur_s": 2.0}, {"t0": 2, "t1": 4, "dur_s": 2.0},
                         {"t0": 4, "t1": 10, "dur_s": 6.0}, {"t0": 10, "t1": 12, "dur_s": 2.0}],
               "cuts": [{"t": 2.0, "kind": "hard", "dur_f": 1}, {"t": 4.0, "kind": "hard", "dur_f": 1},
                        {"t": 7.0, "kind": "jump", "dur_f": 1}, {"t": 10.0, "kind": "gradual", "dur_f": 8}],
               "text": {"boxes": [{"t": 5.0, "box": [0.1, 0.1, 0.5, 0.2], "h_pct": 20.0},
                                  {"t": 5.4, "box": [0.1, 0.1, 0.5, 0.2], "h_pct": 20.0},
                                  {"t": 8.0, "box": [0.1, 0.8, 0.5, 0.03], "h_pct": 3.0}]}}
    sound = {"sfx": [{"t": 4.05, "class_guess": "whoosh"}]}
    out = creator.candidates(measure, sound)
    assert out["median_shot_s"] == 2.0
    src = [(c["t"], c["source"]) for c in out["candidates"]]
    assert src == [(2.0, "cut"), (4.0, "cut"), (4.05, "sfx"), (5.0, "big_text"), (7.0, "jump"), (10.0, "cut")]
    last = [c for c in out["candidates"] if c["t"] == 10.0][0]
    assert last["shot_before_s"] == 6.0 and last["vs_median_shot"] == 3.0
    assert "median shot" in out["rule"]

    pack = tmp_path / "pack"
    d = pack / "analysis" / "V01"
    d.mkdir(parents=True)
    (d / "measure.json").write_text(json.dumps(measure), encoding="utf-8")
    r = run_cli("creator.py", "candidates", pack, "V01")
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(json.loads(r.stdout)["candidates"]) == 5
    assert run_cli("creator.py", "candidates", pack, "V09").returncode == 2


# ---------------------------------------------------------------- synth: story + retention

def test_synth_story_template_and_gaps():
    docs = {
        "V01": make_doc(devices=(2, 5, 9, 13, 17)),                       # gaps 2 3 4 4 4 | 3  -> 4
        "V02": make_doc(devices=(3, 6, 12, 15, 18)),                      # 3 3 6 3 3 | 2       -> 6
        "V03": make_doc(devices=(1, 4, 7, 10, 13, 16, 19)),               # 1 3 3 3 3 3 3 | 1   -> 3
        "V04": make_doc(order=["hook", "setup", "payoff", "cta"], devices=(2, 4, 6, 8, 10, 12, 14)),
        "V05": make_doc(devices=()),                                      # no device: one gap of 20 s
    }
    # V04's structure ends at 16 s, but measure.json says 25 s: the end gap is 11 s
    mind = creator.synth(docs, durations={"V04": 25.0})
    st = mind["story_template"]
    assert st["order"] == PARTS
    assert st["parts"]["turn"]["seen_in"] == 4 and st["parts"]["turn"]["share"] == pytest.approx(0.8)
    assert st["parts"]["turn"]["duration_s"]["median"] == pytest.approx(4.0)
    assert st["most_common_exact"]["order"] == PARTS and st["most_common_exact"]["count"] == 4

    maxes = [4.0, 6.0, 3.0, 11.0, 20.0]
    gap = mind["retention"]["max_gap_s"]
    assert gap["p90"] == pytest.approx(round(float(np.percentile(maxes, 90)), 4))
    assert gap["median"] == pytest.approx(6.0)
    assert gap["n"] == 5 and gap["seen_in"] == 5
    assert mind["retention"]["never_more_than_s"] == pytest.approx(gap["p90"])
    assert mind["retention"]["gap_s"]["seen_in"] == 5          # seen_in counts videos, not gaps
    assert mind["retention"]["gap_s"]["n"] > 5
    assert mind["retention"]["devices_per_10s"]["n"] == 5
    assert mind["hooks"]["types"] == {"question": 5}
    assert mind["principles"][0]["principle"] == "contrast_for_clarity"
    assert mind["principles"][0]["seen_in"] == 5
    assert mind["trained_on"] == ["V01", "V02", "V03", "V04", "V05"]


def test_device_gaps():
    assert creator.device_gaps([2, 5], 10) == [2, 3, 5]
    assert creator.device_gaps([], 8.0) == [8.0]
    assert creator.device_gaps([], None) == []
    assert creator.device_gaps([3], None) == [3]


def test_story_consensus_order():
    orders = [["hook", "setup", "turn", "payoff", "cta"],
              ["hook", "setup", "turn", "payoff", "recap", "cta"],
              ["hook", "setup", "escalation", "turn", "payoff", "cta"],
              ["hook", "setup", "turn", "payoff", "cta", "recap"]]
    mind = creator.synth({f"V0{i}": make_doc(order=o) for i, o in enumerate(orders, 1)})
    st = mind["story_template"]
    assert st["order"] == ["hook", "setup", "turn", "payoff", "recap", "cta"]
    assert st["parts"]["escalation"]["in_template"] is False      # 1 of 4 videos = 25 %
    assert st["parts"]["recap"]["seen_in"] == 2
    assert st["most_common_exact"]["count"] == 1


def test_synth_handles_empty_and_short():
    doc = make_doc(order=[], devices=(), ideas=[], rules=())
    doc["decisions"] = []
    doc["toolchain"] = []
    doc["story"]["open_loops"] = []
    mind = creator.synth({"V01": doc})
    assert mind["retention"]["max_gap_s"]["n"] == 0
    assert mind["story_template"]["order"] == []
    assert mind["ideation"]["methods"] == {}
    assert mind["ideation"]["literal_share"] is None
    assert len(creator.checklist(mind)) >= 5


def test_synth_aggregates_avoidance_promises_loops_hooks():
    docs = {"V01": make_doc(rejected="No music under the confession."),
            "V02": make_doc(rejected="no music under the confession"),
            "V03": make_doc(rejected="a zoom on the punchline")}
    docs["V03"]["story"]["open_loops"].append({"opened_t": 2.0, "closed_t": None, "text": "a second question"})
    mind = creator.synth(docs)
    nd = {x["key"]: x for x in mind["never_do"]}
    assert nd["no music under the confession"]["seen_in"] == 2
    assert mind["audience_promise"][0]["key"] == "you will learn one trick"
    assert mind["audience_promise"][0]["seen_in"] == 3
    lo = mind["open_loops"]
    assert lo["per_video"]["median"] == 1.0 and lo["per_video"]["p90"] == pytest.approx(1.8)
    assert lo["open_to_close_s"]["median"] == pytest.approx(13.5)
    assert lo["closed_share"] == pytest.approx(0.75)
    assert mind["hooks"]["visual_devices"][0]["key"] == "object in hand"
    assert mind["hooks"]["visual_devices"][0]["seen_in"] == 3
    loop_q = [q for q in creator.checklist(mind) if "loop" in q.lower()][0]
    assert "13.5" in loop_q and "1.0" in loop_q


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


def test_toolchain_consensus():
    docs = {"V01": make_doc(tool=("broll", PIPE, 0.7)),
            "V02": make_doc(tool=("broll", PIPE, 0.7)),
            "V03": make_doc(tool=("broll", PIPE, 0.7)),
            "V04": make_doc(tool=("broll", "stock", 0.7))}
    b = creator.synth(docs)["toolchain"]["broll"]
    assert b["consensus"] == "ai_image+ai_i2v"
    assert b["confidence"] == pytest.approx(0.75)
    assert b["seen_in"] == 4
    assert {v["tool"]: v["videos"] for v in b["votes"]}["stock"] == 1


def test_toolchain_weighted_by_stated_confidence():
    docs = {"V01": make_doc(tool=("ui", "screen_capture", 0.9)),
            "V02": make_doc(tool=("ui", "html_motion", 0.2)),
            "V03": make_doc(tool=("ui", "html_motion", 0.2)),
            "V04": make_doc(tool=("ui", "html_motion", 0.2))}
    u = creator.synth(docs)["toolchain"]["ui"]
    assert u["consensus"] == "screen_capture"        # 0.9 outweighs three unsure votes (0.6)
    assert u["confidence"] == pytest.approx(0.6)


# ---------------------------------------------------------------- ideation

def _ideation_docs():
    def ideas(n):
        out = [
            _idea(1.0, f"ten times bigger {n}", "number", "scale_shift", "a coin grows taller than a house",
                  why="size makes the number felt", pattern="Object grows past the frame",
                  principle="scale_for_magnitude", emotion="awe"),
            _idea(4.0, f"half the cost {n}", "number", "scale_shift", "a receipt shrinks to a stamp",
                  why="size makes the number felt", pattern="object grows past the frame!",
                  principle="scale_for_magnitude", emotion="Relief"),
            _idea(8.0, f"before it was slow {n}", "contrast", "before_after", "split screen, old and new",
                  pattern="split screen", principle="contrast_for_clarity"),
            _idea(12.0, f"it just works {n}", "claim", "literal", "the app running on a phone",
                  principle="prove_with_the_real_thing"),
        ]
        if n % 2 == 0:
            out.append(_idea(15.0, f"forty percent {n}", "number", "data_viz", "a bar fills to 40 %",
                             principle="make_the_invisible_visible", emotion="awe"))
        return out
    docs = {f"V0{n}": make_doc(ideas=ideas(n), rules=["Turn every number into a size", "Cut on the breath"],
                               rejected="No music under the confession")
            for n in range(1, 5)}
    return docs


def test_ideation_playbook():
    mind = creator.synth(_ideation_docs())
    ide = mind["ideation"]
    num = ide["playbook"]["number"]
    assert num["n"] == 10 and num["seen_in"] == 4
    s, d = num["methods"][0], num["methods"][1]
    assert s["method"] == "scale_shift" and d["method"] == "data_viz"
    # per video: V01/V03 1.0 scale_shift, V02/V04 2/3 scale_shift + 1/3 data_viz, macro-averaged
    assert s["share"] == pytest.approx((1 + 2 / 3 + 1 + 2 / 3) / 4, abs=1e-4)
    assert d["share"] == pytest.approx((1 / 3 + 1 / 3) / 4, abs=1e-4)
    assert s["pooled_share"] == pytest.approx(0.8) and d["pooled_share"] == pytest.approx(0.2)
    assert s["seen_in"] == 4 and d["seen_in"] == 2
    assert "number" in num["summary"] and "scale_shift 83 %" in num["summary"] and "data_viz 17 %" in num["summary"]
    assert s["examples"][0]["idea"] and s["examples"][0]["line"] and s["examples"][0]["example_id"] == "V01@1"
    assert s["principles"] == {"scale_for_magnitude": 8}
    assert num["emotions"]["awe"] == pytest.approx(0.6) and num["emotions"]["relief"] == pytest.approx(0.4)

    assert ide["methods"]["scale_shift"]["count"] == 8
    assert ide["methods"]["scale_shift"]["seen_in"] == 4
    assert ide["literal_share"] == pytest.approx((0.25 + 0.2 + 0.25 + 0.2) / 4, abs=1e-4)
    assert ide["pooled_literal_share"] == pytest.approx(4 / 18, abs=1e-4)
    assert ide["figurative_share"] == pytest.approx(1 - ide["literal_share"], abs=1e-4)
    pats = {p["key"]: p for p in ide["patterns"]}
    assert pats["object grows past the frame"]["count"] == 8
    assert pats["object grows past the frame"]["seen_in"] == 4
    assert "the app running on a phone" not in pats          # no fallback to the idea text


def test_playbook_macro_average_resists_one_heavy_video():
    heavy = [_idea(float(i), f"n {i}", "number", "data_viz", f"bar {i}") for i in range(10)]
    docs = {"V01": make_doc(ideas=heavy)}
    for n in (2, 3, 4):
        docs[f"V0{n}"] = make_doc(ideas=[_idea(1.0, "x", "number", "scale_shift", "giant coin")])
    num = creator.synth(docs)["ideation"]["playbook"]["number"]
    assert num["top"] == "scale_shift"
    assert num["methods"][0]["share"] == pytest.approx(0.75)
    dv = [m for m in num["methods"] if m["method"] == "data_viz"][0]
    assert dv["pooled_share"] == pytest.approx(10 / 13, abs=1e-4) and dv["seen_in"] == 1


def test_ideate_brief():
    mind = creator.synth(_ideation_docs())
    brief = creator.ideate(mind, "the new tool is five times cheaper", "number")
    tops = [m["method"] for m in brief["top_methods"]]
    assert tops == ["scale_shift", "data_viz"]
    ex = brief["top_methods"][0]["examples"][0]
    assert ex["line"] and ex["idea"] and ex["example_id"]
    wild = brief["wild_card"]["method"]
    assert wild not in tops and wild not in ("literal", "other")
    assert wild in creator.FIT["number"]
    assert "3 ideas" in brief["rule"] and wild in brief["rule"]
    assert "size makes the number felt" in brief["why_patterns"]
    assert brief["craft_notes"]["scale_shift"]["motion"]
    assert brief["principles"][0]["principle"] == "scale_for_magnitude"
    assert brief["rules_of_thumb"][0]["rule"] == "Turn every number into a size"
    assert brief["rules_of_thumb"][0]["relevant"] is True
    assert brief["emotions"]["awe"] == pytest.approx(0.6)
    assert brief["patterns"][0]["pattern"] == "Object grows past the frame"
    assert brief["never_do"][0]["avoid"] == "No music under the confession"

    with pytest.raises(ValueError):
        creator.ideate(mind, "x", "banana")
    fb = creator.ideate(mind, "click the link below", "cta")
    assert fb["fallback"] == "overall" and fb["top_methods"]


def test_ideate_wild_card_fits_message_type():
    mind = creator.synth(_ideation_docs())
    for i in range(40):
        w = creator.ideate(mind, f"it felt like losing a friend {i}", "emotion")["wild_card"]["method"]
        assert w in creator.FIT["emotion"] and w != "data_viz"


def test_ideate_rotates_used_methods_and_examples():
    mind = creator.synth(_ideation_docs())
    first = creator.ideate(mind, "a line", "number")
    used = [{"method": "scale_shift", "example_id": e["example_id"]}
            for e in mind["ideation"]["playbook"]["number"]["methods"][0]["examples"]]
    b = creator.ideate(mind, "a line", "number", used=used)
    assert [m["method"] for m in b["top_methods"]] == ["data_viz", "scale_shift"]
    assert b["top_methods"][1]["examples"] == []
    assert any("ran out of reference examples for scale_shift" in n for n in b["rotation"]["notes"])
    assert first["top_methods"][0]["examples"]
    both = used + [{"method": "data_viz", "example_id": "V02@15"}]
    c = creator.ideate(mind, "a line", "number", used=both)
    assert any(n.startswith("ran out: every method") for n in c["rotation"]["notes"])


def test_ideate_cli(tmp_path):
    pack = write_pack(tmp_path, _ideation_docs())
    r = run_cli("creator.py", "synth", pack, "--all")
    assert r.returncode == 0, r.stdout + r.stderr
    mind = json.loads((pack / "creator-mind.json").read_text(encoding="utf-8"))
    assert mind["schema"] == "creator-mind/1"
    used = tmp_path / "used.json"
    used.write_text(json.dumps([{"method": "scale_shift", "example_id": "V01@1"}]), encoding="utf-8")
    r = run_cli("creator.py", "ideate", pack, "--line", "פי שלושה מהר יותר", "--type", "number", "--used", used)
    assert r.returncode == 0, r.stdout + r.stderr
    brief = json.loads(r.stdout)
    assert brief["line"] == "פי שלושה מהר יותר"
    assert brief["top_methods"][0]["method"] == "data_viz"
    r = run_cli("creator.py", "ideate", tmp_path / "nopack", "--line", "x", "--type", "number")
    assert r.returncode == 2


def test_synth_cli_refuses_bad_files(tmp_path):
    bad = make_doc()
    del bad["visual_ideas"][0]["message"]
    pack = write_pack(tmp_path, {"V01": make_doc(), "V02": bad})
    r = run_cli("creator.py", "synth", pack)
    assert r.returncode == 1
    assert not (pack / "creator-mind.json").exists()
    assert run_cli("creator.py", "synth", pack, "--train", "V01,V09").returncode == 2
    r = run_cli("creator.py", "synth", pack, "--train", "V01")
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads((pack / "creator-mind.json").read_text(encoding="utf-8"))["trained_on"] == ["V01"]


def test_synth_cli_dedupes_and_uses_measured_duration(tmp_path):
    pack = write_pack(tmp_path, {"V01": make_doc(), "V02": make_doc(), "V03": make_doc(rules=["other rule"])})
    for vid, sha in (("V01", "a" * 64), ("V02", "a" * 64), ("V03", "b" * 64)):
        (pack / "analysis" / vid / "measure.json").write_text(
            json.dumps({"video": {"sha256": sha, "duration_s": 30.0}}), encoding="utf-8")
    r = run_cli("creator.py", "synth", pack)
    assert r.returncode == 0, r.stdout + r.stderr
    mind = json.loads((pack / "creator-mind.json").read_text(encoding="utf-8"))
    assert mind["trained_on"] == ["V01", "V03"]
    assert mind["dupes"] == [["V01", "V02"]]
    assert mind["retention"]["max_gap_s"]["median"] == pytest.approx(13.0)   # last device 17 s, end 30 s


# ---------------------------------------------------------------- structure scoring

def test_score_structure():
    actual = [{"part": "hook", "t0": 0.0, "t1": 3.0}, {"part": "setup", "t0": 3.0, "t1": 10.0},
              {"part": "turn", "t0": 10.0, "t1": 15.0}, {"part": "payoff", "t0": 15.0, "t1": 22.0}]
    pred = [{"part": "hook", "t0": 0.0, "t1": 3.5},      # the opening boundary: not scored
            {"part": "setup", "t0": 3.5, "t1": 12.0},    # 0.5 off: match
            {"part": "turn", "t0": 12.0, "t1": 16.0},    # 2.0 off: no match
            {"part": "cta", "t0": 16.0, "t1": 22.0}]     # wrong part
    s = creator.score_structure(pred, actual)
    assert s["tp"] == 1 and s["n_pred"] == 3 and s["n_actual"] == 3
    assert s["precision"] == pytest.approx(1 / 3) and s["recall"] == pytest.approx(1 / 3)
    assert s["overall_f1"] == pytest.approx(1 / 3)
    assert s["timing_mae_s"] == pytest.approx(0.5)
    assert s["order_match"] is False and s["first_part_match"] is True
    assert creator.score_structure(pred, actual, tol_s=2.0)["tp"] == 2
    perfect = creator.score_structure(copy.deepcopy(actual), actual)
    assert perfect["overall_f1"] == pytest.approx(1.0) and perfect["order_match"] is True
    only_hook = creator.score_structure([{"part": "hook", "t0": 0.0, "t1": 22.0}], actual)
    assert only_hook["tp"] == 0 and only_hook["overall_f1"] == 0.0     # no free match for the start
    empty = creator.score_structure([], [])
    assert empty["overall_f1"] == 0.0 and empty["timing_mae_s"] is None


def test_retention_events_for_prove():
    evs = creator.retention_events([{"t": 1.0, "device": "reveal"}, {"t": "x"}])
    assert evs == [{"t": 1.0, "type": "reveal"}]


# ---------------------------------------------------------------- checklist

def test_checklist_non_empty_and_generic():
    docs = _ideation_docs()
    for d in docs.values():
        d["rules_of_thumb"] = ["Zebra stripes at every corner"]
        d["toolchain"][0]["how"] = "frobnicator steps"
        d["decisions"][0]["principle_note"] = "quokka note"
        d["decisions"][0]["alternative_rejected"] = "no walrus"
        d["story"]["audience_promise"] = "a platypus promise"
    mind = creator.synth(docs)
    items = creator.checklist(mind)
    assert len(items) >= 5
    assert all(isinstance(i, str) and i.strip() for i in items)
    text = " ".join(items).lower()
    for word in ("zebra", "frobnicator", "quokka", "walrus", "platypus", "coin", "receipt", "v01", "broll"):
        assert word not in text
    assert any("gap" in i.lower() for i in items)
    assert any("loop" in i.lower() for i in items)
    assert any("principle" in i.lower() for i in items)
    assert any("message" in i.lower() for i in items)
    assert str(mind["retention"]["never_more_than_s"]) in text
