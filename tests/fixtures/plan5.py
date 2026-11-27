"""Synthetic Plan 5 world: a `paper-3d` style with a pack (laws, fingerprint, recipes), the `example-shop`
client with a palette, and a project with plan.json and edit_ready.words.json. Plan 4 code is never imported."""
import json
from pathlib import Path

import store

LAW = {"when": "word:number", "then": "counter_card", "anchor": "word", "confidence": 0.82,
       "support_videos": 5, "occurrences": 11, "lag": {"median": 0.3, "p10": 0.1, "p90": 0.6, "n": 9, "seen_in": 5},
       "status": "rule-candidate"}
LAW_OPT = {"when": "question", "then": "big_text", "anchor": "word", "confidence": 0.4, "support_videos": 2,
           "occurrences": 4, "lag": {"median": 0.2, "p10": 0.1, "p90": 0.4, "n": 3, "seen_in": 2},
           "status": "option"}

FINGERPRINT = {"schema": "fingerprint/1", "trained_on": ["V01", "V02", "V03"], "held_out": [],
               "layers": {"motion": {"source": "refs", "eases": [
                   {"name": "power2.out", "bezier": [0.215, 0.61, 0.355, 1], "share": 0.6, "seen_in": 3},
                   {"name": "power3.out", "bezier": [0.165, 0.84, 0.44, 1], "share": 0.3, "seen_in": 2}],
                   "metrics": {"dur_f": {"median": 14, "p10": 10, "p90": 20, "n": 3, "seen_in": 3},
                               "peak_at": {"median": 0.3, "p10": 0.2, "p90": 0.45, "n": 3, "seen_in": 3}}},
                   "look": {"source": "refs", "metrics": {"grain_std": {"median": 2.0, "p10": 1.5, "p90": 2.5,
                                                                        "n": 3, "seen_in": 3}}}}}

RECIPES = {"recipes": [
    {"kind": "paper-cutout", "how": "ai_image+i2v",
     "look": {"grain": "fine", "vignette": "soft", "blur": None, "palette_ref": None, "lens": "35mm",
              "light": "soft window light", "composition": None},
     "prompt_template": "{subject}, {look}, colours {palette}", "negative": "blurry",
     "motion": {"camera_ease": "M0,0 C0.165,0.84 0.44,1 1,1", "dur_s": {"median": 3.4, "p10": 2.8, "p90": 4.0},
                "subject_motion": "the paper pieces drift apart"},
     "tools": [{"capability": "image_generator", "level": "required", "fallback": "stock"}],
     "test": {"status": "passed", "sample_paths": [], "notes": ""}, "source_evidence": ["V01@2.0"]},
    {"kind": "era-scene", "how": "ai_video", "look": {}, "prompt_template": "{subject} in the {era}",
     "negative": "", "motion": {"camera_ease": "power2.inOut", "dur_s": 5, "subject_motion": "people walk"},
     "tools": [{"capability": "video_generator", "level": "required", "fallback": "stock"}],
     "test": {"status": "passed"}, "source_evidence": []},
    {"kind": "code-card", "how": "code", "look": {}, "prompt_template": "", "negative": "", "motion": {},
     "tools": [], "test": {"status": "untested"}, "source_evidence": []}]}

WORDS = [("יש", 0.0, 0.3), ("3", 0.3, 0.5), ("דרכים", 0.5, 1.0), ("להתחיל", 1.0, 1.9),
         ("ומה", 2.0, 2.4), ("עושים", 2.4, 2.9), ("עכשיו", 2.9, 3.4), ("?", 3.4, 3.5), ("בואו", 3.6, 5.9)]

BEATS = [{"n": 1, "start": 0.0, "end": 2.0, "script": "יש 3 דרכים להתחיל", "beat_types": ["list"]},
         {"n": 2, "start": 2.0, "end": 3.6, "script": "ומה עושים עכשיו ?", "beat_types": []},
         {"n": 3, "start": 3.6, "end": 6.0, "script": "בואו", "beat_types": []}]


def write_style(home, tmp_path, *, laws=(LAW, LAW_OPT), fingerprint=FINGERPRINT, recipes=RECIPES, kind="trained"):
    pack = Path(tmp_path) / "pack"
    pack.mkdir(parents=True, exist_ok=True)
    if laws is not None:
        store.write_json(pack / "laws.json", {"laws": list(laws)})
    if fingerprint is not None:
        store.write_json(pack / "fingerprint.json", fingerprint)
    if recipes is not None:
        store.write_json(pack / "recipes.json", recipes)
    Path(home).mkdir(parents=True, exist_ok=True)
    (Path(home) / "styles.json").write_text(json.dumps({"version": 2, "styles": [
        {"slug": "paper-3d", "name": "Paper 3D", "pack": str(pack), "aspect": "9:16", "kind": kind,
         "status": "ready"}]}), encoding="utf-8")
    return pack


def write_client(run_cli, name="example-shop", palette=True):
    r = run_cli("clients.py", "add", "--name", name, "--color", "#008754")
    assert r.returncode == 0, r.stderr
    if palette:
        r = run_cli("clients.py", "palette", name, "--pick", "accent")
        assert r.returncode == 0, r.stderr


def write_project(tmp_path, name="proj", client="example-shop", beats=BEATS, length=6.0):
    import project
    d = Path(tmp_path) / name
    project.init(d, "launch", "clip.mp4", "paper-3d", client)
    store.write_json(d / "plan.json", {"title": "launch", "lang": "he",
                                       "clip": {"file": "clip.mp4", "length_s": length},
                                       "beats": [dict(b) for b in beats]})
    store.write_json(d / "edit_ready.words.json", {"language": "he", "words": [
        {"word": w, "start": a, "end": b} for w, a, b in WORDS]})
    return d


def option(beat, k, *, source="invention", wild=False, concept=None, set_no=1, fit=0.5, tags=(), law=None,
           idea_id=None, recipe=None, category="B-roll", short_q=None):
    dq = {q: f"answer for {q} of option {k} on beat {beat}" for q in ("point", "real_thing", "invisible",
                                                                       "reference", "new")}
    if short_q:
        dq[short_q] = "short"
    return {"id": f"B{beat:02d}-p{k}", "key": "ABCDEFGHIJ"[(k - 1) % 10], "set": set_no,
            "title": f"concept {beat}.{k}", "category": category, "scene_type": "overlay",
            "desc": "d", "source": source, "law": law, "idea_id": idea_id, "recipe": recipe, "wild": wild,
            "concept_key": concept or f"concept-{beat}-{k}", "tags": list(tags), "fit": fit, "design_q": dq}


def good_set(beat, set_no=1, start=1, idea_id=None):
    """A valid set: a law option (beat 1 fires the number law), an idea or invention, a wild invention."""
    opts = []
    if beat == 1:
        opts.append(option(beat, start, source="law", law="number->counter_card", set_no=set_no, fit=0.6))
    else:
        opts.append(option(beat, start, set_no=set_no, fit=0.6))
    if idea_id:
        opts.append(option(beat, start + 1, source="idea", idea_id=idea_id, set_no=set_no, fit=0.5))
    else:
        opts.append(option(beat, start + 1, set_no=set_no, fit=0.5))
    opts.append(option(beat, start + 2, wild=True, set_no=set_no, fit=0.4))
    return opts


def set_options(d, by_beat):
    def go(plan):
        for b in plan["beats"]:
            if b["n"] in by_beat:
                b["options"] = by_beat[b["n"]]
    store.update_json(Path(d) / "plan.json", {"beats": []}, go)
