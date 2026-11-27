#!/usr/bin/env python3
"""Rebuild the eval fixtures (generic content only): ready-style/, en-style/, notes-project/,
profiles/creator30.json. conversation-a0.md is hand-written. Run from anywhere; writes next to this file."""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tests" / "fixtures"))
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
import plan5  # noqa: E402


def w(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, str):
        p.write_text(obj, encoding="utf-8")
    else:
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


RULES = """# Example Clean (סגנון לדוגמה)

Generic eval style: clean talking-head edits with calm boxes and big numbers.

## Rules
1. Cut on the word, never in the middle of a word; pauses up to 0.3 s.
2. Captions: centred, two lines at most, the key word in the accent colour.
3. Boxes enter with power3.out over 14 frames (0.56 s at 25 fps) and leave in 8 frames.
4. A number in the speech gets a counter card within 0.3 s of the word.
5. One idea on screen at a time; the previous graphic leaves before the next enters.
6. Background: deep dark with one accent colour; no more than 3 colours in a frame.
7. B-roll is full screen, never a floating card.
8. Camera: one slow push-in per beat, never a zoom jump to hide a cut.

## Review round
"""


def pack(dst, lang):
    if dst.exists():
        shutil.rmtree(dst)
    fp = json.loads(json.dumps(plan5.FINGERPRINT))
    fp.update({"lang": lang, "trained_on": ["V01", "V02", "V03", "V04", "V05", "V06"], "held_out": ["V07"]})
    if lang == "en":
        fp["elements"] = ["progress_bar", "slide_in", "numbers", "logo", "reading_order", "clock"]
        fp["caption"] = {"emphasis": ["italic", "weight"]}
    w(dst / "fingerprint.json", fp)
    w(dst / "laws.json", {"laws": [plan5.LAW, plan5.LAW_OPT]})
    w(dst / "recipes.json", {"recipes": [r for r in plan5.RECIPES["recipes"] if r["how"] == "code"]})
    w(dst / "layers.json", {"layers": {"motion": {"source": "refs"}, "look": {"source": "refs"},
                                       "type": {"source": "refs"}, "sound": {"source": "refs"}}})
    w(dst / "proof.json", {"recreation": {"passed": True, "score": 0.86, "clips": ["V02"]},
                           "holdout": {"passed": True, "score": 0.81, "clips": ["V07"]}, "checked": "2026-11-27"})
    w(dst / "board.json", {"name": "Example Clean", "lang": lang, "verdict": "approved",
                           "colors": ["#0b0910", "#7c5cff", "#f4f1fb"], "fonts": [{"name": "Heebo", "role": "main"}]})
    w(dst / "library" / "index.json", {"items": [
        {"id": "L1", "kind": "counter_card", "file": "library/counter.html"},
        {"id": "L2", "kind": "caption", "file": "library/caption.html"},
        {"id": "L3", "kind": "box_in", "file": "library/box.html"}]})
    w(dst / "STYLE.md", RULES if lang == "he" else RULES.replace("(סגנון לדוגמה)", "(English)"))


def notes_project(dst):
    if dst.exists():
        shutil.rmtree(dst)
    w(dst / "plan.json", {"title": "example", "lang": "he", "clip": {"file": "draft.mp4", "length_s": 6.0},
                          "beats": plan5.BEATS})
    w(dst / "edit_ready.words.json", {"language": "he", "words": [
        {"word": a, "start": b, "end": c} for a, b, c in plan5.WORDS]})
    w(dst / "README.md", "Draft round 1 of a generic example video. The render is draft.mp4 (made at prepare).\n")


def creator30(dst):
    entries = []
    for i in range(30):
        views = 1000 + ((i * 7919) % 97) * 1000
        entries.append({"id": f"73000000000000001{i:02d}", "url": f"https://www.tiktok.com/@example.creator/video/73000000000000001{i:02d}",
                        "title": f"post {i + 1}", "view_count": views, "like_count": views // 20,
                        "duration": 20 + i % 40, "upload_date": f"202610{(i % 28) + 1:02d}",
                        "thumbnails": [{"url": f"https://example.com/t/{i}.jpg"}]})
    w(dst, {"id": "example.creator", "uploader": "example.creator", "_type": "playlist", "entries": entries})


if __name__ == "__main__":
    pack(HERE / "ready-style", "he")
    pack(HERE / "en-style", "en")
    notes_project(HERE / "notes-project")
    creator30(HERE / "profiles" / "creator30.json")
    print("ok")
