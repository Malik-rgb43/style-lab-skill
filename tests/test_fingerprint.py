import json
import shutil
import subprocess
from pathlib import Path

import pytest

import code_tokens
import fingerprint as fp


def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def measure(sha, dur=20.0, cuts=(), size=9.0):
    return {"schema": "measure/1",
            "video": {"id": sha[:12], "sha256": sha, "duration_s": dur, "fps": 30, "width": 1920, "height": 1080},
            "cuts": [{"t": t, "kind": "hard", "dur_f": 1} for t in cuts],
            "shots": [{"t0": 0, "t1": dur, "dur_s": dur, "palette": [{"hex": "#223344", "share": 1}],
                       "face_share": 0.5, "headroom": 0.1}],
            "cadence": {"repeat_fraction": 0, "unique_fps": 30, "on_twos": False},
            "text": {"size_pct_h": {"median": size, "p10": size, "p90": size, "n": 1}},
            "loudness": {"lufs": -16.0, "tp": -1.0}, "words_per_s": 2.5, "longest_hold_s": 2.0,
            "not_run": []}


def add_video(pack, vid, sha=None, words=None, lang="en", manual=None, motion=None, sound=None, **kw):
    d = Path(pack) / "analysis" / vid
    write(d / "measure.json", measure(sha or ("%064x" % (abs(hash(vid)) + 1)), **kw))
    if words is not None:
        write(d / "words.json", {"language": lang, "words": [{"w": w, "start": s, "end": s + 0.2}
                                                              for w, s in words]})
    if manual is not None:
        write(d / "events.manual.json", manual)
    if motion is not None:
        write(d / "motion.json", motion)
    if sound is not None:
        write(d / "sound.json", sound)
    return d


def test_stat_and_seen_in():
    s = fp.stat([1, 2, 3, None, float("nan")], 10)
    assert s["n"] == 3 and s["seen_in"] == 3 and s["median"] == 2 and s["share"] == 0.3
    empty = fp.stat([], 4)
    assert empty["n"] == 0 and empty["median"] is None and empty["share"] == 0.0
    # a trait in 5 of 20 videos is an option, in 6 of 20 a rule candidate
    assert fp.trait_status(5, 20) == "option"
    assert fp.trait_status(6, 20) == "rule-candidate"


def test_traits_share_and_status(tmp_path):
    for k in range(20):
        add_video(tmp_path, f"V{k:02d}", sha="%064x" % (k + 1),
                  manual=[{"t": 1.0, "type": "prop"}] if k < 5 else [])
    out = fp.build(tmp_path)
    t = {x["name"]: x for x in out["traits"]}["prop"]
    assert t["seen_in"] == 5 and t["share"] == 0.25 and t["status"] == "option"


def test_duplicates_by_sha_and_thumbs(tmp_path):
    for vid, sha in (("A", "a" * 64), ("B", "a" * 64), ("C", "c" * 64), ("D", "d" * 64)):
        add_video(tmp_path, vid, sha=sha)
    a = fp.load_all(tmp_path)
    kept, dupes = fp.dedupe(["A", "B", "C", "D"], a)
    assert kept == ["A", "C", "D"] and dupes == [("A", "B")]
    # a re-upload with another hash but near-identical thumbnails
    a["C"]["thumbs"] = ["ffff0000ffff0000"] * 10
    a["D"]["thumbs"] = ["ffff0000ffff0001"] * 10
    kept, dupes = fp.dedupe(["A", "C", "D"], a)
    assert kept == ["A", "C"] and dupes == [("C", "D")]
    a["D"]["thumbs"] = ["0000ffff0000ffff"] * 10
    assert fp.dedupe(["C", "D"], a)[1] == []


def test_layer_source_code_overrides(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "index.html").write_text(
        "<html><head><style>h1{font-size:96px}p{font-size:32px}</style></head>"
        "<body><h1>a</h1><p>b</p><script></script></body></html>", encoding="utf-8")
    pack = tmp_path / "pack"
    for k in range(2):
        add_video(pack, f"V{k}", sha="%064x" % (k + 1), size=9.0)
    tokens = code_tokens.read_project(proj)
    write(pack / "tokens.json", tokens)
    code_tokens.set_layer_source(pack, "type", "code", str(proj))
    out = fp.build(pack)
    assert out["layers"]["type"]["source"] == "code"
    assert out["layers"]["type"]["metrics"]["levels"] == tokens["type"]["levels"]
    assert "text_height_pct" not in out["layers"]["type"]["metrics"]
    assert out["layers"]["structure"]["source"] == "refs"


def test_code_layer_with_other_projects_tokens_is_refused(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "index.html").write_text("<html><body><p>x</p></body></html>", encoding="utf-8")
    pack = tmp_path / "pack"
    add_video(pack, "V0")
    write(pack / "tokens.json", {"schema": "tokens/1", "source": {"kind": "code", "path_hint": "other"},
                                 "type": {"levels": []}})
    code_tokens.set_layer_source(pack, "type", "code", str(proj))
    with pytest.raises(fp.BadInput):
        fp.build(pack)
    assert fp.main([str(pack)]) == 2


def test_held_out_not_trained(tmp_path):
    for k in range(4):
        add_video(tmp_path, f"V{k:02d}", sha="%064x" % (k + 1))
    out = fp.build(tmp_path, exclude=["V03"])
    assert "V03" in out["held_out"] and "V03" not in out["trained_on"]
    assert out["schema"] == "fingerprint/1"
    with pytest.raises(fp.BadInput):
        fp.build(tmp_path, exclude=["V99"])


def test_short_static_video_is_valid(tmp_path):
    add_video(tmp_path, "V0", dur=6.0)
    out = fp.build(tmp_path)
    m = out["layers"]["motion"]["metrics"]
    assert m["camera_moves_per_min"]["n"] == 0 and m["camera_moves_per_min"]["median"] is None
    assert out["layers"]["structure"]["metrics"]["cuts_per_min"]["n"] == 1
    json.dumps(out)
    assert fp.main([str(tmp_path)]) == 0
    assert (tmp_path / "fingerprint.json").exists()


def test_events_unified_stream():
    a = {"measure": {"cuts": [{"t": 1.0, "kind": "hard", "dur_f": 1}, {"t": 5.0, "kind": "gradual", "dur_f": 9}],
                     "text": {"boxes": [{"t": 2.0, "h_pct": 12}, {"t": 3.0, "h_pct": 4}]}},
         "motion": {"camera_moves": [{"t0": 2.5, "prop": "scale", "dur_f": 12, "ease_name": "expoOut"}],
                    "elements": [{"id": "e1", "t0": 3.5}, {"t0": "bad"}]},
         "sound": {"sfx": [{"t": 4.0, "class_guess": "whoosh"}]},
         "manual": [{"t": 6.0, "type": "newspaper_prop", "label": "paper"}],
         "words": {"language": "en", "words": [{"word": "in", "start": 7.0}, {"word": "2020", "start": 7.2}]}}
    ev = fp.events(a)
    types = [e["type"] for e in ev]
    assert types.count("big_text") == 1
    assert {"cut", "transition", "camera:scale", "entrance", "sfx:whoosh", "newspaper_prop",
            "word:number"} <= set(types)
    assert [e["t"] for e in ev] == sorted(e["t"] for e in ev)
    assert next(e for e in ev if e["type"] == "word:number")["attrs"]["lang"] == "en"
    assert "word:number" not in [e["type"] for e in fp.events(a, words=False)]
    assert fp.events({"measure": {}}) == []


def test_thumbs_for_and_cli(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg missing")
    v = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=10:duration=3",
                    "-pix_fmt", "yuv420p", str(v)], check=True)
    th = fp.thumbs_for(v)
    assert len(th) == 10 and all(len(h) == 16 for h in th)
    pack = tmp_path / "pack"
    d = add_video(pack, "V0")
    m = json.loads((d / "measure.json").read_text(encoding="utf-8"))
    m["source"] = str(v)
    write(d / "measure.json", m)
    add_video(pack, "V1")
    assert fp.main([str(pack), "--thumbs"]) == 0
    assert (d / "thumbs.json").exists()
    assert not (pack / "analysis" / "V1" / "thumbs.json").exists()


def test_motion_eases_and_metrics(tmp_path):
    mv = lambda n, b: {"t0": 1.0, "prop": "scale", "dur_f": 12, "peak_at": 0.3, "ease_name": n, "bezier": b}  # noqa: E731
    add_video(tmp_path, "A", sha="1" * 64, motion={"camera_moves": [mv("expoOut", [0.2, 0.9, 0.3, 1.0])],
                                                    "elements": [{"id": "e", "t0": 2, "moves": [mv("quadOut", [0.3, 0.7, 0.4, 1.0])]}]})
    add_video(tmp_path, "B", sha="2" * 64, motion={"camera_moves": [mv("expoOut", [0.4, 0.9, 0.3, 1.0])]})
    m = fp.build(tmp_path)["layers"]["motion"]
    e = {x["name"]: x for x in m["eases"]}
    assert e["expoOut"]["seen_in"] == 2 and e["expoOut"]["share"] == 1.0
    assert e["expoOut"]["bezier"] == [0.3, 0.9, 0.3, 1.0]
    assert e["quadOut"]["seen_in"] == 1
    assert m["metrics"]["dur_f"]["median"] == 12 and m["metrics"]["peak_at"]["n"] == 2


def test_low_entropy_thumbs_are_not_duplicates(tmp_path):
    add_video(tmp_path, "A", sha="a" * 64)
    add_video(tmp_path, "B", sha="b" * 64)
    a = fp.load_all(tmp_path)
    a["A"]["thumbs"] = ["0000000000000000"] * 10
    a["B"]["thumbs"] = ["0000000000000000"] * 10
    assert fp.dedupe(["A", "B"], a)[1] == []
    a["A"]["thumbs"] = [None, None] + ["ffff0000ffff0000"] * 8
    a["B"]["thumbs"] = [None, "ffff0000ffff0000"] + ["ffff0000ffff0000"] * 8
    assert fp.dedupe(["A", "B"], a)[1] == [("A", "B")]


def test_bad_manual_events_exit_2(tmp_path):
    d = add_video(tmp_path, "A", manual=[{"t": 1, "type": "x", "attrs": "oops"}])
    assert fp.main([str(tmp_path)]) == 2
    write(d / "events.manual.json", [{"t": "1", "type": "x"}])
    assert fp.main([str(tmp_path)]) == 2
    write(d / "events.manual.json", [{"t": 1, "type": "x", "attrs": {"a": 1}}])
    assert fp.main([str(tmp_path)]) == 0


def test_video_without_language_listed(tmp_path):
    add_video(tmp_path, "A", sha="1" * 64, words=[("5", 1.0)])
    add_video(tmp_path, "B", sha="2" * 64, words=[("5", 1.0)], lang="xx")
    out = fp.build(tmp_path)
    assert any(n.startswith("word_laws:B") for n in out["not_run"])
    assert not any(n.startswith("word_laws:A") for n in out["not_run"])


def test_manual_copy_of_a_measured_cut_is_dropped():
    """Found by the train-folder eval: manual 'cut' entries doubled the measured cuts and halved recall."""
    import fingerprint
    a = {"measure": {"cuts": [{"t": 2.0}]}, "manual": [{"t": 2.05, "type": "cut"}, {"t": 3.0, "type": "cut"},
                                                      {"t": 2.0, "type": "pose_change"}]}
    ev = [(e["t"], e["type"]) for e in fingerprint.events(a, words=False)]
    assert ev.count((2.0, "cut")) == 1 and (2.05, "cut") not in ev
    assert (3.0, "cut") in ev and (2.0, "pose_change") in ev
