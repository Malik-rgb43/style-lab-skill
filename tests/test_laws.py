import json

import pytest

import laws
from test_fingerprint import add_video


def build_pack(tmp_path, per_video, followed, lang="en", with_words=True):
    """per_video: number-word times of each video; followed: {(video index, time)} followed by a card."""
    for k, times in enumerate(per_video):
        words = [("5", t) for t in times] if with_words else None
        manual = [{"t": t + 0.18, "type": "counter_card"} for t in times if (k, t) in followed]
        add_video(tmp_path, f"V{k}", sha="%064x" % (k + 1), words=words, lang=lang, manual=manual,
                  cuts=(2.0, 9.0))
    return tmp_path


FULL = [[1.0, 5.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]       # 9 occurrences in 8 videos
FOLLOWED = {(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0)}   # 7 of 9


def test_law_support_and_confidence(tmp_path):
    build_pack(tmp_path, FULL, FOLLOWED)
    doc = laws.build(tmp_path)
    law = next(l for l in doc["laws"] if l["when"] == "word:number" and l["then"] == "counter_card")
    assert law["support_videos"] == 7 and law["occurrences"] == 9
    assert law["pooled_confidence"] == pytest.approx(0.7778, abs=0.001)
    assert law["confidence"] == pytest.approx(0.8125, abs=0.001)   # macro over the 8 videos
    assert law["lift"] >= laws.LIFT_MIN
    assert law["status"] == "rule-candidate" and law["anchor"] == "word"
    assert law["lag"]["median"] == pytest.approx(0.18, abs=0.001)
    assert law["lag"]["seen_in"] == 7


def test_law_below_support_is_option(tmp_path):
    build_pack(tmp_path, [[1.0]] * 3, {(0, 1.0), (1, 1.0)})
    doc = laws.build(tmp_path)
    law = next(l for l in doc["laws"] if l["when"] == "word:number" and l["then"] == "counter_card")
    assert law["support_videos"] == 2 and law["status"] == "option"


def test_duplicates_do_not_double_support(tmp_path):
    # three real videos, the third also copied under a second id
    for vid, sha in (("A", "1" * 64), ("B", "2" * 64), ("C", "3" * 64), ("C2", "3" * 64)):
        add_video(tmp_path, vid, sha=sha, words=[("5", 1.0)], manual=[{"t": 1.18, "type": "counter_card"}])
    doc = laws.build(tmp_path)
    assert doc["dupes"] == [["C", "C2"]] and doc["trained_on"] == ["A", "B", "C"]
    law = next(l for l in doc["laws"] if l["when"] == "word:number")
    assert law["support_videos"] == 3 and law["occurrences"] == 3
    # without the dedupe the support would be 4: two real videos plus the copy is not a rule
    two = laws.build(tmp_path, exclude=["A"])
    law2 = next(l for l in two["laws"] if l["when"] == "word:number")
    assert law2["support_videos"] == 2 and law2["status"] == "option"


def test_no_words_skips_word_laws(tmp_path):
    build_pack(tmp_path, [[1.0]] * 3, set(), with_words=False)
    for k in range(3):   # a visual law: a flash 0.18 s after every cut in 3 videos
        (tmp_path / "analysis" / f"V{k}" / "events.manual.json").write_text(
            json.dumps([{"t": 2.18, "type": "flash"}, {"t": 9.18, "type": "flash"}]), encoding="utf-8")
    doc = laws.build(tmp_path)
    assert any(n.startswith("word_laws") for n in doc["not_run"])
    assert not any(l["anchor"] == "word" for l in doc["laws"])
    vis = next(l for l in doc["laws"] if l["when"] == "cut" and l["then"] == "flash")
    assert vis["anchor"] == "visual" and vis["status"] == "rule-candidate"


def test_mixed_languages_skip_word_laws(tmp_path):
    build_pack(tmp_path, FULL, FOLLOWED)
    add_video(tmp_path, "V9", sha="9" * 64, words=[("5", 1.0)], lang="he")
    doc = laws.build(tmp_path)
    assert any("mixed languages" in n for n in doc["not_run"])
    assert not any(l["anchor"] == "word" for l in doc["laws"])


def test_language_without_profile_skips_word_laws(tmp_path):
    build_pack(tmp_path, FULL, FOLLOWED, lang="xx")
    doc = laws.build(tmp_path)
    assert any(n.startswith("word_laws") for n in doc["not_run"])
    assert not any(l["anchor"] == "word" for l in doc["laws"])


def test_held_out_not_mined_and_cli(tmp_path):
    build_pack(tmp_path, FULL, FOLLOWED)
    assert laws.main([str(tmp_path), "--exclude", "V0"]) == 0
    doc = json.loads((tmp_path / "laws.json").read_text(encoding="utf-8"))
    assert doc["held_out"] == ["V0"] and "V0" not in doc["trained_on"]
    law = next(l for l in doc["laws"] if l["when"] == "word:number" and l["then"] == "counter_card")
    assert law["occurrences"] == 7
    assert laws.main([str(tmp_path / "nope")]) == 2
    assert laws.main([str(tmp_path), "--train", "V99"]) == 2


def test_mine_empty_and_single_type():
    assert laws.mine({}) == []
    assert laws.mine({"a": [{"t": 1, "kind": "visual", "type": "cut", "attrs": {}}]}) == []


def ev(t, typ, src=None):
    return {"t": t, "kind": "visual", "type": typ, "src": src or f"{typ}{t}", "attrs": {}}


def test_one_to_one_matching():
    streams = {v: [ev(1.0, "entrance"), ev(1.2, "entrance"), ev(1.4, "entrance"), ev(1.5, "whoosh")]
               for v in ("a", "b")}
    law = next(l for l in laws.mine(streams) if l["when"] == "entrance" and l["then"] == "whoosh")
    assert law["confidence"] == pytest.approx(1 / 3, abs=0.001) and law["followed"] == 2


def test_frequent_y_is_not_a_rule(tmp_path):
    for k in range(3):
        add_video(tmp_path, f"V{k}", sha="%064x" % (k + 1), words=[("5", 1.0)],
                  cuts=tuple(i * 0.5 for i in range(1, 40)))
    law = next(l for l in laws.build(tmp_path)["laws"] if l["when"] == "word:number" and l["then"] == "cut")
    assert law["confidence"] == 1.0 and law["lift"] < laws.LIFT_MIN and law["status"] == "option"


def test_same_source_makes_no_law(tmp_path):
    for k in range(3):
        add_video(tmp_path, f"V{k}", sha="%064x" % (k + 1),
                  motion={"elements": [{"id": "e1", "t0": 1.0, "h_pct": 12}, {"id": "e2", "t0": 4.0, "h_pct": 12}]})
    doc = laws.build(tmp_path)
    assert not any({l["when"], l["then"]} == {"entrance", "big_text"} and l["lag"]["median"] == 0
                   for l in doc["laws"])
    assert not any(l["when"] == "big_text" and l["then"] == "entrance" for l in doc["laws"])


def test_simultaneous_events_one_direction_only():
    streams = {v: [ev(1.0, "aaa"), ev(1.0, "bbb")] for v in ("a", "b", "c")}
    ls = laws.mine(streams)
    assert [(l["when"], l["then"]) for l in ls] == [("aaa", "bbb")]


def test_per_video_support_40_vs_1_vs_1():
    big = [e for k in range(40) for e in (ev(k * 3.0, "X"), ev(k * 3.0 + 0.2, "Y"))]
    small = lambda: [ev(0.0, "X"), ev(0.2, "Y"), ev(5.0, "X"), ev(9.0, "X")] + [ev(30.0, "Z")]  # noqa: E731
    ls = laws.mine({"a": big, "b": small(), "c": small()}, spans={"a": 130, "b": 30, "c": 30})
    law = next(l for l in ls if l["when"] == "X" and l["then"] == "Y")
    assert law["pooled_confidence"] > 0.8 and law["confidence"] < 0.6
    assert law["status"] == "option"


def test_text_boxes_only_where_no_element(tmp_path):
    from fingerprint import events
    a = {"measure": {"text": {"boxes": [{"t": 1.0, "box": [0, 0, 1, 1], "h_pct": 12},
                                         {"t": 1.2, "box": [0, 0, 1, 1], "h_pct": 12},
                                         {"t": 5.0, "box": [0, 0, 1, 1], "h_pct": 12}]}},
         "motion": {"elements": [{"id": "e", "t0": 0.9, "h_pct": 12}]}}
    big = [e for e in events(a) if e["type"] == "big_text"]
    assert [e["src"].split(":")[0] for e in big] == ["el", "box"] and big[1]["t"] == 5.0


def test_unmeasured_layer_is_skipped_not_zero(tmp_path):
    for k in range(5):
        snd = {"sfx": [{"t": 2.3, "class_guess": "whoosh"}]} if k < 3 else None
        add_video(tmp_path, f"V{k}", sha="%064x" % (k + 1), cuts=(2.0,), sound=snd)
    doc = laws.build(tmp_path)
    law = next(l for l in doc["laws"] if l["when"] == "cut" and l["then"] == "sfx:whoosh")
    assert law["confidence"] == 1.0 and law["support_videos"] == 3 and law["status"] == "rule-candidate"


def test_co_occurrence_is_not_a_law():
    c = lambda t, typ, layer: {"t": t, "type": typ, "src": f"{typ}{t}", "layer": layer}  # noqa: E731
    streams = {v: [c(1.0, "cut", "cut"), c(1.01, "camera:scale", "motion")] for v in ("a", "b")}
    ls, co = laws._mine(streams)
    assert ls == [] and co == [{"a": "camera:scale", "b": "cut", "occurrences": 2, "videos": 2}]


def test_zero_duration_uses_stream_extent():
    streams = {v: [ev(1.0, "X"), ev(1.2, "Y")] for v in ("a", "b")}
    ls = laws.mine(streams, spans={"a": 0, "b": -3})
    assert ls and 0 < ls[0]["chance"] <= 1


def test_box_merge_by_position():
    from fingerprint import events
    boxes = [{"t": 1.0, "box": [0.1, 0.1, 0.3, 0.1], "h_pct": 10},
             {"t": 1.5, "box": [0.12, 0.11, 0.3, 0.1], "h_pct": 10},      # same bucket, merged
             {"t": 1.8, "box": [0.1, 0.6, 0.3, 0.1], "h_pct": 10}]       # other place: its own event
    big = [e for e in events({"measure": {"text": {"boxes": boxes}}}) if e["type"] == "big_text"]
    assert [e["t"] for e in big] == [1.0, 1.8]


def test_lag0_word_sfx_is_a_law_cut_camera_is_co_occurrence():
    c = lambda t, typ, layer: {"t": t, "type": typ, "src": f"{typ}{t}", "layer": layer}  # noqa: E731
    streams = {v: [c(1.0, "word:number", "word"), c(1.0, "sfx:hit", "sound"),
                   c(4.0, "cut", "cut"), c(4.0, "camera:scale", "motion")] for v in ("a", "b", "c")}
    ls, co = laws._mine(streams)
    pairs = {(l["when"], l["then"]): l for l in ls}
    assert pairs[("word:number", "sfx:hit")]["lag"]["median"] == 0
    assert not any({l["when"], l["then"]} == {"cut", "camera:scale"} for l in ls)
    assert co == [{"a": "camera:scale", "b": "cut", "occurrences": 3, "videos": 3}]
