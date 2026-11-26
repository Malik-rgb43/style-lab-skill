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
    assert law["confidence"] == pytest.approx(0.7778, abs=0.001)
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
