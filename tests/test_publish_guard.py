from conftest import REPO_ROOT
from publish_guard import find_mentions, find_private


def test_no_private_files_in_repo():
    assert find_private(REPO_ROOT) == []
    assert find_mentions(REPO_ROOT) == []


def test_guard_detects_private(tmp_path):
    (tmp_path / "private").mkdir()
    (tmp_path / "private" / "x.txt").write_text("x", encoding="utf-8")
    assert find_private(tmp_path) == ["private/"]


def test_guard_detects_private_mention(tmp_path):
    (tmp_path / "SKILL.md").write_text("see references/private-notes.md", encoding="utf-8")
    assert find_mentions(tmp_path) == ["SKILL.md: private-notes.md"]
