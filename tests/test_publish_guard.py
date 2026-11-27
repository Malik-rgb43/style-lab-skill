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


def test_guard_scans_js_css_yaml_svg_toml(tmp_path):
    for ext in (".js", ".css", ".yml", ".yaml", ".svg", ".toml"):
        (tmp_path / f"a{ext}").write_text("see private-notes.md", encoding="utf-8")
    assert len(find_mentions(tmp_path)) == 6


def test_machine_paths_flagged(tmp_path):
    import publish_guard as pg
    (tmp_path / "a.md").write_text("see C:\\Users\\someone\\a.txt and /home/someone/a\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("run scripts/a.py, see https://example.com/x and C:/Program Files/Tool\n",
                                   encoding="utf-8")
    hits = pg.find_machine_paths(tmp_path)
    assert any(h.startswith("a.md") for h in hits) and not any(h.startswith("b.md") for h in hits)


def test_names_file_flagged(tmp_path):
    import publish_guard as pg
    names = tmp_path / "names.txt"
    names.write_text("# private\nSomeone Example\n", encoding="utf-8")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "x.md").write_text("made with someone example's help\n", encoding="utf-8")
    assert pg.find_names(root, pg.load_names(names)) == ["x.md: Someone Example"]


def test_repo_has_no_machine_paths():
    import publish_guard as pg
    from conftest import REPO_ROOT
    assert pg.find_machine_paths(REPO_ROOT) == []
