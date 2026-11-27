import json
import os
from pathlib import Path

import pytest

import install
import publish_guard
from conftest import REPO_ROOT

FIXTURES = REPO_ROOT / "tests" / "fixtures"
PRIV_NOTES = publish_guard.PRIVATE[0]            # a private file inside a managed folder
PRIV_DIR_FILE = publish_guard.PRIVATE[3] + "tpl/x.html"   # a private folder outside the managed ones
OLD_PRIVATE = install.PRIVATE_HEADING + " (this machine only)\n- local note\n"


@pytest.fixture
def paths():
    base = Path.home() / ".claude"
    return base / "skills" / "style-lab", base / "skills-archive"


def tree(*roots):
    out = {}
    for r in roots:
        r = Path(r)
        if r.is_dir():
            for f in sorted(r.rglob("*")):
                if f.is_file():
                    out[str(f.relative_to(r.parent))] = f.read_bytes()
    return out


def run(target, archive, *extra):
    return install.main(["--target", str(target), "--archive", str(archive), *map(str, extra)])


def make_live(target, skill="# old skill\n\nold body\n" + "\n" + OLD_PRIVATE):
    (target / "references").mkdir(parents=True)
    (target / "scripts").mkdir()
    (target / "SKILL.md").write_bytes(skill.encode("utf-8"))
    (target / PRIV_NOTES).write_bytes(b"secret notes\n")
    (target / PRIV_DIR_FILE).parent.mkdir(parents=True)
    (target / PRIV_DIR_FILE).write_bytes(b"<p>private</p>\n")
    (target / "scripts" / "store.py").write_bytes(b"# old store\n")


def archive_dirs(archive):
    if not Path(archive).is_dir():
        return []
    return sorted(p for p in Path(archive).glob("style-lab-*") if not p.name.startswith("style-lab-store-"))


def store_snaps(archive):
    return sorted(Path(archive).glob("style-lab-store-*")) if Path(archive).is_dir() else []


def test_managed_files_selection(tmp_path):
    for rel in ("SKILL.md", "VERSION", "README.md", "CHANGELOG.md", "references/a.md", "scripts/x.py",
                "scripts/__pycache__/x.pyc", "scripts/y.tmp", "scripts/y.lock", "assets/b.json",
                "tests/t.py", "evals/e.json", ".git/config", PRIV_NOTES):
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x", encoding="utf-8")
    assert install.managed_files(tmp_path) == [
        "SKILL.md", "VERSION", "assets/b.json", "references/a.md", "scripts/x.py"]


def test_fresh_install(paths, capsys):
    target, archive = paths
    assert run(target, archive) == 0
    assert archive_dirs(archive) == []
    for rel in install.managed_files(REPO_ROOT):
        assert (target / rel).is_file(), rel
    assert (target / "SKILL.md").read_bytes() == (REPO_ROOT / "SKILL.md").read_bytes()
    rec = json.loads((target / ".install.json").read_text(encoding="utf-8"))
    assert rec["version"] == "2.0.0-rc.1"
    assert rec["backup"] is None
    assert set(rec) == {"version", "commit", "installed", "backup"}
    assert not (target / "tests").exists()


def test_update_backs_up_and_keeps_private(paths, capsys):
    target, archive = paths
    make_live(target)
    old_skill = (target / "SKILL.md").read_bytes()
    private_before = {p: (target / p).read_bytes() for p in (PRIV_NOTES, PRIV_DIR_FILE)}
    assert run(target, archive) == 0
    (bak,) = archive_dirs(archive)
    assert (bak / "SKILL.md").read_bytes() == old_skill
    for p, b in private_before.items():
        assert (target / p).read_bytes() == b
        assert (bak / p).read_bytes() == b
    live = (target / "SKILL.md").read_bytes().decode("utf-8").replace("\r\n", "\n")
    repo = (REPO_ROOT / "SKILL.md").read_bytes().decode("utf-8").replace("\r\n", "\n")
    assert live.startswith(repo.rstrip("\n"))
    assert live.count(install.PRIVATE_HEADING) == 1
    assert live.endswith(OLD_PRIVATE)
    out = capsys.readouterr().out
    assert "rollback:" in out and str(bak) in out


@pytest.mark.parametrize("eol", ["\n", "\r\n"])
def test_second_install_no_duplicate_section(paths, eol):
    target, archive = paths
    skill = ("# old\n\n" + OLD_PRIVATE + "\n\n\n").replace("\n", eol)
    make_live(target, skill)
    assert run(target, archive) == 0
    first = (target / "SKILL.md").read_bytes()
    assert run(target, archive) == 0
    live = (target / "SKILL.md").read_bytes()
    assert live == first
    text = live.decode("utf-8").replace("\r\n", "\n")     # the file keeps the repo's own endings
    assert text.count(install.PRIVATE_HEADING) == 1
    assert text.endswith(OLD_PRIVATE)


def test_split_and_merge_units():
    assert install.split_private("a\nb\n") == ("a\nb\n", "")
    before, priv = install.split_private("x\r\n" + install.PRIVATE_HEADING + "\r\n- a\r\n\r\n")
    assert before == "x" and priv == install.PRIVATE_HEADING + "\n- a\n\n"
    assert install.merge_skill_md("repo\n", "live\n") == "repo\n"
    assert install.merge_skill_md("repo\n\n", "l\n" + install.PRIVATE_HEADING + "\n- k\n\n") == \
        "repo\n\n" + install.PRIVATE_HEADING + "\n- k\n"


def test_stale_kept_unless_prune(paths, capsys):
    target, archive = paths
    make_live(target)
    old = target / "scripts" / "old_tool.py"
    old.write_bytes(b"# stale\n")
    assert run(target, archive) == 0
    assert old.exists()
    assert "scripts/old_tool.py" in capsys.readouterr().out
    assert run(target, archive, "--prune") == 0
    assert not old.exists()
    assert (target / PRIV_NOTES).read_bytes() == b"secret notes\n"
    assert any((b / "scripts" / "old_tool.py").exists() for b in archive_dirs(archive))


def test_dry_run_changes_nothing(paths, capsys):
    target, archive = paths
    make_live(target)
    before = tree(target, archive)
    assert run(target, archive, "--dry-run") == 0
    assert tree(target, archive) == before
    assert not archive.exists()
    assert "dry-run" in capsys.readouterr().out


def _make_link(link: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "sentinel.txt").write_text("keep", encoding="utf-8")
    if os.name == "nt":
        try:
            import _winapi
            _winapi.CreateJunction(str(dest), str(link))
            return
        except Exception:
            pass
    try:
        os.symlink(dest, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create a junction or symlink here")


@pytest.mark.parametrize("where", ["target", "archive"])
def test_refuses_reparse_point(paths, tmp_path, capsys, where):
    target, archive = paths
    make_live(target)
    archive.mkdir(parents=True)
    dest = tmp_path / "elsewhere"
    link = (target / "references" / "lnk") if where == "target" else (archive / "style-lab-20200101-0000")
    _make_link(link, dest)
    try:
        before = tree(target, archive)
        assert run(target, archive) == 2
        assert tree(target, archive) == before
        assert (dest / "sentinel.txt").read_text(encoding="utf-8") == "keep"
        assert "junctions/symlinks" in capsys.readouterr().err
        assert install.find_reparse_points(link.parent) != [] or where == "archive"
    finally:
        try:
            os.rmdir(link)       # removes the link itself, never what it points to
        except OSError:
            try:
                os.unlink(link)
            except OSError:
                pass


def test_hebrew_target_path(tmp_path):
    target = tmp_path / "כלים שלי" / "style-lab"
    archive = tmp_path / "ארכיון"
    make_live(target)
    assert run(target, archive) == 0
    assert (target / "SKILL.md").read_bytes().decode("utf-8").replace("\r\n", "\n").endswith(OLD_PRIVATE)
    assert len(archive_dirs(archive)) == 1
    assert (target / ".install.json").is_file()


def test_seed_lexicon_brand_sound(paths, tmp_path):
    target, archive = paths
    home = Path(os.environ["STYLE_LAB_HOME"])
    lex = home / "lexicon" / "he.json"
    lex.parent.mkdir(parents=True)
    lex.write_text(json.dumps({"example": "mine"}, ensure_ascii=False), encoding="utf-8")
    seed = FIXTURES / "seed_example.json"

    res = install.apply_seed(seed, home)
    assert "lexicon:he:sample" in res["added"] and "lexicon:he:example" in res["kept"]
    assert json.loads(lex.read_text(encoding="utf-8")) == {"example": "mine", "sample": "סמפל"}
    assert "brand.md" in res["added"] and "sound:example-sfx" in res["added"]

    (home / "brand.md").write_text("edited by hand\n", encoding="utf-8")
    res2 = install.apply_seed(seed, home)
    assert res2["added"] == [] and "brand.md" in res2["kept"]
    assert (home / "brand.md").read_text(encoding="utf-8") == "edited by hand\n"
    libs = json.loads((home / "settings.json").read_text(encoding="utf-8"))["sound_libraries"]
    assert [x["name"] for x in libs] == ["example-sfx"]

    assert run(target, archive, "--seed", seed) == 0       # through the CLI too
    assert (home / "brand.md").read_text(encoding="utf-8") == "edited by hand\n"


def test_seed_brand_file_relative(tmp_path):
    (tmp_path / "b.md").write_text("# from file\n", encoding="utf-8")
    seed = tmp_path / "s.json"
    seed.write_text(json.dumps({"schema": install.SEED_SCHEMA, "brand_md_file": "b.md"}), encoding="utf-8")
    home = tmp_path / "h"
    assert install.apply_seed(seed, home)["added"] == ["brand.md"]
    assert (home / "brand.md").read_text(encoding="utf-8") == "# from file\n"


@pytest.mark.parametrize("bad", [
    {"schema": install.SEED_SCHEMA, "bogus": 1},
    {"schema": install.SEED_SCHEMA, "lexicon": {"he": {"w": 3}}},
    {"schema": install.SEED_SCHEMA, "sound_libraries": [{"name": "x"}]},
    {"schema": install.SEED_SCHEMA, "settings": []},
    {"schema": "other/1"},
])
def test_seed_invalid_exit2_before_any_change(paths, tmp_path, bad, capsys):
    target, archive = paths
    make_live(target)
    seed = tmp_path / "bad.json"
    seed.write_text(json.dumps(bad), encoding="utf-8")
    before = tree(target, archive)
    assert run(target, archive, "--seed", seed) == 2
    assert tree(target, archive) == before
    assert archive_dirs(archive) == []
    assert not (target / "scripts" / "install.py").exists()
    assert "seed invalid" in capsys.readouterr().err


def test_migrates_v1_registry(paths):
    target, archive = paths
    home = Path(os.environ["STYLE_LAB_HOME"])
    home.mkdir(parents=True)
    (home / "styles.json").write_bytes((FIXTURES / "styles_v1.json").read_bytes())
    assert run(target, archive) == 0
    assert json.loads((home / "styles.json").read_text(encoding="utf-8"))["version"] == 2
    assert list(home.glob("styles.json.bak-*"))


def test_rollback_restores_keeps_extras_and_undo_works(paths, capsys):
    target, archive = paths
    make_live(target)
    pre = tree(target)
    assert run(target, archive) == 0
    out = capsys.readouterr().out
    (bak,) = archive_dirs(archive)
    cmd = install.rollback_command(bak, target)
    assert cmd in out and f'"{bak}"' in cmd and f'"{target}"' in cmd
    assert "NOT rolled back" in out
    after_install = tree(target)

    assert install.main(["--rollback", str(bak), "--target", str(target), "--archive", str(archive)]) == 0
    rout = capsys.readouterr().out
    assert "NOT rolled back" in rout and "store snapshot" in rout
    assert "left in place (not in the backup" in rout
    now = tree(target)
    for rel, data in pre.items():                       # everything from the backup is back
        assert now[rel] == data
    assert (target / "scripts" / "install.py").exists()  # an extra from the install stays
    assert len(archive_dirs(archive)) == 2
    replaced = [b for b in archive_dirs(archive) if b != bak][0]
    copy = replaced / install.INSTALLER_DIR / "install.py"
    assert copy.is_file() and f'"{copy}"' in rout        # the undo command points at a script that exists
    assert "undo this rollback" in rout

    # undo: restoring the replaced state brings the installed files back, without the helper copy
    assert install.main(["--rollback", str(replaced), "--target", str(target), "--archive", str(archive)]) == 0
    now = tree(target)
    for rel, data in after_install.items():
        if not rel.endswith(".install.json"):
            assert now[rel] == data
    assert not (target / install.INSTALLER_DIR).exists()

    assert install.main(["--rollback", str(archive / "nope"), "--target", str(target),
                         "--archive", str(archive)]) == 2


def test_rollback_prune_removes_extras_after_backup(paths, capsys):
    target, archive = paths
    make_live(target)
    assert run(target, archive) == 0
    (bak,) = archive_dirs(archive)
    capsys.readouterr()
    assert install.main(["--rollback", str(bak), "--target", str(target), "--archive", str(archive),
                         "--prune"]) == 0
    assert not (target / "scripts" / "install.py").exists()
    replaced = [b for b in archive_dirs(archive) if b != bak][0]
    assert (replaced / "scripts" / "install.py").exists()


def test_install_prints_store_snapshot_before_seed(paths, capsys):
    target, archive = paths
    home = Path(os.environ["STYLE_LAB_HOME"])
    (home / "settings.json").parent.mkdir(parents=True)
    (home / "settings.json").write_text('{"keep": 1}', encoding="utf-8")
    assert run(target, archive, "--seed", FIXTURES / "seed_example.json") == 0
    out = capsys.readouterr().out
    (snap,) = store_snaps(archive)
    assert str(snap) in out and "NOT rolled back" in out
    assert json.loads((snap / "settings.json").read_text(encoding="utf-8")) == {"keep": 1}
    assert snap.parent == archive                         # a sibling of the skill backups


def test_nothing_to_change_skips_backup(paths, capsys):
    target, archive = paths
    assert run(target, archive) == 0
    rec = (target / ".install.json").read_bytes()
    assert run(target, archive) == 0
    assert archive_dirs(archive) == []
    assert (target / ".install.json").read_bytes() == rec
    assert "nothing to change" in capsys.readouterr().out


def test_repo_endings_kept_with_private_section(paths, tmp_path):
    src = tmp_path / "src"
    (src / "scripts").mkdir(parents=True)
    (src / "scripts" / "store.py").write_text("", encoding="utf-8")
    (src / "SKILL.md").write_bytes(b"# repo\r\n\r\nbody\r\n")
    target, archive = paths
    make_live(target)
    assert install.main(["--src", str(src), "--target", str(target), "--archive", str(archive)]) == 0
    live = (target / "SKILL.md").read_bytes()
    assert live.startswith(b"# repo\r\n\r\nbody\r\n\r\n" + install.PRIVATE_HEADING.encode())
    assert b"\n" not in live.replace(b"\r\n", b"")


def test_bad_utf8_skill_exit2_no_traceback(paths, capsys):
    target, archive = paths
    make_live(target)
    (target / "SKILL.md").write_bytes(b"\xff\xfe\x00bad")
    before = tree(target, archive)
    assert run(target, archive) == 2
    assert tree(target, archive) == before
    assert "Traceback" not in capsys.readouterr().err


def test_oserror_exits_1_one_line(paths, monkeypatch, capsys):
    target, archive = paths
    make_live(target)

    def boom(*a, **k):
        raise PermissionError("denied")
    monkeypatch.setattr(install.shutil, "copytree", boom)
    assert run(target, archive) == 1
    err = capsys.readouterr().err
    assert "denied" in err and "Traceback" not in err


def test_seed_lang_validated(tmp_path):
    seed = tmp_path / "s.json"
    seed.write_text(json.dumps({"schema": install.SEED_SCHEMA, "lexicon": {"../x": {"a": "b"}}}),
                    encoding="utf-8")
    with pytest.raises(install.InstallError):
        install.load_seed(seed)
    seed.write_text(json.dumps({"schema": install.SEED_SCHEMA, "lexicon": {"he": {"Word": "b"}}}),
                    encoding="utf-8")
    home = tmp_path / "h"
    install.apply_seed(seed, home)
    assert json.loads((home / "lexicon" / "he.json").read_text(encoding="utf-8")) == {"word": "b"}


def _mini_src(tmp_path):
    src = tmp_path / "src"
    (src / "scripts").mkdir(parents=True)
    (src / "references").mkdir()
    (src / "scripts" / "store.py").write_text("", encoding="utf-8")
    (src / "references" / "a.md").write_text("a", encoding="utf-8")
    (src / "SKILL.md").write_text("# repo\n", encoding="utf-8")
    return src


def test_source_reparse_point_refused_detector(paths, tmp_path, monkeypatch, capsys):
    src = _mini_src(tmp_path)
    target, archive = paths
    real = install._is_reparse
    monkeypatch.setattr(install, "_is_reparse", lambda p: Path(p).name == "a.md" or real(p))
    assert install.main(["--src", str(src), "--target", str(target), "--archive", str(archive)]) == 2
    err = capsys.readouterr().err
    assert "source tree" in err and "a.md" in err
    assert not target.exists()


def test_source_real_junction_refused(paths, tmp_path, capsys):
    src = _mini_src(tmp_path)
    target, archive = paths
    link = src / "references" / "lnk"
    _make_link(link, tmp_path / "elsewhere")
    try:
        assert install.main(["--src", str(src), "--target", str(target), "--archive", str(archive)]) == 2
        assert "lnk" in capsys.readouterr().err
        assert not target.exists()
        assert not any(f.startswith("references/lnk") for f in install.managed_files(src))
    finally:
        try:
            os.rmdir(link)
        except OSError:
            os.unlink(link)


def test_repo_skill_with_private_heading_refused(paths, tmp_path):
    src = tmp_path / "src"
    (src / "scripts").mkdir(parents=True)
    (src / "scripts" / "store.py").write_text("", encoding="utf-8")
    (src / "SKILL.md").write_text("x\n" + install.PRIVATE_HEADING + "\n", encoding="utf-8")
    target, archive = paths
    assert install.main(["--src", str(src), "--target", str(target), "--archive", str(archive)]) == 2
    assert not target.exists()


def test_publish_guard_skips_only_install(tmp_path):
    assert "scripts/install.py" in publish_guard.SKIP
    (tmp_path / "README.md").write_text(install.PRIVATE_HEADING + "\n", encoding="utf-8")
    hits = publish_guard.find_mentions(tmp_path)
    assert len(hits) == 1 and hits[0].startswith("README.md: ")
