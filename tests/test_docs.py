import re
import subprocess
import sys

from conftest import REPO_ROOT

SKILL = REPO_ROOT / "SKILL.md"
README = REPO_ROOT / "README.md"


def _md_files():
    return [SKILL, README, REPO_ROOT / "CHANGELOG.md"] + sorted((REPO_ROOT / "references").glob("*.md"))


def test_skill_md_limits():
    s = SKILL.read_text(encoding="utf-8")
    assert len(s.splitlines()) < 500
    desc = re.search(r"^description: (.*)$", s, re.M).group(1)
    assert len(desc) <= 1024


def test_references_linked_both_ways():
    s = SKILL.read_text(encoding="utf-8")
    ref = s[s.index("## References"):]
    listed = set(re.findall(r"references/([a-z0-9-]+\.md)", ref))
    have = {p.name for p in (REPO_ROOT / "references").glob("*.md")}
    assert have <= listed, have - listed
    assert listed <= have, listed - have


def test_every_script_documented():
    docs = "".join(p.read_text(encoding="utf-8") for p in _md_files())
    missing = [p.name for p in (REPO_ROOT / "scripts").glob("*.py") if p.name not in docs]
    assert missing == []


def test_documented_scripts_exist():
    names = {p.name for p in REPO_ROOT.rglob("*.py") if ".git" not in p.parts}
    for f in _md_files():
        for n in set(re.findall(r"\b([a-z_][a-z0-9_]*\.py)\b", f.read_text(encoding="utf-8"))):
            assert n in names, f"{f.name} names {n}, which does not exist"


def test_readme_connections_block_current():
    text = README.read_text(encoding="utf-8")
    block = text.split("<!-- connections:start -->")[1].split("<!-- connections:end -->")[0].strip()
    r = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "doctor.py"), "stages", "--markdown"],
                       capture_output=True, text=True, encoding="utf-8")
    assert block == r.stdout.strip()


def test_changelog_has_version():
    v = (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    assert f"## [{v}]" in (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


def test_readme_flags_exist():
    text = README.read_text(encoding="utf-8")
    for script in ("install.py", "profile_fetch.py"):
        flags = set()
        for line in text.splitlines():
            if script in line:
                flags |= set(re.findall(r"(--[a-z][a-z-]+)", line))
        helptext = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / script), "--help"],
                                  capture_output=True, text=True, encoding="utf-8").stdout
        for sub in ("list", "page", "pick", "download") if script == "profile_fetch.py" else ():
            helptext += subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / script), sub, "--help"],
                                       capture_output=True, text=True, encoding="utf-8").stdout
        for f in flags:
            assert f in helptext, f"{script}: README shows {f}, --help does not"
