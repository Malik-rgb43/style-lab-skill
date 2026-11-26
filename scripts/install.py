#!/usr/bin/env python3
"""Install or update the live style-lab skill from this repo, with a dated backup and a rollback.

    install.py [--target DIR] [--archive DIR] [--home DIR] [--seed FILE] [--prune] [--dry-run] [--src DIR]
    install.py --rollback BACKUP_DIR [--target DIR] [--archive DIR]

Defaults: target ~/.claude/skills/style-lab, archive ~/.claude/skills-archive, store home
STYLE_LAB_HOME (or the store default).

What it does, in order:
  1. validates everything first (source, repo SKILL.md, junctions/symlinks, seed); any failure is exit 2
     and nothing has been written;
  2. backs up the live skill to <archive>/style-lab-YYYYMMDD-HHMM;
  3. copies only the managed files (SKILL.md, VERSION, LICENSE, references/, scripts/, assets/),
     through a temp file and os.replace, skipping byte-identical files;
  4. keeps the machine-private files and the private section at the end of the live SKILL.md
     (re-appended after the update); stale files are listed and kept, removed only with --prune
     (they stay in the backup);
  5. verifies every managed file and every private file, seeds the store with what is missing,
     migrates a v1 registry, writes <target>/.install.json, prints the rollback command.

Never creates links, never follows them: a junction or symlink inside the target, the archive or on
their paths is refused. --dry-run prints the plan and writes nothing, not even a backup.
Exit codes: 0 ok, 1 check failed, 2 bad input, 3 missing tool.

Seed file {"schema": "style-lab-seed/1", ...} keys: lexicon {lang: {word: spoken}}, brand_md or
brand_md_file (relative to the seed file), sound_libraries [{name, path, kinds}], settings {key: value}.
Only what is missing is added; existing values are kept and reported.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import publish_guard  # noqa: E402
import store  # noqa: E402

PRIVATE_HEADING = "## Private extras"
MANAGED_TOP = ("references", "scripts", "assets")
MANAGED_ROOT_FILES = ("SKILL.md", "VERSION", "LICENSE")
SEED_SCHEMA = "style-lab-seed/1"
SEED_KEYS = {"schema", "lexicon", "brand_md", "brand_md_file", "sound_libraries", "settings"}


class InstallError(Exception):
    def __init__(self, msg, code=2, backup=None):
        super().__init__(msg)
        self.code = code
        self.backup = backup


# ---------- small helpers ----------

def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_junk(rel: str) -> bool:
    parts = rel.split("/")
    return "__pycache__" in parts or rel.endswith((".pyc", ".tmp", ".lock"))


def _is_private(rel: str) -> bool:
    for p in publish_guard.PRIVATE:
        if p.endswith("/"):
            if rel.startswith(p):
                return True
        elif rel == p:
            return True
    return False


def _is_reparse(p: Path) -> bool:
    try:
        if p.is_symlink():
            return True
        isj = getattr(p, "is_junction", None)
        if isj is not None and isj():
            return True
        st = os.lstat(p)
    except OSError:
        return False
    return bool(getattr(st, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def find_reparse_points(root: Path) -> list[str]:
    """Symlinks and Windows junctions at or below root (POSIX relative paths, '.' for root itself).
    Never descends into one."""
    root = Path(root)
    found = []
    if not os.path.lexists(root):
        return found
    if _is_reparse(root):
        return ["."]
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError:
            continue
        for e in entries:
            p = Path(e.path)
            if _is_reparse(p):
                found.append(p.relative_to(root).as_posix())
            elif e.is_dir(follow_symlinks=False):
                stack.append(p)
    return sorted(found)


def _reparse_on_path(p: Path) -> list[str]:
    """Existing ancestors (and p itself) that are links."""
    p = Path(os.path.abspath(p))
    bad = []
    for q in [p, *p.parents]:
        if q == q.parent:
            continue
        if os.path.lexists(q) and _is_reparse(q):
            bad.append(str(q))
    return bad


def _files_under(root: Path) -> list[str]:
    out = []
    root = Path(root)
    for dp, dns, fns in os.walk(root):
        for fn in fns:
            out.append((Path(dp) / fn).relative_to(root).as_posix())
    return sorted(out)


def _atomic_write(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), prefix=dest.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        store._retry_perm(lambda: os.replace(tmp, dest))
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


# ---------- managed files, SKILL.md merge ----------

def managed_files(src: Path) -> list[str]:
    """POSIX relative paths the installer owns: SKILL.md, VERSION, LICENSE (when present) and every
    file under references/, scripts/, assets/ (no caches, temp or lock files, no private files)."""
    src = Path(src)
    out = []
    for name in MANAGED_ROOT_FILES:
        if (src / name).is_file():
            out.append(name)
    for top in MANAGED_TOP:
        base = src / top
        if not base.is_dir():
            continue
        for rel in _files_under(base):
            full = f"{top}/{rel}"
            if not _is_junk(full) and not _is_private(full):
                out.append(full)
    return sorted(out)


def split_private(md: str) -> tuple[str, str]:
    """(text before the private section, the section). The section runs from the first line starting
    with PRIVATE_HEADING to the end, line endings normalised to \\n; ("...", "") when there is none."""
    norm = md.replace("\r\n", "\n").replace("\r", "\n")
    lines = norm.split("\n")
    for i, line in enumerate(lines):
        if line.startswith(PRIVATE_HEADING):
            return "\n".join(lines[:i]), "\n".join(lines[i:])
    return md, ""


def merge_skill_md(repo_md: str, live_md: str) -> str:
    """The repo SKILL.md with the live file's private section re-appended at the end."""
    private = split_private(live_md)[1]
    if not private:
        return repo_md
    return split_private(repo_md)[0].rstrip("\n") + "\n\n" + private.rstrip("\n") + "\n"


def _read_text(p: Path) -> str:
    return p.read_bytes().decode("utf-8")


# ---------- backup / rollback ----------

def _tree_hashes(root: Path) -> dict:
    return {rel: _sha(Path(root) / rel) for rel in _files_under(root)}


def backup(target: Path, archive: Path) -> Path:
    target, archive = Path(target), Path(archive)
    archive.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    dest = archive / f"style-lab-{stamp}"
    n = 2
    while os.path.lexists(dest):
        dest = archive / f"style-lab-{stamp}-{n}"
        n += 1
    shutil.copytree(target, dest, symlinks=False)
    if _tree_hashes(target) != _tree_hashes(dest):
        raise InstallError(f"backup verification failed: {dest}", code=1)
    return dest


def rollback_command(backup: Path, target: Path) -> str:
    return f'python "{target}/scripts/install.py" --rollback "{backup}" --target "{target}"'


def _check_no_links(target: Path, archive: Path, extra=()):
    """Refuse links on the paths of target/archive/extra, inside target/extra, and directly in archive."""
    bad = []
    for p in (target, archive, *extra):
        bad += _reparse_on_path(p)
    for root in (target, *extra):
        bad += [f"{root}/{x}" for x in find_reparse_points(root) if x != "."]
    if os.path.isdir(archive):
        try:
            bad += [e.path for e in os.scandir(archive) if _is_reparse(Path(e.path))]
        except OSError:
            pass
    if bad:
        raise InstallError("refusing junctions/symlinks (nothing was changed): " + "; ".join(sorted(set(bad))))


def _remove_empty_dirs(root: Path) -> None:
    for dp, dns, fns in os.walk(root, topdown=False):
        p = Path(dp)
        if p != root and not os.listdir(p) and not _is_reparse(p):
            try:
                os.rmdir(p)
            except OSError:
                pass


def rollback(backup_dir: Path, target: Path, archive: Path) -> Path:
    """Make target equal to backup_dir. The replaced state is backed up first (returned)."""
    backup_dir, target, archive = Path(backup_dir), Path(target), Path(archive)
    if not backup_dir.is_dir() or not (backup_dir / "SKILL.md").is_file():
        raise InstallError(f"not a style-lab backup (no SKILL.md): {backup_dir}")
    _check_no_links(target, archive, extra=(backup_dir,))
    bak = backup(target, archive) if target.exists() else None
    keep = set(_files_under(backup_dir))
    if target.exists():
        for rel in _files_under(target):
            if rel not in keep:
                (target / rel).unlink()  # a copy is in `bak`
    for rel in sorted(keep):
        _atomic_write(target / rel, (backup_dir / rel).read_bytes())
    if target.exists():
        _remove_empty_dirs(target)
    if _tree_hashes(target) != _tree_hashes(backup_dir):
        raise InstallError("rollback verification failed", code=1, backup=bak)
    return bak


# ---------- install ----------

def _plan_file(src: Path, target: Path, rel: str, live_md):
    """(new bytes, changed?) for one managed file."""
    if rel == "SKILL.md":
        repo_md = _read_text(src / rel)
        data = (merge_skill_md(repo_md, live_md) if live_md else repo_md).encode("utf-8")
    else:
        data = (src / rel).read_bytes()
    dest = target / rel
    changed = not (dest.is_file() and dest.read_bytes() == data)
    return data, changed


def install(src, target, *, archive, prune=False, dry_run=False) -> dict:
    src, target, archive = Path(src), Path(target), Path(archive)
    # 1. validate everything first
    if not (src / "SKILL.md").is_file() or not (src / "scripts" / "store.py").is_file():
        raise InstallError(f"source is not a style-lab repo (needs SKILL.md and scripts/store.py): {src}")
    repo_md = _read_text(src / "SKILL.md")
    if split_private(repo_md)[1]:
        raise InstallError("the repo SKILL.md contains the private-extras heading; it must stay machine-local")
    _check_no_links(target, archive)
    files = managed_files(src)
    live_md = _read_text(target / "SKILL.md") if (target / "SKILL.md").is_file() else ""
    private_kept = split_private(live_md)[1] != ""

    plan = {rel: _plan_file(src, target, rel, live_md) for rel in files}
    copied = [r for r in files if plan[r][1]]
    unchanged = [r for r in files if not plan[r][1]]
    stale = []
    for top in MANAGED_TOP:
        base = target / top
        if base.is_dir():
            for rel in _files_under(base):
                full = f"{top}/{rel}"
                if full not in plan and not _is_junk(full) and not _is_private(full):
                    stale.append(full)
    stale.sort()
    out = {"backup": None, "copied": copied, "unchanged": unchanged, "stale": stale,
           "pruned": [], "private_kept": private_kept}
    if dry_run:
        return out

    # 2. backup
    bak = backup(target, archive) if target.exists() else None
    out["backup"] = bak
    private_before = {r: _sha(target / r) for r in _files_under(target) if _is_private(r)} if target.exists() else {}

    try:
        # 3. copy
        for rel in copied:
            _atomic_write(target / rel, plan[rel][0])
        # 4. prune (only stale, only with --prune; copies are in the backup)
        if prune:
            for rel in stale:
                (target / rel).unlink()
                out["pruned"].append(rel)
            for top in MANAGED_TOP:
                if (target / top).is_dir():
                    _remove_empty_dirs(target / top)
        # 5. verify
        for rel in files:
            if (target / rel).read_bytes() != plan[rel][0]:
                raise InstallError(f"verification failed: {rel}", code=1, backup=bak)
        for rel, h in private_before.items():
            if not (target / rel).is_file() or _sha(target / rel) != h:
                raise InstallError(f"private file changed: {rel}", code=1, backup=bak)
    except InstallError:
        raise
    except OSError as e:
        raise InstallError(f"install failed: {e}", code=1, backup=bak)
    return out


# ---------- seed ----------

def load_seed(seed: Path) -> dict:
    """Parse and validate a seed file; InstallError(2) on any problem."""
    seed = Path(seed)
    try:
        d = json.loads(seed.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise InstallError(f"seed unreadable: {seed} ({e})")

    def bad(msg):
        raise InstallError(f"seed invalid: {msg}")

    if not isinstance(d, dict):
        bad("top level must be an object")
    if d.get("schema") != SEED_SCHEMA:
        bad(f'"schema" must be "{SEED_SCHEMA}"')
    extra = set(d) - SEED_KEYS
    if extra:
        bad("unknown key(s): " + ", ".join(sorted(extra)))
    lex = d.get("lexicon", {})
    if not isinstance(lex, dict) or not all(
            isinstance(k, str) and isinstance(v, dict)
            and all(isinstance(w, str) and isinstance(s, str) for w, s in v.items())
            for k, v in lex.items()):
        bad('"lexicon" must be {lang: {word: spoken}} with text values')
    if "brand_md" in d and "brand_md_file" in d:
        bad('use "brand_md" or "brand_md_file", not both')
    if "brand_md" in d and not isinstance(d["brand_md"], str):
        bad('"brand_md" must be text')
    if "brand_md_file" in d:
        if not isinstance(d["brand_md_file"], str):
            bad('"brand_md_file" must be a path (text)')
        f = seed.parent / d["brand_md_file"]
        if not f.is_file():
            bad(f'"brand_md_file" not found: {f}')
    libs = d.get("sound_libraries", [])
    if not isinstance(libs, list):
        bad('"sound_libraries" must be a list')
    for x in libs:
        if not (isinstance(x, dict) and isinstance(x.get("name"), str) and x["name"]
                and isinstance(x.get("path"), str) and x["path"]
                and isinstance(x.get("kinds", ["sfx"]), list)
                and all(isinstance(k, str) for k in x.get("kinds", ["sfx"]))
                and set(x) <= {"name", "path", "kinds"}):
            bad('each sound library needs text "name" and "path" and an optional "kinds" list')
    st = d.get("settings", {})
    if not isinstance(st, dict):
        bad('"settings" must be an object')
    if "sound_libraries" in st:
        bad('put sound libraries under "sound_libraries", not "settings"')
    return d


def apply_seed(seed: Path, home: Path) -> dict:
    """Add what is missing to the store. Returns {added: [...], kept: [...]}."""
    seed, home = Path(seed), Path(home)
    d = load_seed(seed)
    added, kept = [], []

    for lang, words in d.get("lexicon", {}).items():
        def fn(data, lang=lang, words=words):
            for w, spoken in words.items():
                k = w.lower()
                if k not in data:
                    data[k] = spoken
                    added.append(f"lexicon:{lang}:{k}")
                elif data[k] != spoken:
                    kept.append(f"lexicon:{lang}:{k}")
        store.update_json(home / "lexicon" / f"{lang}.json", {}, fn)

    text = d.get("brand_md")
    if "brand_md_file" in d:
        text = (seed.parent / d["brand_md_file"]).read_text(encoding="utf-8")
    if text is not None:
        brand = home / "brand.md"
        if brand.exists():
            kept.append("brand.md")
        else:
            _atomic_write(brand, text.encode("utf-8"))
            added.append("brand.md")

    libs, st = d.get("sound_libraries", []), d.get("settings", {})
    if libs or st:
        def fn_set(data):
            have = data.setdefault("sound_libraries", [])
            if not isinstance(have, list):
                have = data["sound_libraries"] = []
            names = {x.get("name") for x in have if isinstance(x, dict)}
            for x in libs:
                if x["name"] in names:
                    kept.append(f"sound:{x['name']}")
                else:
                    have.append({"name": x["name"], "path": x["path"], "kinds": x.get("kinds", ["sfx"])})
                    names.add(x["name"])
                    added.append(f"sound:{x['name']}")
            for k, v in st.items():
                if k in data:
                    if data[k] != v:
                        kept.append(f"settings:{k}")
                else:
                    data[k] = v
                    added.append(f"settings:{k}")
        store.update_json(home / "settings.json", {}, fn_set)
    return {"added": added, "kept": kept}


# ---------- migrate, record ----------

def _registry_version(home: Path):
    p = home / "styles.json"
    if not p.is_file():
        return None
    try:
        v = json.loads(p.read_text(encoding="utf-8")).get("version", 1)
    except (OSError, ValueError, AttributeError):
        return None
    return v if isinstance(v, int) else 1


def migrate_registry(target: Path, home: Path) -> str:
    """Run the installed styles.py migrate against `home`. Returns a one-line note."""
    v = _registry_version(home)
    if v is None or v >= 2:
        return ""
    env = dict(os.environ)
    env["STYLE_LAB_HOME"] = str(home)
    env.pop("STYLE_LAB_REGISTRY", None)
    r = subprocess.run([sys.executable, str(target / "scripts" / "styles.py"), "migrate"],
                       env=env, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise InstallError("registry migration failed: " + (r.stderr or r.stdout).strip(), code=1)
    return r.stdout.strip()


def _commit(src: Path):
    try:
        r = subprocess.run(["git", "-C", str(src), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, encoding="utf-8", timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() or None if r.returncode == 0 else None


def _version(src: Path):
    v = src / "VERSION"
    return v.read_text(encoding="utf-8").strip() if v.is_file() else None


# ---------- CLI ----------

def main(argv=None) -> int:
    store.print_utf8()
    ap = argparse.ArgumentParser(description="install, update or roll back the live style-lab skill")
    ap.add_argument("--target")
    ap.add_argument("--archive")
    ap.add_argument("--home")
    ap.add_argument("--seed")
    ap.add_argument("--src", help="source repo (default: the repo this script is in)")
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rollback", metavar="BACKUP_DIR")
    a = ap.parse_args(argv)

    claude = Path.home() / ".claude"
    target = Path(os.path.abspath(a.target)) if a.target else claude / "skills" / "style-lab"
    archive = Path(os.path.abspath(a.archive)) if a.archive else claude / "skills-archive"
    src = Path(os.path.abspath(a.src)) if a.src else Path(__file__).resolve().parent.parent
    bak = None
    try:
        if a.rollback:
            bak = rollback(Path(os.path.abspath(a.rollback)), target, archive)
            print(f"rolled back {target} from {a.rollback}")
            if bak:
                print(f"replaced state saved to {bak}")
                print("to undo this rollback: " + rollback_command(bak, target))
            return 0

        seed = Path(os.path.abspath(a.seed)) if a.seed else None
        if seed:
            load_seed(seed)  # validate before anything is written
        home = Path(os.path.abspath(a.home)) if a.home else None
        if home is None:
            env = os.environ.get("STYLE_LAB_HOME")
            home = Path(os.path.abspath(os.path.expanduser(env))) if env else claude / "style-lab"

        res = install(src, target, archive=archive, prune=a.prune, dry_run=a.dry_run)
        bak = res["backup"]
        tag = "[dry-run] " if a.dry_run else ""
        print(f"{tag}target: {target}")
        print(f"{tag}would copy {len(res['copied'])}, unchanged {len(res['unchanged'])}" if a.dry_run
              else f"copied {len(res['copied'])}, unchanged {len(res['unchanged'])}")
        for r in res["copied"]:
            print(f"  update {r}")
        if res["stale"]:
            verb = "pruned" if res["pruned"] else ("would prune" if a.prune and a.dry_run else "kept (use --prune)")
            print(f"stale files {verb}:")
            for r in res["stale"]:
                print(f"  {r}")
        if res["private_kept"]:
            print("private extras section at the end of SKILL.md: kept")
        if a.dry_run:
            if seed:
                print(f"{tag}seed {seed} would be applied (missing values only)")
            v = _registry_version(home)
            if v is not None and v < 2:
                print(f"{tag}registry v{v} would be migrated to v2")
            print(f"{tag}no backup written, nothing changed")
            return 0

        if seed:
            s = apply_seed(seed, home)
            print(f"seed: added {len(s['added'])}, kept {len(s['kept'])}")
            for x in s["added"]:
                print(f"  added {x}")
            for x in s["kept"]:
                print(f"  kept {x}")
        note = migrate_registry(target, home)
        if note:
            print(note)
        store.write_json(target / ".install.json", {
            "version": _version(src), "commit": _commit(src),
            "installed": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "backup": str(bak) if bak else None})
        if bak:
            print(f"backup: {bak}")
            print("rollback: " + rollback_command(bak, target))
        else:
            print("fresh install, no previous version to back up")
        return 0
    except InstallError as e:
        print(f"error: {e}", file=sys.stderr)
        b = e.backup or bak
        if b:
            print(f"backup: {b}", file=sys.stderr)
            print("rollback: " + rollback_command(b, target), file=sys.stderr)
        return e.code
    except store.LockTimeout:
        print(f"error: {store.LOCK_BUSY}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
