import ast
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
MODULES = ["project", "ideas", "plan", "spend", "genprep", "imgmetrics", "lookcheck", "hf", "frames",
           "motion", "picks", "pickpage", "buildcheck"]
BANNED = {"requests", "urllib.request", "http.client", "httpx", "socket", "aiohttp"}


def _imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            yield mod
            for a in node.names:
                yield f"{mod}.{a.name}"


def _is_banned(name):
    return any(name == b or name.startswith(b + ".") for b in BANNED)


def test_no_paid_calls_guard():
    checked = 0
    for m in MODULES:
        f = SCRIPTS / f"{m}.py"
        if not f.exists():
            continue
        checked += 1
        bad = [n for n in _imports(ast.parse(f.read_text(encoding="utf-8"))) if _is_banned(n)]
        assert not bad, f"{m}.py imports a network client: {bad}"
    assert checked >= 1  # spend.py itself


def test_spend_exists_and_is_checked():
    assert (SCRIPTS / "spend.py").exists()
