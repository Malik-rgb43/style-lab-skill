"""Taste log: every pick and rejection the owner makes, per client. Recommendations read `summary`.

Files: <store home>/taste/<client slug | _none>.jsonl, one JSON object per line.

Commands:
    taste.py log [--client C] [--style S] --step plan|frames|motion|palette|idea --picked ID
                 [--rejected ID ...] --project P [--tags picked=warm,flat rejected=3d,neon] [--note ..]
                 [--sig-picked SIG] [--sig-rejected SIG ...]
    taste.py summary [--client C] [--style S] [--last 50] [--json]

summary: prefers = tags picked >= 3 times and >= 2x as often as rejected; avoids = tags rejected
>= 2 times and never picked; rejected_twice = option ids rejected in 2+ projects.
"""
import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402

STEPS = ("plan", "frames", "motion", "palette", "idea")


class _Bad(Exception):
    pass


def parse_tags(pairs):
    out = {"picked": [], "rejected": []}
    for p in pairs or []:
        key, sep, vals = p.partition("=")
        if not sep or key not in out:
            raise _Bad(f"bad --tags item {p!r}: use picked=a,b or rejected=c,d")
        out[key] += [v.strip() for v in vals.split(",") if v.strip()]
    return out


def _slug(client):
    if client is None:
        return "_none"
    import clients
    c = clients.load_client(client)
    if not c:
        raise _Bad(f"unknown client: {client}")
    return c["slug"]


def _style_slug(style):
    if style is None:
        return None
    import styles
    s = styles.find(styles.load_registry(), style)
    if not s:
        raise _Bad(f"unknown style: {style}")
    return s["slug"]


def _file(slug):
    return store.path("taste", f"{slug}.jsonl")


def log(client, style, step, picked, rejected, project, note="", tags=None, sigs=None):
    if step not in STEPS:
        raise _Bad(f"bad step {step!r}: one of {', '.join(STEPS)}")
    slug = _slug(client)
    rec = {"date": dt.date.today().isoformat(), "client": None if client is None else slug,
           "style": _style_slug(style), "step": step, "project": project, "picked": picked,
           "rejected": list(rejected or []), "tags": tags or {"picked": [], "rejected": []},
           "note": note}
    if sigs is not None:
        rec["sigs"] = {"picked": sigs.get("picked"), "rejected": list(sigs.get("rejected") or [])}
    store.append_jsonl(_file(slug), rec)
    return rec


def _read(slug):
    p = _file(slug)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def used_sigs(client, step, exclude_project=None):
    """Option signatures the client already picked / rejected at `step` (lines without `sigs` are skipped)."""
    out = {"picked": set(), "rejected": set()}
    for r in _read(_slug(client)):
        if r.get("step") != step or (exclude_project is not None and r.get("project") == exclude_project):
            continue
        sg = r.get("sigs")
        if not isinstance(sg, dict):
            continue
        if sg.get("picked"):
            out["picked"].add(sg["picked"])
        out["rejected"].update(x for x in (sg.get("rejected") or []) if x)
    return out


def summary(client, style=None, last=50):
    if last < 0:
        raise _Bad(f"--last must be 0 or more (0 = all entries), got {last}")
    slug = _slug(client)
    st = _style_slug(style)
    rows = [r for r in _read(slug) if st is None or r.get("style") == st]
    rows = rows[-last:] if last else rows
    pk, rj, rej_projects = Counter(), Counter(), {}
    for r in rows:
        t = r.get("tags") or {}
        pk.update(t.get("picked", []))
        rj.update(t.get("rejected", []))
        for o in r.get("rejected", []):
            rej_projects.setdefault(o, set()).add(r.get("project"))
    return {"client": None if client is None else slug, "style": st, "entries": len(rows),
            "picked_tags": dict(pk), "rejected_tags": dict(rj),
            "prefers": sorted(t for t, n in pk.items() if n >= 3 and n >= 2 * rj.get(t, 0)),
            "avoids": sorted(t for t, n in rj.items() if n >= 2 and pk.get(t, 0) == 0),
            "rejected_twice": sorted(o for o, ps in rej_projects.items() if len(ps) >= 2)}


def main(argv=None):
    return store.exit_on_lock_timeout(_main, argv)


def _main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("log")
    a.add_argument("--client"); a.add_argument("--style")
    a.add_argument("--step", required=True, choices=STEPS); a.add_argument("--picked", required=True)
    a.add_argument("--rejected", action="append", default=[]); a.add_argument("--project", required=True)
    a.add_argument("--tags", nargs="*", default=[]); a.add_argument("--note", default="")
    a.add_argument("--sig-picked"); a.add_argument("--sig-rejected", action="append", default=[])
    a = sub.add_parser("summary")
    a.add_argument("--client"); a.add_argument("--style")
    a.add_argument("--last", type=int, default=50); a.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    store.print_utf8()
    try:
        if args.cmd == "log":
            rec = log(args.client, args.style, args.step, args.picked, args.rejected, args.project,
                      args.note, parse_tags(args.tags),
                      {"picked": args.sig_picked, "rejected": args.sig_rejected}
                      if args.sig_picked or args.sig_rejected else None)
            print(json.dumps(rec, ensure_ascii=False))
            return 0
        s = summary(args.client, args.style, args.last)
    except _Bad as e:
        print(str(e), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(s, ensure_ascii=False, indent=2))
    else:
        print(f"entries: {s['entries']}")
        for k in ("prefers", "avoids", "rejected_twice"):
            print(f"{k}: {', '.join(s[k]) or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
