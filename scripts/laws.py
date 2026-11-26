#!/usr/bin/env python3
"""Mine "when X then Y" regularities across the training videos of a pack.

Usage: laws.py <pack> [--train V01,V02 | --all] [--exclude V03,...] [--window 1.5]   write <pack>/laws.json

Events come from fingerprint.events(): word triggers (number, date, question), cuts, transitions, camera
moves, element entrances, big text, SFX by class and manual events. For every ordered pair of different
event types (X, Y): an occurrence of X is "followed" when a Y starts 0..window seconds after it.
  confidence      = occurrences of X followed by Y / occurrences of X
  support_videos  = videos with at least one such pair
  lag             = {median, p10, p90, n, seen_in} of Y - X over the matched pairs only
  chance          = how often Y would follow a random moment (Y count x window / video span, averaged)
  status          = "rule-candidate" when support_videos >= 3 and confidence >= 0.6, else "option"
  anchor          = "word" when X is a word trigger, else "visual" (lang.scale_timing relies on it)
Only pairs seen in at least 2 videos with confidence >= 0.3 are written. Near-duplicate videos count once
(fingerprint.dedupe). Word laws are skipped, and listed under not_run, when no training video has a
transcript or the transcripts are in different languages; the visual and sound laws still run.

Output: {schema "laws/1", trained_on, held_out, dupes, window_s, laws, not_run}.
Exit: 0 ok, 2 bad input.
"""
import argparse
import bisect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fingerprint as fp  # noqa: E402
import store  # noqa: E402

SCHEMA = "laws/1"
WINDOW_S = 1.5
MIN_SUPPORT = 3
MIN_CONFIDENCE = 0.6
LIST_MIN_VIDEOS = 2
LIST_MIN_CONFIDENCE = 0.3


def mine(streams, window_s=WINDOW_S):
    """Laws from {video id: events}. Law = {when, then, anchor, lag, support_videos, occurrences,
    followed, confidence, chance, status}."""
    occ, followed = {}, {}
    lags, vids, chance_acc = {}, {}, {}
    for vid, evs in streams.items():
        evs = sorted((e for e in evs if fp._num(e.get("t")) and e.get("type")), key=lambda e: e["t"])
        if not evs:
            continue
        times = [e["t"] for e in evs]
        span = max(times[-1] - times[0], window_s)
        counts = {}
        for e in evs:
            counts[e["type"]] = counts.get(e["type"], 0) + 1
        for k, e in enumerate(evs):
            x = e["type"]
            occ[x] = occ.get(x, 0) + 1
            seen = {}
            j = bisect.bisect_left(times, e["t"] - 1e-9)
            while j < len(evs) and times[j] <= e["t"] + window_s + 1e-9:
                if j != k and evs[j]["type"] != x and evs[j]["type"] not in seen:
                    seen[evs[j]["type"]] = times[j] - e["t"]
                j += 1
            for y, lag in seen.items():
                key = (x, y)
                followed[key] = followed.get(key, 0) + 1
                lags.setdefault(key, []).append(max(0.0, lag))
                vids.setdefault(key, set()).add(vid)
            for y, c in counts.items():
                if y != x:
                    chance_acc.setdefault((x, y), []).append(min(1.0, c * window_s / span))
    laws = []
    for (x, y), n_f in followed.items():
        sv = len(vids[(x, y)])
        conf = n_f / occ[x]
        if sv < LIST_MIN_VIDEOS or conf < LIST_MIN_CONFIDENCE:
            continue
        lag = fp.stat(lags[(x, y)])
        lag["seen_in"] = sv
        ch = chance_acc.get((x, y)) or [0.0]
        laws.append({"when": x, "then": y, "anchor": "word" if x.startswith("word:") else "visual",
                     "lag": lag, "support_videos": sv, "occurrences": occ[x], "followed": n_f,
                     "confidence": round(conf, 4), "chance": round(sum(ch) / len(ch), 4),
                     "status": "rule-candidate" if sv >= MIN_SUPPORT and conf >= MIN_CONFIDENCE - 1e-9
                     else "option"})
    laws.sort(key=lambda l: (l["status"] != "rule-candidate", -l["support_videos"], -l["confidence"],
                             l["when"], l["then"]))
    return laws


def build(pack, train=None, exclude=None, window_s=WINDOW_S):
    ctx = fp.prepare(pack, train, exclude)
    a = ctx["analyses"]
    streams = {i: fp.events(a[i], words=ctx["word_ok"]) for i in ctx["trained"]}
    return {"schema": SCHEMA, "trained_on": ctx["trained"], "held_out": ctx["held_out"], "dupes": ctx["dupes"],
            "window_s": window_s, "laws": mine(streams, window_s),
            "not_run": ctx["not_run"]}


def main(argv=None):
    store.print_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack")
    fp.add_selection_args(ap)
    ap.add_argument("--window", type=float, default=WINDOW_S, help="seconds Y may lag X (default 1.5)")
    a = ap.parse_args(argv)
    try:
        if not Path(a.pack).is_dir():
            raise fp.BadInput(f"pack folder not found: {Path(a.pack).name}")
        if not a.window > 0:
            raise fp.BadInput("--window must be positive")
        doc = build(a.pack, fp.parse_ids(a.train), fp.parse_ids(a.exclude), a.window)
        store.write_json(Path(a.pack) / "laws.json", doc)
        rc = sum(1 for l in doc["laws"] if l["status"] == "rule-candidate")
        print(f"laws.json: {len(doc['laws'])} laws ({rc} rule candidates) from {len(doc['trained_on'])} videos"
              + (f"; not run: {'; '.join(doc['not_run'])}" if doc["not_run"] else ""))
        return 0
    except (fp.BadInput, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(store.exit_on_lock_timeout(main))
