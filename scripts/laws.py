#!/usr/bin/env python3
"""Mine "when X then Y" regularities across the training videos of a pack.

Usage: laws.py <pack> [--train V01,V02 | --all] [--exclude V03,...] [--window 1.5]   write <pack>/laws.json

Events come from fingerprint.events(): word triggers (number, date, question), cuts, transitions, camera
moves, element entrances, big text, SFX by class and manual events. For every ordered pair of different
event types (X, Y): an occurrence of X is "followed" when a Y starts 0..window seconds after it.
  Matching is one-to-one per video (each Y serves one X); events of one source (same element, same box)
  never pair. confidence = macro-average over videos of matched X / all X (pooled_confidence = all videos
  together); support_videos = videos whose own confidence is at least 0.3; lag = {median, p10, p90, n,
  seen_in} of Y - X over the matched pairs of those videos; chance = share of time with a Y within the
  window (macro-average); lift = confidence / chance.
  status = "rule-candidate" when support_videos >= 3, confidence >= 0.6 and lift >= 1.5, else "option"
  anchor          = "word" when X is a word trigger, else "visual" (lang.scale_timing relies on it)
Only pairs supported by at least 2 videos with confidence >= 0.3 are written. Near-duplicate videos count once
(fingerprint.dedupe). Word laws are skipped, and listed under not_run, when no training video has a
transcript or the transcripts are in different languages; the visual and sound laws still run.

Output: {schema "laws/1", trained_on, held_out, dupes, window_s, laws, co_occurrences, not_run}.
A video is skipped for a pair when the layer of X or Y was not measured in it (not a confidence of 0).
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
LIFT_MIN = 1.5            # a rule candidate must beat the base rate of Y by this factor
VIDEO_CONF_MIN = 0.3      # a video supports a law only if its own confidence is at least this
LIST_MIN_VIDEOS = 2
LIST_MIN_CONFIDENCE = 0.3
LIFT_CAP = 999.0
CO_S = 0.04              # lag-0: events of different layers closer than this are one moment (1 frame)


def _union_len(intervals, lo, hi):
    """Length of the union of [a, b] intervals clipped to [lo, hi]."""
    total, cur_a, cur_b = 0.0, None, None
    for a, b in sorted((max(a, lo), min(b, hi)) for a, b in intervals if b > lo and a < hi):
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        total += cur_b - cur_a
    return total


def _match(xs, ys, window_s):
    """Greedy one-to-one matching in time order: each X takes the earliest unused Y that starts 0..window
    after it. Events of one source never match; at lag 0 only X before Y in type order counts, so two
    simultaneous events never make a law in both directions. Returns the lags of the matched pairs."""
    lags, used = [], set()
    yt = [y["t"] for y in ys]
    for x in xs:
        j = bisect.bisect_left(yt, x["t"] - 1e-9)
        while j < len(ys) and yt[j] <= x["t"] + window_s + 1e-9:
            y = ys[j]
            lag = y["t"] - x["t"]
            same_moment = lag < CO_S and y["layer"] != x["layer"]     # a co-occurrence, not a law
            ok = (j not in used and y.get("src") != x.get("src") and not same_moment
                  and (lag > 1e-6 or x["type"] < y["type"]))
            if ok:
                used.add(j)
                lags.append(max(0.0, lag))
                break
            j += 1
    return lags


def mine(streams, window_s=WINDOW_S, spans=None, layers=None):
    return _mine(streams, window_s, spans, layers)[0]


def _mine(streams, window_s=WINDOW_S, spans=None, layers=None):
    """(laws, co_occurrences) from {video id: events}; spans {video id: duration_s} (default: the stream's
    own extent, also when a duration is 0 or less); layers {video id: set of measured layers}: a video is
    skipped for a pair when the layer of X or of Y was not measured in it (default: all measured).
    co_occurrences [{a, b, occurrences, videos}] are lag-0 pairs of different layers (a cut and a camera
    move, a manual and an automatic event within one frame): one moment, not a rule.

    Per video and (X, Y) pair, X and Y are matched one-to-one (_match); the video's confidence is
    matched X / all X. Law = {when, then, anchor, lag, support_videos, occurrences, followed, confidence,
    pooled_confidence, chance, lift, status}:
      confidence         macro-average of the per-video confidences over the videos that have X
      pooled_confidence  all matched X / all X
      support_videos     videos whose own confidence is at least VIDEO_CONF_MIN
      chance             macro-average over those videos of the share of time that has a Y within the window
      lift               confidence / chance
      status             "rule-candidate" with support >= 3, confidence >= 0.6 and lift >= LIFT_MIN"""
    per = {}        # (x, y) -> {vid: (n_x, n_matched, chance)}
    lags_all = {}   # (x, y) -> [lags of videos that support]  (filled below)
    raw_lags, co = {}, {}
    all_types = {e["type"] for evs in streams.values() for e in evs if e.get("type")}
    for vid, evs in streams.items():
        evs = sorted((dict(e, src=e.get("src", f"#{k}"), layer=e.get("layer") or fp.layer_of(e["type"]))
                      for k, e in enumerate(evs) if fp._num(e.get("t")) and e.get("type")),
                     key=lambda e: (e["t"], e["type"], e["src"]))
        if not evs:
            continue
        if spans and fp._num(spans.get(vid)) and spans[vid] > 0:
            lo, hi = 0.0, float(spans[vid])
        else:
            lo = evs[0]["t"]
            hi = lo + max(evs[-1]["t"] - evs[0]["t"], window_s)
        ok_layers = layers.get(vid) if layers is not None else None
        for a_i, a in enumerate(evs):         # co-occurrences: one moment seen by two layers
            for b in evs[a_i + 1:]:
                if b["t"] - a["t"] >= CO_S:
                    break
                if a["layer"] != b["layer"] and a["src"] != b["src"] and a["type"] != b["type"] and (
                        ok_layers is None or (a["layer"] in ok_layers and b["layer"] in ok_layers)):
                    key = tuple(sorted((a["type"], b["type"])))
                    co.setdefault(key, {})[vid] = co.get(key, {}).get(vid, 0) + 1
        by = {}
        for e in evs:
            by.setdefault(e["type"], []).append(e)
        span = max(hi - lo, 1e-9)
        cover = {y: _union_len([(e["t"] - window_s, e["t"]) for e in lst], lo, hi) / span
                 for y, lst in by.items()}
        for x, xs in by.items():
            for y in all_types:
                if x == y:
                    continue
                if ok_layers is not None and not (by[x][0]["layer"] in ok_layers and
                                                  fp.layer_of(y) in ok_layers):
                    continue
                lags = _match(xs, by.get(y, []), window_s)
                per.setdefault((x, y), {})[vid] = (len(xs), len(lags), min(1.0, cover.get(y, 0.0)))
                raw_lags.setdefault((x, y), {})[vid] = lags
    laws = []
    for (x, y), vids in per.items():
        confs = {v: m / n for v, (n, m, _) in vids.items()}
        sup = [v for v, c in confs.items() if c >= VIDEO_CONF_MIN and vids[v][1] > 0]
        conf = sum(confs.values()) / len(confs)
        if len(sup) < LIST_MIN_VIDEOS or conf < LIST_MIN_CONFIDENCE:
            continue
        n_x = sum(n for n, _, _ in vids.values())
        n_m = sum(m for _, m, _ in vids.values())
        chance = sum(c for _, _, c in vids.values()) / len(vids)
        lift = min(LIFT_CAP, conf / chance) if chance > 1e-9 else LIFT_CAP
        lag = fp.stat([l for v in sup for l in raw_lags[(x, y)][v]])
        lag["seen_in"] = len(sup)
        rule = len(sup) >= MIN_SUPPORT and conf >= MIN_CONFIDENCE - 1e-9 and lift >= LIFT_MIN - 1e-9
        laws.append({"when": x, "then": y, "anchor": "word" if x.startswith("word:") else "visual",
                     "lag": lag, "support_videos": len(sup), "occurrences": n_x, "followed": n_m,
                     "confidence": round(conf, 4), "pooled_confidence": round(n_m / n_x, 4),
                     "chance": round(chance, 4), "lift": round(lift, 3),
                     "status": "rule-candidate" if rule else "option"})
    laws.sort(key=lambda l: (l["status"] != "rule-candidate", -l["support_videos"], -l["confidence"],
                             l["when"], l["then"]))
    cos = [{"a": a, "b": b, "occurrences": sum(v.values()), "videos": len(v)}
           for (a, b), v in sorted(co.items()) if len(v) >= LIST_MIN_VIDEOS]
    return laws, cos


def build(pack, train=None, exclude=None, window_s=WINDOW_S):
    ctx = fp.prepare(pack, train, exclude)
    a = ctx["analyses"]
    streams = {i: fp.events(a[i], words=ctx["word_ok"]) for i in ctx["trained"]}
    spans = {i: fp._dur(a[i]) for i in ctx["trained"]}
    layers = {i: fp.measured_layers(a[i], ctx["word_ok"]) for i in ctx["trained"]}
    lw, co = _mine(streams, window_s, spans, layers)
    return {"schema": SCHEMA, "trained_on": ctx["trained"], "held_out": ctx["held_out"], "dupes": ctx["dupes"],
            "window_s": window_s, "laws": lw, "co_occurrences": co, "not_run": ctx["not_run"]}


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
