#!/usr/bin/env python3
"""Paid-spend gate: prices, one estimate per video, owner-approved batches, release and log.

This script NEVER makes a paid call and never touches the network. Claude makes the call
through an MCP/CLI tool, but only after `next` released the item, and then writes down what
happened with `record`.

Files (project = the project folder):
  <store>/prices.json   {"version": 1, "prices": {"tool|model|unit": {cost, currency, unit, source, date}}}
  <project>/gen/queue.json  {"version": 1, "items": [...]}      every paid call is first an item here
  <project>/spend.json      {"version": 1, "batches": [...]}     estimates and the owner's approvals
  <project>/gen/log.jsonl   one line per paid call: prompt, model, seed, cost, file, sha256

Flow:
  price set ...        list the price of every tool/model/unit (tool estimate, owner or site)
  add / (genprep)      queue an item
  estimate <project>   ONE estimate for the whole video: base + retry allowance, per currency
  approve <project> --batch S01 --owner-words "<the owner's own message>"
                       refuses empty words (exit 2); only this command marks a batch approved
  next <project>       releases the next item only if it is free, or approved and inside the batch
                       total, or a retry inside the allowance; else exit 1 (needs_batch)
  record <project> ... logs the paid call, adds the cost; past the total the batch becomes `over`
                       (the line is still logged: the money is gone) and the exit is 1

Exit codes: 0 ok, 1 a check failed or the gate refused, 2 bad input (also a bad item: no units,
units that are not a positive number, ...: such an item is BLOCKED, never free).

What this gate is, and is not: it protects against mistakes and against agents that forget the
rule. `approve` needs the owner's own words and the total the owner saw; `next` releases only an
item that still matches what was estimated and approved (model, units, price, prompt); a batch ID
is never reused. It can NOT fully stop a determined local edit: anyone who can write spend.json
or queue.json by hand can fake an approval. Treat the files as an audit trail, not as a lock.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import re
import sys
import unicodedata
from pathlib import Path

import store

UNITS = ("image", "second", "clip", "model", "minute")
SOURCES = ("tool-estimate", "owner", "site")
KINDS = ("image", "video", "3d", "asr", "stock", "other")
STALE_DAYS = 30
EPS = 1e-9
SEQ = re.compile(r"^G(\d+)(?:\.r(\d+))?$")


class BadInput(Exception):
    """Exit 2."""


class BadItem(Exception):
    """An item that cannot be priced safely (no/invalid units ...). Blocked, never free (exit 2)."""


class MissingPrice(Exception):
    """One or more price keys are not listed."""

    def __init__(self, keys):
        self.keys = sorted(set(keys))
        super().__init__("missing price: " + ", ".join(self.keys))


class NotReleased(Exception):
    """The gate refused to release an item (exit 1)."""

    def __init__(self, item_id, reason, bad_item=False):
        self.item_id = item_id
        self.reason = reason
        self.bad_item = bad_item
        super().__init__(f"{item_id}: {reason}")


# ---------------------------------------------------------------- paths and files

def _queue_path(d) -> Path:
    return Path(d) / "gen" / "queue.json"


def _spend_path(d) -> Path:
    return Path(d) / "spend.json"


def _log_path(d) -> Path:
    return Path(d) / "gen" / "log.jsonl"


def _prices_path() -> Path:
    return store.path("prices.json")


def _empty_queue():
    return {"version": 1, "items": []}


def _empty_spend():
    return {"version": 1, "batches": []}


def _empty_prices():
    return {"version": 1, "prices": {}}


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _today() -> str:
    return datetime.date.today().isoformat()


def load_prices() -> dict:
    return store.read_json(_prices_path(), _empty_prices()).get("prices", {})


def price_key(tool, model, unit) -> str:
    return f"{tool}|{model}|{unit}"


def _item_key(item) -> str:
    return price_key(item.get("tool"), item.get("model"), item.get("unit"))


def _units(item) -> float:
    """The item's units as a positive finite number, else BadItem."""
    u = item.get("units")
    if isinstance(u, bool) or not isinstance(u, (int, float)) or not math.isfinite(u) or u <= 0:
        raise BadItem(f"bad_item: {item.get('id')} has units {u!r} (must be a positive number)")
    return float(u)


def _unit_cost(item, prices) -> float:
    key = _item_key(item)
    if key not in prices:
        raise KeyError(key)
    c = prices[key].get("cost")
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or c < 0:
        raise BadItem(f"bad_item: price {key} is not a number >= 0")
    return float(c)


def item_cost(item, prices) -> float:
    """units x listed price. KeyError(price key) when the price is not listed; BadItem when the
    units are missing, zero, negative or not a number (never treated as free)."""
    uc = _unit_cost(item, prices)
    return round(_units(item) * uc, 6)


def _prompt_hash(item) -> str:
    raw = json.dumps([item.get("prompt"), item.get("negative")], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _snap(item, prices) -> dict:
    return {"key": _item_key(item), "model": item.get("model"), "units": _units(item),
            "unit_cost": _unit_cost(item, prices), "cost": item_cost(item, prices),
            "prompt_hash": _prompt_hash(item)}


def _currency(item, prices) -> str:
    return prices[_item_key(item)].get("currency") or ""


_INVISIBLE = {"ㅤ", "ᅟ", "ᅠ", "⠀", "ﾠ"}


def _blank(s) -> bool:
    """True for empty text, whitespace, and invisible marks (RTL/LRM, zero-width, NBSP)."""
    return all(c.isspace() or c in _INVISIBLE or unicodedata.category(c) in ("Cf", "M") for c in (s or ""))


# ---------------------------------------------------------------- price commands

def price_set(tool, model, unit, cost, currency, source) -> dict:
    if unit not in UNITS:
        raise BadInput(f"unit must be one of {', '.join(UNITS)}")
    if source not in SOURCES:
        raise BadInput(f"source must be one of {', '.join(SOURCES)}")
    if not tool or not model or "|" in tool or "|" in model:
        raise BadInput("tool and model must be non-empty and not contain '|'")
    if not currency or not currency.strip():
        raise BadInput("currency is required")
    try:
        cost = float(cost)
    except (TypeError, ValueError):
        raise BadInput("cost must be a number")
    if cost < 0 or cost != cost:
        raise BadInput("cost must be 0 or more")
    entry = {"cost": cost, "currency": currency.strip(), "unit": unit, "source": source, "date": _today()}

    def fn(data):
        data.setdefault("version", 1)
        data.setdefault("prices", {})[price_key(tool, model, unit)] = entry

    store.update_json(_prices_path(), _empty_prices(), fn)
    return entry


def stale_prices(prices, keys, today=None) -> list:
    today = today or datetime.date.today()
    out = []
    for k in sorted(set(keys)):
        try:
            age = (today - datetime.date.fromisoformat(str(prices[k].get("date")))).days
        except (ValueError, KeyError, TypeError):
            age = None
        if age is None or age > STALE_DAYS:
            out.append((k, age))
    return out


# ---------------------------------------------------------------- queue helpers

def _next_id(items) -> str:
    n = 0
    for it in items:
        m = SEQ.match(str(it.get("id", "")))
        if m:
            n = max(n, int(m.group(1)))
    return f"G{n + 1:03d}"


def _by_id(items) -> dict:
    return {i["id"]: i for i in items}


def _root_id(item, by_id) -> str:
    seen = set()
    while item.get("retry_of") and item["id"] not in seen:
        seen.add(item["id"])
        parent = by_id.get(item["retry_of"])
        if parent is None:
            return item["retry_of"]
        item = parent
    return item["id"]


def _batch_for(item, by_id, batches):
    """The batch that covers this item: the nearest one up its retry chain (a retry that was
    re-estimated sits in its own batch; any other retry shares the batch of what it retries)."""
    seen = set()
    while item is not None and item["id"] not in seen:
        seen.add(item["id"])
        hits = [b for b in batches if item["id"] in b.get("items", [])]
        if hits:
            return hits[-1]
        item = by_id.get(item.get("retry_of"))
    return None


def add(d, tool, model, units, unit, kind, what) -> dict:
    """Queue any other paid call (cloud ASR, stock, ...)."""
    if unit not in UNITS:
        raise BadInput(f"unit must be one of {', '.join(UNITS)}")
    if kind not in KINDS:
        raise BadInput(f"kind must be one of {', '.join(KINDS)}")
    try:
        units = float(units)
    except (TypeError, ValueError):
        raise BadInput("units must be a number")
    if not math.isfinite(units) or units <= 0:
        raise BadInput("units must be a positive number")
    holder = {}

    def fn(q):
        q.setdefault("version", 1)
        items = q.setdefault("items", [])
        item = {"id": _next_id(items), "beat": None, "stage": "other", "variant": None, "recipe": None,
                "how": None, "tool": tool, "model": model, "kind": kind, "units": units, "unit": unit,
                "prompt": None, "negative": None, "seed": None, "params": {}, "camera_curve": None,
                "attempt": 1, "retry_of": None, "status": "queued", "batch": None, "file": None,
                "cost": None, "what": what}
        items.append(item)
        holder["item"] = item

    store.update_json(_queue_path(d), _empty_queue(), fn)
    return holder["item"]


def retry(d, item_id) -> dict:
    """Queue a retry of an item (G001 -> G001.r2, .r3 ...). It is only released inside the allowance."""
    holder = {}

    def fn(q):
        items = q.setdefault("items", [])
        by_id = _by_id(items)
        if item_id not in by_id:
            raise BadInput(f"no such item: {item_id}")
        root = by_id[_root_id(by_id[item_id], by_id)]
        taken = [int(m.group(2)) for i in items if _root_id(i, by_id) == root["id"]
                 for m in [SEQ.match(str(i["id"]))] if m and m.group(2)]
        n = max(taken + [1]) + 1
        chain = [i for i in items if i["id"] == root["id"] or i.get("retry_of") == root["id"]
                 or _root_id(i, by_id) == root["id"]]
        latest = max(chain, key=lambda i: i.get("attempt", 1))
        new = dict(root)
        new.update({"id": f"{root['id']}.r{n}", "retry_of": latest["id"], "attempt": n, "status": "queued",
                    "file": None, "cost": None})
        items.append(new)
        holder["item"] = new

    store.update_json(_queue_path(d), _empty_queue(), fn)
    return holder["item"]


# ---------------------------------------------------------------- estimate

def estimate(d, retry_allowance=1, warnings=None) -> list:
    """One estimate for the whole video: a draft batch per currency (base + retry allowance)."""
    if retry_allowance < 0:
        raise BadInput("retry allowance must be 0 or more")
    prices = load_prices()
    out = []

    def on_spend(sp):
        def on_queue(q):
            items = q.get("items", [])
            by_id = _by_id(items)
            sp.setdefault("version", 1)
            batches = sp.setdefault("batches", [])
            # an older draft is superseded by this estimate (kept, with its ID: IDs are never reused)
            for b in batches:
                if b.get("status") == "draft":
                    b["status"] = "superseded"
            todo, missing, bad = {}, [], []
            for it in items:
                if it.get("status") not in ("queued", "needs_batch"):
                    continue
                if it.get("retry_of") and it["status"] == "queued":
                    b = _batch_for(it, by_id, batches)
                    if b and b.get("status") == "approved":
                        continue  # covered by that batch's retry allowance
                try:
                    cost = item_cost(it, prices)
                except KeyError as e:
                    missing.append(e.args[0])
                    continue
                except BadItem as e:
                    bad.append(str(e))
                    continue
                if cost <= 0:
                    continue  # free: price explicitly 0 and units a positive number
                todo.setdefault(_currency(it, prices), []).append((it, cost))
            if bad:
                raise BadItem("; ".join(bad))
            if missing:
                raise MissingPrice(missing)
            n = max([int(b["id"][1:]) for b in batches if re.fullmatch(r"S\d+", str(b.get("id")))] + [0])  # never reuse
            for cur in sorted(todo):
                n += 1
                bid = f"S{n:02d}"
                base = round(sum(c for _, c in todo[cur]), 6)
                rt = round(base * retry_allowance, 6)
                used, snap = {}, {}
                for it, _ in todo[cur]:
                    used[_item_key(it)] = prices[_item_key(it)]["cost"]
                    snap[it["id"]] = _snap(it, prices)
                    it["batch"] = bid
                batch = {"id": bid, "items": [it["id"] for it, _ in todo[cur]], "base_total": base,
                         "retry_allowance": retry_allowance, "retry_total": rt, "total": round(base + rt, 6),
                         "currency": cur, "prices_used": used, "snapshot": snap, "status": "draft", "created": _now(),
                         "approved_at": None, "owner_words": None, "spent": 0.0, "retries_used": 0}
                batches.append(batch)
                out.append(batch)
            if warnings is not None:
                keys = {k for b in out for k in b["prices_used"]}
                for k, age in stale_prices(prices, keys):
                    when = "of unknown date" if age is None else f"{age} days old"
                    warnings.append(f"price {k} is {when} (over {STALE_DAYS}): confirm it")

        store.update_json(_queue_path(d), _empty_queue(), on_queue)

    store.update_json(_spend_path(d), _empty_spend(), on_spend)
    return out


# ---------------------------------------------------------------- approve

def _totals(total, n):
    if total is None or (isinstance(total, str) and not total.strip()):
        raise BadInput("--total is required: the total the owner saw (one per batch, comma separated)")
    parts = total if isinstance(total, (list, tuple)) else str(total).split(",")
    try:
        vals = [float(x) for x in parts]
    except (TypeError, ValueError):
        raise BadInput("--total must be a number (one per batch, comma separated)")
    if len(vals) == 1 and n > 1:
        vals = vals * n  # one number can only match when every batch has that total
    if len(vals) != n:
        raise BadInput(f"--total needs {n} value(s), one per batch")
    return vals


def approve(d, batch_ids, owner_words, total=None) -> None:
    """Mark draft batches approved with the owner's own words and the total the owner saw
    (must equal the batch total to the cent); their items become `approved`."""
    if _blank(owner_words):
        raise BadInput("owner words are required: paste the owner's own message that says yes")
    owner_words = owner_words.strip()
    ids = list(dict.fromkeys(batch_ids))
    if not ids:
        raise BadInput("no batch given")
    seen_totals = _totals(total, len(ids))

    def on_spend(sp):
        batches = {b["id"]: b for b in sp.get("batches", [])}
        for i in ids:
            if i not in batches:
                raise BadInput(f"no such batch: {i}")
            if batches[i].get("status") != "draft":
                raise BadInput(f"batch {i} is {batches[i].get('status')}, not a draft: run estimate again")
        for i, t in zip(ids, seen_totals):
            if round(t, 2) != round(batches[i]["total"], 2):
                raise BadInput(f"total {t:.2f} is not the total of batch {i} ({batches[i]['total']:.2f}): "
                               "show the owner the current estimate")

        def on_queue(q):
            for i in ids:
                b = batches[i]
                for it in q.get("items", []):
                    if (it["id"] in b["items"] and it.get("batch") == i
                            and it.get("status") in ("queued", "needs_batch")):
                        it["status"] = "approved"
                b["status"] = "approved"
                b["approved_at"] = _now()
                b["owner_words"] = owner_words

        store.update_json(_queue_path(d), _empty_queue(), on_queue)

    store.update_json(_spend_path(d), _empty_spend(), on_spend)


# ---------------------------------------------------------------- next

def _inflight_cost(items, batch, by_id, batches, prices) -> float:
    total = 0.0
    for it in items:
        if it.get("status") == "in_flight" and _batch_for(it, by_id, batches) is batch:
            try:
                total += item_cost(it, prices)
            except (KeyError, BadItem):
                total += float(batch.get("snapshot", {}).get(it["id"], {}).get("cost", 0) or 0)
    return total


def _snapshot_for(it, by_id, batch):
    """The estimate-time snapshot that covers this item (its own, or its nearest ancestor's)."""
    snaps = batch.get("snapshot") or {}
    seen = set()
    cur = it
    while cur is not None and cur["id"] not in seen:
        seen.add(cur["id"])
        if cur["id"] in snaps:
            return cur["id"], snaps[cur["id"]]
        cur = by_id.get(cur.get("retry_of"))
    return None, None


def _check_release(it, items, by_id, batches, prices):
    """Return (ok, batch_or_None, how, bad_item). The only place that decides a release."""
    try:
        cost = item_cost(it, prices)
    except KeyError as e:
        return False, None, f"price not listed: {e.args[0]}", False
    except BadItem as e:
        return False, None, str(e), True
    batch = _batch_for(it, by_id, batches)
    if batch is None:
        if cost <= 0:
            return True, None, "free", False  # price explicitly 0 and units a positive number
        return False, None, "not in any batch: estimate and get the owner's yes", False
    if batch.get("status") != "approved" or _blank(batch.get("owner_words")) or not batch.get("approved_at"):
        return False, batch, f"batch {batch['id']} is {batch.get('status')}, not approved", False
    # what is released must be what the owner approved: compare with the estimate-time snapshot
    sid, snap = _snapshot_for(it, by_id, batch)
    if snap is None:
        return False, batch, f"batch {batch['id']} has no estimate snapshot for this item", False
    live = _snap(it, prices)
    for k in ("key", "model", "units", "unit_cost", "cost") + (("prompt_hash",) if sid == it["id"] else ()):
        if live[k] != snap.get(k):
            return False, batch, (f"{k} changed since the estimate (was {snap.get(k)!r}, now {live[k]!r}): "
                                  "new estimate and a new yes"), False
    if _currency(it, prices) != batch.get("currency"):
        return False, batch, f"currency differs from batch {batch['id']}", False
    room = batch["total"] - batch.get("spent", 0.0) - _inflight_cost(items, batch, by_id, batches, prices)
    if cost > room + EPS:
        return False, batch, f"batch {batch['id']} has {room:.2f} {batch['currency']} left, this needs {cost:.2f}", False
    if it["id"] in batch["items"]:  # an estimated item (a re-estimated retry counts as one too)
        if it.get("status") != "approved":
            return False, batch, f"item is {it.get('status')}, not approved", False
        return True, batch, "approved", False
    if it.get("retry_of"):
        cap = len(batch["items"]) * batch.get("retry_allowance", 0)
        if batch.get("retries_used", 0) >= cap:
            return False, batch, (f"retry allowance of batch {batch['id']} is used ({cap}): "
                                  "new estimate and a new yes"), False
        return True, batch, "retry", False
    return False, batch, f"item is not in batch {batch['id']}", False


def next_item(d):
    """Release the next item (-> in_flight) or return None when nothing is left.

    Raises NotReleased(item id, reason) when items are waiting but none may be released;
    those items are set to `needs_batch`.
    """
    prices = load_prices()
    res = {}

    def on_spend(sp):
        def on_queue(q):
            items = q.get("items", [])
            by_id = _by_id(items)
            batches = sp.get("batches", [])
            blocked = []
            for it in items:
                if it.get("status") not in ("queued", "approved", "needs_batch"):
                    continue
                ok, batch, how, bad = _check_release(it, items, by_id, batches, prices)
                if ok and how == "approved" and it["status"] != "approved":
                    ok, how = False, f"item is {it['status']}, not approved"
                if not ok:
                    blocked.append((it, how, bad))
                    continue
                if how == "retry":
                    batch["retries_used"] = batch.get("retries_used", 0) + 1
                it["status"] = "in_flight"
                if batch is not None:
                    it["batch"] = batch["id"]
                res["item"] = dict(it)
                break
            if "item" not in res:
                for it, _, _ in blocked:
                    it["status"] = "needs_batch"
                if blocked:
                    res["refused"] = (blocked[0][0]["id"], blocked[0][1], blocked[0][2])

        store.update_json(_queue_path(d), _empty_queue(), on_queue)

    store.update_json(_spend_path(d), _empty_spend(), on_spend)
    if "item" in res:
        return res["item"]
    if "refused" in res:
        raise NotReleased(*res["refused"])
    return None


# ---------------------------------------------------------------- record

def _sha256(p: Path):
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(d, file) -> str:
    p = Path(file)
    if not p.is_absolute():
        p = Path(d) / p
    try:
        return os.path.relpath(p, d).replace("\\", "/")
    except ValueError:  # other drive on Windows
        return str(p).replace("\\", "/")


def record(d, item_id, file, cost, model, seed, job_id=None, status="generated") -> dict:
    """Log one paid call and add its cost to the batch. Past the total the batch becomes `over`;
    the line is logged anyway (the money is gone) and the result has over=True."""
    if status not in ("generated", "failed"):
        raise BadInput("status must be generated or failed")
    try:
        cost = float(cost)
    except (TypeError, ValueError):
        raise BadInput("cost must be a number")
    if cost < 0:
        raise BadInput("cost must be 0 or more")
    prices = load_prices()
    res = {}

    def on_spend(sp):
        def on_queue(q):
            items = q.get("items", [])
            by_id = _by_id(items)
            if item_id not in by_id:
                raise BadInput(f"no such item: {item_id}")
            it = by_id[item_id]
            batch = _batch_for(it, by_id, sp.get("batches", []))
            relfile = _rel(d, file) if file else None
            currency = batch["currency"] if batch else prices.get(_item_key(it), {}).get("currency")
            line = {"type": "gen", "date": _now(), "item": it["id"], "beat": it.get("beat"),
                    "stage": it.get("stage"), "recipe": it.get("recipe"), "tool": it.get("tool"),
                    "model": model, "prompt": it.get("prompt"), "negative": it.get("negative"),
                    "seed": seed, "cost": cost, "currency": currency, "file": relfile,
                    "sha256": _sha256(Path(d) / relfile) if relfile else None, "job_id": job_id,
                    "status": status, "attempt": it.get("attempt", 1)}
            # the log line is written first: whatever happens next, the paid call is on record
            store.append_jsonl(_log_path(d), line)
            was = it.get("status")
            it.update({"status": status, "file": relfile, "cost": cost, "model": model, "seed": seed})
            over = False
            if batch is not None:
                batch["spent"] = round(batch.get("spent", 0.0) + cost, 6)
                if batch["spent"] > batch["total"] + EPS:
                    batch["status"] = "over"
                    over = True
            out = dict(line)
            out["over"] = over
            if was != "in_flight":
                out["warning"] = f"item was {was}, not released by `next`"
            res["line"] = out

        store.update_json(_queue_path(d), _empty_queue(), on_queue)

    store.update_json(_spend_path(d), _empty_spend(), on_spend)
    return res["line"]


# ---------------------------------------------------------------- status

def status(d) -> dict:
    q = store.read_json(_queue_path(d), _empty_queue())
    sp = store.read_json(_spend_path(d), _empty_spend())
    counts = {}
    for it in q.get("items", []):
        counts[it.get("status")] = counts.get(it.get("status"), 0) + 1
    return {"items": counts, "batches": sp.get("batches", [])}


# ---------------------------------------------------------------- CLI

def _fmt(x) -> str:
    return f"{x:.2f}"


def _print_estimate(d, batches, warnings):
    q = store.read_json(_queue_path(d), _empty_queue())
    prices = load_prices()
    for w in warnings:
        print(f"אזהרה: {w}")
    if not batches:
        print("אין פריטים בתשלום שממתינים להערכה.")
        return
    for b in batches:
        print(f"אצווה {b['id']} ({b['currency']})")
        rows = {}
        for it in q["items"]:
            if it["id"] in b["items"]:
                key = (it.get("tool"), it.get("model"), it.get("unit"))
                r = rows.setdefault(key, [0, 0.0, 0.0])
                r[0] += 1
                r[1] += float(it.get("units") or 0)
                r[2] += item_cost(it, prices)
        print("  כלי | מודל | פריטים | יחידות | בסיס")
        for (tool, model, unit), (n, units, c) in sorted(rows.items(), key=lambda kv: str(kv[0])):
            print(f"  {tool} | {model} | {n} | {units:g} {unit} | {_fmt(c)}")
        print(f"  בסיס {_fmt(b['base_total'])} + מרווח לניסיונות חוזרים {_fmt(b['retry_total'])} "
              f"({b['retry_allowance']} לפריט) = סה\"כ {_fmt(b['total'])} {b['currency']}")
    print("הבא: להציג לבעלים ולקבל כן במילים שלו, ואז: spend.py approve <project> --batch "
          + ",".join(b["id"] for b in batches) + " --owner-words \"<ההודעה שלו>\"")


def _build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("price", help="list or set a price")
    ps = pr.add_subparsers(dest="pcmd", required=True)
    s = ps.add_parser("set")
    for n in ("tool", "model", "unit", "currency", "source"):
        s.add_argument("--" + n, required=True)
    s.add_argument("--cost", required=True, type=float)
    ps.add_parser("list")
    a = sub.add_parser("add", help="queue another paid call (cloud ASR, stock...)")
    a.add_argument("project")
    for n in ("tool", "model", "unit", "kind", "what"):
        a.add_argument("--" + n, required=True)
    a.add_argument("--units", required=True, type=float)
    r = sub.add_parser("retry", help="queue a retry of an item")
    r.add_argument("project")
    r.add_argument("--item", required=True)
    e = sub.add_parser("estimate", help="one estimate for the whole video")
    e.add_argument("project")
    e.add_argument("--retry-allowance", type=int, default=1)
    ap_ = sub.add_parser("approve", help="record the owner's yes for batches")
    ap_.add_argument("project")
    ap_.add_argument("--batch", required=True, help="S01 or S01,S02")
    ap_.add_argument("--owner-words", required=True, help="the owner's own message")
    ap_.add_argument("--total", default=None,
                     help="the total the owner saw, per batch, comma separated (must equal the estimate)")
    n_ = sub.add_parser("next", help="release the next item")
    n_.add_argument("project")
    rc = sub.add_parser("record", help="log a paid call")
    rc.add_argument("project")
    rc.add_argument("--item", required=True)
    rc.add_argument("--file", default=None)
    rc.add_argument("--cost", required=True, type=float)
    rc.add_argument("--model", required=True)
    rc.add_argument("--seed", default=None)
    rc.add_argument("--job-id", default=None)
    rc.add_argument("--status", default="generated", choices=("generated", "failed"))
    st = sub.add_parser("status", help="items and batches")
    st.add_argument("project")
    st.add_argument("--json", action="store_true")
    return ap


def _seed(v):
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        return v


def _run(a) -> int:
    if a.cmd == "price":
        if a.pcmd == "set":
            e = price_set(a.tool, a.model, a.unit, a.cost, a.currency, a.source)
            print(f"price {price_key(a.tool, a.model, a.unit)} = {e['cost']:g} {e['currency']} "
                  f"({e['source']}, {e['date']})")
        else:
            for k, v in sorted(load_prices().items()):
                print(f"{k}\t{v['cost']:g} {v['currency']}\t{v['source']}\t{v['date']}")
        return 0
    d = Path(a.project)
    if not d.is_dir():
        raise BadInput(f"project folder not found: {a.project}")
    if a.cmd == "add":
        it = add(d, a.tool, a.model, a.units, a.unit, a.kind, a.what)
        print(f"queued {it['id']}: {a.what}")
    elif a.cmd == "retry":
        it = retry(d, a.item)
        print(f"queued {it['id']} (retry of {it['retry_of']}, attempt {it['attempt']})")
    elif a.cmd == "estimate":
        warnings = []
        try:
            batches = estimate(d, a.retry_allowance, warnings)
        except MissingPrice as e:
            print("חסר מחיר: " + ", ".join(e.keys), file=sys.stderr)
            print("ask the tool's estimate or the owner, then: spend.py price set --tool --model --unit "
                  "--cost --currency --source", file=sys.stderr)
            return 1
        _print_estimate(d, batches, warnings)
    elif a.cmd == "approve":
        ids = [x.strip() for x in a.batch.split(",") if x.strip()]
        approve(d, ids, a.owner_words, a.total)
        print("approved: " + ", ".join(ids))
    elif a.cmd == "next":
        try:
            it = next_item(d)
        except NotReleased as e:
            print(f"{'blocked' if e.bad_item else 'needs_batch'}: {e.item_id}: {e.reason}", file=sys.stderr)
            return 2 if e.bad_item else 1
        print(json.dumps(it if it is not None else {"done": True}, ensure_ascii=False))
    elif a.cmd == "record":
        line = record(d, a.item, a.file, a.cost, a.model, _seed(a.seed), a.job_id, a.status)
        if line["over"]:
            print(f"OVER BUDGET: {a.item} logged, but batch spending is past the approved total; "
                  "stop and ask the owner for a new estimate", file=sys.stderr)
            print(json.dumps({"over": True}))
            return 1
        print(json.dumps({"over": False, "item": a.item}))
        if line.get("warning"):
            print("warning: " + line["warning"], file=sys.stderr)
    elif a.cmd == "status":
        s = status(d)
        if a.json:
            print(json.dumps(s, ensure_ascii=False, indent=2))
        else:
            print("items: " + (", ".join(f"{k}={v}" for k, v in sorted(s["items"].items())) or "none"))
            for b in s["batches"]:
                print(f"{b['id']} {b['status']}: spent {_fmt(b['spent'])} of {_fmt(b['total'])} {b['currency']}, "
                      f"retries {b['retries_used']}/{len(b['items']) * b['retry_allowance']}")
    return 0


def main(argv=None):
    store.print_utf8()
    a = _build_parser().parse_args(argv)
    try:
        return store.exit_on_lock_timeout(_run, a)
    except (BadInput, BadItem) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
