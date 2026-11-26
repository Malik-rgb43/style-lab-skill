import hashlib
import json

import store
from conftest import run_cli

TOOL, MODEL = "example-tool", "paper-model"


def _item(i, **kw):
    it = {"id": f"G{i:03d}", "beat": i, "stage": "frames", "variant": f"B0{i}-f1-a", "recipe": "r",
          "how": "x", "tool": TOOL, "model": MODEL, "kind": "image", "units": 1, "unit": "image",
          "prompt": f"prompt {i}", "negative": "n", "seed": i, "params": {}, "camera_curve": None,
          "attempt": 1, "retry_of": None, "status": "queued", "batch": None, "file": None,
          "cost": None, "what": "w"}
    it.update(kw)
    return it


def _project(tmp_path, n=3, **kw):
    p = tmp_path / "example-shop"
    store.write_json(p / "gen" / "queue.json",
                     {"version": 1, "items": [_item(i, **kw) for i in range(1, n + 1)]})
    return p


def _price(cost=2.5, model=MODEL, cur="USD"):
    r = run_cli("spend.py", "price", "set", "--tool", TOOL, "--model", model, "--unit", "image",
                "--cost", str(cost), "--currency", cur, "--source", "owner")
    assert r.returncode == 0, r.stderr


def _items(p):
    return {i["id"]: i for i in store.read_json(p / "gen" / "queue.json", {})["items"]}


def _approve(p, bid="S01", words="yes, approved"):
    return run_cli("spend.py", "approve", p, "--batch", bid, "--owner-words", words)


def test_estimate_needs_prices(home, tmp_path):
    p = _project(tmp_path)
    r = run_cli("spend.py", "estimate", p)
    assert r.returncode == 1
    assert f"{TOOL}|{MODEL}|image" in r.stderr
    assert "ask the tool's estimate or the owner" in r.stderr
    _price()
    r = run_cli("spend.py", "estimate", p)
    assert r.returncode == 0, r.stderr
    b = store.read_json(p / "spend.json", {})["batches"][0]
    assert b["base_total"] == 7.5 and b["retry_total"] == 7.5 and b["total"] == 15.0
    assert b["status"] == "draft" and b["currency"] == "USD" and b["id"] == "S01"
    assert "15.00" in r.stdout


def test_approve_needs_owner_words(home, tmp_path):
    p = _project(tmp_path)
    _price()
    run_cli("spend.py", "estimate", p)
    for words in ("", "   ", "‏  \n"):
        r = _approve(p, words=words)
        assert r.returncode == 2
    assert all(i["status"] == "queued" for i in _items(p).values())
    assert store.read_json(p / "spend.json", {})["batches"][0]["status"] == "draft"
    assert _approve(p, words="כן, תאשר").returncode == 0
    b = store.read_json(p / "spend.json", {})["batches"][0]
    assert b["status"] == "approved" and b["owner_words"] == "כן, תאשר" and b["approved_at"]
    assert all(i["status"] == "approved" for i in _items(p).values())


def test_next_releases_only_approved(home, tmp_path):
    p = tmp_path / "example-shop"
    store.write_json(p / "gen" / "queue.json", {"version": 1, "items": [
        _item(1), _item(2, model="free-model")]})
    _price()
    r = run_cli("spend.py", "price", "set", "--tool", TOOL, "--model", "free-model", "--unit", "image",
                "--cost", "0", "--currency", "USD", "--source", "owner")
    assert r.returncode == 0
    # the free item is released with no batch, even behind a paid one; the paid one is not
    r = run_cli("spend.py", "next", p)
    assert r.returncode == 0 and json.loads(r.stdout)["id"] == "G002"
    r = run_cli("spend.py", "next", p)
    assert r.returncode == 1 and "needs_batch" in r.stderr
    assert _items(p)["G001"]["status"] == "needs_batch"
    assert run_cli("spend.py", "estimate", p).returncode == 0
    assert run_cli("spend.py", "next", p).returncode == 1  # still a draft
    assert _approve(p).returncode == 0
    r = run_cli("spend.py", "next", p)
    assert r.returncode == 0
    got = json.loads(r.stdout)
    assert got["id"] == "G001" and got["status"] == "in_flight"
    assert json.loads(run_cli("spend.py", "next", p).stdout) == {"done": True}


def test_free_item_after_blocked_one_is_released(home, tmp_path):
    p = tmp_path / "example-shop"
    store.write_json(p / "gen" / "queue.json", {"version": 1, "items": [
        _item(1), _item(2, model="free-model")]})
    _price()
    run_cli("spend.py", "price", "set", "--tool", TOOL, "--model", "free-model", "--unit", "image",
            "--cost", "0", "--currency", "USD", "--source", "owner")
    assert run_cli("spend.py", "estimate", p).returncode == 0
    # the free item sits behind an unapproved paid one and is still released
    r = run_cli("spend.py", "next", p)
    assert r.returncode == 0 and json.loads(r.stdout)["id"] == "G002"


def test_hand_edited_approval_is_not_enough(home, tmp_path):
    p = _project(tmp_path, n=1)
    _price()
    run_cli("spend.py", "estimate", p)
    sp = store.read_json(p / "spend.json", {})
    sp["batches"][0]["status"] = "approved"  # approved with no owner words, no timestamp
    store.write_json(p / "spend.json", sp)
    q = store.read_json(p / "gen" / "queue.json", {})
    q["items"][0]["status"] = "approved"
    store.write_json(p / "gen" / "queue.json", q)
    assert run_cli("spend.py", "next", p).returncode == 1


def test_item_outside_batch_not_released(home, tmp_path):
    p = _project(tmp_path, n=1)
    _price()
    run_cli("spend.py", "estimate", p)
    _approve(p)
    q = store.read_json(p / "gen" / "queue.json", {})
    q["items"].append(_item(2, status="approved"))  # approved by hand, not in the batch
    store.write_json(p / "gen" / "queue.json", q)
    assert run_cli("spend.py", "next", p).returncode == 0  # G001
    r = run_cli("spend.py", "next", p)
    assert r.returncode == 1 and "G002" in r.stderr


def test_inflight_counts_against_total(home, tmp_path):
    p = _project(tmp_path, n=1)
    _price()
    run_cli("spend.py", "estimate", p, "--retry-allowance", "0")
    _approve(p)
    assert run_cli("spend.py", "next", p).returncode == 0
    run_cli("spend.py", "retry", p, "--item", "G001")
    r = run_cli("spend.py", "next", p)  # allowance 0 and G001 still in flight
    assert r.returncode == 1


def test_retry_allowance_then_needs_batch(home, tmp_path):
    p = _project(tmp_path, n=1)
    _price()
    run_cli("spend.py", "estimate", p)
    _approve(p)
    assert json.loads(run_cli("spend.py", "next", p).stdout)["id"] == "G001"
    assert run_cli("spend.py", "record", p, "--item", "G001", "--cost", "2.5", "--model", MODEL,
                   "--seed", "1", "--status", "failed").returncode == 0
    assert run_cli("spend.py", "retry", p, "--item", "G001").returncode == 0
    r = run_cli("spend.py", "next", p)
    assert r.returncode == 0 and json.loads(r.stdout)["id"] == "G001.r2"
    assert run_cli("spend.py", "record", p, "--item", "G001.r2", "--cost", "2.5", "--model", MODEL,
                   "--seed", "2").returncode == 0
    run_cli("spend.py", "retry", p, "--item", "G001")
    r = run_cli("spend.py", "next", p)
    assert r.returncode == 1 and "needs_batch" in r.stderr and "G001.r3" in r.stderr
    assert _items(p)["G001.r3"]["status"] == "needs_batch"
    # a new estimate covers it and a new yes releases it
    assert run_cli("spend.py", "estimate", p).returncode == 0
    assert _approve(p, "S02", "yes").returncode == 0
    assert json.loads(run_cli("spend.py", "next", p).stdout)["id"] == "G001.r3"


def test_record_over_budget_logs_and_flags(home, tmp_path):
    p = _project(tmp_path, n=1)
    _price()
    run_cli("spend.py", "estimate", p)
    _approve(p)
    assert run_cli("spend.py", "next", p).returncode == 0
    r = run_cli("spend.py", "record", p, "--item", "G001", "--cost", "100", "--model", MODEL, "--seed", "1")
    assert r.returncode == 1
    assert json.loads(r.stdout) == {"over": True}
    lines = (p / "gen" / "log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["cost"] == 100.0
    assert store.read_json(p / "spend.json", {})["batches"][0]["status"] == "over"
    q = store.read_json(p / "gen" / "queue.json", {})
    q["items"].append(_item(2, status="approved", batch="S01"))
    store.write_json(p / "gen" / "queue.json", q)
    assert run_cli("spend.py", "next", p).returncode == 1


def test_log_fields_complete(home, tmp_path):
    p = _project(tmp_path, n=1)
    _price()
    run_cli("spend.py", "estimate", p)
    _approve(p)
    run_cli("spend.py", "next", p)
    f = p / "gen" / "out" / "G001.png"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"png-bytes")
    r = run_cli("spend.py", "record", p, "--item", "G001", "--file", f, "--cost", "2.5", "--model", MODEL,
                "--seed", "7", "--job-id", "J1")
    assert r.returncode == 0, r.stderr
    line = json.loads((p / "gen" / "log.jsonl").read_text(encoding="utf-8").splitlines()[0])
    for k in ("type", "date", "item", "beat", "stage", "recipe", "tool", "model", "prompt", "negative", "seed",
              "cost", "currency", "file", "sha256", "job_id", "status", "attempt"):
        assert k in line, k
    assert line["prompt"] == "prompt 1" and line["model"] == MODEL and line["seed"] == 7
    assert line["cost"] == 2.5 and line["file"] == "gen/out/G001.png" and line["job_id"] == "J1"
    assert line["sha256"] == hashlib.sha256(b"png-bytes").hexdigest()
    sp = store.read_json(p / "spend.json", {})["batches"][0]
    assert sp["spent"] == 2.5 and sp["status"] == "approved"
    assert _items(p)["G001"]["status"] == "generated"


def test_mixed_currencies_one_batch_each_one_yes(home, tmp_path):
    p = tmp_path / "example-shop"
    store.write_json(p / "gen" / "queue.json", {"version": 1, "items": [
        _item(1), _item(2, model="eur-model")]})
    _price()
    _price(cost=1.0, model="eur-model", cur="EUR")
    assert run_cli("spend.py", "estimate", p).returncode == 0
    bs = store.read_json(p / "spend.json", {})["batches"]
    assert sorted(b["currency"] for b in bs) == ["EUR", "USD"]
    assert _approve(p, "S01,S02").returncode == 0
    assert all(b["status"] == "approved" for b in store.read_json(p / "spend.json", {})["batches"])


def test_add_and_stale_price_warning(home, tmp_path):
    p = tmp_path / "example-shop"
    p.mkdir()
    r = run_cli("spend.py", "add", p, "--tool", "cloud-asr", "--model", "m", "--units", "3", "--unit", "minute",
                "--kind", "asr", "--what", "transcript")
    assert r.returncode == 0, r.stderr
    run_cli("spend.py", "price", "set", "--tool", "cloud-asr", "--model", "m", "--unit", "minute",
            "--cost", "0.1", "--currency", "USD", "--source", "site")
    pp = store.path("prices.json")
    data = store.read_json(pp, {})
    data["prices"]["cloud-asr|m|minute"]["date"] = "2020-01-01"
    store.write_json(pp, data)
    r = run_cli("spend.py", "estimate", p)
    assert r.returncode == 0 and "אזהרה" in r.stdout
    assert store.read_json(p / "spend.json", {})["batches"][0]["total"] == 0.6
    assert run_cli("spend.py", "status", p, "--json").returncode == 0
