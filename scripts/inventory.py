"""Episode inventory: what was run, per source directory, model, test condition, graph and seed.

  python scripts/inventory.py results/raw/{api_premise,api_visible,confirm_hidden,confirm_visible}/*.jsonl
"""
import json
import sys
from collections import defaultdict

rows = defaultdict(lambda: {"n": 0, "ok": 0, "res": 0, "cost": 0.0, "tasks": set()})
for path in sys.argv[1:]:
    src = path.split("/")[-2]
    last = {}
    for line in open(path):
        e = json.loads(line)
        last[(e["episode_id"], e.get("tests", "hidden"))] = e
    for e in last.values():
        r = rows[(src, e["model"], e.get("tests", "hidden"), e["graph"], e["seed"])]
        r["n"] += 1
        if e["status"] == "ok":
            r["ok"] += 1
            r["res"] += bool(e["resolved"])
            r["cost"] += float(e.get("cost_usd") or 0)
            r["tasks"].add(e["task_id"])
print("src model tests graph seed n ok tasks resolve cost_per_ep")
for k in sorted(rows):
    r = rows[k]
    print(*k, r["n"], r["ok"], len(r["tasks"]), round(r["res"] / max(r["ok"], 1), 3), round(r["cost"] / max(r["ok"], 1), 4))
