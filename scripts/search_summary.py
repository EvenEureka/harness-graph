"""What the live whole-grammar searches did: health of their episodes, the steps they took, what they recommend.

The mean a search saw for its own recommendation is not its value: the graph was kept because it looked good on
those episodes.  Its value is measured on other tasks (scripts/select_graph.py).

  python scripts/search_summary.py results/search/local_*/ --out results/summary/live_search.json
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from harness_graph.grammar import Config

p = argparse.ArgumentParser()
p.add_argument("runs", nargs="+")
p.add_argument("--deadline", type=float, default=3600)
p.add_argument("--out", default="")
a = p.parse_args()

FIELDS = list(Config().__dict__)
runs, by_method = [], defaultdict(list)
for d in sorted(a.runs):
    d = Path(d)
    r = json.loads((d / "result.json").read_text())
    ends, late, errors, n, tasks, secs = Counter(), 0, 0, 0, set(), []
    for line in open(d / "episodes.jsonl"):
        e = json.loads(line)
        if e.get("status") != "ok":
            errors += 1
            continue
        n += 1
        ends[e.get("exit_reason", "")] += 1
        tasks.add(e["task_id"])
        secs.append(e.get("latency_s") or 0)
        late += (e.get("latency_s") or 0) > a.deadline
    start, rec = Config(), Config(**r["recommendation_config"])
    changed = [f"{k}={getattr(rec, k)}" for k in FIELDS if getattr(rec, k) != getattr(start, k)]
    steps = []
    for t in r["trace"]:
        c0, c1 = Config(**r["configs"][t["from"]]), Config(**r["configs"][t["candidate"]])
        move = [f"{k}={getattr(c1, k)}" for k in FIELDS if getattr(c1, k) != getattr(c0, k)]
        steps.append({"move": move, "paired_diff": t["paired_diff"], "accepted": t["accepted"]})
    rec_name = r["recommendation"]
    row = {"run": d.name, "method": r["method"], "seed": r["args"]["seed"], "spent": r["spent"], "scored_episodes": n,
           "engine_error_records": errors, "engine_errors_retried": r["engine_errors"], "exit_reasons": dict(ends),
           "past_deadline": int(late), "median_s": float(np.median(secs)), "distinct_tasks": len(tasks),
           "graphs_run": len(r["pulls"]), "steps": len(steps), "accepted": sum(s["accepted"] for s in steps),
           "recommendation": rec_name, "changed_from_plain_loop": changed, "has_enforced_check": bool(rec.gate),
           "pulls_of_recommendation": r["pulls"][rec_name], "mean_seen_by_search": r["means"][rec_name],
           "plain_loop_seen": {"pulls": r["pulls"][start.name], "mean": r["means"][start.name]},
           "first_move": steps[0]["move"] if steps else [], "trace": steps}
    runs.append(row)
    by_method[r["method"]].append(row)

print(f"{'run':18s} spent ok err late steps acc graphs  seen (pulls)  recommendation")
for r in runs:
    print(f"{r['run']:18s} {r['spent']:5d} {r['scored_episodes']:4d} {r['engine_error_records']:3d} {r['past_deadline']:4d} "
          f"{r['steps']:5d} {r['accepted']:3d} {r['graphs_run']:6d}  {r['mean_seen_by_search']:.3f} ({r['pulls_of_recommendation']:3d})  "
          f"{r['recommendation']}  [{', '.join(r['changed_from_plain_loop'])}]")
summary = {}
for m, rs in by_method.items():
    moves = Counter(mv for r in rs for s in r["trace"] if s["accepted"] for mv in s["move"])
    first = Counter(", ".join(r["first_move"]) for r in rs)
    summary[m] = {"runs": len(rs), "with_enforced_check": sum(r["has_enforced_check"] for r in rs),
                  "distinct_recommendations": len({r["recommendation"] for r in rs}),
                  "accepted_per_run": [r["accepted"] for r in rs], "steps_per_run": [r["steps"] for r in rs],
                  "accepted_moves": dict(moves.most_common()), "first_moves": dict(first),
                  "exit_reasons": dict(sum((Counter(r["exit_reasons"]) for r in rs), Counter()))}
    print(f"\n{m}: {summary[m]}")
if a.out:
    Path(a.out).write_text(json.dumps({"runs": runs, "by_method": summary}, indent=1))
