"""How the episodes of result files ended: exit reasons, engine errors, episodes past the soft deadline, completeness.

  python scripts/table_health.py FILE... [--deadline 3600]
"""
import argparse
import json
from collections import Counter

p = argparse.ArgumentParser()
p.add_argument("files", nargs="+")
p.add_argument("--deadline", type=float, default=3600)
a = p.parse_args()
ends, cells, late, errors, secs = Counter(), Counter(), [], 0, []
graphs, tasks = set(), set()
for path in a.files:
    for line in open(path):
        e = json.loads(line)
        graphs.add(e["graph"]); tasks.add(e["task_id"])
        if e.get("error"):
            errors += 1
            continue
        ends[e.get("exit_reason", "")] += 1
        cells[(e["graph"], e["task_id"])] += 1
        t = e.get("latency_s") or e.get("wall_s") or 0
        secs.append(t)
        if t > a.deadline:
            late.append((e["graph"], round(t)))
missing = len(graphs) * len(tasks) - len(cells)
secs.sort()
print(f"{len(graphs)} graphs x {len(tasks)} tasks; {sum(ends.values())} scored episodes, {errors} engine errors, "
      f"{missing} cells missing, {sum(v > 1 for v in cells.values())} cells with more than one episode")
print("exit reasons:", dict(ends.most_common()))
if secs:
    print(f"seconds per episode: median {secs[len(secs) // 2]:.0f}, 95th percentile {secs[int(.95 * len(secs))]:.0f}, max {secs[-1]:.0f}")
print(f"past the {a.deadline:.0f} s deadline: {len(late)}", dict(Counter(g for g, _ in late).most_common(8)))
