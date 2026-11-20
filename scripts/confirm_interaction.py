"""Stage 5, exploratory (added after the confirmation results were seen; not a frozen contrast).

How much larger is a contrast with visible tests than with hidden tests?
    (a - b with visible tests) - (a - b with hidden tests)
on the confirmation tasks.  Same conventions as scripts/confirm_analysis.py: rollouts are averaged inside a task and
condition, a task is redrawn with both conditions together, inside its repository (4,000 replicates, seed 0).

  python scripts/confirm_interaction.py --visible results/raw/confirm/r0/visible/qwen3.8-27b.jsonl \
      results/raw/confirm/r1/visible/qwen3.8-27b.jsonl --hidden results/raw/confirm/h_a/hidden/qwen3.8-27b.jsonl \
      results/raw/confirm/h_b/hidden/qwen3.8-27b.jsonl --tasks data/v2/confirm.jsonl \
      --selection results/summary/selection.json --out results/summary/confirmation_exploratory.json
"""
import argparse
import json
from pathlib import Path

from harness_graph import analysis as A
from harness_graph.grammar import Config

H01, H04 = Config().name, Config(gate=True).name

p = argparse.ArgumentParser()
p.add_argument("--visible", nargs="+", required=True)
p.add_argument("--hidden", nargs="+", required=True)
p.add_argument("--tasks", required=True)
p.add_argument("--selection", required=True)
p.add_argument("--reps", type=int, default=4000)
p.add_argument("--boot-seed", type=int, default=0)
p.add_argument("--out", default="")
a = p.parse_args()

tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in open(a.tasks) if l.strip()}
repo_of = {t: d["repo_key"] for t, d in tasks.items()}
searched = json.loads(Path(a.selection).read_text())["searched_graph"]
label = {H01: "h01", H04: "h04", searched: "searched"}
order = sorted(tasks)
idx = A.strat_index(order, repo_of, a.reps, a.boot_seed)
res = {c: A.scored(A.load_episodes(*files)).pivot_table(index="task_id", columns="graph", values="resolved",
                                                        aggfunc="mean").loc[order]
       for c, files in (("visible", a.visible), ("hidden", a.hidden))}
out = {"note": "exploratory: computed after the confirmation results were seen; not a frozen contrast",
       "n_tasks": len(order), "reps": a.reps, "boot_seed": a.boot_seed, "visible_minus_hidden": {}}
for x, y in [(H04, H01), (searched, H01), (searched, H04)]:
    d = A.paired_diff((res["visible"][x] - res["visible"][y]) - (res["hidden"][x] - res["hidden"][y]), idx)
    out["visible_minus_hidden"][f"{label[x]} - {label[y]}"] = d
    print(f"({label[x]} - {label[y]}) visible minus hidden: {d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")
if a.out:
    Path(a.out).write_text(json.dumps(out, indent=1))
