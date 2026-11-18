"""Paired contrasts between two graphs, and how the agent-callable checker was used.

  python scripts/contrasts.py results/raw/confirm_visible/*.jsonl --tasks data/confirm_tasks.jsonl \
      --pairs h17_checker_tool_gate:h16_checker_tool h04_check_retry:h01_react --seeds 1:2 --out results/summary/x.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from harness_graph import analysis as A
from harness_graph import prompts as P


def checker_use(paths, tasks, seeds):
    """Per (model, graph): share of episodes that called the checker, and that finished right after a passed check."""
    c = defaultdict(lambda: defaultdict(int))
    for path in paths:
        for line in open(path):
            e = json.loads(line)
            if e.get("status") != "ok" or e["task_id"] not in tasks or (seeds and e["seed"] not in seeds):
                continue
            ev = e.get("events", [])
            calls = [i for i, x in enumerate(ev) if x["role"] == "command" and x["content"].strip() == P.CHECK_COMMAND]
            k = c[(e["model"], e.get("tests", "hidden"), e["graph"])]
            k["n"] += 1
            k["called"] += bool(calls)
            k["calls"] += len(calls)
            last_ok = bool(calls) and len(ev) > calls[-1] + 1 and ev[calls[-1] + 1]["content"].startswith(P.CHECK_PASSED[:20])
            k["last_call_passed"] += last_ok
            k["resolved"] += bool(e["resolved"])
            k["resolved_and_last_passed"] += bool(e["resolved"]) and last_ok
            gates = [l for l in e.get("node_logs", []) if l["type"] == "check"]
            k["gate_runs"] += len(gates)
            k["gate_first_fail"] += bool(gates) and gates[0]["outcome"] == "fail"
    return [{"model": m, "tests": t, "graph": g, **dict(v)} for (m, t, g), v in sorted(c.items())]


def main(a):
    tasks = {json.loads(l)["instance_id"] for l in open(a.tasks)}
    seeds = {int(s) for s in a.seeds.split(":")} if a.seeds else set()
    df = A.load_episodes(*a.files)
    df = df[df.task_id.isin(tasks)]
    if seeds:
        df = df[df.seed.isin(seeds)]
    out = {"seeds": sorted(seeds), "contrasts": [], "checker_use": checker_use(a.files, tasks, seeds)}
    for (model, tests), d in df.groupby(["model", "tests"]):
        for pair in a.pairs:
            x, y = pair.split(":")
            if {x, y} <= set(d.graph):
                r = {"model": model, "tests": tests, **A.contrast(d, x, y)}
                out["contrasts"].append(r)
                print(f"{model:15s} {tests:8s} {x} - {y}: n={r['n_tasks']} {r['rate_a']:.3f} vs {r['rate_b']:.3f}  "
                      f"delta {r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]  wins {r['wins']} losses {r['losses']}  "
                      f"errors {r['engine_errors']}  {r['verdict']}")
    for r in out["checker_use"]:
        print(json.dumps(r))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--tasks", required=True)
    p.add_argument("--pairs", nargs="+", required=True, help="graph_a:graph_b")
    p.add_argument("--seeds", default="", help="colon-separated; default all")
    p.add_argument("--out", default="")
    main(p.parse_args())
