"""E1 infrastructure pilot: does the pipeline do what the protocol says?  No outcome is compared between graphs.

  python scripts/e1_pilot_check.py [--runs results/raw/e1_pilot/visible/qwen3.8-27b.jsonl] [--tasks data/e1/pilot.jsonl]

Counts, over all pilot runs together: missing, repeated or foreign (task, graph, seed) records; runs that ended in an
engine error; graded runs without per-part counts or with counts that disagree with the task; check-node visits whose
log does not say that a baseline was measured; runs in which a command's output names a withheld file path (a path
can be named without the file existing: these are listed for a look).  ``defects`` is the sum that has to be zero.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_analysis import CHECK, PLAIN, split_problem  # noqa: E402

SEEDS = (90, 91)


def main(a):
    tasks = {t["instance_id"]: t for t in (json.loads(l) for l in open(a.tasks) if l.strip())}
    runs = [json.loads(l) for l in open(a.runs) if l.strip()]
    want = {(t, g, s) for t in tasks for g in (PLAIN, CHECK) for s in SEEDS}
    keys = Counter((r.get("task_id"), r.get("graph"), r.get("seed")) for r in runs if r.get("status") == "ok")
    out = Counter({"records": len(runs), "expected runs": len(want), "missing runs": len(want - set(keys)),
                   "foreign runs": sum(k not in want for k in keys), "repeated runs": sum(n > 1 for n in keys.values())})
    named = []
    for r in runs:
        out["status: " + str(r.get("status"))] += 1
        if r.get("status") != "ok" or r["task_id"] not in tasks:
            continue
        t, g = tasks[r["task_id"]], r.get("grade") or {}
        if r["diff"].strip():
            s = g.get("split")
            if not s:
                out["graded without per-part counts"] += 1
            elif split_problem(t, s):
                out["per-part counts disagree with the task"] += 1
            elif s["shown"]["missing"] or s["withheld"]["missing"] or s["p2p"]["missing"]:
                out["a designated test was not run by the grader"] += 1
            out["grader timeouts"] += bool(g.get("timeout"))
        else:
            out["empty diff, not graded"] += 1
        for l in r.get("node_logs", []):
            if l.get("node") == "check":
                out["check visits"] += 1
                out["check visits without a measured baseline"] += not str(l.get("baseline", "")).startswith("measured:")
                out["checks passed without consulting the tests"] += l.get("summary") == "pass_untested"
        seen = [f for f in t["withheld_files"] for e in r.get("events", []) if e.get("role") == "observation" and f in (e.get("content") or "")]
        if seen:
            named.append((r["episode_id"], sorted(set(seen))[:3]))
    bad = ["missing runs", "foreign runs", "repeated runs", "status: engine_error", "graded without per-part counts",
           "per-part counts disagree with the task", "a designated test was not run by the grader", "grader timeouts",
           "check visits without a measured baseline", "checks passed without consulting the tests"]
    res = {"counts": dict(out), "tasks": len(tasks), "runs_whose_command_output_names_a_withheld_file": named[:20],
           "n_such_runs": len(named), "defects": sum(out[k] for k in bad) + len(named)}
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=str(ROOT / "results/raw/e1_pilot/visible/qwen3.8-27b.jsonl"))
    p.add_argument("--tasks", default=str(ROOT / "data/e1/pilot.jsonl"))
    main(p.parse_args())
