"""E2 read-out: which part of check-and-retry carries its gain?  Four arms on the same tasks, tests visible, 50 calls.

  python scripts/e2_analysis.py [--runs ...] [--tasks data/a1/tasks.jsonl] [--out results/summary/e2_decomposition.json]

  plain loop            the agent's hand-in is submitted
  full gate             check-and-retry: failed check, its diagnostic text goes back, at most two retries
  binary gate           the same check; on failure the agent is told only that it failed
  forced continuation   no check; the agent is sent back twice whatever it has done
Primary: full gate minus binary gate in full-grader repair (the effect of automatically supplied diagnostic text).
Secondary: binary minus forced, forced minus plain, full minus plain.  Seeds averaged within task, tasks redrawn inside
repositories.  The caps are equal; realized calls and tokens are reported and are not equal.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_analysis import boot, load  # noqa: E402

ARMS = {"plain_loop": "g-p0-fn-t0-vn-k0-c0-snon", "full_gate": "g-p0-fn-t0-vn-kcon2-c0-snon",
        "binary_gate": "e2-binary-gate", "forced_continuation": "e2-forced-continue"}
CONTRASTS = [("full_gate", "binary_gate", "primary"), ("binary_gate", "forced_continuation", "secondary"),
             ("forced_continuation", "plain_loop", "secondary"), ("full_gate", "plain_loop", "secondary")]
MODEL, SEEDS, CAP, WALL = "qwen3.8-27b", (4, 5), 50, 3600
RAW = str(ROOT / "results" / "raw" / "e2" / "s{}" / "visible" / "qwen3.8-27b.jsonl")


def main(a):
    listed = [json.loads(l) for l in open(a.tasks) if l.strip()]
    tasks = {t["instance_id"]: t["repo_key"] for t in listed}
    if len(tasks) != len(listed):
        raise SystemExit(f"{len(listed) - len(tasks)} repeated task ids in {a.tasks}")
    names = {g: n for n, g in ARMS.items()}
    cells, problems, errors = {}, [], 0
    for r in load(a.runs):
        key = (r.get("task_id"), names.get(r.get("graph")), r.get("seed"))
        b = r.get("budget") or {}
        if key[0] not in tasks or key[1] is None or key[2] not in SEEDS or r.get("model") != MODEL or r.get("tests") != "visible":
            problems.append(f"foreign record {r.get('episode_id')}")
        elif r.get("status") != "ok":
            errors += 1
        elif b.get("model_calls") != CAP or b.get("wall_s") != WALL or "snapshot_at" in b:
            problems.append(f"wrong budget {r['episode_id']}")
        elif key in cells:
            problems.append(f"two scored records {key}")
        else:
            cells[key] = r
    problems += [f"missing {(t, n, s)}" for t in tasks for n in ARMS for s in SEEDS if (t, n, s) not in cells]
    out = {"note": "attribution control of the check-and-retry package; not a complete factorial decomposition",
           "arms": ARMS, "model": MODEL, "seeds": list(SEEDS), "cap": CAP, "tasks": len(tasks),
           "stored_record_problems": problems[:50], "n_problems": len(problems), "engine_error_records_rerun": errors}
    if problems:
        out["verdict"] = "incomplete"
    else:
        per = {n: {t: float(np.mean([bool(cells[(t, n, s)]["resolved"]) for s in SEEDS])) for t in tasks} for n in ARMS}
        out["repair"] = {n: boot(per[n], tasks) for n in ARMS}
        out["contrasts"] = {}
        for x, y, role in CONTRASTS:
            d = boot({t: per[x][t] - per[y][t] for t in tasks}, tasks)
            d.update(role=role, tasks_better_worse=[sum(per[x][t] > per[y][t] for t in tasks), sum(per[x][t] < per[y][t] for t in tasks)],
                     reading="benefit supported" if d["lo"] > 0 else "harm supported" if d["hi"] < 0 else "direction unresolved")
            out["contrasts"][f"{x}_minus_{y}"] = d
        runs = lambda n: [cells[(t, n, s)] for t in tasks for s in SEEDS]
        checks = lambda r: [l for l in r.get("node_logs", []) if l.get("node") == "check"]
        out["per_run"] = {n: {"model_calls": float(np.mean([r["model_calls"] for r in runs(n)])),
                              "tokens": float(np.mean([r["prompt_tokens"] + r["completion_tokens"] for r in runs(n)])),
                              "ended_by": dict(Counter(r["exit_reason"] for r in runs(n))),
                              "check_visits": float(np.mean([len(checks(r)) for r in runs(n)])),
                              "runs_sent_back_at_least_once": sum(any(l.get("outcome") == "fail" for l in checks(r)) for r in runs(n)),
                              "checks_passed_without_consulting_the_tests": sum(l.get("summary") == "pass_untested" for r in runs(n) for l in checks(r))}
                          for n in ARMS}
        out["verdict"] = "complete"
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=[RAW.format(s) for s in SEEDS])
    p.add_argument("--tasks", default=str(ROOT / "data" / "a1" / "tasks.jsonl"))
    p.add_argument("--out", default=str(ROOT / "results" / "summary" / "e2_decomposition.json"))
    main(p.parse_args())
