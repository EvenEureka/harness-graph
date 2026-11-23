"""A1 (cap-extension audit): does the 50-call cap stop repairs that more calls would complete?  Exploratory.

Check-and-retry is run with a 100-call cap; the diff as it stands after call 50 is kept and graded like a final diff
(Budget.snapshot_at).  Read-out, fixed before the runs:
  primary   repair at the end of the 100-call run minus repair of the state after call 50, per task (mean over seeds),
            then over tasks; 95% interval from redrawing tasks inside repositories (4,000 draws, seed 0);
  also      gains and losses run by run, how many runs reach call 51, how the runs end, calls and tokens after call 50,
            runs that finish solved in each block of ten calls after 50 (finishing is not the moment a correct patch
            first existed), and the stored 50-call runs of the same tasks next to the state after call 50.
Any missing, doubled or foreign record makes the result incomplete.

  python scripts/a1_analysis.py --out results/summary/a1_cap_extension.json      (the other paths have defaults)
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

GRAPH, MODEL, SEEDS, CAP, SNAP, WALL, REPS, BOOT_SEED = "g-p0-fn-t0-vn-kcon2-c0-snon", "qwen3.8-27b", (2, 3), 100, 50, 3600, 4000, 0
STORED_SEEDS, STORED_CAP = (0, 1), 50


def load(files):
    for f in files:
        for line in open(f):
            if line.strip():
                yield json.loads(line)


def boot(per_task: dict, repo_of: dict) -> dict:
    """Mean over tasks of a per-task value, with a 95% interval from redrawing tasks inside repositories."""
    by_repo = defaultdict(list)
    for t, v in per_task.items():
        by_repo[repo_of[t]].append(v)
    groups = [np.array(v, float) for _, v in sorted(by_repo.items())]
    rng = np.random.default_rng(BOOT_SEED)
    n = sum(len(g) for g in groups)
    draws = [sum(g[rng.integers(0, len(g), len(g))].sum() for g in groups) / n for _ in range(REPS)]
    return {"mean": float(np.mean(list(per_task.values()))), "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)), "n_tasks": n}


def main(a):
    tasks, problems, cells, engine_errors = {}, [], {}, 0
    for line in open(a.tasks):
        if line.strip():
            t = json.loads(line)
            if t["instance_id"] in tasks:
                problems.append(f"task listed twice {t['instance_id']}")
            tasks[t["instance_id"]] = t["repo_key"]
    for r in load(a.runs):
        key = (r["task_id"], r["seed"])
        b, sn = r.get("budget") or {}, r.get("snapshot") or {}
        if (r["task_id"] not in tasks or r["seed"] not in SEEDS or r["graph"] != GRAPH or r.get("tests") != "visible"
                or r.get("model") != MODEL):
            problems.append(f"foreign record {r['episode_id']}")
        elif r.get("status") != "ok":
            engine_errors += 1                                 # an engine error is rerun; the scored record decides
        elif b.get("model_calls") != CAP or b.get("snapshot_at") != SNAP or b.get("wall_s") != WALL or sn.get("at") != SNAP:
            problems.append(f"wrong budget or no snapshot {r['episode_id']}")
        elif (not sn["terminal"] and sn["model_calls"] != SNAP) or (sn["terminal"] and (
                sn["model_calls"] != r["model_calls"] or r["model_calls"] > SNAP or bool(sn["resolved"]) != bool(r["resolved"]))):
            problems.append(f"snapshot does not agree with the run {r['episode_id']}")
        elif key in cells:
            problems.append(f"two scored records {key}")
        else:
            cells[key] = r
    problems += [f"missing {(t, s)}" for t in tasks for s in SEEDS if (t, s) not in cells]
    out = {"note": "exploratory audit; nothing is built on it and it has no pass or fail",
           "graph": GRAPH, "model": MODEL, "tasks": len(tasks), "seeds": list(SEEDS), "cap": CAP, "state_kept_after_call": SNAP,
           "stored_record_problems": problems[:50], "n_problems": len(problems), "scored_runs": len(cells),
           "engine_error_records_rerun": engine_errors}
    if problems:
        out["verdict"] = "incomplete"
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
        return
    at50 = {k: bool(r["snapshot"]["resolved"]) for k, r in cells.items()}
    end = {k: bool(r["resolved"]) for k, r in cells.items()}
    per = lambda d: {t: float(np.mean([d[(t, s)] for s in SEEDS])) for t in tasks}
    p50, pend = per(at50), per(end)
    out["repair_of_the_state_after_call_50"] = boot(p50, tasks)
    out["repair_at_the_end_of_the_100_call_run"] = boot(pend, tasks)
    out["primary_end_minus_state_after_call_50"] = boot({t: pend[t] - p50[t] for t in tasks}, tasks)
    tr = Counter(("solved" if at50[k] else "unsolved") + " after call 50 -> " + ("solved" if end[k] else "unsolved") + " at the end" for k in cells)
    out["runs_by_outcome"] = dict(sorted(tr.items()))
    out["tasks_better_worse"] = {"better": sum(pend[t] > p50[t] for t in tasks), "worse": sum(pend[t] < p50[t] for t in tasks)}
    # the boundary: the run had made 50 calls and was about to make another.  The time limit or the context limit can
    # still stop it there, so runs with a completed 51st call are counted separately; all stay in the primary read-out.
    went_on = {k: r for k, r in cells.items() if not r["snapshot"]["terminal"]}
    out["runs_that_reached_the_boundary_after_call_50"] = {
        "n": len(went_on), "share": len(went_on) / len(cells),
        "with_at_least_one_further_model_call": sum(r["model_calls"] > SNAP for r in went_on.values()),
        "stopped_at_the_boundary_by": dict(Counter(r["exit_reason"] for r in went_on.values() if r["model_calls"] == SNAP)),
        "state_after_call_50_already_solved": sum(at50[k] for k in went_on),
        "unsolved_after_call_50": sum(not at50[k] for k in went_on),
        "of_those_solved_at_the_end": sum(end[k] and not at50[k] for k in went_on),
        "solved_after_call_50_but_not_at_the_end": sum(at50[k] and not end[k] for k in went_on),
        "ended_by": dict(Counter(r["exit_reason"] for r in went_on.values())),
        "ended_by_when_unsolved_at_the_end": dict(Counter(r["exit_reason"] for k, r in went_on.items() if not end[k])),
        "model_calls_after_call_50_mean": float(np.mean([r["model_calls"] - SNAP for r in went_on.values()])) if went_on else None,
        "tokens_after_call_50_mean": float(np.mean([r["prompt_tokens"] + r["completion_tokens"] - r["snapshot"]["prompt_tokens"]
                                                    - r["snapshot"]["completion_tokens"] for r in went_on.values()])) if went_on else None,
        "tokens_up_to_call_50_mean": float(np.mean([r["snapshot"]["prompt_tokens"] + r["snapshot"]["completion_tokens"]
                                                    for r in went_on.values()])) if went_on else None,
        "largest_prompt_tokens_mean": float(np.mean([r["max_prompt_tokens"] for r in went_on.values()])) if went_on else None}
    open_ = [r for k, r in went_on.items() if not at50[k]]
    out["unsolved_after_call_50_finishing_solved_by_block"] = {
        f"{c + 1}-{c + 10}": {"not_yet_terminated_and_unsolved_at_the_50_call_snapshot": sum(r["model_calls"] > c for r in open_),
                              "finish_solved": sum(r["resolved"] and c < r["model_calls"] <= c + 10 for r in open_)}
        for c in range(SNAP, CAP, 10)}
    out["all_runs_ended_by"] = dict(Counter(r["exit_reason"] for r in cells.values()))
    # exec_s stops before the final diff is read and graded; latency_s would also count grading
    out["graph_ran_past_the_3600_s_limit"] = sum(r["exec_s"] > WALL for r in cells.values())
    out["seconds_until_the_graph_stopped"] = {"mean": float(np.mean([r["exec_s"] for r in cells.values()])),
                                              "max": float(max(r["exec_s"] for r in cells.values()))}
    if a.stored:
        st = defaultdict(dict)
        for r in load(a.stored):
            if (r["graph"] == GRAPH and r["task_id"] in tasks and r.get("status") == "ok" and r.get("tests") == "visible"
                    and r.get("model") == MODEL and (r.get("budget") or {}).get("model_calls") == STORED_CAP):
                st[r["task_id"]][r["seed"]] = bool(r["resolved"])
        if set(st) == set(tasks) and all(set(v) == set(STORED_SEEDS) for v in st.values()):
            out["stored_50_call_runs_of_the_same_tasks"] = boot({t: float(np.mean(list(v.values()))) for t, v in st.items()}, tasks)
        else:
            out["stored_50_call_runs_of_the_same_tasks"] = "not shown: the stored runs are not exactly seeds 0 and 1 of every task"
    out["verdict"] = "complete"
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    raw = "results/raw/{}/visible/qwen3.8-27b.jsonl"
    p.add_argument("--runs", nargs="+", default=[raw.format("a1/s2"), raw.format("a1/s3")])
    p.add_argument("--tasks", default="data/a1/tasks.jsonl")
    p.add_argument("--stored", nargs="*", default=[raw.format("c1/r0"), raw.format("c1/r1")],
                   help="the stored 50-call runs of the follow-up, shown beside the state after call 50")
    p.add_argument("--out", required=True)
    main(p.parse_args())
