"""E1 read-out: does the gain of check-and-retry carry to failing tests the agent never saw?

  python scripts/e1_analysis.py [--runs results/raw/e1/s0/visible/qwen3.8-27b.jsonl results/raw/e1/s1/visible/qwen3.8-27b.jsonl]
                                [--tasks data/e1/tasks.jsonl data/e1/tasks_incomplete.jsonl] [--out results/summary/e1_split_test.json]

Each task shows the agent part of its failing tests and withholds the files of the rest.  Two graphs, two seeds.
Primary (primary block): check-and-retry minus plain loop in the share of tasks whose final patch passes every
withheld failing test and every pass-to-pass test; seeds averaged within task, tasks redrawn inside repositories.
Fixed reading: interval above 0 = positive transfer advantage; upper end below 5 points = a gain of five points is
excluded; upper end at or below 0 = contradicts a positive advantage; otherwise inconclusive.
The incomplete block (no split lets the shown tests detect every component) is described, with no verdict.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_analysis import BOOT_SEED, REPS, boot, load  # noqa: E402

PLAIN, CHECK = "g-p0-fn-t0-vn-k0-c0-snon", "g-p0-fn-t0-vn-kcon2-c0-snon"
MODEL, SEEDS, CAP, WALL, MARGIN = "qwen3.8-27b", (0, 1), 50, 3600, 0.05
RAW = str(ROOT / "results" / "raw" / "e1" / "s{}" / "visible" / "qwen3.8-27b.jsonl")


def split_problem(task: dict, s: dict) -> str | None:
    """Does a stored per-part record agree with the task's manifest and with itself?"""
    gone = set(task["withheld_files"])
    want = {"shown": len(task["shown_f2p"]), "withheld": len(task["withheld_f2p"]), "p2p": len(task["PASS_TO_PASS"]),
            "p2p_available": sum(t.split("::")[0] not in gone for t in task["PASS_TO_PASS"])}
    for k, n in want.items():
        d = s.get(k) or {}
        if d.get("total") != n:
            return f"{k} total {d.get('total')} is not the task's {n}"
        if not (0 <= d["pass"] <= n and 0 <= d["missing"] <= n and d["pass"] + d["missing"] <= n):
            return f"{k} counts out of range"
    full = lambda d: d["total"] > 0 and d["pass"] == d["total"]
    flags = {"shown_f2p_success": full(s["shown"]), "withheld_f2p_success": full(s["withheld"]),
             "withheld_success": full(s["withheld"]) and s["p2p"]["pass"] == s["p2p"]["total"],
             "partial_evaluator_accepts": full(s["shown"]) and s["p2p_available"]["pass"] == s["p2p_available"]["total"]}
    for k, v in flags.items():
        if s.get(k) is not v:
            return f"{k} does not follow from the counts"
    return None


def values(r: dict) -> dict:
    """What one run contributes.  An empty diff is not graded and passes nothing."""
    s = (r.get("grade") or {}).get("split")
    frac = lambda d: d["pass"] / d["total"] if d["total"] else 0.0
    if not s:
        return {"withheld": 0.0, "full": float(bool(r["resolved"])), "shown": 0.0, "withheld_frac": 0.0, "shown_frac": 0.0,
                "accepts": False, "withheld_f2p": False, "graded": False, "missing": 0}
    return {"withheld": float(s["withheld_success"]), "full": float(bool(r["resolved"])), "shown": float(s["shown_f2p_success"]),
            "withheld_frac": frac(s["withheld"]), "shown_frac": frac(s["shown"]), "accepts": bool(s["partial_evaluator_accepts"]),
            "withheld_f2p": bool(s["withheld_f2p_success"]), "graded": True,
            "missing": s["shown"]["missing"] + s["withheld"]["missing"] + s["p2p"]["missing"]}


def ratio_boot(num: dict, den: dict, repo_of: dict) -> dict:
    """Pooled ratio over runs with tasks redrawn inside repositories; a draw with an empty denominator is undefined."""
    by_repo = defaultdict(list)
    for t in den:
        by_repo[repo_of[t]].append((num[t], den[t]))
    groups = [np.array(v, float) for _, v in sorted(by_repo.items())]
    rng, draws, undefined = np.random.default_rng(BOOT_SEED), [], 0
    for _ in range(REPS):
        n = d = 0.0
        for g in groups:
            pick = g[rng.integers(0, len(g), len(g))]
            n, d = n + pick[:, 0].sum(), d + pick[:, 1].sum()
        if d:
            draws.append(n / d)
        else:
            undefined += 1
    N, D = sum(num.values()), sum(den.values())
    return {"rejected": int(N), "accepted": int(D), "rate": N / D if D else None,
            "lo": float(np.percentile(draws, 2.5)) if draws else None, "hi": float(np.percentile(draws, 97.5)) if draws else None,
            "undefined_draws": undefined}


def block(tasks: dict, cells: dict) -> dict:
    repo_of = {t: v["repo_key"] for t, v in tasks.items()}
    per = lambda g, k: {t: float(np.mean([values(cells[(t, g, s)])[k] for s in SEEDS])) for t in tasks}
    out = {"tasks": len(tasks), "runs": 2 * len(SEEDS) * len(tasks)}
    for k, name in (("withheld", "every_withheld_failing_test_and_every_passing_test"), ("full", "full_grader"),
                    ("shown", "every_shown_failing_test"),
                    ("withheld_frac", "share_of_withheld_failing_tests_passed"), ("shown_frac", "share_of_shown_failing_tests_passed")):
        p, c = per(PLAIN, k), per(CHECK, k)
        out[name] = {"plain_loop": boot(p, repo_of), "check_and_retry": boot(c, repo_of),
                     "check_minus_plain": boot({t: c[t] - p[t] for t in tasks}, repo_of),
                     "tasks_better_worse": [sum(c[t] > p[t] for t in tasks), sum(c[t] < p[t] for t in tasks)]}
    rej = {}
    for g, name in ((PLAIN, "plain_loop"), (CHECK, "check_and_retry")):
        v = {t: [values(cells[(t, g, s)]) for s in SEEDS] for t in tasks}
        den = {t: sum(x["accepts"] for x in v[t]) for t in tasks}
        # rejected = an accepted run that does not pass every withheld failing test
        rej[name] = ratio_boot({t: sum(x["accepts"] and not x["withheld_f2p"] for x in v[t]) for t in tasks}, den, repo_of)
        # the same with the primary criterion, which also counts a broken passing test inside a withheld file
        rej[name]["with_passing_tests_of_withheld_files"] = ratio_boot(
            {t: sum(x["accepts"] and not x["withheld"] for x in v[t]) for t in tasks}, den, repo_of)
    out["withheld_test_rejection_among_partial_evaluator_acceptances"] = rej
    runs = lambda g: [cells[(t, g, s)] for t in tasks for s in SEEDS]
    out["cost_per_run"] = {name: {"model_calls": float(np.mean([r["model_calls"] for r in runs(g)])),
                                  "tokens": float(np.mean([r["prompt_tokens"] + r["completion_tokens"] for r in runs(g)])),
                                  "ended_by": dict(Counter(r["exit_reason"] for r in runs(g))),
                                  "not_graded_empty_diff": sum(not values(r)["graded"] for r in runs(g)),
                                  "grader_timeouts": sum(bool((r.get("grade") or {}).get("timeout")) for r in runs(g)),
                                  "graded_runs_with_a_test_the_grader_did_not_run": sum(values(r)["missing"] > 0 for r in runs(g)),
                                  "check_visits": sum(l.get("node") == "check" for r in runs(g) for l in r.get("node_logs", [])),
                                  "check_visits_without_a_measured_baseline": sum(
                                      l.get("node") == "check" and not str(l.get("baseline", "")).startswith("measured:")
                                      for r in runs(g) for l in r.get("node_logs", []))}
                           for g, name in ((PLAIN, "plain_loop"), (CHECK, "check_and_retry"))}
    return out


def reading(d: dict) -> str:
    lo, hi = d["lo"], d["hi"]
    said = []
    if lo > 0:
        said.append("positive transfer advantage supported")
    if hi <= 0:
        said.append("contradicts a positive advantage in this population")
    elif hi < MARGIN:
        said.append("a gain of five points is excluded")
    return "; ".join(said) or "inconclusive"


def main(a):
    tasks = {}
    for f in a.tasks:
        for t in (json.loads(l) for l in open(f) if l.strip()):
            assert t["instance_id"] not in tasks
            tasks[t["instance_id"]] = t
    cells, problems, errors = {}, [], 0
    for r in load(a.runs):
        key = (r.get("task_id"), r.get("graph"), r.get("seed"))
        b = r.get("budget") or {}
        if key[0] not in tasks or key[1] not in (PLAIN, CHECK) or key[2] not in SEEDS or r.get("model") != MODEL or r.get("tests") != "visible":
            problems.append(f"foreign record {r.get('episode_id')}")
        elif r.get("status") != "ok":
            errors += 1
        elif b.get("model_calls") != CAP or b.get("wall_s") != WALL or "snapshot_at" in b:
            problems.append(f"wrong budget {r['episode_id']}")
        elif key in cells:
            problems.append(f"two scored records {key}")
        elif r["diff"].strip() and "split" not in (r.get("grade") or {}):
            problems.append(f"graded without per-part counts {r['episode_id']}")
        elif r["diff"].strip() and split_problem(tasks[key[0]], r["grade"]["split"]):
            problems.append(f"per-part counts: {split_problem(tasks[key[0]], r['grade']['split'])} {r['episode_id']}")
        else:
            cells[key] = r
    problems += [f"missing {(t, g, s)}" for t in tasks for g in (PLAIN, CHECK) for s in SEEDS if (t, g, s) not in cells]
    out = {"note": "validity control of the check-and-retry finding; tasks from the follow-up set, used as development data",
           "graphs": {"plain_loop": PLAIN, "check_and_retry": CHECK}, "model": MODEL, "seeds": list(SEEDS), "cap": CAP,
           "margin": MARGIN, "stored_record_problems": problems[:50], "n_problems": len(problems), "engine_error_records_rerun": errors}
    if problems:
        out["verdict"] = "incomplete"
    else:
        prim = {t: v for t, v in tasks.items() if v["e1"]["block"] == "primary"}
        inc = {t: v for t, v in tasks.items() if v["e1"]["block"] == "incomplete"}
        out["primary_block"] = block(prim, cells)
        d = out["primary_block"]["every_withheld_failing_test_and_every_passing_test"]["check_minus_plain"]
        out["primary"] = {**d, "reading": reading(d)}
        repos = sorted({v["repo_key"] for v in prim.values()})
        loo = []
        for r in repos:
            sub = {t: v for t, v in prim.items() if v["repo_key"] != r}
            loo.append(block(sub, cells)["every_withheld_failing_test_and_every_passing_test"]["check_minus_plain"]["mean"])
        out["primary_leaving_one_repository_out"] = {"min": min(loo), "max": max(loo)}
        full = {t: v for t, v in prim.items() if v["e1"]["withheld_tests_detect_every_component"]}
        if full:
            out["secondary_withheld_tests_detect_every_component"] = block(full, cells)
        if inc:
            out["incomplete_shown_component_detection_block"] = block(inc, cells)
        out["verdict"] = "complete"
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=[RAW.format(s) for s in SEEDS])
    p.add_argument("--tasks", nargs="+", default=[str(ROOT / "data" / "e1" / "tasks.jsonl"), str(ROOT / "data" / "e1" / "tasks_incomplete.jsonl")])
    p.add_argument("--out", default=str(ROOT / "results" / "summary" / "e1_split_test.json"))
    main(p.parse_args())
