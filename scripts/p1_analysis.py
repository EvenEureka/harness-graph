"""P1, step 4: how well do the generated checks separate stored wrong patches from correct trees?  No model call.

Frozen in refine-logs/EXPERIMENT_PLAN.md (section P1) before any generator run.  Every number is a mean over tasks
of a per-task value (the task's mean over its generation seeds and its patches); tasks are redrawn with replacement
inside repositories (4,000 replicates, seed 0) for 95% percentile intervals.

  validity                share of (task, seed) that delivered a check
  detection               a wrong patch is detected if a delivered check rejects it; no check = not detected
  false rejection,        among delivered checks: share of reference runs (two per check) rejected; tasks with no
    reference             delivered check are left out and counted
  false rejection,        among delivered checks: share of grader-accepted patches rejected
    accepted patches
A rejection is "behavioural" when one of the check's own asserts failed, otherwise "crash" (invalid, time-out).

Verdict: pass if detection >= 0.30, reference false rejection <= 0.02 and accepted-patch false rejection <= 0.05
on point estimates and each conditional rate rests on at least MIN_TASKS tasks.  Inconclusive whenever a stored
record is missing, foreign, made with another check, or a patch did not apply (no verdict from partial data), and
when the thresholds are met but a conditional denominator is too small.  Fail otherwise.

  python scripts/p1_analysis.py --gen results/raw/p1/gen_G1.jsonl --eval results/raw/p1/eval_G1.jsonl \
      --tasks data/v2/search.jsonl --patches data/p1/patches.jsonl --audit data/spec_audit.jsonl \
      --out results/summary/p1_G1.json
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from harness_graph import analysis as A

DETECT_MIN, REF_MAX, ACC_MAX, MIN_TASKS, SEEDS, REF_RUNS = 0.30, 0.02, 0.05, 94, (0, 1), 2
SINGLE = {"func_basic", "func_pm_ctrl_shuffle", "func_pm_remove_cond", "func_pm_class_rm_funcs", "func_pm_remove_loop",
          "func_pm_op_swap", "func_pm_ctrl_invert_if", "func_pm_remove_assign", "func_pm_remove_wrapper",
          "func_pm_op_change", "func_pm_op_break_chains", "func_pm_op_change_const", "func_pm_class_rm_base",
          "func_pm_class_shuffle_funcs"}


def family(s: str) -> str:
    if s in ("combine_module", "combine_file"):
        return "combine strategies"
    if s == "lm_rewrite":
        return "lm_rewrite"
    if s.startswith("pr_"):
        return "pr"
    if s in SINGLE:
        return "single-function strategies"
    raise SystemExit(f"strategy {s!r} is in no family")


def read(path):
    out = []
    for line in open(path):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def mean_ci(values: dict, repo_of: dict, reps: int, seed: int) -> dict:
    """Mean over tasks of one value per task, with a percentile interval from redrawing tasks inside repositories."""
    order = sorted(values)
    if not order:
        return {"mean": None, "lo": None, "hi": None, "n_tasks": 0}
    v = np.array([values[t] for t in order], dtype=float)
    idx = A.strat_index(order, repo_of, reps, seed)
    lo, hi = np.percentile(v[idx].mean(1), [2.5, 97.5])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "n_tasks": len(order)}


def build(gen_rows, eval_rows, tasks, patches, runner_of=""):
    """-> (checks by (task, seed), runs by (task, seed, tree, patch id, rep), problems found in the stored records)."""
    problems = []
    ok = defaultdict(list)
    for e in gen_rows:
        if e.get("status") == "ok":
            ok[(e["task_id"], e["seed"])].append(e)
    checks = {}
    for t in tasks:
        for s in SEEDS:
            rows = ok.get((t, s), [])
            if len(rows) != 1:
                problems.append(f"generator cell ({t}, seed {s}) holds {len(rows)} scored records")
            if rows:
                checks[(t, s)] = rows[-1]
    foreign = [k for k in ok if k[0] not in tasks or k[1] not in SEEDS]
    if foreign:
        problems.append(f"{len(foreign)} generator records outside the cells")
    from harness_graph import gencheck as G
    shas = {r.get("runner_sha") for r in eval_rows}
    versions = {runner_of} if runner_of else {c["generator"] for c in checks.values()}
    if shas != {None}:                           # files written before runner hashes were stored carry none at all
        if len(versions) != 1 or shas != {G.runner_sha(next(iter(versions)))}:
            problems.append(f"evaluation runs were not all made with the expected runner: {sorted(map(str, shas))}")
    runs, infra = {}, Counter()
    expected = {(t, s, "unrepaired", "", 0) for (t, s), c in checks.items() if c["valid"]} | \
               {(t, s, "reference", "", r) for (t, s), c in checks.items() if c["valid"] for r in range(REF_RUNS)} | \
               {(t, s, "patch", p["patch_id"], 0) for (t, s), c in checks.items() if c["valid"] for p in patches.get(t, [])}
    for r in eval_rows:
        key = (r["task_id"], r["seed"], r["tree"], r["patch_id"], r["rep"])
        if key not in expected:
            problems.append(f"run {key} belongs to no delivered check of this analysis")
            continue
        c = checks[key[:2]]
        if r["episode_id"] != c["episode_id"] or r.get("artifact_sha") != hashlib.sha256(c["artifact"].encode()).hexdigest()[:16]:
            problems.append(f"run {key} was made with another check than the one stored for its generator episode")
            continue
        if r.get("status") == "ok":
            if key in runs:
                problems.append(f"run {key} is stored twice")
            runs[key] = r
            if not r["applied"]:
                problems.append(f"run {key}: the patch did not apply ({str(r.get('apply'))[:80]})")
        else:
            infra[key] += 1
    for (t, s), c in checks.items():
        if not c["valid"]:
            continue
        want = [(t, s, "unrepaired", "", 0)] + [(t, s, "reference", "", r) for r in range(REF_RUNS)] + \
               [(t, s, "patch", p["patch_id"], 0) for p in patches.get(t, [])]
        missing = [k for k in want if k not in runs]
        if missing:
            problems.append(f"check ({t}, seed {s}): {len(missing)} of {len(want)} runs missing"
                            f" ({sum(k in infra for k in missing)} after infrastructure errors)")
    return checks, runs, problems


def measures(checks, runs, task_ids, patches, repo_of, reps, seed, only_patches=lambda p: True):
    """All P1 measures on the given tasks.  ``only_patches`` restricts the stored patches (for breakdowns)."""
    rej = lambda r: r["verdict"] != "fixed"
    behav = lambda r: r["verdict"] == "problem_present"
    valid = {t: float(np.mean([bool(checks.get((t, s), {}).get("valid")) for s in SEEDS])) for t in task_ids}
    first = {t: float(np.mean([bool(checks.get((t, s), {}).get("first_valid")) for s in SEEDS])) for t in task_ids}
    out = {"validity": mean_ci(valid, repo_of, reps, seed), "validity_at_first_validation": mean_ci(first, repo_of, reps, seed)}
    not_applied = Counter()
    for pool, name in (("primary", "detection_primary"), ("secondary", "detection_secondary"), ("correct", "false_rejection_accepted")):
        any_, beh, crash = {}, {}, {}
        for t in task_ids:
            ps = [p for p in patches.get(t, []) if p["pool"] == pool and only_patches(p)]
            vals_any, vals_beh = [], []
            for p in ps:
                got = {s: runs.get((t, s, "patch", p["patch_id"], 0)) for s in SEEDS}
                if any(r is not None and not r["applied"] for r in got.values()):
                    not_applied[f"{pool}:{next(r['apply'] for r in got.values() if r is not None and not r['applied'])[:40]}"] += 1
                    continue
                for s in SEEDS:
                    delivered = bool(checks.get((t, s), {}).get("valid"))
                    if pool == "correct":
                        if delivered and got[s] is not None:                 # conditional on a delivered check
                            vals_any.append(rej(got[s]))
                            vals_beh.append(behav(got[s]))
                    else:                                                    # no check counts as not detected
                        r = got[s] if delivered else None
                        vals_any.append(bool(r is not None and rej(r)))
                        vals_beh.append(bool(r is not None and behav(r)))
            if vals_any:
                any_[t], beh[t] = float(np.mean(vals_any)), float(np.mean(vals_beh))
                crash[t] = any_[t] - beh[t]
        out[name] = {**mean_ci(any_, repo_of, reps, seed), "behavioural": mean_ci(beh, repo_of, reps, seed)["mean"],
                     "crash_or_timeout": mean_ci(crash, repo_of, reps, seed)["mean"]}
        if pool == "primary":
            out["validity_on_primary_tasks"] = mean_ci({t: valid[t] for t in any_}, repo_of, reps, seed)
    ref_c, ref_u, ref_any, ref_dis, unrep = {}, {}, {}, {}, {}
    for t in task_ids:
        per, any_r, dis, un = [], [], [], []
        for s in SEEDS:
            if not checks.get((t, s), {}).get("valid"):
                continue
            rr = [runs.get((t, s, "reference", "", r)) for r in range(REF_RUNS)]
            if any(r is None for r in rr):
                continue
            x = [rej(r) for r in rr]
            per.append(float(np.mean(x)))
            any_r.append(any(x))
            dis.append(len(set(x)) > 1)
            u = runs.get((t, s, "unrepaired", "", 0))
            if u is not None:
                un.append(u["verdict"] == "problem_present")
        ref_u[t] = float(np.sum(per)) / len(SEEDS)                          # per generation attempt: no check = no rejection
        if per:
            ref_c[t], ref_any[t], ref_dis[t] = float(np.mean(per)), float(np.mean(any_r)), float(np.mean(dis))
        if un:
            unrep[t] = float(np.mean(un))
    out["false_rejection_reference"] = {**mean_ci(ref_c, repo_of, reps, seed),
                                        "any_of_two_runs": mean_ci(ref_any, repo_of, reps, seed)["mean"],
                                        "two_runs_disagree": mean_ci(ref_dis, repo_of, reps, seed)["mean"]}
    out["false_rejection_reference_per_attempt"] = mean_ci(ref_u, repo_of, reps, seed)
    out["still_rejects_unrepaired_tree"] = mean_ci(unrep, repo_of, reps, seed)
    out["patches_left_out_not_applied"] = dict(not_applied)
    return out


def verdict(m: dict, problems: list) -> dict:
    d, r, c = m["detection_primary"], m["false_rejection_reference"], m["false_rejection_accepted"]
    met = {"detection_at_least_0.30": d["mean"] is not None and d["mean"] >= DETECT_MIN,
           "reference_rejection_at_most_0.02": r["mean"] is not None and r["mean"] <= REF_MAX,
           "accepted_patch_rejection_at_most_0.05": c["mean"] is not None and c["mean"] <= ACC_MAX}
    enough = {"reference": r["n_tasks"] >= MIN_TASKS, "accepted_patches": c["n_tasks"] >= MIN_TASKS}
    if problems:
        v = "inconclusive"                       # missing, foreign or unusable records: no verdict from partial data
    elif not met["detection_at_least_0.30"]:
        v = "fail"                               # too few wrong patches caught, whatever the other rates are
    elif not all(enough.values()):
        v = "inconclusive" if (met["reference_rejection_at_most_0.02"] or not enough["reference"]) and \
            (met["accepted_patch_rejection_at_most_0.05"] or not enough["accepted_patches"]) else "fail"
    else:
        v = "pass" if all(met.values()) else "fail"
    return {"verdict": v, "thresholds_met": met, "denominators_at_least_%d_tasks" % MIN_TASKS: enough,
            "stored_record_problems": len(problems)}


def main(a):
    tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in open(a.tasks) if l.strip()}
    repo_of = {t: d["repo_key"] for t, d in tasks.items()}
    patches = defaultdict(list)
    for p in read(a.patches):
        patches[p["task_id"]].append(p)
    gen_rows = read(a.gen)
    checks, runs, problems = build(gen_rows, read(a.eval), tasks, patches, a.runner_of)
    ids = sorted(tasks)
    M = lambda ts, only=lambda p: True: measures(checks, runs, ts, patches, repo_of, a.reps, a.boot_seed, only)
    main_ = M(ids)
    label = {json.loads(l)["instance_id"]: json.loads(l)["spec"] for l in open(a.audit) if l.strip()}
    groups = {"issue audit: " + k: [t for t in ids if label.get(t, "not audited") == k] for k in ("full", "partial", "not audited")}
    groups.update({"strategy: " + k: [t for t in ids if family(tasks[t]["strategy"]) == k]
                   for k in sorted({family(d["strategy"]) for d in tasks.values()})})
    cost = [c for c in checks.values()]
    both = [t for t in ids if all(checks.get((t, s), {}).get("valid") for s in SEEDS)]
    rejects = lambda t, s, p: runs[(t, s, "patch", p["patch_id"], 0)]["verdict"] != "fixed"
    agree = [rejects(t, 0, p) == rejects(t, 1, p) for t in both for p in patches.get(t, [])
             if all((t, s, "patch", p["patch_id"], 0) in runs and runs[(t, s, "patch", p["patch_id"], 0)]["applied"] for s in SEEDS)]
    out = {"generator": sorted({c["generator"] for c in checks.values()}), "runner_of": a.runner_of or "the generator's own", "n_tasks": len(ids), "seeds": list(SEEDS),
           "reps": a.reps, "boot_seed": a.boot_seed,
           "thresholds": {"detection_min": DETECT_MIN, "reference_rejection_max": REF_MAX,
                          "accepted_patch_rejection_max": ACC_MAX, "min_tasks_for_a_conditional_rate": MIN_TASKS},
           "stored_record_problems": problems[:50], "measures": main_, "decision": verdict(main_, problems),
           "by_group": {g: M(ts) for g, ts in groups.items() if ts},
           "primary_by_patch_kind": {
               "leaves target tests failing": M(ids, lambda p: p["pool"] != "primary" or p["f2p_fail"] > 0)["detection_primary"],
               "breaks only other tests": M(ids, lambda p: p["pool"] != "primary" or p["f2p_fail"] == 0)["detection_primary"]},
           "two_seeds_agree_on_a_patch": {"share": float(np.mean(agree)) if agree else None, "n_patch_pairs": len(agree),
                                          "tasks_with_two_checks": len(both)},
           "generator_cost": {"episodes": len(cost), "model_calls_mean": float(np.mean([c["model_calls"] for c in cost])) if cost else None,
                              "tokens_mean": float(np.mean([c["prompt_tokens"] + c["completion_tokens"] for c in cost])) if cost else None,
                              "exit_reasons": dict(Counter(c["exit_reason"] for c in cost)),
                              "why_first_validation_failed": dict(Counter(c["validations"][0]["why"] for c in cost
                                                                          if c["validations"] and not c["validations"][0]["valid"])),
                              "changed_the_repository": int(sum(bool(c["repo_changed"]) for c in cost))}}
    f = lambda d: "n/a" if d["mean"] is None else f'{d["mean"]:.3f} [{d["lo"]:.3f}, {d["hi"]:.3f}] ({d["n_tasks"]} tasks)'
    for k in ("validity", "validity_at_first_validation", "validity_on_primary_tasks", "detection_primary", "detection_secondary",
              "false_rejection_reference", "false_rejection_reference_per_attempt", "false_rejection_accepted",
              "still_rejects_unrepaired_tree"):
        extra = "" if "behavioural" not in main_[k] else f'  behavioural {main_[k]["behavioural"]}  crash {main_[k]["crash_or_timeout"]}'
        print(f"{k:40s} {f(main_[k])}{extra}")
    print("left out (patch did not apply):", main_["patches_left_out_not_applied"])
    print("problems in the stored records:", len(problems), problems[:5])
    for g, m in out["by_group"].items():
        print(f"  {g:36s} detection {f(m['detection_primary'])}  reference {f(m['false_rejection_reference'])}")
    print("DECISION:", json.dumps(out["decision"]))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gen", required=True)
    p.add_argument("--eval", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--patches", required=True)
    p.add_argument("--audit", required=True)
    p.add_argument("--runner-of", default="", help="the evaluation used this generator version's runner (sensitivity runs)")
    p.add_argument("--reps", type=int, default=4000)
    p.add_argument("--boot-seed", type=int, default=0)
    p.add_argument("--out", default="")
    main(p.parse_args())
