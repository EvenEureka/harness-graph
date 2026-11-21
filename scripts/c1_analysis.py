"""C1: the selected graph against check-and-retry on combine-strategy tasks of unused repositories.  No model call.

Frozen in refine-logs/EXPERIMENT_PLAN.md (items 16-21) before any C1 episode:
  - cells: selected graph and check-and-retry, rollouts 0 and 1 of every task; plain loop, rollout 0; visible tests;
    exactly one scored record per cell, none outside;
  - primary: selected - check-and-retry, rollouts averaged inside a task, tasks redrawn inside repositories
    (4,000 replicates, seed 0), 95% percentile interval.  Verdicts, each on its own: lower bound > 0 (better on the
    sampled repository mixture), lower bound > 0.05 (a gain above five points), upper bound < 0.05 (a five-point gain
    excluded);
  - secondary, descriptive: check-and-retry rollout 0 - plain loop rollout 0; model calls and tokens of the primary
    pair; the primary contrast by strategy and by repository; with repositories redrawn (all tasks of a drawn
    repository kept, task-weighted mean); with each repository left out in turn.

  python scripts/c1_analysis.py results/raw/c1/*/visible/qwen3.8-27b.jsonl --tasks data/c1/confirm.jsonl \
      --selection results/summary/selection.json --graph-dir graphs_confirm --out results/summary/c1.json
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from harness_graph import analysis as A
from harness_graph.grammar import Config

H01, H04 = Config().name, Config(gate=True).name


def verdicts(d, margin):
    return {"better_lower_bound_above_zero": d["lo"] > 0, "gain_above_margin_lower_bound": d["lo"] > margin,
            "margin_gain_excluded_upper_bound": d["hi"] < margin}


def main(a):
    tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in open(a.tasks) if l.strip()}
    repo_of = {t: d["repo_key"] for t, d in tasks.items()}
    sel = json.loads(Path(a.selection).read_text())["searched_graph"]
    if sel == H04:
        raise SystemExit("the selected graph is check-and-retry: nothing to compare")
    label = {H01: "h01", H04: "h04", sel: "selected"}
    try:
        cells = {"primary_arms": A.audit_cells(a.files, a.model, [H04, sel], list(tasks), {"visible": [0, 1]}),
                 "plain_loop": A.audit_cells(a.files, a.model, [H01], list(tasks), {"visible": [0]})}
    except ValueError as e:
        raise SystemExit(str(e))
    if cells["primary_arms"]["records_of_other_graphs"] != cells["plain_loop"]["scored_records"] + cells["plain_loop"]["engine_error_records"]:
        raise SystemExit(f"records of graphs other than the three of C1: {cells}")
    print(f"cells: {cells}")

    df = A.scored(A.load_episodes(*a.files))
    df = df[df.tests == "visible"]
    order = sorted(tasks)
    two = df[df.graph.isin([H04, sel])]
    m = {c: two.pivot_table(index="task_id", columns="graph", values=c, aggfunc="mean").loc[order] for c in ("resolved", "model_calls", "tokens")}
    r0 = df[df.seed == 0].pivot_table(index="task_id", columns="graph", values="resolved", aggfunc="mean").loc[order]
    idx = A.strat_index(order, repo_of, a.reps, a.boot_seed)
    d = (m["resolved"][sel] - m["resolved"][H04])

    primary = A.paired_diff(d, idx)
    primary.update(rate_selected=float(m["resolved"][sel].mean()), rate_h04=float(m["resolved"][H04].mean()), **verdicts(primary, a.margin))
    by_seed = two.pivot_table(index="graph", columns="seed", values="resolved", aggfunc="mean")
    graphs = {}
    for g in (H01, H04, sel):
        x = df[df.graph == g]
        graphs[label[g]] = {"graph": g, "resolved": float(x.groupby("task_id").resolved.mean().loc[order].mean()),
                            "by_rollout": {int(s): float(v) for s, v in x.groupby("seed").resolved.mean().items()},
                            "model_calls": float(x.groupby("task_id").model_calls.mean().mean()),
                            "tokens": float(x.groupby("task_id").tokens.mean().mean()),
                            "exit_reasons": dict(Counter(x.exit_reason)), "past_deadline": int((x.latency_s > a.deadline).sum())}
    out = {"model": a.model, "selected_graph": sel, "n_tasks": len(order), "repositories": sorted(set(repo_of.values())),
           "strategies": dict(Counter(t["strategy"] for t in tasks.values())), "cells": cells, "reps": a.reps,
           "boot_seed": a.boot_seed, "margin": a.margin, "graphs": graphs,
           "sha256": {**{str(f): hashlib.sha256(Path(f).read_bytes()).hexdigest()[:16] for f in [a.tasks, *sorted(Path(a.graph_dir).glob("*.yaml"))]}},
           "primary": {"contrast": "selected - h04, visible tests, all C1 tasks, rollouts 0 and 1 averaged", **primary},
           "secondary": {}}
    S = out["secondary"]
    k = A.paired_diff(r0[H04] - r0[H01], idx)
    S["h04 - h01, rollout 0"] = {**k, "rate_h04": float(r0[H04].mean()), "rate_h01": float(r0[H01].mean())}
    S["cost selected / h04"] = {c: A.paired_ratio(m[c][sel], m[c][H04], idx) for c in ("model_calls", "tokens")}
    S["by_strategy"], S["by_repository"], S["leave_one_repository_out"] = {}, {}, {}
    for s in sorted(out["strategies"]):
        ts = [t for t in order if tasks[t]["strategy"] == s]
        S["by_strategy"][s] = {**A.paired_diff(d.loc[ts], A.strat_index(ts, repo_of, a.reps, a.boot_seed)),
                               "rate_selected": float(m["resolved"][sel].loc[ts].mean()), "rate_h04": float(m["resolved"][H04].loc[ts].mean())}
    repos = out["repositories"]
    for r in repos:
        ts = [t for t in order if repo_of[t] == r]
        rest = [t for t in order if repo_of[t] != r]
        S["by_repository"][r] = {"n_tasks": len(ts), "mean": float(d.loc[ts].mean()), "better": int((d.loc[ts] > 0).sum()),
                                 "worse": int((d.loc[ts] < 0).sum()), "rate_selected": float(m["resolved"][sel].loc[ts].mean()),
                                 "rate_h04": float(m["resolved"][H04].loc[ts].mean())}
        S["leave_one_repository_out"][r] = A.paired_diff(d.loc[rest], A.strat_index(rest, repo_of, a.reps, a.boot_seed))
    # repositories redrawn with replacement; every task of a drawn repository is kept; task-weighted mean
    rng = np.random.default_rng(a.boot_seed)
    sums = np.array([d.loc[[t for t in order if repo_of[t] == r]].sum() for r in repos])
    sizes = np.array([sum(repo_of[t] == r for t in order) for r in repos])
    draw = rng.integers(0, len(repos), size=(a.reps, len(repos)))
    boot = sums[draw].sum(1) / sizes[draw].sum(1)
    S["repositories_redrawn"] = {"mean": float(d.mean()), "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)),
                                 "n_repositories": len(repos)}

    print(f"\n{len(order)} tasks, {len(repos)} repositories, strategies {out['strategies']}")
    for name, g in graphs.items():
        print(f"  {name:9s} {g['resolved']:.3f}  by rollout {g['by_rollout']}  calls {g['model_calls']:5.1f}  tokens {g['tokens']:9.0f}  "
              f"past deadline {g['past_deadline']}  exits {g['exit_reasons']}")
    p = out["primary"]
    print(f"\nPRIMARY selected - h04: {p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]  {p['better']} better / {p['worse']} worse of {p['n_tasks']}"
          f"\n  better (lower bound > 0): {p['better_lower_bound_above_zero']}   gain above {a.margin:.2f} (lower bound): "
          f"{p['gain_above_margin_lower_bound']}   gain of {a.margin:.2f} excluded (upper bound): {p['margin_gain_excluded_upper_bound']}")
    print(f"h04 - h01, rollout 0: {k['mean']:+.4f} [{k['lo']:+.4f}, {k['hi']:+.4f}]")
    print(f"repositories redrawn: [{S['repositories_redrawn']['lo']:+.4f}, {S['repositories_redrawn']['hi']:+.4f}]")
    for s, v in S["by_strategy"].items():
        print(f"  {s:16s} n={v['n_tasks']:4d} {v['mean']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]")
    for r, v in S["by_repository"].items():
        l = S["leave_one_repository_out"][r]
        print(f"  {r:44s} n={v['n_tasks']:3d} {v['mean']:+.4f} ({v['better']}/{v['worse']})   without it: {l['mean']:+.4f} [{l['lo']:+.4f}, {l['hi']:+.4f}]")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--tasks", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--graph-dir", default="graphs_confirm")
    p.add_argument("--model", default="qwen3.8-27b")
    p.add_argument("--reps", type=int, default=4000)
    p.add_argument("--boot-seed", type=int, default=0)
    p.add_argument("--margin", type=float, default=0.05)
    p.add_argument("--deadline", type=float, default=3600)
    p.add_argument("--out", default="")
    main(p.parse_args())
