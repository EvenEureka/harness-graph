"""Statements about the panel table (stage 1) with their uncertainty.  No model call.

  1. check-and-retry (h04) minus the plain loop (h01): resolve rate, model calls and tokens, on all search tasks and
     on the tasks no earlier run had used;
  2. headroom of the panel over h04: every other graph minus h04 with a simultaneous upper bound (bootstrap of the
     maximum deviation), so "no graph in the panel is 5 points better" can be checked for all graphs at once;
  3. the same two numbers when the episodes that were replaced after a time-limit exit are put back.
Tasks are redrawn with replacement inside repositories (paired across graphs), 95% percentile intervals.

  python scripts/table_claims.py results/raw/search_table/a/visible/qwen3.8-27b.jsonl \
      results/raw/search_table/b/visible/qwen3.8-27b.jsonl --tasks data/v2/search.jsonl --seen data/tasks.jsonl \
      --replaced results/raw/search_table/discarded_wall_time.jsonl --out results/summary/table_claims.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from harness_graph import analysis as A
from harness_graph.grammar import Config

H01, H04 = Config().name, Config(gate=True).name


strat_index, diff, ratio = A.strat_index, A.paired_diff, A.paired_ratio


def h04_h01(df, tasks, repo_of, a):
    df = df[df.task_id.isin(tasks)]
    m = {c: df.pivot_table(index="task_id", columns="graph", values=c, aggfunc="mean") for c in ("resolved", "model_calls", "tokens")}
    order = sorted(tasks)
    idx = strat_index(order, repo_of, a.reps, a.boot_seed)
    R = m["resolved"].loc[order]
    return {"h04": float(R[H04].mean()), "h01": float(R[H01].mean()), "difference": diff(R[H04] - R[H01], idx),
            "model_calls": {"h04": float(m["model_calls"].loc[order, H04].mean()), "h01": float(m["model_calls"].loc[order, H01].mean()),
                            **ratio(m["model_calls"].loc[order, H04], m["model_calls"].loc[order, H01], idx)},
            "tokens": {"h04": float(m["tokens"].loc[order, H04].mean()), "h01": float(m["tokens"].loc[order, H01].mean()),
                       **ratio(m["tokens"].loc[order, H04], m["tokens"].loc[order, H01], idx)}}


def headroom(df, tasks, repo_of, a):
    order = sorted(tasks)
    R = df.pivot_table(index="task_id", columns="graph", values="resolved", aggfunc="mean").loc[order]
    others = [g for g in R.columns if g != H04]
    D = R[others].to_numpy() - R[[H04]].to_numpy()                 # tasks x graphs
    idx = strat_index(order, repo_of, a.reps, a.boot_seed)
    obs = D.mean(0)
    star = D[idx].mean(1)                                           # reps x graphs
    c = float(np.percentile((obs[None, :] - star).max(1), 95))      # one-sided, simultaneous over the graphs
    top = int(np.argmax(obs))
    lo, hi = np.percentile(star[:, top], [2.5, 97.5])
    return {"n_other_graphs": len(others), "top_graph": others[top], "top_minus_h04": float(obs[top]),
            "top_unadjusted": [float(lo), float(hi)], "simultaneous_margin_95": c,
            "upper_bound_for_every_graph": float(obs.max() + c),
            "graphs_above_h04": int((obs > 0).sum()),
            "each": {g: float(v) for g, v in sorted(zip(others, obs), key=lambda kv: -kv[1])}}


def main(a):
    tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in open(a.tasks) if l.strip()}
    repo_of = {t: d["repo_key"] for t, d in tasks.items()}
    seen = {json.loads(l)["instance_id"] for l in open(a.seen) if l.strip()}
    df = A.scored(A.load_episodes(*a.files))
    df = df[(df.tests == "visible") & df.task_id.isin(tasks)]
    cells = df.groupby(["graph", "task_id"]).size()
    if len(cells) != df.graph.nunique() * len(tasks) or (cells != 1).any():
        raise SystemExit("the table must hold one scored episode per graph and task")
    new = [t for t in tasks if t not in seen]
    out = {"n_tasks": len(tasks), "n_graphs": int(df.graph.nunique()), "n_tasks_never_used_before": len(new),
           "h04_minus_h01": {"all_tasks": h04_h01(df, list(tasks), repo_of, a), "never_used_before": h04_h01(df, new, repo_of, a),
                             "used_before": h04_h01(df, [t for t in tasks if t in seen], repo_of, a)},
           "headroom_over_h04": headroom(df, list(tasks), repo_of, a)}
    if a.replaced:                                  # put the first, time-limited episode of those cells back
        old = A.scored(A.load_episodes(a.replaced))
        old = old[(old.tests == "visible") & old.task_id.isin(tasks)]
        key = set(zip(old.graph, old.task_id))
        keep = df[[(g, t) not in key for g, t in zip(df.graph, df.task_id)]]
        import pandas as pd
        back = pd.concat([keep, old])
        out["with_replaced_episodes_put_back"] = {
            "n_cells": len(key), "resolved_in_originals": int(old.resolved.sum()),
            "resolved_in_reruns": int(df[[(g, t) in key for g, t in zip(df.graph, df.task_id)]].resolved.sum()),
            "h04_minus_h01": h04_h01(back, list(tasks), repo_of, a)["difference"],
            "headroom_over_h04": {k: v for k, v in headroom(back, list(tasks), repo_of, a).items() if k != "each"}}
    print(json.dumps({k: v for k, v in out.items()}, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--tasks", required=True)
    p.add_argument("--seen", required=True, help="tasks used by runs before this study")
    p.add_argument("--replaced", default="")
    p.add_argument("--reps", type=int, default=4000)
    p.add_argument("--boot-seed", type=int, default=0)
    p.add_argument("--out", default="")
    main(p.parse_args())
