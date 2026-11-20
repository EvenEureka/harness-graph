"""Stage 5: the confirmation contrasts.  No model call.

Frozen in refine-logs/EXPERIMENT_PLAN.md before any confirmation task was run:
  - every (graph, task, rollout) cell must hold exactly one scored record (visible rollouts 0 and 1, hidden rollout 0);
  - rollouts are averaged inside a task; tasks are redrawn with replacement inside repositories, the same draws for
    every graph and both conditions (4,000 replicates, seed 0), 95% percentile intervals;
  - primary contrast: searched graph - check-and-retry (h04), visible tests, all confirmation tasks.  Two verdicts
    are reported separately: "better" (lower bound > 0) and "no gain of 5 points" (upper bound < 0.05);
  - secondary: h04 - h01 and searched - h01; the same contrasts with hidden tests; each contrast inside the
    repositories that also supplied search tasks ("old") and inside those that did not ("new"); model calls and
    tokens of each pair.

  python scripts/confirm_analysis.py --visible results/raw/confirm/r0/visible/qwen3.8-27b.jsonl \
      results/raw/confirm/r1/visible/qwen3.8-27b.jsonl --hidden results/raw/confirm/h_a/hidden/qwen3.8-27b.jsonl \
      results/raw/confirm/h_b/hidden/qwen3.8-27b.jsonl --tasks data/v2/confirm.jsonl --search-tasks data/v2/search.jsonl \
      --selection results/summary/selection.json --out results/summary/confirmation.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from harness_graph import analysis as A
from harness_graph.grammar import Config

H01, H04 = Config().name, Config(gate=True).name


def main(a):
    tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in open(a.tasks) if l.strip()}
    repo_of = {t: d["repo_key"] for t, d in tasks.items()}
    old_repos = {json.loads(l)["repo_key"] for l in open(a.search_tasks) if l.strip()}
    searched = json.loads(Path(a.selection).read_text())["searched_graph"]
    graphs = list(dict.fromkeys([H01, H04, searched]))
    seeds_of = {"visible": [int(x) for x in a.visible_seeds.split(":")], "hidden": [int(x) for x in a.hidden_seeds.split(":")]}
    files = {"visible": a.visible, "hidden": a.hidden}
    try:
        cells = {c: A.audit_cells(files[c], a.model, graphs, list(tasks), {c: seeds_of[c]}) for c in files}
    except ValueError as e:
        raise SystemExit(str(e))
    if any(v["records_of_other_graphs"] for v in cells.values()):
        raise SystemExit(f"records of graphs other than {graphs}: {cells}")
    print(f"cells: {cells}")

    order = sorted(tasks)
    groups = {"all": order, "old_repositories": [t for t in order if repo_of[t] in old_repos],
              "new_repositories": [t for t in order if repo_of[t] not in old_repos]}
    idx = {g: A.strat_index(ts, repo_of, a.reps, a.boot_seed) for g, ts in groups.items()}
    pairs = [(searched, H04), (H04, H01), (searched, H01)] if searched != H04 else [(H04, H01)]
    label = {H01: "h01", H04: "h04", searched: "searched"} if searched != H04 else {H01: "h01", H04: "h04"}
    out = {"model": a.model, "searched_graph": searched, "searched_is_h04": searched == H04, "n_tasks": len(order),
           "repositories": {"old": sorted({repo_of[t] for t in groups["old_repositories"]}),
                            "new": sorted({repo_of[t] for t in groups["new_repositories"]})},
           "n_tasks_by_group": {g: len(ts) for g, ts in groups.items()}, "rollouts": seeds_of, "cells": cells,
           "reps": a.reps, "boot_seed": a.boot_seed, "sesoi": a.sesoi, "conditions": {}}
    for c in ("visible", "hidden"):
        df = A.scored(A.load_episodes(*files[c]))
        m = {col: df.pivot_table(index="task_id", columns="graph", values=col, aggfunc="mean") for col in ("resolved", "model_calls", "tokens")}
        by_seed = df.pivot_table(index="graph", columns="seed", values="resolved", aggfunc="mean")
        res = {"graphs": {label[g]: {"graph": g, "resolved": float(m["resolved"].loc[order, g].mean()),
                                     "by_rollout": {int(s): float(by_seed.loc[g, s]) for s in seeds_of[c]},
                                     "model_calls": float(m["model_calls"].loc[order, g].mean()),
                                     "tokens": float(m["tokens"].loc[order, g].mean()),
                                     "exit_reasons": dict(Counter(df[df.graph == g].exit_reason)),
                                     "past_deadline": int((df[df.graph == g].latency_s > a.deadline).sum())} for g in graphs},
               "contrasts": {}}
        for x, y in pairs:
            name = f"{label[x]} - {label[y]}"
            res["contrasts"][name] = {}
            for g, ts in groups.items():
                d = A.paired_diff(m["resolved"].loc[ts, x] - m["resolved"].loc[ts, y], idx[g])
                d.update(rate_a=float(m["resolved"].loc[ts, x].mean()), rate_b=float(m["resolved"].loc[ts, y].mean()),
                         better_lower_bound_above_zero=d["lo"] > 0, no_gain_of_sesoi=d["hi"] < a.sesoi)
                if g == "all":
                    d["model_calls"] = A.paired_ratio(m["model_calls"].loc[ts, x], m["model_calls"].loc[ts, y], idx[g])
                    d["tokens"] = A.paired_ratio(m["tokens"].loc[ts, x], m["tokens"].loc[ts, y], idx[g])
                res["contrasts"][name][g] = d
        out["conditions"][c] = res
    if searched != H04:
        p = out["conditions"]["visible"]["contrasts"]["searched - h04"]["all"]
        out["primary"] = {"contrast": "searched - h04, visible tests, all confirmation tasks", **{k: p[k] for k in (
            "mean", "lo", "hi", "better", "worse", "n_tasks", "better_lower_bound_above_zero", "no_gain_of_sesoi")}}
    else:
        out["primary"] = {"contrast": "searched - h04", "note": "the searched graph is h04: zero by construction"}

    for c, res in out["conditions"].items():
        print(f"\n== tests {c}")
        for k, g in res["graphs"].items():
            print(f"  {k:9s} {g['resolved']:.3f}  calls {g['model_calls']:5.1f}  tokens {g['tokens']:9.0f}  "
                  f"past deadline {g['past_deadline']}  {g['graph']}")
        for name, by in res["contrasts"].items():
            for g, d in by.items():
                print(f"  {name:16s} {g:17s} {d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]  {d['better']}/{d['worse']} of {d['n_tasks']}"
                      f"  better={d['better_lower_bound_above_zero']} no_gain_5={d['no_gain_of_sesoi']}")
    print(f"\nprimary: {out['primary']}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--visible", nargs="+", required=True)
    p.add_argument("--hidden", nargs="+", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--search-tasks", required=True, help="repositories in this file count as old")
    p.add_argument("--selection", required=True)
    p.add_argument("--model", default="qwen3.8-27b")
    p.add_argument("--visible-seeds", default="0:1")
    p.add_argument("--hidden-seeds", default="0")
    p.add_argument("--reps", type=int, default=4000)
    p.add_argument("--boot-seed", type=int, default=0)
    p.add_argument("--sesoi", type=float, default=0.05)
    p.add_argument("--deadline", type=float, default=3600)
    p.add_argument("--out", default="")
    main(p.parse_args())
