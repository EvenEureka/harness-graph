"""Exploratory (after stage 5-6): where could a different graph, or a graph per task, still gain?  No model call.

Reads episodes that already exist and answers four questions:
  1. ceiling    - in the 32-graph table, on how many tasks does the choice of graph change the outcome at all?
  2. repeat     - with two runs per (graph, task): do two graphs differ on a task more than one graph differs from
                  itself?  Is the best graph of a task in run 0 still the best in run 1?
  3. failures   - how do the runs that check-and-retry does not solve end, and on which repositories?
  4. hidden gap - tasks solved with visible tests but not with hidden tests, and how those runs end.

  python scripts/headroom_diagnostics.py --out results/summary/headroom_diagnostics.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from harness_graph import analysis as A
from harness_graph.grammar import Config

R = Path("results/raw")
M = "qwen3.8-27b.jsonl"
H01, H04 = Config().name, Config(gate=True).name
p = argparse.ArgumentParser()
p.add_argument("--out", default="")
a = p.parse_args()
out = {"note": "exploratory diagnostics on existing episodes; nothing here is a pre-specified contrast"}


def cube(df, graphs, tasks, seeds):
    """Y[graph, task, run] of 0/1."""
    m = df.pivot_table(index=["graph", "task_id"], columns="seed", values="resolved", aggfunc="mean")
    return np.array([[[float(m.loc[(g, t), s]) for s in seeds] for t in tasks] for g in graphs])


def repeat(Y, graphs, name):
    """Y[g, t, 2].  Disagreement inside a graph (two runs) against disagreement between graphs (same task)."""
    G, T, _ = Y.shape
    within = float(np.mean(Y[:, :, 0] != Y[:, :, 1]))
    between = float(np.mean([np.mean(Y[i, :, 0] != Y[j, :, 1]) for i in range(G) for j in range(G) if i != j]))
    inter = []
    for r in (0, 1):
        y = Y[:, :, r]
        inter.append(y - y.mean(0, keepdims=True) - y.mean(1, keepdims=True) + y.mean())
    corr = float(np.corrcoef(inter[0].ravel(), inter[1].ravel())[0, 1])
    res = {"graphs": G, "tasks": T, "disagreement_same_graph_two_runs": within,
           "disagreement_two_graphs_two_runs": between, "interaction_correlation_between_runs": corr}
    # pick on one run, score on the other
    cv = {"fixed_best": [], "best_per_task": [], "hindsight_per_task": []}
    for tr, te in ((0, 1), (1, 0)):
        order = np.argsort(-Y[:, :, tr].mean(1), kind="stable")           # graphs by their mean in the picking run
        fixed = order[0]
        pick = [next(g for g in order if Y[g, t, tr] == Y[:, t, tr].max()) for t in range(T)]   # ties -> the better graph overall
        cv["fixed_best"].append(float(Y[fixed, :, te].mean()))
        cv["best_per_task"].append(float(np.mean([Y[g, t, te] for t, g in enumerate(pick)])))
        cv["hindsight_per_task"].append(float(Y[:, :, te].max(0).mean()))
    res["pick_on_one_run_score_on_the_other"] = {k: float(np.mean(v)) for k, v in cv.items()}
    # tasks by what the graphs do on them, both runs pooled
    solved = Y.mean(2)                                                    # [g, t] in {0, .5, 1}
    res["tasks_every_graph_solves_in_both_runs"] = int((solved.min(0) == 1).sum())
    res["tasks_no_graph_solves_in_any_run"] = int((solved.max(0) == 0).sum())
    res["tasks_in_between"] = T - res["tasks_every_graph_solves_in_both_runs"] - res["tasks_no_graph_solves_in_any_run"]
    out[name] = res
    print(f"\n== repeat: {name}\n" + json.dumps(res, indent=1))


# 1. ceiling in the stage-1 table (32 graphs x 188 tasks x 1 run, visible tests)
t1 = A.scored(A.load_episodes(R / "search_table/a/visible" / M, R / "search_table/b/visible" / M))
t1 = t1[t1.tests == "visible"]
m = t1.pivot_table(index="task_id", columns="graph", values="resolved", aggfunc="mean").dropna()
gated = [g for g in m.columns if "-k0-" not in g]
n = m.sum(1)
res = {"graphs": m.shape[1], "tasks": m.shape[0], "graphs_with_enforced_check": len(gated),
       "tasks_solved_by_every_graph": int((n == m.shape[1]).sum()), "tasks_solved_by_no_graph": int((n == 0).sum()),
       "tasks_solved_by_every_graph_with_check": int((m[gated].sum(1) == len(gated)).sum()),
       "tasks_solved_by_no_graph_with_check": int((m[gated].sum(1) == 0).sum()),
       "tasks_solved_by_10_to_90_percent_of_graphs": int(((n >= 0.1 * m.shape[1]) & (n <= 0.9 * m.shape[1])).sum()),
       "h01": float(m[H01].mean()), "h04": float(m[H04].mean()), "best_graph": float(m.mean().max()),
       "hindsight_best_per_task": float(m.max(1).mean()),
       "h04_fails_and_some_graph_solves": int(((m[H04] == 0) & (m.max(1) == 1)).sum()),
       "h04_fails_and_at_least_a_quarter_of_graphs_solve": int(((m[H04] == 0) & (n >= 0.25 * m.shape[1])).sum()),
       "solved_count_histogram": {int(k): int(v) for k, v in sorted(Counter(n.astype(int)).items())}}
out["table_ceiling"] = res
print("== ceiling: stage-1 table\n" + json.dumps(res, indent=1))

# 2. repeatability with two runs
sel_tasks = sorted(json.loads(l)["instance_id"] for l in open("data/v2/select.jsonl") if l.strip())
conf = {json.loads(l)["instance_id"]: json.loads(l) for l in open("data/v2/confirm.jsonl") if l.strip()}
conf_tasks = sorted(conf)
sel = A.scored(A.load_episodes(R / "select/r0/visible" / M, R / "select/r1/visible" / M))
sg = sorted(sel.graph.unique())
repeat(cube(sel, sg, sel_tasks, [0, 1]), sg, "selection_14_graphs_56_tasks")
repeat(cube(sel, [g for g in sg if g != H01], sel_tasks, [0, 1]), sg, "selection_13_graphs_with_check_56_tasks")
cv = A.scored(A.load_episodes(R / "confirm/r0/visible" / M, R / "confirm/r1/visible" / M))
cg = sorted(cv.graph.unique())
repeat(cube(cv, cg, conf_tasks, [0, 1]), cg, "confirmation_3_graphs_232_tasks_visible")

# 3. how the unsolved runs end, and where (confirmation, visible tests)
other = next(g for g in cg if g not in (H01, H04))
label = {H01: "h01", H04: "h04", other: "searched"}
fail = {}
for g in cg:
    d = cv[cv.graph == g]
    f = d[~d.resolved.astype(bool)]
    fail[label[g]] = {"runs": len(d), "unsolved": len(f), "unsolved_by_exit": dict(Counter(f.exit_reason)),
                      "solved_by_exit": dict(Counter(d[d.resolved.astype(bool)].exit_reason))}
cv = cv.assign(repo=cv.task_id.map(lambda t: conf[t]["repo_key"].split(".")[0]))
by_repo = cv.pivot_table(index="repo", columns="graph", values="resolved", aggfunc="mean").rename(columns=label)
calls = cv.pivot_table(index="repo", columns="graph", values="model_calls", aggfunc="mean").rename(columns=label)
Yc = cube(cv, cg, conf_tasks, [0, 1])
hard = int((Yc.max((0, 2)) == 0).sum())
out["confirmation_visible_failures"] = {"by_graph": fail, "tasks_no_graph_solves_in_any_run": hard,
                                        "repair_rate_by_repository": by_repo.round(3).to_dict("index"),
                                        "model_calls_by_repository": calls.round(1).to_dict("index")}
print("\n== unsolved runs, confirmation, visible tests\n" + json.dumps(fail, indent=1))
print(f"tasks that no graph solves in any run: {hard} of {len(conf_tasks)}")
print(by_repo.round(3).to_string())

# 4. hidden gap (confirmation)
ch = A.scored(A.load_episodes(R / "confirm/h_a/hidden" / M, R / "confirm/h_b/hidden" / M))
ch = ch.assign(repo=ch.task_id.map(lambda t: conf[t]["repo_key"].split(".")[0]))
vis = cv.pivot_table(index="task_id", columns="graph", values="resolved", aggfunc="mean").loc[conf_tasks]
hid = ch.pivot_table(index="task_id", columns="graph", values="resolved", aggfunc="mean").loc[conf_tasks]
gap = {}
for g in cg:
    d = ch[ch.graph == g]
    f = d[~d.resolved.astype(bool)]
    gap[label[g]] = {"hidden_unsolved": len(f), "hidden_unsolved_by_exit": dict(Counter(f.exit_reason)),
                     "solved_in_both_visible_runs_but_not_hidden": int(((vis[g] == 1) & (hid[g] == 0)).sum()),
                     "solved_hidden_but_in_no_visible_run": int(((vis[g] == 0) & (hid[g] == 1)).sum())}
Yh = np.array([[float(hid.loc[t, g]) for t in conf_tasks] for g in cg])
agree = {"tasks_all_three_solve": int((Yh.min(0) == 1).sum()), "tasks_none_solves": int((Yh.max(0) == 0).sum()),
         "tasks_graphs_disagree": int(((Yh.min(0) == 0) & (Yh.max(0) == 1)).sum()),
         "hindsight_best_per_task": float(Yh.max(0).mean())}
hid_repo = ch.pivot_table(index="repo", columns="graph", values="resolved", aggfunc="mean").rename(columns=label)
out["confirmation_hidden"] = {"by_graph": gap, "agreement": agree, "repair_rate_by_repository": hid_repo.round(3).to_dict("index")}
print("\n== hidden tests, confirmation\n" + json.dumps(gap, indent=1) + "\n" + json.dumps(agree, indent=1))
print(hid_repo.round(3).to_string())
if a.out:
    Path(a.out).write_text(json.dumps(out, indent=1))
