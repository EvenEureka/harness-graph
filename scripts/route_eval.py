"""Does choosing the graph from pre-run observables beat one fixed graph?  Offline, on stored episodes.

A unit is one task under one test condition (tests hidden or visible); rollouts of a unit are averaged.

  fit       fit FixedBest, RuleRouter[tests_visible] and FeatureRouter on the given episodes and write them to a
            file.  That file is committed before the tasks that will score it are run.
  score     load such a file and score the three policies on other episodes: value of each policy and every
            pairwise difference, paired task bootstrap inside each repository.
  crossfit  exploration on one data set: fit on half of the tasks and score on the other half, many splits.

  python scripts/route_eval.py fit   SEARCH_HIDDEN.jsonl SEARCH_VISIBLE.jsonl --tasks data/v2/search.jsonl \
      --graphs A:B:C --out results/summary/router_frozen.json
  python scripts/route_eval.py score CONFIRM_HIDDEN.jsonl CONFIRM_VISIBLE.jsonl --tasks data/v2/confirm.jsonl \
      --policies results/summary/router_frozen.json --out results/summary/route_confirm.json

``fit`` and ``score`` stop if a unit lacks one of the graphs (``score --allow-incomplete`` leaves such units out
and lists them).  Rewards are resolve rates; ``--cost-weight w``
subtracts w x (model calls / 50) per episode.
"""
import argparse
import json
from pathlib import Path

from harness_graph import analysis as A
from harness_graph.router import FeatureRouter, FixedBest, Rewards, RuleRouter, crossfit, freeze, held_out, observe, thaw

task_of = lambda u: u.split("|")[0]


def rewards(a, portfolio=None, allow_incomplete=False):
    tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in open(a.tasks) if l.strip()}
    df = A.scored(A.load_episodes(*a.files))
    df = df[df.task_id.isin(tasks)]
    if a.model:
        df = df[df.model == a.model]
    models = sorted(df.model.unique())
    if len(models) != 1:
        raise SystemExit(f"one model per call (use --model); found {models}")
    rows, obs = [], {}
    for r in df.to_dict("records"):
        u = f"{r['task_id']}|{r['tests']}"
        obs[u] = observe(tasks[r["task_id"]], r["tests"] == "visible", models[0])
        rows.append({"unit": u, "graph": r["graph"],
                     "resolved": float(r["resolved"]) - a.cost_weight * r["model_calls"] / 50.0})
    R = Rewards(rows, obs, reward=lambda r: r["resolved"], portfolio=portfolio, allow_incomplete=allow_incomplete)
    n = {c: sum(u.endswith("|" + c) for u in R.units) for c in ("hidden", "visible")}
    print(f"units: {n['hidden']} with hidden tests, {n['visible']} with visible tests")
    return R, models[0], tasks


def show(res):
    for k, v in res.items():
        if isinstance(v, dict) and "lo" in v:
            print(f"  {k:36s} {v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]")


def fit(a):
    R, model, _ = rewards(a, portfolio=a.graphs.split(":"))
    pol = {"fixed_best": FixedBest(), "rule[tests_visible]": RuleRouter("tests_visible"), "feature_router": FeatureRouter()}
    out = {"model": model, "graphs": R.graphs, "fitted_on": {"files": a.files, "tasks": a.tasks, "n_units": len(R.units)},
           "cost_weight": a.cost_weight, "value_on_fit_data": {g: R.value(g, R.units) for g in R.graphs},
           "policies": {name: freeze(p.fit(R, R.units)) for name, p in pol.items()}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"{model}: fitted on {len(R.units)} units, {len(R.graphs)} graphs -> {a.out}")
    print(f"  fixed best: {pol['fixed_best'].graph}; rule: {pol['rule[tests_visible]'].table}")


def score(a):
    frozen = json.loads(Path(a.policies).read_text())
    if a.cost_weight != frozen["cost_weight"]:
        raise SystemExit("--cost-weight differs from the one the policies were fitted with")
    R, model, tasks = rewards(a, portfolio=frozen["graphs"], allow_incomplete=a.allow_incomplete)
    if model != frozen["model"]:
        raise SystemExit(f"policies were fitted on {frozen['model']}, episodes are from {model}")
    fitted = {json.loads(l)["instance_id"] for l in open(frozen["fitted_on"]["tasks"]) if l.strip()}
    if fitted & set(tasks):
        raise SystemExit(f"{len(fitted & set(tasks))} scoring tasks were used to fit the policies")
    res = held_out({n: thaw(d) for n, d in frozen["policies"].items()}, R, group_of=task_of,
                   stratum_of=lambda t: tasks[t]["repo_key"], reps=a.reps)
    res.update(model=model, policies_file=a.policies, graphs=R.graphs, units_left_out=R.dropped,
               graph_values={g: R.value(g, R.units) for g in R.graphs})
    print(f"== {model}: {res['n_units']} units, {res['n_groups']} tasks, {res['n_strata']} repositories; "
          f"{len(R.dropped)} units left out")
    show(res)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def explore(a):
    R, model, _ = rewards(a, portfolio=a.graphs.split(":") if a.graphs else None, allow_incomplete=True)
    if R.dropped:
        print(f"  note: {len(R.dropped)} units without every graph are left out")
    res = crossfit(R, {"rule[tests_visible]": lambda: RuleRouter("tests_visible"),
                       "rule[repo]": lambda: RuleRouter("repo"), "feature_router": FeatureRouter},
                   group_of=task_of, splits=a.splits)
    full = RuleRouter("tests_visible").fit(R, R.units)
    res["rule_table_all_data"] = {str(k): v for k, v in full.table.items()}
    res["n_units"], res["graphs"], res["model"] = len(R.units), R.graphs, model
    print(f"== {model}: {len(R.units)} units, {len(R.graphs)} graphs; rule on all data: {res['rule_table_all_data']}")
    show(res)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("fit", fit), ("score", score), ("crossfit", explore)):
        s = sub.add_parser(name)
        s.add_argument("files", nargs="+")
        s.add_argument("--tasks", required=True)
        s.add_argument("--model", default="")
        s.add_argument("--cost-weight", type=float, default=0.0)
        s.add_argument("--out", default="", required=name == "fit")
        if name != "score":
            s.add_argument("--graphs", default="", required=name == "fit")
        if name == "score":
            s.add_argument("--policies", required=True)
            s.add_argument("--reps", type=int, default=4000)
            s.add_argument("--allow-incomplete", action="store_true",
                           help="leave out units that lack a graph instead of stopping; they are listed in the output")
        if name == "crossfit":
            s.add_argument("--splits", type=int, default=400)
        s.set_defaults(fn=fn)
    a = p.parse_args()
    a.fn(a)
