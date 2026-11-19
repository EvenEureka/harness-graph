"""Stage 4, step 2: choose the searched graph from the episodes on the selection tasks.  No model call.

Rule (frozen in refine-logs/EXPERIMENT_PLAN.md): rollouts are averaged within a task; the searched graph is the
candidate with the highest mean over the selection tasks; ties go to the graph with fewer model calls per episode,
then to the first graph name in alphabetical order.  Every cell (graph, task, rollout) must hold exactly one scored
episode of the expected model; an episode that ended in an engine error is run again, a scored one never is.
If check-and-retry (h04) wins, the primary contrast of stage 5 is zero by construction.

Also reported, on the same tasks: every candidate minus h04 and minus h01, and the mean value of the graphs each
search method recommended (five live runs per local method, one replay run per panel method), with paired task
bootstraps inside repositories.  The selection tasks were not used by any search, so these values are not inflated
by the search itself; the value of the *chosen* graph is, because it is the maximum of several noisy means.

  python scripts/select_graph.py results/raw/select/*/visible/qwen3.8-27b.jsonl --tasks data/v2/select.jsonl \
      --candidates graphs_select/candidates.json --out results/summary/selection.json
"""
import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from harness_graph import analysis as A
from harness_graph.grammar import Config


def main(a):
    tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in open(a.tasks) if l.strip()}
    cand = json.loads(Path(a.candidates).read_text())
    graphs = list(cand["configs"])
    seeds = [int(s) for s in a.seeds.split(":")]
    raw = Counter()                               # every stored record, before duplicates of an episode are merged
    for path in a.files:
        for line in open(path):
            try:
                e = json.loads(line)
            except json.JSONDecodeError:              # a line cut off by a kill
                continue
            if e.get("tests") == "visible" and e["task_id"] in tasks and e["graph"] in graphs and e["seed"] in seeds:
                raw["ok" if e["status"] == "ok" else "engine_error"] += 1
    df = A.load_episodes(*a.files)
    foreign = df[~(df.task_id.isin(tasks) & df.graph.isin(graphs) & df.seed.isin(seeds) & (df.model == a.model)
                   & (df.tests == "visible"))]
    if len(foreign):
        raise SystemExit(f"{len(foreign)} records are not candidate graphs x selection tasks x rollouts {seeds} "
                         f"of {a.model} with visible tests, e.g. {foreign.episode_id.iloc[0]}")
    bad = df[df.status != "ok"]
    ok = A.scored(df)
    per_cell = ok.groupby(["graph", "task_id", "seed"]).size()
    missing = [(g, t, s) for g in graphs for t in tasks for s in seeds if (g, t, s) not in per_cell.index]
    print(f"{len(graphs)} graphs x {len(tasks)} tasks x {len(seeds)} rollouts: {len(ok)} scored episodes, "
          f"{len(missing)} cells missing, {len(bad)} cells ending in an engine error; "
          f"stored records: {raw['ok']} scored, {raw['engine_error']} engine errors (retried)")
    if missing or len(bad) or (per_cell != 1).any() or len(ok) != len(graphs) * len(tasks) * len(seeds):
        raise SystemExit(f"selection needs exactly one scored episode per cell; missing: {missing[:5]}, "
                         f"engine errors: {list(bad.episode_id[:5])}, cells with several: {int((per_cell != 1).sum())}")

    order = sorted(tasks)
    res = ok.pivot_table(index="task_id", columns="graph", values="resolved", aggfunc="mean").loc[order, graphs]
    calls = ok.groupby("graph").model_calls.mean()
    tokens = ok.groupby("graph").tokens.mean()
    by_seed = ok.pivot_table(index="graph", columns="seed", values="resolved", aggfunc="mean")
    h01, h04 = Config().name, Config(gate=True).name

    strata = defaultdict(list)
    for i, t in enumerate(order):
        strata[tasks[t]["repo_key"]].append(i)
    rng = np.random.default_rng(a.boot_seed)
    idx = np.concatenate([rng.choice(ix, size=(a.reps, len(ix))) for ix in map(np.array, strata.values())], axis=1)

    def stat(v):                                  # v: one value per task, in ``order``
        v = np.asarray(v, dtype=float)
        lo, hi = np.percentile(v[idx].mean(1), [2.5, 97.5])
        return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi)}

    table = []
    for g in graphs:
        d4, d1 = res[g] - res[h04], res[g] - res[h01]
        table.append({"graph": g, "sources": cand["sources"][g], "value": stat(res[g]),
                      "by_rollout": {int(s): float(by_seed.loc[g, s]) for s in seeds},
                      "model_calls": float(calls[g]), "tokens": float(tokens[g]),
                      "minus_h04": {**stat(d4), "better": int((d4 > 0).sum()), "worse": int((d4 < 0).sum())},
                      "minus_h01": {**stat(d1), "better": int((d1 > 0).sum()), "worse": int((d1 < 0).sum())}})
    table.sort(key=lambda r: (-r["value"]["mean"], r["model_calls"], r["graph"]))     # last resort: graph name
    top = table[0]["value"]["mean"]
    tied = [r["graph"] for r in table if abs(r["value"]["mean"] - top) < 1e-12]
    chosen = table[0]["graph"]

    # value of what each method recommended: the mean over its runs, per task
    recs = defaultdict(list)
    for run, r in cand["live"].items():
        recs[r["method"]].append(r["recommendation"])
    for m, r in cand["panel_runs"].items():
        recs["panel " + m].append(r["recommendation"])
    per_task = {m: res[gs].mean(axis=1) for m, gs in recs.items()}
    methods = {m: {"recommendations": gs, "each": [float(res[g].mean()) for g in gs], "value": stat(per_task[m]),
                   "minus_h04": stat(per_task[m] - res[h04]), "minus_h01": stat(per_task[m] - res[h01])}
               for m, gs in recs.items()}
    pairs = {}
    if {"local_search_guided", "local_search_random"} <= set(per_task):
        pairs["local_search_guided - local_search_random"] = stat(per_task["local_search_guided"]
                                                                  - per_task["local_search_random"])
    # maximum over the candidates other than h04, minus h04: how far the best of them can look ahead by chance
    others = [g for g in graphs if g != h04]
    M = res[others].to_numpy() - res[[h04]].to_numpy()
    mx = M[idx].mean(1).max(1)
    headroom = {"max_minus_h04": float(M.mean(0).max()), "lo": float(np.percentile(mx, 2.5)),
                "hi": float(np.percentile(mx, 97.5)), "note": "descriptive: uncentred bootstrap of the maximum, not a simultaneous bound or a null distribution"}

    print(f"\n{'graph':34s} value [95%]            calls  -h04 [95%]              sources")
    for r in table:
        v, d = r["value"], r["minus_h04"]
        print(f"{r['graph']:34s} {v['mean']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]  {r['model_calls']:5.1f}  "
              f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]  {'; '.join(r['sources'])}")
    print(f"\nsearched graph: {chosen}" + (f"  (tie among {tied}, fewer model calls)" if len(tied) > 1 else "")
          + ("  -> h04 itself: the primary contrast is zero by construction" if chosen == h04 else ""))
    print("\nvalue of the recommendations of each method (mean over its runs):")
    for m, r in methods.items():
        v, d = r["value"], r["minus_h04"]
        print(f"  {m:28s} {v['mean']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]  -h04 {d['mean']:+.3f} "
              f"[{d['lo']:+.3f}, {d['hi']:+.3f}]  each: {[round(x, 3) for x in r['each']]}")
    for k, v in pairs.items():
        print(f"  {k}: {v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]")
    print(f"best other candidate - h04: {headroom['max_minus_h04']:+.3f}; bootstrap of the maximum "
          f"[{headroom['lo']:+.3f}, {headroom['hi']:+.3f}]")
    if a.graphs_out:                              # the graphs of stages 5 and 6: h01, h04 and the searched graph
        src, dst = Path(a.candidates).parent, Path(a.graphs_out)
        dst.mkdir(parents=True, exist_ok=True)
        for f in dst.glob("*.yaml"):
            f.unlink()
        for g in dict.fromkeys([h01, h04, chosen]):
            shutil.copyfile(src / f"{g}.yaml", dst / f"{g}.yaml")
    if a.out:
        Path(a.out).write_text(json.dumps(
            {"searched_graph": chosen, "tied": tied, "is_h04": chosen == h04, "config": cand["configs"][chosen],
             "n_tasks": len(order), "rollouts": seeds, "model": a.model, "stored_records": dict(raw), "files": a.files,
             "candidates": table, "methods": methods, "method_pairs": pairs, "headroom": headroom}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--tasks", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--model", default="qwen3.8-27b")
    p.add_argument("--seeds", default="0:1")
    p.add_argument("--reps", type=int, default=4000)
    p.add_argument("--boot-seed", type=int, default=0)
    p.add_argument("--graphs-out", default="", help="directory for the YAML of h01, h04 and the searched graph")
    p.add_argument("--out", default="")
    main(p.parse_args())
