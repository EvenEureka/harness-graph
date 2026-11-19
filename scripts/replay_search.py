"""Compare search methods offline, on episodes that were already run (no model call).

For each repetition the tasks are split in two: methods search on one half through a replay of the stored
episodes, and the graph each one recommends is scored on the other half.  Reported per method and budget:
the mean held-out resolve rate of the recommendation (the primary number), its gap to the best graph of the
held-out half, and how often it is within ``--tol`` of that graph.  "Best" is the highest observed mean in a
finite table, so it is itself noisy.  A budget larger than the table reuses stored episodes; the share of such
pulls is reported.  Intervals are over task splits and are not sampling confidence intervals.

  python scripts/replay_search.py results/raw/confirm_visible/gpt-6-luna.jsonl --tests visible \
      --graphs h01_react:h02_plan:... --budgets 64:128:256:512 --reps 200 --out results/summary/replay_search.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from harness_graph import analysis as A
from harness_graph.grammar import HAND_WRITTEN, features
from harness_graph.search import METHODS, Replay, local_search, remedy_proposer


def main(a):
    df = A.scored(A.load_episodes(*a.files))
    df = df[df.tests == a.tests]
    if a.model:
        df = df[df.model == a.model]
    if a.graphs:
        df = df[df.graph.isin(a.graphs.split(":"))]
    if a.seeds:
        df = df[df.seed.isin([int(s) for s in a.seeds.split(":")])]
    rows = df.to_dict("records")
    full = Replay(rows, np.random.default_rng(0), a.cost_weight)
    arms, tasks = full.arms, full.tasks
    budgets = [int(b) for b in a.budgets.split(":")]
    cfg = dict(HAND_WRITTEN)
    if a.panel:                                           # graphs named by the grammar: configs from the frozen panel file
        from harness_graph.grammar import Config
        cfg.update({n: Config(**d) for n, d in json.loads(Path(a.panel).read_text())["configs"].items()})
    feats = {g: features(cfg[g]) for g in arms}
    by_cfg = {cfg[g].name: g for g in arms}
    propose_cfg = remedy_proposer(lambda g: cfg[g], lambda c: by_cfg.get(c.name, "?"))
    print(f"{len(arms)} graphs, {len(tasks)} tasks with every graph, {len(rows)} episodes; truth on all tasks:")
    for g, v in sorted(full.truth().items(), key=lambda kv: -kv[1]):
        print(f"  {g:26s} {v:.3f}")
    acc = defaultdict(lambda: defaultdict(list))
    reuse = defaultdict(list)
    from harness_graph.grammar import Config
    fixed = [g for g in ("h01_react", "h04_check_retry", Config().name, Config(gate=True).name) if g in arms]
    splits = []
    for rep in range(a.reps):
        rng = np.random.default_rng(1000 + rep)
        perm = list(rng.permutation(tasks))
        search, held = perm[: len(perm) // 2], perm[len(perm) // 2:]
        truth = Replay(rows, rng, a.cost_weight).truth(held)
        best = max(truth.values())
        runs = {}
        for name, fn in METHODS.items():
            kw = {"features": feats} if name == "linear_thompson" else {}
            if name == "successive_halving":              # its allocation depends on the total budget: one run per budget
                at = {}
                for b in budgets:
                    r = np.random.default_rng(5000 + rep)
                    rp = Replay(rows, r, a.cost_weight)
                    at[b] = fn(arms, search, rp.pull, b, r).recommendation
                    reuse[(name, b)].append(rp.reused / max(rp.pulls, 1))
                runs[name] = type("R", (), {"checkpoints": at, "recommendation": at[budgets[-1]], "method": name})()
                continue
            r = np.random.default_rng(5000 + rep)
            rp = Replay(rows, r, a.cost_weight)
            runs[name] = fn(arms, search, rp.pull, budgets[-1], r, checkpoints=budgets, **kw)
            for b in budgets:                             # share of reused draws among the first b
                reuse[(name, b)].append(rp.reused_after[b - 1] / b)
        for guided in ((False, True) if a.local else ()):
            r = np.random.default_rng(5000 + rep)
            res = local_search(a.start, search, Replay(rows, r, a.cost_weight).pull, budgets[-1], r,
                               lambda g: [x for x in arms if x != g], propose_cfg if guided else None,
                               batch=a.batch, checkpoints=budgets)
            runs[res.method] = res
        splits.append({"rep": rep, "held_out_value": truth,
                       "picks": {f"{name}@{b}": res.checkpoints.get(b, res.recommendation)
                                 for name, res in runs.items() for b in budgets}})
        for g in fixed:                                   # always the same graph, scored on the same held-out half
            for b in budgets:
                acc[("always " + g, b)]["value"].append(truth[g])
                acc[("always " + g, b)]["regret"].append(best - truth[g])
                acc[("always " + g, b)]["near_best"].append(best - truth[g] <= a.tol)
                acc[("always " + g, b)]["pick"].append(g)
        for name, res in runs.items():
            for b in budgets:
                g = res.checkpoints.get(b, res.recommendation)
                acc[(name, b)]["value"].append(truth[g])
                acc[(name, b)]["regret"].append(best - truth[g])
                acc[(name, b)]["near_best"].append(best - truth[g] <= a.tol)
                acc[(name, b)]["pick"].append(g)
    out = []
    print("\nmethod budget held-out value  regret  P(within tol)  most recommended")
    for (name, b), d in sorted(acc.items()):
        picks = {g: d["pick"].count(g) / len(d["pick"]) for g in set(d["pick"])}
        top = max(picks, key=picks.get)
        row = {"method": name, "budget": b, "value": float(np.mean(d["value"])), "regret": float(np.mean(d["regret"])),
               "regret_lo": float(np.percentile(d["regret"], 2.5)), "regret_hi": float(np.percentile(d["regret"], 97.5)),
               "near_best": float(np.mean(d["near_best"])), "picks": picks,
               "reused_share": float(np.mean(reuse[(name, b)])) if reuse[(name, b)] else None}
        out.append(row)
        print(f"{name:22s} {b:5d}  {row['value']:.3f}  {row['regret']:.3f}  {row['near_best']:.2f}  {top} ({picks[top]:.2f})")
    # the same held-out half for both sides of every difference; the range is over task splits of this table
    paired = []
    names = sorted({n for n, _ in acc})
    refs = ["uniform"] + ["always " + g for g in fixed]
    print("\npaired differences on the same splits: mean [2.5%, 97.5%] over splits")
    for b in budgets:
        for ref in refs:
            for name in names:
                if name == ref or name.startswith("always "):
                    continue
                d = np.array(acc[(name, b)]["value"]) - np.array(acc[(ref, b)]["value"])
                row = {"budget": b, "a": name, "b": ref, "mean": float(d.mean()),
                       "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
                       "share_a_higher": float((d > 0).mean()), "share_b_higher": float((d < 0).mean())}
                paired.append(row)
                print(f"  B={b:5d} {name:20s} - {ref:36s} {row['mean']:+.3f} [{row['lo']:+.3f}, {row['hi']:+.3f}]")
    if a.out:
        Path(a.out).write_text(json.dumps({"args": vars(a), "arms": arms, "n_tasks": len(tasks),
                                           "truth_all_tasks": full.truth(), "results": out, "paired": paired,
                                           "splits": splits}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--tests", default="visible")
    p.add_argument("--model", default="")
    p.add_argument("--graphs", default="")
    p.add_argument("--seeds", default="")
    p.add_argument("--budgets", default="64:128:256:512")
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--start", default="h01_react")
    p.add_argument("--panel", default="", help="panel.json written by make_panel.py (for grammar-named graphs)")
    p.add_argument("--local", action="store_true", help="also replay local search restricted to the stored graphs")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--tol", type=float, default=0.03)
    p.add_argument("--cost-weight", type=float, default=0.0)
    p.add_argument("--out", default="")
    main(p.parse_args())
