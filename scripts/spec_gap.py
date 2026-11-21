"""Exploratory: is the visible-hidden gap tied to what the issue text says?  No model call.

Uses the issue audit made on 2026-11 before any outcome was looked at (data/spec_audit.jsonl: "full" = every planted
defect is described in the issue, "partial" = at least one is not; 108 of the 188 search tasks) and the injection
strategy of every task.  Rows: repair rate with visible and with hidden tests for the plain loop, check-and-retry
and the selected graph.

  python scripts/spec_gap.py --out results/summary/spec_gap.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from harness_graph import analysis as A
from harness_graph.grammar import Config

R, M = Path("results/raw"), "qwen3.8-27b.jsonl"
p = argparse.ArgumentParser()
p.add_argument("--out", default="")
a = p.parse_args()
sel = json.loads(Path("results/summary/selection.json").read_text())["searched_graph"]
label = {Config().name: "h01", Config(gate=True).name: "h04", sel: "searched"}
audit = {json.loads(l)["instance_id"]: json.loads(l) for l in open("data/spec_audit.jsonl")}


SINGLE = {"func_basic", "func_pm_ctrl_shuffle", "func_pm_remove_cond", "func_pm_class_rm_funcs", "func_pm_remove_loop",
          "func_pm_op_swap", "func_pm_ctrl_invert_if", "func_pm_remove_assign", "func_pm_remove_wrapper",
          "func_pm_op_change", "func_pm_op_break_chains", "func_pm_op_change_const", "func_pm_class_rm_base",
          "func_pm_class_shuffle_funcs"}
MODEL = "qwen3.8-27b"


def family(s):
    """Strategy families; an unknown strategy stops the script instead of being counted somewhere."""
    if s in ("combine_module", "combine_file"):
        return "combine strategies"
    if s == "lm_rewrite":
        return "lm_rewrite"
    if s.startswith("pr_"):
        return "pr"
    if s in SINGLE:
        return "single-function strategies"
    raise SystemExit(f"strategy {s!r} is in no family")


def complete(files, tasks, cond, seeds):
    """Exactly one scored record of each of the three graphs in every (task, run) cell."""
    try:
        return A.audit_cells([str(f) for f in files], MODEL, list(label), list(tasks), {cond: seeds})
    except ValueError as e:
        raise SystemExit(str(e))


def table(name, tasks, vis_files, hid_files, vis_seeds):
    T = {json.loads(l)["instance_id"]: json.loads(l) for l in open(tasks) if l.strip()}
    rows = []
    for cond, files, seeds in (("visible", vis_files, vis_seeds), ("hidden", hid_files, [0])):
        complete(files, T, cond, seeds)
        df = A.scored(A.load_episodes(*files))
        df = df[df.graph.isin(label) & df.task_id.isin(T) & (df.tests == cond)]
        m = df.pivot_table(index="task_id", columns="graph", values="resolved", aggfunc="mean").rename(columns=label)
        m["cond"] = cond
        rows.append(m.reset_index())
    d = pd.concat(rows)
    d["spec"] = d.task_id.map(lambda t: audit[t]["spec"] if t in audit else "not audited")
    d["family"] = d.task_id.map(lambda t: family(T[t]["strategy"]))
    d["gold_files"] = d.task_id.map(lambda t: "1 file" if int(T[t]["gold_files"]) == 1 else "2+ files")
    res = {}
    for by in ("spec", "family", "gold_files"):
        g = d.groupby([by, "cond"])[["h01", "h04", "searched"]].mean().round(3)
        n = d[d.cond == "hidden"].groupby(by).size()
        print(f"\n== {name}: by {by}\n{g.to_string()}\n n tasks: {n.to_dict()}")
        res[by] = {"n_tasks": {k: int(v) for k, v in n.items()},
                   "repair_rate": {f"{k[0]} | {k[1]}": v for k, v in g.to_dict("index").items()}}
    # cross: audited tasks, spec x family
    x = d[(d.spec != "not audited")].groupby(["spec", "family", "cond"])[["h04"]].agg(["mean", "size"]).round(3)
    if len(x):
        print(f"\n== {name}: audited tasks, spec x family, h04\n{x.to_string()}")
    return res


out = {"note": "exploratory; the audit labels were written before outcomes were seen, the split by label was not pre-specified",
       "search_tasks": table("search tasks (188; visible = stage-1 table, 1 run; hidden 1 run)", "data/v2/search.jsonl",
                             [R / "search_table/a/visible" / M, R / "search_table/b/visible" / M],
                             [R / "search_hidden/a/hidden" / M, R / "search_hidden/b/hidden" / M], [0]),
       "confirmation_tasks": table("confirmation tasks (232; visible 2 runs; hidden 1 run)", "data/v2/confirm.jsonl",
                                   [R / "confirm/r0/visible" / M, R / "confirm/r1/visible" / M],
                                   [R / "confirm/h_a/hidden" / M, R / "confirm/h_b/hidden" / M], [0, 1])}
if a.out:
    Path(a.out).write_text(json.dumps(out, indent=1))

# contrasts inside strategy families, visible tests, two runs each: selection tasks (which chose the graph),
# confirmation tasks, and both pooled.  Pointwise, unadjusted, exploratory; the groups are not all disjoint.
import numpy as np

SETS = {"selection": ("data/v2/select.jsonl", [R / "select/r0/visible" / M, R / "select/r1/visible" / M]),
        "confirmation": ("data/v2/confirm.jsonl", [R / "confirm/r0/visible" / M, R / "confirm/r1/visible" / M])}
T, frames = {}, []
for name, (tf, files) in SETS.items():
    t = {json.loads(l)["instance_id"]: json.loads(l) for l in open(tf) if l.strip()}
    if name == "confirmation":
        complete(files, t, "visible", [0, 1])
    d = A.scored(A.load_episodes(*files))
    d = d[d.graph.isin(label) & d.task_id.isin(t) & (d.tests == "visible") & d.seed.isin([0, 1])]
    assert len(d) == 3 * len(t) * 2, (name, len(d))
    frames.append(d.assign(split=name))
    T.update(t)
df = pd.concat(frames)
split_of = df.drop_duplicates("task_id").set_index("task_id").split
m = df.pivot_table(index="task_id", columns="graph", values="resolved", aggfunc="mean").rename(columns=label)
calls = df.pivot_table(index="task_id", columns="graph", values="model_calls", aggfunc="mean").rename(columns=label)
fam = {"combine strategies": lambda t: family(T[t]["strategy"]) == "combine strategies",
       "lm_rewrite": lambda t: T[t]["strategy"] == "lm_rewrite",
       "single-function strategies": lambda t: family(T[t]["strategy"]) == "single-function strategies",
       "several gold files (overlaps the others)": lambda t: int(T[t]["gold_files"]) > 1}
out["visible_by_family"] = {"note": "pointwise 95% intervals, unadjusted, exploratory; selection tasks helped choose the graph"}
print("\n== visible tests, two runs: contrasts inside strategy families (pointwise, unadjusted, exploratory)")
for name, keep in fam.items():
    for part in ("selection", "confirmation", "pooled"):
        ts = [t for t in m.index if keep(t) and (part == "pooled" or split_of[t] == part)]
        if len(ts) < 2:
            continue
        idx = A.strat_index(ts, {t: T[t]["repo_key"] for t in ts}, 4000, 0)
        s_, k_ = m.loc[ts, "searched"] - m.loc[ts, "h04"], m.loc[ts, "h04"] - m.loc[ts, "h01"]
        r = {"n_tasks": len(ts), "repositories": len({T[t]["repo_key"] for t in ts}),
             "repositories_with_a_task_where_searched_differs_from_h04": len({T[t]["repo_key"] for t in ts if s_[t] != 0}),
             "rates": {g: float(m.loc[ts, g].mean()) for g in ("h01", "h04", "searched")},
             "model_calls": {g: float(calls.loc[ts, g].mean()) for g in ("h01", "h04", "searched")},
             "searched - h04": A.paired_diff(s_, idx), "h04 - h01": A.paired_diff(k_, idx)}
        out["visible_by_family"][f"{name} | {part}"] = r
        s, k = r["searched - h04"], r["h04 - h01"]
        print(f"  {name:40s} {part:12s} n={len(ts):3d} repos={r['repositories']:2d}  h01 {r['rates']['h01']:.3f} h04 {r['rates']['h04']:.3f} "
              f"searched {r['rates']['searched']:.3f}  searched-h04 {s['mean']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}] {s['better']}/{s['worse']}"
              f"  h04-h01 {k['mean']:+.3f} [{k['lo']:+.3f}, {k['hi']:+.3f}]")
# is the lead of check-and-retry over the plain loop larger in the combine family?  (direct interaction, pooled)
pool = [t for t in m.index if family(T[t]["strategy"]) in ("combine strategies", "single-function strategies")]
idx = A.strat_index(pool, {t: T[t]["repo_key"] for t in pool}, 4000, 0)
d = (m.loc[pool, "h04"] - m.loc[pool, "h01"]).to_numpy()
c = np.array([family(T[t]["strategy"]) == "combine strategies" for t in pool])
boot = np.array([d[i][c[i]].mean() - d[i][~c[i]].mean() for i in idx if c[i].any() and (~c[i]).any()])
out["visible_by_family"]["(h04 - h01) combine minus single-function, pooled"] = {
    "mean": float(d[c].mean() - d[~c].mean()), "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5))}
print("  (h04 - h01) in combine minus in single-function, pooled:", out["visible_by_family"]["(h04 - h01) combine minus single-function, pooled"])
if a.out:
    Path(a.out).write_text(json.dumps(out, indent=1))
