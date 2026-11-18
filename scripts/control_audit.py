"""Checks on the enforcement control that need no new runs: the episode ledger, resources actually used,
per-seed contrasts, the model-by-instruction interaction, gate passes that the grader rejected, and how
voluntary-checker episodes ended.

  python scripts/control_audit.py results/raw/confirm_visible/*.jsonl --tasks data/confirm_tasks.jsonl --out results/summary/control_audit.json
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from harness_graph import analysis as A
from harness_graph import prompts as P

GRAPHS = ["h01_react", "h04_check_retry", "h16_checker_tool", "h17_checker_tool_gate"]
PAIRS = [("h17_checker_tool_gate", "h16_checker_tool"), ("h04_check_retry", "h01_react"), ("h16_checker_tool", "h01_react")]


def ratio(df, a, b, col, rng):
    m = A.scored(df)
    m = m[m.graph.isin([a, b])].pivot_table(index="task_id", columns="graph", values=col, aggfunc="mean").dropna()
    x, y = m[a].to_numpy(), m[b].to_numpy()
    idx = rng.integers(0, len(m), (4000, len(m)))
    r = x[idx].sum(1) / y[idx].sum(1)
    return {"a": a, "b": b, "col": col, "mean_a": float(x.mean()), "mean_b": float(y.mean()),
            "ratio": float(x.sum() / y.sum()), "lo": float(np.percentile(r, 2.5)), "hi": float(np.percentile(r, 97.5))}


def last_records(paths, tasks):
    """Last record per (episode, test condition): the same rule the contrast loader uses."""
    last, n = {}, Counter()
    for path in paths:
        for line in open(path):
            e = json.loads(line)
            if e["task_id"] in tasks and e["graph"] in GRAPHS and e.get("tests") == "visible":
                n[e["episode_id"]] += 1
                last[e["episode_id"]] = e
    return last, n


def ending(e):
    """How a voluntary-checker episode ended, from its command stream."""
    ev = e.get("events", [])
    cmds = [i for i, x in enumerate(ev) if x["role"] == "command"]
    calls = [i for i in cmds if ev[i]["content"].strip() == P.CHECK_COMMAND]
    if not calls:
        return "never_called"
    passed = len(ev) > calls[-1] + 1 and ev[calls[-1] + 1]["content"].startswith(P.CHECK_PASSED[:20])
    after = sum(i > calls[-1] for i in cmds)
    if not passed:
        return "last_check_failed"
    return "passed_then_finished" if after <= 1 else "passed_then_more_commands"


def main(a):
    rng = np.random.default_rng(0)
    tasks = {json.loads(l)["instance_id"] for l in open(a.tasks)}
    df = A.load_episodes(*a.files)
    df = df[df.task_id.isin(tasks) & (df.tests == "visible")]
    new = df[df.seed.isin([1, 2]) & df.graph.isin(GRAPHS)]
    last, nrec = last_records(a.files, tasks)
    out = {"ledger": [], "resources": [], "per_seed": [], "model_interaction": {}, "gate": [], "endings": [], "matched": []}

    for (model, g), d in new.groupby(["model", "graph"]):
        out["ledger"].append({"model": model, "graph": g, "episodes": len(d), "ok": int((d.status == "ok").sum()),
                              "errors": int((d.status != "ok").sum()),
                              "tasks_with_both_seeds_ok": int((A.scored(d).groupby("task_id").seed.nunique() == 2).sum()),
                              "rewritten": sum(nrec[i] > 1 for i in d.episode_id)})
    for model, d in new.groupby("model"):
        for x, y in PAIRS:
            for col in ("model_calls", "tokens", "cost_usd", "latency_s"):
                out["resources"].append({"model": model, **ratio(d, x, y, col, rng)})
        # sensitivity: only tasks where both graphs have both seeds scored
        ok = A.scored(d)
        full = ok[ok.graph.isin(PAIRS[0])].groupby(["task_id", "graph"]).seed.nunique().unstack()
        keep = full[(full == 2).all(axis=1)].index
        out["per_seed"].append({"model": model, "seeds": "1+2, complete tasks only", **A.contrast(d[d.task_id.isin(keep)], *PAIRS[0])})
    for model, d in df[df.graph.isin(GRAPHS)].groupby("model"):
        for s in sorted(d.seed.unique()):
            for x, y in PAIRS:
                ds = d[d.seed == s]
                if {x, y} <= set(ds.graph):
                    out["per_seed"].append({"model": model, "seeds": int(s), **A.contrast(ds, x, y)})

    # does instruction + tool help the two models differently?  paired by task
    per = {}
    for model, d in new.groupby("model"):
        m = A.scored(d).pivot_table(index="task_id", columns="graph", values="resolved", aggfunc="mean")
        per[model] = {f"{x}-{y}": (m[x] - m[y]) for x, y in PAIRS}
    models = sorted(per)
    if len(models) == 2:
        for k in per[models[0]]:
            dd = (per[models[1]][k] - per[models[0]][k]).dropna().to_numpy()
            lo, hi = A.boot(dd, rng)
            out["model_interaction"][k] = {"models": f"{models[1]} minus {models[0]}", "n_tasks": len(dd),
                                           "delta": float(dd.mean()), "lo": lo, "hi": hi}

    # gate verdict against the grader; a pass the grader rejects bounds what a silent pass can hide
    c = defaultdict(Counter)
    for e in last.values():
        if e["status"] != "ok" or e["seed"] not in (1, 2):
            continue
        k = c[(e["model"], e["graph"])]
        gates = [l for l in e.get("node_logs", []) if l["type"] == "check"]
        k["n"] += 1
        k["resolved"] += bool(e["resolved"])
        k["exit_" + e["exit_reason"]] += 1
        if gates:
            k["gated"] += 1
            k["last_gate_pass"] += gates[-1]["outcome"] == "pass"
            k["last_gate_pass_unresolved"] += gates[-1]["outcome"] == "pass" and not e["resolved"]
            k["last_gate_fail_resolved"] += gates[-1]["outcome"] == "fail" and bool(e["resolved"])
            k["gate_slow_over_250s"] += any(float(l["latency_s"]) > 250 for l in gates)
            k["ended_without_final_gate"] += e["path"][-1] != gates[-1]["node"]
        if e["graph"] == "h16_checker_tool":
            end = ending(e)
            k["end_" + end] += 1
            k["end_" + end + "_resolved"] += bool(e["resolved"])
    for (model, g), k in sorted(c.items()):
        (out["endings"] if g == "h16_checker_tool" else out["gate"]).append({"model": model, "graph": g, **dict(sorted(k.items()))})

    # matched episodes: voluntary failed, enforced resolved (same task and seed) -> how did the voluntary one end?
    by = {(e["model"], e["graph"], e["task_id"], e["seed"]): e for e in last.values() if e["status"] == "ok" and e["seed"] in (1, 2)}
    m = defaultdict(Counter)
    for (model, g, t, s), e in by.items():
        f = by.get((model, "h17_checker_tool_gate", t, s))
        if g == "h16_checker_tool" and f is not None:
            key = ("h16_fail" if not e["resolved"] else "h16_ok") + "_" + ("h17_ok" if f["resolved"] else "h17_fail")
            m[model][key] += 1
            if key == "h16_fail_h17_ok":
                m[model]["rescued_" + ending(e)] += 1
                m[model]["rescued_h17_first_gate_failed"] += any(
                    l["type"] == "check" and l["outcome"] == "fail" for l in f.get("node_logs", []))
    out["matched"] = [{"model": k, **dict(sorted(v.items()))} for k, v in sorted(m.items())]

    for name, rows in out.items():
        print(f"\n== {name}")
        for r in (rows if isinstance(rows, list) else [{"contrast": k, **v} for k, v in rows.items()]):
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--tasks", required=True)
    p.add_argument("--out", default="")
    main(p.parse_args())
