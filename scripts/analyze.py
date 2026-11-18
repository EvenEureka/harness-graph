"""Summarise episode files: prints the tables and writes them as JSON (the site and figures read that file).

  python scripts/analyze.py results/raw/formal/*.jsonl --dev data/dev_tasks.jsonl --out results/summary/formal.json

With --dev, tasks in that file are the development split (the harness battery was designed while looking
at them) and all other tasks are the confirmation split; every table is reported per split.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from harness_graph import analysis as A

pd.set_option("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3f}".format)
MARGIN = 0.05      # equivalence margin on the paired resolve-rate difference, stated before the formal run


def records(t: pd.DataFrame) -> list:
    return json.loads(t.reset_index().to_json(orient="records"))


def summarise(d: pd.DataFrame, label: str) -> dict:
    ok = A.scored(d)
    print(f"\n----- {label}: {len(d)} episodes, {len(d) - len(ok)} engine errors, {d.task_id.nunique()} tasks, "
          f"seeds {sorted(int(s) for s in d.seed.unique())}")
    gt = A.graph_table(d)
    gt["equivalent"] = (gt.dlo > -MARGIN) & (gt.dhi < MARGIN)
    print(gt.to_string())
    fm = A.failure_modes(d)
    gate = A.gate_table(d)
    if len(gate):
        print(gate.to_string())
    cost = [A.cost_ratio(d, g, c) for g in gt.index if g != A.BASELINE for c in ("tokens", "model_calls")]
    orc = A.oracle(d)
    print("head-room:", json.dumps(orc))
    out = {"episodes": len(d), "engine_errors": int(len(d) - len(ok)), "tasks": int(d.task_id.nunique()),
           "seeds": sorted(int(s) for s in d.seed.unique()), "tokens_total": int(ok.tokens.sum()),
           "graphs": records(gt), "failure_modes": records(fm), "by_multi_file": records(A.slice_table(d, "multi_file")),
           "by_spec": records(A.slice_table(d, "spec")) if "spec" in d else [],
           "exits": records(A.exit_table(d)), "gate": records(gate) if len(gate) else [], "cost_ratio": cost,
           "oracle": orc, "seed_noise": A.seed_noise(d),
           "routing": [A.routing(d, k) for k in ("multi_file", "repo")]}
    if "spec" in d:
        print(A.slice_table(d, "spec").assign(n_full=(d.drop_duplicates("task_id").spec == "full").sum()).to_string())
    for r in out["routing"]:
        print("routing:", json.dumps(r))
    return out


def main(a):
    df = A.load_episodes(*a.files)
    if a.spec:
        spec = {r["instance_id"]: r["spec"] for r in map(json.loads, open(a.spec))}
        df["spec"] = df.task_id.map(spec)
    if a.tasks:
        df = df[df.task_id.isin({json.loads(l)["instance_id"] for l in open(a.tasks)})]
    dev = {json.loads(l)["instance_id"] for l in open(a.dev)} if a.dev else set()
    out = {"files": [str(f) for f in a.files], "margin": MARGIN, "models": {}}
    for (model, tests), d in df.groupby(["model", "tests"]):
        model = model if tests == "hidden" else f"{model}, tests {tests}"
        print(f"\n===== {model}")
        splits = {"all": d}
        if dev:
            splits["dev"], splits["confirm"] = d[d.task_id.isin(dev)], d[~d.task_id.isin(dev)]
        out["models"][model] = {k: summarise(v, k) for k, v in splits.items() if len(v)}
    out["conditions"] = {}
    for model, d in df.groupby("model"):
        if d.tests.nunique() == 2:
            d = d[~d.task_id.isin(dev)] if dev else d       # with a dev split, the comparison is on confirmation tasks
            t = A.conditions(d)
            print(f"\n===== {model}: hidden vs visible tests ({'confirmation' if dev else 'all'} tasks)\n{t.to_string()}")
            out["conditions"][model] = records(t)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
        print("\nwrote", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--tasks", default="", help="score only episodes of tasks in this file")
    p.add_argument("--spec", default="", help="labels from scripts/spec_audit.py; adds a full/partial breakdown")
    p.add_argument("--dev", default="", help="task file of the development split")
    p.add_argument("--out", default="")
    main(p.parse_args())
