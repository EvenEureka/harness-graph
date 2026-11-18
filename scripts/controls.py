"""Grader controls for a task file: the reference fix must resolve every task and the unchanged repository none.

  python scripts/controls.py --tasks data/tasks.jsonl --out results/summary/controls.json

A task that fails either control cannot be scored fairly and is listed, so it can be dropped before any run.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from harness_graph.swe.container import ensure_sif
from harness_graph.swe.grading import grade


def main(a):
    tasks = [json.loads(l) for l in open(a.tasks)]
    sifs = {img: ensure_sif(img) for img in sorted({t["image_name"] for t in tasks})}

    def one(t):
        g, e = (grade(t, sifs[t["image_name"]], m) for m in ("gold", "empty"))
        row = {"instance_id": t["instance_id"], "gold_resolved": g["resolved"], "empty_resolved": e["resolved"],
               "gold_f2p": f"{g['f2p_pass']}/{g['f2p_total']}", "gold_p2p": f"{g['p2p_pass']}/{g['p2p_total']}",
               "empty_f2p": f"{e['f2p_pass']}/{e['f2p_total']}", "ungradable": g.get("ungradable"),
               "gold_apply_ok": g["apply_ok"], "gold_timeout": g["timeout"]}
        print(json.dumps(row), flush=True)
        return row

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, tasks))
    bad = [r["instance_id"] for r in rows if not r["gold_resolved"] or r["empty_resolved"]]
    print(f"\n{len(rows)} tasks, {len(bad)} fail a control")
    for b in bad:
        print("  ", b)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"tasks": a.tasks, "failed": bad, "rows": rows}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--workers", type=int, default=6)
    main(p.parse_args())
