"""P1, exploratory (written after the G1 result): the checks that reject the healthy repository, laid out for reading.

For every delivered check that rejected the reference tree: the assertion that failed (with the lines before it), what
the healthy code produced, and the issue text.  Used to label why the check and the healthy code disagree.

  python scripts/p1_rejections.py --gen results/raw/p1/gen_G1.jsonl --eval results/raw/p1/eval_G1.jsonl \
      --tasks data/p1/tasks.jsonl --out results/summary/p1_G1_reference_rejections.json
"""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--gen", required=True)
p.add_argument("--eval", required=True)
p.add_argument("--tasks", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in open(a.tasks) if l.strip()}
gen = {(g["task_id"], g["seed"]): g for g in map(json.loads, open(a.gen)) if g["status"] == "ok"}
rows = {}
for r in map(json.loads, open(a.eval)):
    if r["tree"] == "reference" and r["status"] == "ok" and r["verdict"] != "fixed" and r["rep"] == 0:
        rows[(r["task_id"], r["seed"])] = r
out = []
for i, ((t, s), r) in enumerate(sorted(rows.items())):
    art = gen[(t, s)]["artifact"].split("\n")
    line = int(r["reason"].rsplit("_", 1)[-1]) if r["reason"].startswith("assert_line_") else None
    out.append({"case": i, "task_id": t, "seed": s, "strategy": tasks[t]["strategy"], "verdict": r["verdict"], "reason": r["reason"],
                "on_healthy_code": r["detail"][:1200],
                "check_lines_before_the_failed_assert": art[max(0, line - 12):line + 2] if line else art[:60],
                "issue_to_assertions_header": [l for l in art[:25] if l.startswith("#")],
                "issue": tasks[t]["problem_statement"][:4000]})
Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(len(out), "checks reject the healthy repository;", len({o["task_id"] for o in out}), "tasks")
