"""P1, step 1: the stored patches a generated check is evaluated on.  No model call, no container.

Reads episodes that already exist on the search tasks and writes, per task, the distinct patches of three kinds:
  correct  a model patch the grader accepted (a correct alternative to the reference fix);
  wrong    a model patch the grader rejected, with a complete grade;
  (the reference fix itself is not listed here: every task has one.)
A wrong patch is in the primary pool when at least one hidden-test run that produced it was ended by the agent:
the situation in which a self-written check would be the only check.  The secondary pool holds every other rejected
hidden-run patch, filled up to PER_KIND per task with rejected visible-run patches.  "Correct" means accepted by
the grader.  A diff the grader both accepted and rejected (in different runs) is dropped and counted.
Patches are picked by fixed rules and a fixed seed, never by what a generated check says about them.

  python scripts/p1_patches.py --tasks data/v2/search.jsonl --hidden results/raw/search_hidden/a/hidden/qwen3.8-27b.jsonl \
      results/raw/search_hidden/b/hidden/qwen3.8-27b.jsonl --visible results/raw/search_table/a/visible/qwen3.8-27b.jsonl \
      results/raw/search_table/b/visible/qwen3.8-27b.jsonl --out data/p1/patches.jsonl
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

PER_KIND, SEED = 3, 0          # at most this many correct, and this many secondary wrong, patches per task


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def kind_of(rec: dict) -> str | None:
    """'correct', 'wrong', or None when the record cannot serve as a patch with a trusted label."""
    g = rec.get("grade") or {}
    if rec.get("status") != "ok" or not (rec.get("diff") or "").strip() or not g:
        return None
    complete = (g.get("grader_version") == "v2" and g.get("apply_ok") and not g.get("bug_apply_fail") and not g.get("timeout")
                and not g.get("ungradable") and g.get("test_rc") is not None and g.get("n_parsed")
                and g.get("f2p_missing") == 0 and g.get("p2p_missing") == 0)
    if not complete:
        return None
    return "correct" if g.get("resolved") else "wrong"


def collect(files: list[str], tests: str, tasks: set[str]) -> tuple[dict[str, dict[str, dict]], int]:
    """task -> patch id -> patch row with every occurrence of this exact diff.  Diffs graded both ways are dropped."""
    out: dict[str, dict[str, dict]] = {}
    for f in files:
        for line in open(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["task_id"] not in tasks or rec.get("tests") != tests:
                continue
            kind = kind_of(rec)
            if kind is None:
                continue
            g = rec["grade"]
            row = out.setdefault(rec["task_id"], {}).setdefault(sha(rec["diff"]), {
                "task_id": rec["task_id"], "patch_id": sha(rec["diff"]), "kinds": set(), "tests": tests, "occurrences": [],
                "f2p_fail": g["f2p_total"] - g["f2p_pass"], "p2p_fail": g["p2p_total"] - g["p2p_pass"], "diff": rec["diff"]})
            row["kinds"].add(kind)
            row["occurrences"].append({"graph": rec["graph"], "seed": rec["seed"], "exit_reason": rec["exit_reason"]})
    conflicts = 0
    for t in out:
        for pid in [p for p, r in out[t].items() if len(r["kinds"]) > 1]:
            del out[t][pid]
            conflicts += 1
        for r in out[t].values():
            r["kind"] = r.pop("kinds").pop()
            r["occurrences"].sort(key=lambda o: (o["graph"], o["seed"]))
            r["ended_by_agent"] = any(o["exit_reason"] == "end" for o in r["occurrences"])
    return out, conflicts


def pick(rows: list[dict], n: int, rng) -> list[dict]:
    rows = sorted(rows, key=lambda r: r["patch_id"])
    return rows if len(rows) <= n else [rows[i] for i in sorted(rng.choice(len(rows), n, replace=False))]


def main(a):
    tasks = sorted(json.loads(l)["instance_id"] for l in open(a.tasks) if l.strip())
    (hid, c_h), (vis, c_v) = collect(a.hidden, "hidden", set(tasks)), collect(a.visible, "visible", set(tasks))
    out, n = [], Counter()
    for i, t in enumerate(tasks):
        rng = np.random.default_rng([SEED, i])
        h, v = list(hid.get(t, {}).values()), [r for p, r in vis.get(t, {}).items() if p not in hid.get(t, {})]
        h = sorted(h, key=lambda r: r["patch_id"])
        cross = [p for p, r in vis.get(t, {}).items() if p in hid.get(t, {}) and r["kind"] != hid[t][p]["kind"]]
        n["graded differently with hidden and visible tests"] += len(cross)
        h = [r for r in h if r["patch_id"] not in cross]
        # primary: rejected, and at least one hidden-test run that produced it was ended by the agent
        primary = [dict(r, pool="primary") for r in h if r["kind"] == "wrong" and r["ended_by_agent"]]
        # secondary: every other rejected hidden-run patch, then rejected visible-run patches up to PER_KIND in all
        second = [r for r in h if r["kind"] == "wrong" and not r["ended_by_agent"]]
        second += pick([r for r in v if r["kind"] == "wrong"], max(0, PER_KIND - len(second)), rng)
        correct = [r for r in h if r["kind"] == "correct"]
        correct = pick(correct, PER_KIND, rng) if len(correct) >= PER_KIND else \
            correct + pick([r for r in v if r["kind"] == "correct"], PER_KIND - len(correct), rng)
        rows = primary + [dict(r, pool="secondary") for r in second] + [dict(r, pool="correct") for r in correct]
        out += rows
        for r in rows:
            n[r["pool"]] += 1
        for pool in ("primary", "secondary", "correct"):
            n[f"tasks with {pool}"] += any(r["pool"] == pool for r in rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    files = {f: hashlib.sha256(Path(f).read_bytes()).hexdigest()[:16] for f in [a.tasks, *a.hidden, *a.visible]}
    summary = {"tasks": len(tasks), "per_kind": PER_KIND, "seed": SEED, "counts": dict(n), "inputs_sha256": files,
               "dropped_same_diff_graded_both_ways": {"hidden": c_h, "visible": c_v},
               "primary_wrong_with_failing_target_tests": sum(r["pool"] == "primary" and r["f2p_fail"] > 0 for r in out),
               "sha256": hashlib.sha256(Path(a.out).read_bytes()).hexdigest()[:16]}
    Path(a.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True)
    p.add_argument("--hidden", nargs="+", required=True)
    p.add_argument("--visible", nargs="+", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
