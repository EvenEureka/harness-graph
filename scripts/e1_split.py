"""E1, step 2: split each decomposable task's failing tests into a shown and a withheld part.  No container, no model.

  python scripts/e1_split.py [--components data/e1/components.jsonl] [--tasks data/c1/confirm.jsonl] [--out data/e1/candidates.jsonl]

Units are the groups of test files that import one another (from step 1).  For split seed s = 0..9 the units are
ordered by sha256("<task id>|<unit files joined by |>|<s>") and the first ceil(n/2) are SHOWN; the other units are
WITHHELD, their files removed from the agent's workspace.  A split is complete when every component of the bug makes
at least one shown test fail.
  primary block      the first seed that gives a complete split
  incomplete block   no seed among the ten gives one: seed 0 is used ("incomplete shown-component detection")
Tasks with a single unit cannot be split and are left out.  Nothing here reads an agent outcome.
"""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(10)


def split(tid: str, units: list[list[str]], seed: int) -> tuple[list[str], list[str]]:
    order = sorted(units, key=lambda u: hashlib.sha256(f"{tid}|{'|'.join(u)}|{seed}".encode()).hexdigest())
    k = math.ceil(len(order) / 2)
    return sorted(f for u in order[:k] for f in u), sorted(f for u in order[k:] for f in u)


def choose(rec: dict, task: dict) -> dict | None:
    if len(rec["units"]) < 2:
        return None
    f2p = task["FAIL_TO_PASS"]
    options = []
    for s in SEEDS:
        shown_files, withheld_files = split(rec["instance_id"], rec["units"], s)
        shown = [t for t in f2p if t.split("::")[0] in shown_files]
        withheld = [t for t in f2p if t.split("::")[0] in withheld_files]
        assert shown and withheld and len(shown) + len(withheld) == len(f2p)
        missed = sum(not set(c["detected_by"]) & set(shown) for c in rec["components"])
        options.append((s, shown_files, withheld_files, shown, withheld, missed))
    complete = [o for o in options if o[5] == 0]
    s, shown_files, withheld_files, shown, withheld, missed = complete[0] if complete else options[0]
    seen_by_withheld = sum(bool(set(c["detected_by"]) & set(withheld)) for c in rec["components"])
    return {"block": "primary" if complete else "incomplete", "split_seed": s, "shown_files": shown_files,
            "withheld_files": withheld_files, "shown_f2p": shown, "withheld_f2p": withheld,
            "components": len(rec["components"]), "components_missed_by_shown_tests": missed,
            "components_detected_by_withheld_tests": seen_by_withheld,
            "withheld_tests_detect_every_component": seen_by_withheld == len(rec["components"])}


def main(a):
    tasks = {t["instance_id"]: t for t in (json.loads(l) for l in open(a.tasks) if l.strip())}
    recs = [json.loads(l) for l in open(a.components) if l.strip()]
    assert all(r["status"] != "error" for r in recs), "rerun step 1: some tasks ended in an error"
    out, why = [], Counter()
    for r in recs:
        if r["status"] != "decomposable":
            why["not decomposable: " + r["reason"]] += 1
            continue
        e1 = choose(r, tasks[r["instance_id"]])
        if e1 is None:
            why["one import unit, no split"] += 1
            continue
        why[e1["block"]] += 1
        out.append({**tasks[r["instance_id"]], "withheld_files": e1["withheld_files"], "shown_f2p": e1["shown_f2p"],
                    "withheld_f2p": e1["withheld_f2p"], "e1": e1})
    Path(a.out).write_text("".join(json.dumps(t) + "\n" for t in out))
    summary = {"candidates_with_two_or_more_failing_test_files": len(recs), "outcome": dict(why),
               "components_sha256": hashlib.sha256(Path(a.components).read_bytes()).hexdigest()[:16],
               "sha256": hashlib.sha256(Path(a.out).read_bytes()).hexdigest()[:16]}
    Path(a.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--components", default=str(ROOT / "data" / "e1" / "components.jsonl"))
    p.add_argument("--tasks", default=str(ROOT / "data" / "c1" / "confirm.jsonl"))
    p.add_argument("--out", default=str(ROOT / "data" / "e1" / "candidates.jsonl"))
    main(p.parse_args())
