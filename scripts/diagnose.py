"""Where do unresolved episodes fail?  Splits them by how they ended and by whether the edit hit the right files.

  python scripts/diagnose.py --tasks data/dev_tasks.jsonl results/raw/premise/*.jsonl
"""
import argparse
import json
import re
from collections import Counter, defaultdict

FILE = re.compile(r"^diff --git a/(\S+) b/", re.M)


def files_of(patch: str) -> set:
    return set(FILE.findall(patch or ""))


def kind(e) -> str:
    if e["resolved"]:
        return "resolved"
    if not (e.get("diff") or "").strip():
        return "no_edit"
    return "wrong_patch_submitted" if e["exit_reason"] == "end" else "cut_off_with_edit"


def main(a):
    gold, tasks = {}, {}
    for l in open(a.tasks):
        t = json.loads(l)
        gold[t["instance_id"]], tasks[t["instance_id"]] = files_of(t["bug_patch"]), t
    for path in a.files:
        eps = [json.loads(l) for l in open(path)]
        eps = [e for e in eps if e["status"] == "ok" and e["task_id"] in gold]
        print(f"\n===== {path}: {len(eps)} episodes")
        for label, sel in (("all", eps), ("single-file", [e for e in eps if len(gold[e["task_id"]]) == 1]),
                           ("multi-file", [e for e in eps if len(gold[e["task_id"]]) > 1])):
            c = Counter(kind(e) for e in sel)
            print(f"  {label:12s} n={len(sel):4d} " + "  ".join(f"{k}={v / len(sel):.2f}" for k, v in sorted(c.items())))
        # localisation among unresolved episodes that have an edit
        loc = Counter()
        for e in eps:
            if e["resolved"] or not (e.get("diff") or "").strip():
                continue
            g, d = gold[e["task_id"]], files_of(e["diff"])
            multi = "multi" if len(g) > 1 else "single"
            loc[(multi, "all_gold_files" if g <= d else "some_gold_files" if g & d else "no_gold_file")] += 1
        print("  unresolved with an edit, gold files touched:", dict(sorted(loc.items())))
        # partial credit: fail-to-pass and pass-to-pass on unresolved graded episodes
        f2p, p2p_broken, n = 0.0, 0, 0
        for e in eps:
            g = e.get("grade") or {}
            if e["resolved"] or not g.get("f2p_total"):
                continue
            n += 1
            f2p += g["f2p_pass"] / g["f2p_total"]
            p2p_broken += g["p2p_pass"] < g["p2p_total"]
        if n:
            print(f"  unresolved & graded n={n}: mean F2P fixed {f2p / n:.2f}, broke a passing test {p2p_broken / n:.2f}")
        # reproduction scripts and the gate
        by = defaultdict(lambda: Counter())
        for e in eps:
            cmds = [ev["content"] for ev in e["events"] if ev["role"] == "command"]
            wrote = any("/tmp/repro.py" in c for c in cmds)
            ran_tests = any("pytest" in c for c in cmds)
            by[e["graph"]]["n"] += 1
            by[e["graph"]]["wrote_repro"] += wrote
            by[e["graph"]]["ran_pytest"] += ran_tests
            if kind(e) == "wrong_patch_submitted":
                by[e["graph"]]["wrong_submitted"] += 1
                by[e["graph"]]["wrong_submitted_with_repro"] += wrote
        print("  per graph: share that wrote /tmp/repro.py, ran pytest, submitted a wrong patch (and of those, had a repro)")
        for g in sorted(by):
            c = by[g]
            print(f"    {g:22s} repro={c['wrote_repro'] / c['n']:.2f} pytest={c['ran_pytest'] / c['n']:.2f} "
                  f"wrong_submitted={c['wrong_submitted'] / c['n']:.2f} with_repro={c['wrong_submitted_with_repro']}/{c['wrong_submitted']}")
        # calls used by outcome
        calls = defaultdict(list)
        for e in eps:
            calls[kind(e)].append(e["model_calls"])
        print("  mean model calls:", {k: round(sum(v) / len(v), 1) for k, v in sorted(calls.items())})
        # per-task: solved by how many of the graph x seed runs
        per = defaultdict(list)
        for e in eps:
            per[e["task_id"]].append(e["resolved"])
        hist = Counter(round(sum(v) / len(v), 1) for v in per.values())
        print("  tasks by share of runs resolved:", dict(sorted(hist.items())))
        # per-task table: is an unsolved task a near miss (most hidden tests fixed) or a miss?
        print("  per task: runs resolved, hidden failing tests, mean share of them fixed, runs that broke a passing test")
        rows = []
        for tid, v in per.items():
            gs = [e["grade"] for e in eps if e["task_id"] == tid and (e.get("grade") or {}).get("f2p_total")]
            part = sum(g["f2p_pass"] / g["f2p_total"] for g in gs) / max(1, len(gs))
            broke = sum(g["p2p_pass"] < g["p2p_total"] for g in gs)
            rows.append((sum(v) / len(v), tid, len(gold[tid]), gs[0]["f2p_total"] if gs else 0, part, broke, len(gs)))
        for r in sorted(rows):
            print(f"    {r[0]:.2f}  files={r[2]} f2p={r[3]:3d} fixed={r[4]:.2f} broke_p2p={r[5]}/{r[6]}  {r[1]}")
        for _, tid, *_ in [r for r in sorted(rows) if r[0] == 0][: a.show]:
            t = tasks[tid]
            e = next(e for e in eps if e["task_id"] == tid and (e.get("diff") or "").strip())
            print(f"\n  ----- never solved: {tid}\n  PROBLEM STATEMENT\n{t['problem_statement'][:2500]}\n  BUG PATCH (the fix is its reverse)\n"
                  f"{t['bug_patch'][:3500]}\n  ONE SUBMITTED DIFF ({e['graph']})\n{e['diff'][:3500]}\n  FAIL_TO_PASS {t['FAIL_TO_PASS'][:12]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--tasks", required=True)
    p.add_argument("--show", type=int, default=0, help="print statement, bug patch and one diff for this many never-solved tasks")
    main(p.parse_args())
