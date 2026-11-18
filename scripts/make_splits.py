"""Task splits for the search study, drawn from control-validated SWE-smith pools.

  step 1  python scripts/make_splits.py candidates --pools A.jsonl B.jsonl --out data/pool_candidates.jsonl
  step 2  scripts/controls.py on that file (reference fix resolves, unchanged repository does not)
  step 3  python scripts/make_splits.py split --controls results/summary/controls_pool.json   -> data/v2/{search,select,confirm}.jsonl

Splits (fixed seed, stratified by repository):
  search    the 108 tasks already used (their results shaped the design, so they can only be search data) plus
            new tasks of the same nine repositories
  select    new tasks of the nine repositories; used to choose among recommendations
  confirm   new tasks: half from the nine repositories, half from three repositories no run has touched
A task is dropped if the grader cannot grade it or if its failing tests live inside a source file (hiding the
tests would delete the module).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from harness_graph.swe.grading import is_test_path, touched_files, ungradable_reason

SEEN = "data/tasks.jsonl"


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def usable(t):
    if ungradable_reason(t):
        return False
    return all(is_test_path(f) for f in t.get("test_files", []))


def candidates(a):
    seen = {t["instance_id"] for t in load(SEEN)}
    old_repos = {t["repo_key"] for t in load(SEEN)}
    rng = np.random.default_rng(a.seed)
    by_repo = defaultdict(list)
    for p in a.pools:
        for t in load(p):
            if t["instance_id"] not in seen and usable(t) and t["instance_id"] not in {x["instance_id"] for x in by_repo[t["repo_key"]]}:
                t["gold_files"] = len(touched_files(t["bug_patch"]))
                by_repo[t["repo_key"]].append(t)
    out = []
    for repo, ts in sorted(by_repo.items()):
        if repo not in old_repos and len(ts) < 100:
            continue                                      # a repository dropped earlier (in-file tests)
        ts = sorted(ts, key=lambda t: t["instance_id"])
        keep = ts if repo in old_repos else [ts[i] for i in sorted(rng.choice(len(ts), size=a.per_new_repo, replace=False))]
        out += keep
        print(f"{repo:45s} {'old' if repo in old_repos else 'NEW'} {len(keep)} of {len(ts)}")
    Path(a.out).write_text("".join(json.dumps(t) + "\n" for t in out))
    print(f"{len(out)} candidates -> {a.out}")


def split(a):
    bad = set(json.loads(Path(a.controls).read_text())["failed"])
    old = load(SEEN)
    old_repos = {t["repo_key"] for t in old}
    cand = [t for t in load(a.candidates) if t["instance_id"] not in bad]
    rng = np.random.default_rng(a.seed)
    same, new = defaultdict(list), defaultdict(list)
    for t in cand:
        (same if t["repo_key"] in old_repos else new)[t["repo_key"]].append(t)
    search, select, confirm = list(old), [], []
    n_s, n_sel, n_c = a.search_new // len(same), a.select // len(same), a.confirm_same // len(same)
    for repo, ts in sorted(same.items()):
        ts = [ts[i] for i in rng.permutation(len(ts))]
        search += ts[:n_s]
        select += ts[n_s:n_s + n_sel]
        confirm += ts[n_s + n_sel:n_s + n_sel + n_c]
    for repo, ts in sorted(new.items()):
        ts = [ts[i] for i in rng.permutation(len(ts))]
        confirm += ts[: a.confirm_new // len(new)]
    for name, ts in (("search", search), ("select", select), ("confirm", confirm)):
        Path(f"data/v2/{name}.jsonl").write_text("".join(json.dumps(t) + "\n" for t in ts))
        repos = defaultdict(int)
        for t in ts:
            repos[t["repo_key"]] += 1
        print(f"{name:8s} {len(ts):4d} tasks, {len(repos)} repositories: {dict(sorted(repos.items()))}")
    ids = [t["instance_id"] for t in search + select + confirm]
    assert len(ids) == len(set(ids)), "a task is in two splits"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--pools", nargs="+", required=True)
    c.add_argument("--out", default="data/pool_candidates.jsonl")
    c.add_argument("--per-new-repo", type=int, default=60)
    c.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("split")
    s.add_argument("--controls", required=True)
    s.add_argument("--candidates", default="data/pool_candidates.jsonl")
    s.add_argument("--search-new", type=int, default=81)
    s.add_argument("--select", type=int, default=63)
    s.add_argument("--confirm-same", type=int, default=117)
    s.add_argument("--confirm-new", type=int, default=120)
    s.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    {"candidates": candidates, "split": split}[a.cmd](a)
