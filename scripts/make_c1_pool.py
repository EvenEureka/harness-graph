"""Task pool for C1: combine-strategy tasks from repositories the project has never used.  No model call.

Every rule below was written before any agent ran on these tasks; none of them looks at an agent outcome.  The pool
is therefore: computationally tractable, grader-validated combine-strategy tasks from unused repositories that have
enough such tasks.  It is a repository-capped sample, not a simple random sample of all such tasks.

  population   SWE-smith tasks whose strategy is exactly combine_module or combine_file, with an issue text, that the
               grader can grade, with at least one failing test, and with every listed test (failing and passing)
               in a test file
  repositories no repository of the same name (any revision) is in a task file of this project; at least MIN_TASKS
               population tasks; put in a random order (seed 0).  A repository is accepted when
                 (1) its image can be pulled and a workspace can be set up,
                 (2) the full test suite of the healthy repository finishes (some tests may fail) in under SUITE_S
                     seconds, measured with nothing else running in the probe job,
                 (3) of up to CAND randomly drawn tasks at least ACCEPT_MIN pass both grader controls: the
                     reference fix applies, does not time out and resolves the task; on the unchanged repository
                     the run does not time out, the defect applies, and at least one failing test is observed to
                     fail or error (a missing result does not count)
  tasks        an accepted repository gives min(PER_REPO, tasks that passed) tasks, drawn at random.  Repositories
               are added in the random order until TARGET tasks are reached; the last one is cut at random.  If
               every eligible repository has been probed and fewer than TARGET tasks exist, the pool is what there
               is, provided it has at least MIN_TOTAL tasks from MIN_REPOS repositories; otherwise C1 is not run.
  failures     an exception in a control is retried once for that task and then counts as a failed control; a
               repository that fails step (1) is probed once more (attempt 2, same candidates).  The probe log is
               append-only.

  python scripts/make_c1_pool.py stats --raw <swesmith dump>     # eligible repositories and their order
  python scripts/make_c1_pool.py probe --raw <swesmith dump>     # checks (1)-(3) in order, resumable, stops at TARGET
  python scripts/make_c1_pool.py final --raw <swesmith dump>     # -> data/c1/confirm.jsonl, data/c1/pool.json
"""
import argparse
import ast
import asyncio
import glob
import hashlib
import json
import os
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from harness_graph.swe.grading import is_test_path, touched_files, ungradable_reason

STRATEGIES = ("combine_module", "combine_file")
USED_FILES = ["data/tasks.jsonl", "data/dev_tasks.jsonl", "data/confirm_tasks.jsonl", "data/pool_candidates.jsonl",
              "data/commit0.jsonl", "data/v2/search.jsonl", "data/v2/select.jsonl", "data/v2/confirm.jsonl"]
MIN_TASKS, CAND, ACCEPT_MIN, PER_REPO, SUITE_S, SEED = 60, 90, 40, 60, 120.0, 0
TARGET, MIN_TOTAL, MIN_REPOS = 900, 600, 10


def _list(x):
    if isinstance(x, str):
        if not x.strip():
            return []
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return list(ast.literal_eval(x))
    return list(x) if x is not None else []


def strategy(iid):
    tail = iid.split(".")[-1]
    return tail.rsplit("__", 1)[0] if "__" in tail else tail


def names(key):
    """A repository under every name it can be matched by: 'pallets__click.fde47b4b' -> {'pallets__click', 'click'}."""
    k = key.split("/", 1)[-1].lower()
    full = k.rsplit(".", 1)[0] if "__" in k and "." in k else k
    return {full, full.split("__")[-1]}


def load_pool(raw):
    """({repo_key: [task rows]} of the population with used repositories left out, the names that were excluded)."""
    import pyarrow.parquet as pq
    used = set()
    for f in USED_FILES:
        for l in open(f):
            if l.strip():
                t = json.loads(l)
                used |= names(t.get("repo_key") or t["repo"])
    by, seen = defaultdict(list), set()
    cols = ["instance_id", "patch", "FAIL_TO_PASS", "PASS_TO_PASS", "image_name", "repo", "problem_statement"]
    for f in sorted(glob.glob(os.path.join(raw, "data", "*.parquet"))):
        for r in pq.read_table(f, columns=cols).to_pylist():
            rk = r["repo"].split("/", 1)[-1]
            if names(rk) & used or strategy(r["instance_id"]) not in STRATEGIES or r["instance_id"] in seen:
                continue
            if not (r.get("problem_statement") or "").strip():
                continue
            f2p, p2p = _list(r["FAIL_TO_PASS"]), _list(r["PASS_TO_PASS"])
            t = {"instance_id": r["instance_id"], "repo_key": rk, "repo": r["repo"], "image_name": r["image_name"],
                 "strategy": strategy(r["instance_id"]), "problem_statement": r["problem_statement"],
                 "bug_patch": r["patch"], "FAIL_TO_PASS": f2p, "PASS_TO_PASS": p2p,
                 "f2p_files": sorted({x.split("::", 1)[0] for x in f2p}),
                 "test_files": sorted({x.split("::", 1)[0] for x in f2p + p2p})}
            if not f2p or ungradable_reason(t) or not all(is_test_path(x) for x in t["test_files"]):
                continue
            t["gold_files"] = len(touched_files(t["bug_patch"]))
            seen.add(t["instance_id"])
            by[rk].append(t)
    return {rk: sorted(ts, key=lambda t: t["instance_id"]) for rk, ts in by.items()}, used


def order(by):
    eligible = sorted(rk for rk, ts in by.items() if len(ts) >= MIN_TASKS)
    return [eligible[i] for i in np.random.default_rng(SEED).permutation(len(eligible))]


def candidates(by, seq, pos):
    """The candidate tasks of the repository at ``pos`` and the generator that later draws its tasks."""
    ts = by[seq[pos]]
    rng = np.random.default_rng([SEED, pos])
    return [ts[i] for i in sorted(rng.choice(len(ts), size=min(CAND, len(ts)), replace=False))], rng


def frozen_inputs(raw, by, used, seq):
    """Everything the random draws depend on.  Written once; every later step must find it unchanged."""
    sha = {os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]
           for f in sorted(glob.glob(os.path.join(raw, "data", "*.parquet")))}
    return {"dataset_files": sha, "excluded_names": sorted(used), "eligible_in_order": seq, "numpy": np.__version__,
            "rules": {"strategies": list(STRATEGIES), "min_tasks": MIN_TASKS, "candidates": CAND, "accept_min": ACCEPT_MIN,
                      "per_repo": PER_REPO, "suite_s": SUITE_S, "seed": SEED, "target": TARGET,
                      "min_total": MIN_TOTAL, "min_repos": MIN_REPOS},
            "population_tasks": {rk: len(by[rk]) for rk in seq},
            "candidate_ids": {seq[p]: [t["instance_id"] for t in candidates(by, seq, p)[0]] for p in range(len(seq))}}


def check_inputs(a, by, used, seq):
    path, now = Path(a.out) / "pool_inputs.json", frozen_inputs(a.raw, by, used, seq)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(now, indent=1))
    elif json.loads(path.read_text()) != now:
        raise SystemExit(f"the inputs differ from {path}: dataset, exclusions, rules or numpy changed")


def image_digest(image):
    """Registry digest of image:latest (a HEAD request; it is not counted as a pull)."""
    try:
        tok = json.load(urllib.request.urlopen(
            f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{image}:pull", timeout=30))["token"]
        req = urllib.request.Request(f"https://registry-1.docker.io/v2/{image}/manifests/latest", method="HEAD", headers={
            "Authorization": f"Bearer {tok}", "Accept": "application/vnd.docker.distribution.manifest.v2+json, "
            "application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, "
            "application/vnd.oci.image.index.v1+json"})
        return urllib.request.urlopen(req, timeout=30).headers.get("Docker-Content-Digest")
    except Exception as e:
        return f"unavailable: {type(e).__name__}"


def control(t, sif, grade):
    """(passes both controls, why not, the facts) for one task; an exception is tried once more."""
    for attempt in (1, 2):
        try:
            g, e = (grade(t, sif, m) for m in ("gold", "empty"))
            break
        except Exception as ex:
            if attempt == 2:
                return False, f"exception: {type(ex).__name__}: {str(ex)[-120:]}", {}
    observed_fail = e["f2p_total"] - e["f2p_pass"] - e["f2p_missing"]
    facts = {"gold": [g["apply_ok"], g["timeout"], g["resolved"], g["f2p_pass"], g["f2p_total"], g["p2p_pass"], g["p2p_total"]],
             "empty": [e["bug_apply_fail"], e["timeout"], e["resolved"], e["f2p_pass"], e["f2p_missing"], e["f2p_total"]]}
    why = ("gold patch does not apply" if not g["apply_ok"] else "gold run timed out" if g["timeout"]
           else "gold does not resolve" if not g["resolved"] else "defect does not apply" if e["bug_apply_fail"]
           else "unchanged run timed out" if e["timeout"] else "unchanged repository resolves" if e["resolved"]
           else "no failing test observed to fail" if observed_fail < 1 else "")
    return why == "", why, facts


def read_log(path):
    """{repo_key: terminal attempt row}; at most two attempts, the second only after an infrastructure failure."""
    rows = defaultdict(list)
    if path.exists():
        for l in open(path):
            r = json.loads(l)
            rows[r["repo_key"]].append(r)
    for rk, rs in rows.items():
        assert [r["attempt"] for r in rs] == list(range(1, len(rs) + 1)) and len(rs) <= 2, f"attempts of {rk}"
        assert len(rs) == 1 or rs[0].get("infrastructure"), f"{rk} was probed again without an infrastructure failure"
    return {rk: rs[-1] for rk, rs in rows.items()}


def stats(a):
    by, used = load_pool(a.raw)
    seq = order(by)
    print(f"{len(by)} unused repositories have population tasks; {len(seq)} have at least {MIN_TASKS}")
    for i, rk in enumerate(seq):
        print(f"{i:3d} {rk:50s} {len(by[rk]):5d}  {dict(Counter(t['strategy'] for t in by[rk]))}")
    print("excluded names:", sorted(used))


def probe(a):
    from harness_graph.check import visible_failures
    from harness_graph.swe.container import LOCAL, Workspace, ensure_sif, sif_name
    from harness_graph.swe.grading import grade
    by, used = load_pool(a.raw)
    seq = order(by)
    check_inputs(a, by, used, seq)
    log_path = Path(a.out) / "pool_log.jsonl"

    def one(pos, attempt):
        rk, ts = seq[pos], by[seq[pos]]
        cand, rng = candidates(by, seq, pos)
        row = {"repo_key": rk, "position": pos, "attempt": attempt, "population_tasks": len(ts), "accepted": False,
               "image": ts[0]["image_name"], "image_digest": image_digest(ts[0]["image_name"])}
        local = os.path.join(LOCAL, "sif", sif_name(ts[0]["image_name"]))
        was_there, sif, t0 = os.path.lexists(local), None, time.time()
        try:
            sif = ensure_sif(ts[0]["image_name"])
            row["image_s"] = round(time.time() - t0, 1)
            ws = Workspace(sif, tag="probe")
            try:
                ws.setup(ts[0], True, False)                # healthy repository, all tests present
                t1 = time.time()
                base = asyncio.run(visible_failures(ws, timeout=300))
                secs = time.time() - t1
                row.update(suite_s=secs, suite_finished=base is not None, healthy_failures=None if base is None else len(base))
            finally:
                ws.close()
        except Exception as e:                              # step (1): image or workspace
            row.update(infrastructure=True, reason=f"infrastructure: {type(e).__name__}: {str(e)[-300:]}")
        if "reason" not in row and not (row["suite_finished"] and row["suite_s"] < SUITE_S):
            row["reason"] = "suite does not finish in time"
        if "reason" not in row:
            with ThreadPoolExecutor(a.workers) as ex:
                res = list(ex.map(lambda t: control(t, sif, grade), cand))
            good = [t for t, (ok, _, _) in zip(cand, res) if ok]
            row.update(candidates=len(cand), pass_controls=len(good), controls_s=round(time.time() - t0, 1),
                       control_failures=dict(Counter(why for ok, why, _ in res if not ok)),
                       controls={t["instance_id"]: {"ok": ok, "why": why, **facts} for t, (ok, why, facts) in zip(cand, res)})
            if len(good) >= ACCEPT_MIN:
                keep = [good[i] for i in sorted(rng.choice(len(good), size=min(PER_REPO, len(good)), replace=False))]
                row.update(accepted=True, tasks=[t["instance_id"] for t in keep])
            else:
                row["reason"] = "too few tasks pass the grader controls"
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if sif and not was_there and not os.path.islink(sif) and os.path.exists(sif):
            os.remove(sif)                                  # pulled by this probe into this job's own folder
        print(json.dumps({k: v for k, v in row.items() if k not in ("tasks", "controls")}), flush=True)

    def total(rows, upto=None):
        return sum(len(r["tasks"]) for r in rows.values() if r["accepted"] and (upto is None or r["position"] < upto))

    while True:                                             # in order; an owed second attempt goes first
        rows = read_log(log_path)
        retry = [r["position"] for r in rows.values() if r.get("infrastructure") and r["attempt"] == 1]
        nxt = next((p for p, rk in enumerate(seq) if rk not in rows), None)
        if retry:
            one(min(retry), 2)
        elif nxt is not None and total(rows, nxt) < TARGET:
            one(nxt, 1)
        else:
            break
    print(f"probed {len(rows)} of {len(seq)} repositories; {total(rows)} tasks in accepted repositories")


def final(a):
    by, used = load_pool(a.raw)
    seq = order(by)
    check_inputs(a, by, used, seq)
    out = Path(a.out)
    frozen = json.loads((out / "pool_inputs.json").read_text())
    rows = read_log(out / "pool_log.jsonl")
    index = {t["instance_id"]: t for rk in by for t in by[rk]}
    tasks, per_repo = [], {}
    for pos, rk in enumerate(seq):
        if len(tasks) >= TARGET:
            break
        if rk not in rows:
            raise SystemExit(f"{rk} (position {pos}) has not been probed and the target is not reached")
        r = rows[rk]
        assert r["position"] == pos and not (r.get("infrastructure") and r["attempt"] == 1), f"{rk}: retry owed"
        if not r["accepted"]:
            continue
        assert set(r["controls"]) == set(frozen["candidate_ids"][rk]), f"{rk}: candidates differ from the frozen draw"
        assert all(r["controls"][i]["ok"] for i in r["tasks"])
        ids = r["tasks"]
        if len(ids) > TARGET - len(tasks):
            rng = np.random.default_rng([SEED, pos, 1])
            ids = [ids[i] for i in sorted(rng.choice(len(ids), size=TARGET - len(tasks), replace=False))]
        tasks += [index[i] for i in ids]
        per_repo[rk] = len(ids)
    for pos, rk in enumerate(seq):
        r = rows.get(rk)
        print(f"  {pos:2d} {rk:46s} " + ("not probed (target reached)" if r is None else
              f"{'accepted' if r['accepted'] else 'rejected':8s} attempt {r['attempt']}  suite {r.get('suite_s') and round(r['suite_s'], 1)} s  "
              f"controls {r.get('pass_controls')}/{r.get('candidates')}  {r.get('reason', '')[:70]}"))
    if len(tasks) < MIN_TOTAL or len(per_repo) < MIN_REPOS:
        raise SystemExit(f"{len(tasks)} tasks from {len(per_repo)} repositories: below {MIN_TOTAL} tasks / {MIN_REPOS} "
                         f"repositories, C1 is not run on this population")
    assert len({t["instance_id"] for t in tasks}) == len(tasks)
    (out / "confirm.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    (out / "pool.json").write_text(json.dumps(
        {"n_tasks": len(tasks), "repositories": per_repo, "strategies": dict(Counter(t["strategy"] for t in tasks)),
         "gold_files": dict(sorted(Counter(t["gold_files"] for t in tasks).items())),
         "images": {rows[rk]["image"]: rows[rk]["image_digest"] for rk in per_repo},
         "sha256_confirm_jsonl": hashlib.sha256((out / "confirm.jsonl").read_bytes()).hexdigest()[:16],
         "probes": [{k: v for k, v in rows[rk].items() if k not in ("tasks", "controls")} for rk in seq if rk in rows]}, indent=1))
    print(f"{len(tasks)} tasks, {len(per_repo)} repositories {per_repo} -> {out / 'confirm.jsonl'}; "
          f"strategies {dict(Counter(t['strategy'] for t in tasks))}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("stats", stats), ("probe", probe), ("final", final)):
        s = sub.add_parser(name)
        s.add_argument("--raw", required=True)
        s.add_argument("--out", default="data/c1")
        s.add_argument("--workers", type=int, default=14)
        s.set_defaults(fn=fn)
    a = p.parse_args()
    a.fn(a)
