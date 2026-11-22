"""P1, step 3: run every delivered check on the unrepaired tree, the reference tree (twice) and the stored patches.

No model call.  One fresh hidden-test workspace per run.  Resumable: a (check, tree, repetition) already stored
with status ok is skipped.  A run whose workspace cannot be built or whose check cannot be placed is tried once
more and then stored as an infrastructure error; such rows are counted by the analysis, never scored.

  python scripts/p1_evaluate.py --gen results/raw/p1/gen_G1.jsonl --tasks data/v2/search.jsonl \
      --patches data/p1/patches.jsonl --out results/raw/p1/eval_G1.jsonl --workers 14
"""
import argparse
import concurrent.futures
import hashlib
import json
import threading
import time
from pathlib import Path

from harness_graph import gencheck as G
from harness_graph.swe.container import ensure_sif

REFERENCE_RUNS = 2


def load_checks(path):
    """The last ok record of each generator episode."""
    out = {}
    for line in open(path):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("status") == "ok":
            out[e["episode_id"]] = e
    return out


def jobs_for(check, patches):
    yield ("unrepaired", "", 0, "")
    for r in range(REFERENCE_RUNS):
        yield ("reference", "", r, "")
    for p in patches:
        yield ("patch", p["patch_id"], 0, p["diff"])


def main(a):
    tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in open(a.tasks) if l.strip()}
    patches = {}
    for line in open(a.patches):
        p = json.loads(line)
        patches.setdefault(p["task_id"], []).append(p)
    checks = {k: v for k, v in load_checks(a.gen).items() if v["valid"]}
    out = Path(a.out)
    versions = {a.runner_of} if a.runner_of else {c["generator"] for c in checks.values()}
    if len(versions) != 1:
        raise SystemExit(f"checks of several generator versions in one file: {sorted(versions)}")
    runner_sha = G.runner_sha(versions.pop())
    done = set()
    if out.exists():
        for line in open(out):
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("runner_sha") != runner_sha:              # never mix runs made with different runners in one file
                raise SystemExit(f"{out} holds runs made with another runner ({e.get('runner_sha')}); use another output file")
            if e.get("status") == "ok":
                done.add((e["episode_id"], e["tree"], e["patch_id"], e["rep"]))
    todo = [(c, *j) for c in checks.values() for j in jobs_for(c, patches.get(c["task_id"], []))
            if (c["episode_id"], j[0], j[1], j[2]) not in done]
    sifs = {img: ensure_sif(img) for img in sorted({tasks[c["task_id"]]["image_name"] for c, *_ in todo})}
    print(f"{len(checks)} delivered checks; {len(done)} runs stored, {len(todo)} to do", flush=True)
    lock, n, t0 = threading.Lock(), [0, 0], time.time()

    def one(c, tree, pid, rep, diff):
        t = tasks[c["task_id"]]
        row = {"episode_id": c["episode_id"], "task_id": c["task_id"], "seed": c["seed"], "tree": tree, "patch_id": pid, "rep": rep,
               "artifact_sha": hashlib.sha256(c["artifact"].encode()).hexdigest()[:16], "runner_sha": runner_sha}
        for attempt in (1, 2):
            try:
                runner = G.GENERATORS[a.runner_of or c["generator"]]["runner"]
                row.update(status="ok", attempt=attempt, **G.run_on_tree(t, sifs[t["image_name"]], c["artifact"], tree, diff, runner))
                break
            except Exception as e:
                row.update(status="infra_error", attempt=attempt, error=f"{type(e).__name__}: {e}"[:500])
        with lock:
            with open(out, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n[0] += 1
            n[1] += row["status"] != "ok"
            if n[0] % 50 == 0 or n[0] == len(todo):
                print(f"[{time.strftime('%H:%M:%S')}] {n[0]}/{len(todo)} runs, {n[1]} infrastructure errors, "
                      f"{time.time() - t0:.0f}s", flush=True)

    with concurrent.futures.ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda j: one(*j), todo))
    print(f"DONE {n[0]} runs, {n[1]} infrastructure errors", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gen", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--patches", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--runner-of", default="", help="use this generator version's runner instead of the check's own (sensitivity runs)")
    main(p.parse_args())
