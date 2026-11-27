"""E1, step 4: the task sets.  No container, no model, no agent outcome.

  python scripts/e1_sample.py [--candidates data/e1/candidates.jsonl] [--validated data/e1/validated.jsonl] [--out data/e1]

Every list is in the order of sha256("e1|<task id>").
  primary      valid candidates of the primary block.  Target 240 with equal repository quotas: 240 // R each, one
               more for the first 240 % R repositories in alphabetical order, each quota filled in hash order.  A
               repository that cannot fill its quota passes the shortfall on, one task at a time, round-robin over
               the repositories in alphabetical order that still have valid candidates.  Fewer than 240 valid
               candidates: all of them are used; fewer than 160: the experiment is not run.
  incomplete   the first 120 valid candidates of the incomplete block.
  pilot        the first 16 tasks of the primary set (infrastructure check with its own seeds; they stay in the set).
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET, FLOOR, INCOMPLETE, PILOT = 240, 160, 120, 16
key = lambda tid: hashlib.sha256(f"e1|{tid}".encode()).hexdigest()


def quota_sample(ids_by_repo: dict[str, list[str]], target: int) -> list[str]:
    repos = sorted(ids_by_repo)
    pools = {r: sorted(ids_by_repo[r], key=key) for r in repos}
    if sum(map(len, pools.values())) <= target:
        return sorted((t for p in pools.values() for t in p), key=key)
    base, extra = divmod(target, len(repos))
    take = {r: min(len(pools[r]), base + (i < extra)) for i, r in enumerate(repos)}
    while sum(take.values()) < target:
        for r in repos:
            if sum(take.values()) < target and take[r] < len(pools[r]):
                take[r] += 1
    return sorted((t for r in repos for t in pools[r][:take[r]]), key=key)


def main(a):
    cand = {t["instance_id"]: t for t in (json.loads(l) for l in open(a.candidates) if l.strip())}
    val = {r["instance_id"]: r for r in (json.loads(l) for l in open(a.validated) if l.strip())}
    assert set(val) == set(cand) and all(r["status"] != "error" for r in val.values()), "validation is not complete"
    ok = [t for tid, t in cand.items() if val[tid]["status"] == "valid"]
    by_repo = defaultdict(list)
    for t in ok:
        if t["e1"]["block"] == "primary":
            by_repo[t["repo_key"]].append(t["instance_id"])
    n_primary = sum(map(len, by_repo.values()))
    primary = quota_sample(by_repo, TARGET) if n_primary >= FLOOR else []
    incomplete = sorted((t["instance_id"] for t in ok if t["e1"]["block"] == "incomplete"), key=key)[:INCOMPLETE]
    out = Path(a.out)
    dump = lambda name, ids: (out / name).write_text("".join(json.dumps(cand[i]) + "\n" for i in ids))
    dump("tasks.jsonl", primary)
    dump("tasks_incomplete.jsonl", incomplete)
    dump("pilot.jsonl", primary[:PILOT])
    sha = lambda name: hashlib.sha256((out / name).read_bytes()).hexdigest()[:16]
    summary = {"valid_primary_candidates": n_primary, "valid_incomplete_candidates": sum(t["e1"]["block"] == "incomplete" for t in ok),
               "invalid_by_block": dict(Counter(cand[i]["e1"]["block"] for i, r in val.items() if r["status"] != "valid")),
               "target": TARGET, "floor": FLOOR, "feasible": n_primary >= FLOOR,
               "primary": {"n": len(primary), "per_repository": dict(Counter(cand[i]["repo_key"] for i in primary)),
                           "components_per_task": dict(Counter(cand[i]["e1"]["components"] for i in primary)),
                           "split_seed": dict(Counter(cand[i]["e1"]["split_seed"] for i in primary)),
                           "withheld_tests_detect_every_component": sum(cand[i]["e1"]["withheld_tests_detect_every_component"] for i in primary),
                           "sha256": sha("tasks.jsonl")},
               "incomplete": {"n": len(incomplete), "per_repository": dict(Counter(cand[i]["repo_key"] for i in incomplete)),
                              "sha256": sha("tasks_incomplete.jsonl")},
               "pilot": {"n": min(PILOT, len(primary)), "sha256": sha("pilot.jsonl")}}
    (out / "tasks.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", default=str(ROOT / "data" / "e1" / "candidates.jsonl"))
    p.add_argument("--validated", default=str(ROOT / "data" / "e1" / "validated.jsonl"))
    p.add_argument("--out", default=str(ROOT / "data" / "e1"))
    main(p.parse_args())
