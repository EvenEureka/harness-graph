"""A1 (cap-extension audit): the task sample.  No model call, no outcome is read.

200 of the 900 follow-up tasks, spread over the 15 repositories: 13 per repository and one more for the first five
repositories in alphabetical order, drawn with a fixed seed from each repository's sorted task ids.

  python scripts/a1_sample.py --tasks data/c1/confirm.jsonl --out data/a1/tasks.jsonl
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

N, SEED = 200, 0
p = argparse.ArgumentParser()
p.add_argument("--tasks", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
rows = {}
for line in open(a.tasks):
    if line.strip():
        t = json.loads(line)
        rows[t["instance_id"]] = line if line.endswith("\n") else line + "\n"
        rows[t["instance_id"]] = (t["repo_key"], rows[t["instance_id"]])
by_repo = defaultdict(list)
for tid, (repo, _) in rows.items():
    by_repo[repo].append(tid)
repos = sorted(by_repo)
base, extra = divmod(N, len(repos))
keep = []
for i, repo in enumerate(repos):
    ids = sorted(by_repo[repo])
    k = base + (i < extra)
    rng = np.random.default_rng([SEED, i])
    keep += [ids[j] for j in sorted(rng.choice(len(ids), k, replace=False))]
with open(a.out, "w") as f:
    for tid in keep:
        f.write(rows[tid][1])
summary = {"source": a.tasks, "source_sha256": hashlib.sha256(Path(a.tasks).read_bytes()).hexdigest()[:16], "source_tasks": len(rows),
           "n": len(keep), "seed": SEED, "per_repository": {r: sum(rows[t][0] == r for t in keep) for r in repos},
           "sha256": hashlib.sha256(Path(a.out).read_bytes()).hexdigest()[:16]}
Path(a.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
