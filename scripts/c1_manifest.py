"""C1: record what the runs will use, before the first episode.  No model call.

  python scripts/c1_manifest.py --out data/c1/manifest.json
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--out", default="data/c1/manifest.json")
a = p.parse_args()
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()[:16]
pool = json.loads(Path("data/c1/pool.json").read_text())
assert sha("data/c1/confirm.jsonl") == pool["sha256_confirm_jsonl"], "the task file is not the one the pool step wrote"
graphs = {}
for d in ("graphs_c1", "graphs_c1_plain"):
    for f in sorted(Path(d).glob("*.yaml")):
        assert sha(f) == sha(Path("graphs_confirm") / f.name), f"{f} differs from the graph confirmed in stage 5"
        graphs[str(f)] = sha(f)
assert len(graphs) == 3
selected = json.loads(Path("results/summary/selection.json").read_text())["searched_graph"]
assert f"graphs_c1/{selected}.yaml" in graphs
snap = sorted(Path(os.environ["HF_HOME"], "hub", "models--Qwen--Qwen3.8-27B", "snapshots").iterdir())
man = {"written": datetime.datetime.now().isoformat(timespec="seconds"), "tasks": "data/c1/confirm.jsonl",
       "sha256_tasks": sha("data/c1/confirm.jsonl"), "n_tasks": pool["n_tasks"], "repositories": pool["repositories"],
       "strategies": pool["strategies"], "graphs": graphs, "selected_graph": selected,
       "sha256_scripts": {f: sha(f) for f in ("scripts/c1_analysis.py", "scripts/make_c1_pool.py", "harness_graph/analysis.py",
                                              "harness_graph/executor.py", "harness_graph/check.py", "harness_graph/runner.py")},
       "code_version": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
       "uncommitted_changes": subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.split("\n")[:-1],
       "model": "Qwen/Qwen3.8-27B", "model_snapshot": [s.name for s in snap], "images": pool["images"],
       "run": {"thinking": False, "model_calls": 50, "deadline_s": 3600, "workers_per_gpu": 16, "tests": "visible",
               "rollouts": {"selected and h04": [0, 1], "h01": [0]}}}
Path(a.out).write_text(json.dumps(man, indent=1))
print(json.dumps({k: v for k, v in man.items() if k not in ("images",)}, indent=1))
