"""P1: record what a generator version's run will use, before its first episode on the 188 tasks.  No model call.

  python scripts/p1_manifest.py --out data/p1/manifest_G1.json
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

from harness_graph import gencheck as G

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--tasks", default="data/p1/tasks.jsonl")
p.add_argument("--generator", default="G1", choices=sorted(G.GENERATORS))
a = p.parse_args()
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()[:16]
text = lambda s: hashlib.sha256(s.encode()).hexdigest()[:16]
pool = json.loads(Path("data/p1/patches.summary.json").read_text())
assert sha("data/p1/patches.jsonl") == pool["sha256"], "the patch file is not the one the pool step wrote"
assert sha(a.tasks) == pool["inputs_sha256"][a.tasks], "the task file changed after the pool was built"
tasks = [json.loads(l) for l in open(a.tasks) if l.strip()]
import importlib.util
spec = importlib.util.spec_from_file_location("pool", "scripts/make_c1_pool.py")
pool_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool_mod)
from harness_graph.swe.container import SIF_DIR, sif_name
images = {}
for img in sorted({t["image_name"] for t in tasks}):
    shared = Path(SIF_DIR, sif_name(img))
    assert shared.exists(), f"no pre-pulled image file for {img}: the run could not be pinned"
    h = hashlib.sha256()
    with open(shared, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    images[img] = {"sif_sha256": h.hexdigest()[:16], "sif_bytes": shared.stat().st_size, "registry_digest_now": pool_mod.image_digest(img)}
snap = sorted(Path(os.environ["HF_HOME"], "hub", "models--Qwen--Qwen3.8-27B", "snapshots").iterdir())
man = {"written": datetime.datetime.now().isoformat(timespec="seconds"), "generator": a.generator,
       "tasks": a.tasks, "sha256_tasks": sha(a.tasks), "n_tasks": len(tasks), "generation_seeds": [0, 1],
       "patches": {"file": "data/p1/patches.jsonl", **pool},
       "sha256_text": {"task prompt": text(G.GENERATORS[a.generator]["task"]), "feedback prompt": text(G.INVALID_FEEDBACK), "feedback reasons": text(json.dumps(G.REASONS, sort_keys=True)),
                       "trusted runner": text(G.GENERATORS[a.generator]["runner"])},
       "may_decline_to_write_a_check": G.GENERATORS[a.generator]["may_abstain"],
       "sha256_files": {f: sha(f) for f in ("harness_graph/gencheck.py", "harness_graph/prompts.py", "harness_graph/swe/container.py",
                                            "harness_graph/swe/grading.py", "harness_graph/llm.py", "harness_graph/analysis.py",
                                            "scripts/p1_generate.py", "scripts/p1_evaluate.py", "scripts/p1_analysis.py", "scripts/p1_patches.py", "scripts/p1_tasks.py", "scripts/p1_leak_audit.py")},
       "task_file_rules": json.loads(Path(a.tasks).with_suffix(".summary.json").read_text()),
       "code_version": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
       "uncommitted_changes": subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.split("\n")[:-1],
       "model": "Qwen/Qwen3.8-27B", "model_snapshot": [s.name for s in snap], "images": images,
       "generator_settings": {"thinking": False, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "model_calls": G.MODEL_CALLS,
                              "model_calls_per_visit": G.STEPS_PER_VISIT, "visits": 2, "command_timeout_s": 180,
                              "validation_runs": G.VALIDATION_RUNS, "check_timeout_s": G.RUN_TIMEOUT, "max_check_characters": G.MAX_ARTIFACT,
                              "workers_per_gpu": 16},
       "evaluation_settings": {"reference_runs": 2, "check_timeout_s": G.RUN_TIMEOUT, "fresh_workspace_per_run": True,
                               "patch_application": "grader v2 file filter, git apply, then GNU patch --fuzz=5",
                               "infrastructure_error": "one more attempt, then stored as an error and reported; never scored"},
       "decision": {"pass": "detection >= 0.30, reference false rejection <= 0.02, accepted-patch false rejection <= 0.05 (point "
                            "estimates), each conditional rate on >= 94 tasks, and no problem in the stored records",
                    "inconclusive": "any missing, foreign or unusable record (a run made with another check or runner, a patch "
                                    "that did not apply); or thresholds met where they can be judged but a conditional rate on < 94 tasks",
                    "fail": "with complete records: detection < 0.30, or a conditional rate with >= 94 tasks above its limit"},
       "runner_sha_in_evaluation_records": G.runner_sha(a.generator),
       "development_history": "refine-logs/EXPERIMENT_PLAN.md, section P1 (debugging rounds and prompt changes of this generator)"}
assert not man["uncommitted_changes"], f"commit first: {man['uncommitted_changes']}"
Path(a.out).write_text(json.dumps(man, indent=1))
print(json.dumps(man, indent=1))
