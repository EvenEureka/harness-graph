"""P1: the task file.  The search tasks minus those where the workspace holds the answer.  No model call.

Rules, applied to the audit of scripts/p1_leak_audit.py (which looks at containers, never at outcomes):
  - a repository is left out when a copy of a file its reference fixes touch, with different content, sits inside
    the repository directory or is installed as the package itself (an egg or a package directory directly under
    site-packages): the healthy code is then on disk next to the defective code;
  - a repository is left out when the check runner would import the package from somewhere else than the repository;
  - a task is left out when a backup copy of one of its hidden test files remains in the repository directory.
A copy vendored inside another project (for example under setuptools/_vendor) is recorded, not a reason to leave out.

  python scripts/p1_tasks.py --tasks data/v2/search.jsonl --audit results/summary/p1_leak_audit.json --out data/p1/tasks.jsonl
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--tasks", required=True)
p.add_argument("--audit", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
audit = json.loads(Path(a.audit).read_text())
tasks = [json.loads(l) for l in open(a.tasks) if l.strip()]
INSTALLED = re.compile(r"/site-packages/([^/]+\.egg/)?[^/]+/")


def healthy_copy(c: dict) -> bool:
    if c["content"] != "DIFFERENT" or "/" not in c["file"]:
        return False
    top = Path(c["file"]).parts[0] if Path(c["file"]).parts[0] != "src" else Path(c["file"]).parts[1]
    m = INSTALLED.search(c["copy"])
    installed = bool(m) and c["copy"][m.start():].split("/")[2 + bool(m.group(1))] == top
    return c["copy"].startswith("/testbed/") or installed


repos = {r: sorted({c["copy"] for c in v["other_copies"] if healthy_copy(c)}) for r, v in audit.items()}
repos = {r: v for r, v in repos.items() if v}
elsewhere = {r: v["import_origin"] for r, v in audit.items() if not v["import_origin"] or
             any(not o.startswith("/testbed/") for o in v["import_origin"].values())}
for r, o in elsewhere.items():                 # the check would not exercise the repository under test
    repos.setdefault(r, []).append(f"imported from outside the repository: {o}")
keep, left_out = [], {}
for t in tasks:
    stray = [s for s in audit[t["repo_key"]]["stray_backup_files"] if any(s.startswith(f) for f in t["f2p_files"])]
    if t["repo_key"] in repos:
        left_out[t["instance_id"]] = "healthy copy of the package on disk"
    elif stray:
        left_out[t["instance_id"]] = "backup copy of a hidden test file: " + ", ".join(stray)
    else:
        keep.append(t)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w") as f:
    for t in keep:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
summary = {"source": a.tasks, "tasks": len(tasks), "kept": len(keep), "left_out": left_out,
           "repositories_with_a_healthy_copy": repos, "sha256": hashlib.sha256(Path(a.out).read_bytes()).hexdigest()[:16]}
Path(a.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
