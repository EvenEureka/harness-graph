"""P1 safeguard: can the generator reach a healthy copy of the code it is writing a check for?  No model call.

For every repository of the task file: one hidden-test workspace (defect applied), then for every file that a
reference fix of that repository touches, look for other files with the same name and parent directory anywhere in
the container (installed copies, build directories, caches, alternate checkouts) and compare them with the
working-tree file.  Also reported: where the package is imported from, compiled .pyc files of the touched modules
(bytecode of the healthy code, compiled when the image was built), and leftover version-control data.

  python scripts/p1_leak_audit.py --tasks data/v2/search.jsonl --out results/summary/p1_leak_audit.json
"""
import argparse
import json
import shlex
from collections import defaultdict
from pathlib import Path

from harness_graph.swe.container import Workspace, ensure_sif
from harness_graph.swe.grading import touched_files


def main(a):
    tasks = [json.loads(l) for l in open(a.tasks) if l.strip()]
    by_repo = defaultdict(list)
    for t in tasks:
        by_repo[t["repo_key"]].append(t)
    out = {}
    for repo, ts in sorted(by_repo.items()):
        gold = sorted({f for t in ts for f in touched_files(t["bug_patch"])})
        ws = Workspace(ensure_sif(ts[0]["image_name"]), tag="leak")
        try:
            ws.setup(ts[0], tests_visible=False, bug=True)
            names = sorted({Path(f).name for f in gold})
            expr = " -o ".join(f"-name {shlex.quote(n)}" for n in names)
            found = ws.execute(f"find / -xdev \\( -path /proc -o -path /sys -o -path /dev \\) -prune -o -type f \\( {expr} \\) -print "
                               f"2>/dev/null; find /testbed /tmp -type f \\( {expr} \\) -print 2>/dev/null", 600)["output"].split("\n")
            found = sorted({p for p in found if p.strip()})
            copies = []
            for f in gold:
                tail = "/".join(Path(f).parts[-2:])
                for p in found:
                    if p != f"/testbed/{f}" and p.endswith("/" + tail):
                        same = ws.execute(f"cmp -s {shlex.quote(p)} {shlex.quote('/testbed/' + f)} && echo SAME || echo DIFFERENT", 60)["output"].strip()
                        copies.append({"file": f, "copy": p, "content": same})
            where = ws.execute("python - <<'PY'\nimport importlib.util, json, sys\nfrom importlib import metadata\n"
                               "out = {}\nfor d in metadata.distributions():\n    f = d.read_text('direct_url.json')\n"
                               "    if f and 'testbed' in f:\n        out[d.metadata['Name']] = json.loads(f)\nprint(json.dumps(out))\nPY", 120)["output"].strip()
            pyc = ws.execute("find /testbed -name '*.pyc' | wc -l; find /testbed -maxdepth 2 \\( -name '.git' -o -name '*.orig' -o -name '*.rej' "
                             "-o -name 'build' -o -name 'dist' -o -name '*.egg-info' \\) | head -20", 120)["output"].split("\n")
            gold_pyc = [f for f in gold if ws.execute(
                f"ls {shlex.quote('/testbed/' + str(Path(f).parent))}/__pycache__/{shlex.quote(Path(f).stem)}.*.pyc 2>/dev/null | head -1", 60)["output"].strip()]
            pkgs = sorted({(Path(f).parts[1] if Path(f).parts[0] == "src" else Path(f).parts[0]) for f in gold
                           if len(Path(f).parts) > 1 and f.endswith(".py")})
            origin = {}
            for pk in pkgs:                    # where the check runner would import the package from (not executed)
                if "PKG" not in ws.execute(f"[ -f /testbed/{pk}/__init__.py ] || [ -f /testbed/src/{pk}/__init__.py ] && echo PKG", 60)["output"]:
                    continue
                o = ws.execute("printf 'import os, sys, importlib.util\\nsys.path.insert(0, os.getcwd())\\nprint(importlib.util.find_spec(\"%s\").origin)\\n' "
                               f"{shlex.quote(pk)} > /tmp/_origin.py && python /tmp/_origin.py 2>&1 | tail -n 1", 120)["output"].strip()
                origin[pk] = o
            stray = ws.execute("find /testbed -type f \\( -name '*.orig' -o -name '*.rej' -o -name '*.bak' -o -name '*~' \\) | head -200", 120)["output"].split()
            log = ws.execute("git log --oneline | wc -l; git stash list | wc -l; git branch -a | wc -l", 60)["output"].split()
        finally:
            ws.close()
        out[repo] = {"tasks": len(ts), "files_touched_by_reference_fixes": len(gold), "other_copies": copies,
                     "other_copies_with_different_content": sum(c["content"] == "DIFFERENT" for c in copies),
                     "editable_installs_from_testbed": where[-600:], "pyc_files_in_testbed": pyc[0].strip(),
                     "touched_modules_with_pyc": len(gold_pyc), "leftovers": [x for x in pyc[1:] if x.strip()],
                     "git_commits_stashes_branches": log, "import_origin": origin,
                     "stray_backup_files": [x[len("/testbed/"):] for x in stray]}
        print(repo, json.dumps({k: v for k, v in out[repo].items() if k != "other_copies"}), flush=True)
        for c in copies[:10]:
            print("   ", c, flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
