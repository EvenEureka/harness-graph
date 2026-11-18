"""Execution-based grading of a SWE-smith task in a clean container (a fresh /testbed copy, never the agent's).

Protocol (mirrors the SWE-smith harness):
  1. fresh /testbed from the image; apply the bug patch (the hidden test files are present here)
  2. apply the candidate diff with test files and test-runner hooks excluded, so edits to them are discarded:
     structural test paths (tests/ dirs, test_*.py, *_test(s).py) and hooks (conftest.py, pytest.ini, tox.ini,
     setup.cfg, pyproject.toml, .coveragerc, sitecustomize.py, usercustomize.py, *.pth)
  3. run the full test suite and parse `pytest -v` status lines
  resolved <=> every FAIL_TO_PASS test and every PASS_TO_PASS test reports PASSED/XFAIL.
Modes: "model" (a candidate diff), "gold" (the reference fix submitted as a diff: positive control),
"empty" (no change: negative control).  A task whose own reference fix touches an excluded path is ungradable.
"""
import json
import os
import re
import shlex
import shutil
import uuid

from .container import LOCAL, extract_testbed, run_script

GRADER_VERSION = os.environ.get("HG_GRADER", "v2")

# SWE-smith PythonProfile default; the three T0 repos use it unchanged (checked against swesmith/profiles/python.py)
TEST_CMD = {"default": "pytest --disable-warnings --color=no --tb=no --verbose"}
STATUS_RE = re.compile(r"^(\S+)(\s+)(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)\b")
OK = {"PASSED", "XFAIL"}
TEST_PATH_RE = re.compile(r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*\.py$|_tests?\.py$|(^|/)conftest\.py$")
HOOK_RE = re.compile(r"(^|/)(conftest\.py|pytest\.ini|tox\.ini|setup\.cfg|pyproject\.toml|\.coveragerc|"
                     r"sitecustomize\.py|usercustomize\.py|[^/]+\.pth)$")
DIFF_FILE_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.M)

EVAL_SH = """set -uo pipefail
cd /testbed
git apply --whitespace=nowarn /io/bug.patch || {{ echo "@@BUG_APPLY_FAIL"; exit 90; }}
MODE={mode}
if [ "$MODE" = gold ]; then
  git apply -R --whitespace=nowarn /io/bug.patch || {{ echo "@@PATCH_APPLY_FAIL"; exit 91; }}
elif [ "$MODE" = model ] && [ -s /io/model.patch ]; then
  if ! git apply --whitespace=nowarn {excludes} /io/model.patch 2>/tmp/apply.err; then
    cat /tmp/apply.err
    # fallback used by SWE-bench harnesses: GNU patch with fuzz
    git checkout -q -- . 2>/dev/null; git apply --whitespace=nowarn /io/bug.patch
    patch --batch --fuzz=5 -p1 -i /io/model_filtered.patch || {{ echo "@@PATCH_APPLY_FAIL"; exit 91; }}
  fi
fi
echo "@@APPLY_OK"
git status --short | head -50
source /opt/miniconda3/bin/activate
conda activate testbed
echo "@@TEST_START"
timeout -s KILL {test_timeout} {test_cmd}
echo "@@TEST_END rc=$?"
"""


def filter_patch(patch, exclude):
    """Drop file sections touching excluded paths (for the GNU-patch fallback)."""
    keep, out = True, []
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git "):
            m = re.match(r"diff --git a/(\S+) b/(\S+)", line)
            keep = not (m and (m.group(1) in exclude or m.group(2) in exclude))
        if keep:
            out.append(line)
    return "".join(out)


def touched_files(patch):
    return sorted({m.group(2) for m in DIFF_FILE_RE.finditer(patch or "")})


def _touched_both(patch):
    out = set()
    for m in DIFF_FILE_RE.finditer(patch or ""):
        out.update((m.group(1), m.group(2)))
    return out


def is_test_path(p):
    return bool(TEST_PATH_RE.search(p))


def is_hook_path(p):
    return bool(HOOK_RE.search(p))


def structural_test_files(task):
    """task["test_files"] restricted to real test files (drops source modules that only host in-file tests)."""
    return sorted(f for f in set(task.get("test_files", [])) if is_test_path(f))


def ungradable_reason(task):
    """Why the v2 grader cannot grade this task (its own gold fix touches an excluded path), or None."""
    fix = touched_files(task.get("bug_patch") or "")
    hooks = [f for f in fix if is_hook_path(f)]
    tests = [f for f in fix if is_test_path(f) and f not in hooks]
    if hooks:
        return "fix_touches_grader_hook:" + ",".join(hooks)
    if tests:
        return "fix_touches_test_path:" + ",".join(tests)
    return None


def reverse_patch(patch):
    """Textual reverse of a unified git diff (swap +/- lines, ---/+++ headers, hunk ranges, new/deleted modes):
    the gold fix as a model would submit it."""
    out, held = [], None
    for l in (patch or "").splitlines():
        if l.startswith("--- ") and held is None:
            held = l[4:]
            continue
        if l.startswith("+++ ") and held is not None:
            out += ["--- " + l[4:], "+++ " + held]
            held = None
            continue
        if l.startswith("new file mode"):
            out.append(l.replace("new file mode", "deleted file mode"))
        elif l.startswith("deleted file mode"):
            out.append(l.replace("deleted file mode", "new file mode"))
        elif l.startswith("@@"):
            m = re.match(r"^@@ -(\d+(?:,\d+)?) \+(\d+(?:,\d+)?) @@(.*)$", l)
            out.append(f"@@ -{m.group(2)} +{m.group(1)} @@{m.group(3)}" if m else l)
        elif l.startswith("+"):
            out.append("-" + l[1:])
        elif l.startswith("-"):
            out.append("+" + l[1:])
        else:
            out.append(l)
    return "\n".join(out) + "\n"


def untracked_report(ec):
    """Untracked files in an EpisodeContainer that is still running (call before ec.stop()): all, and the
    test/hook-looking subset. Scratch scripts are normal; untracked conftest.py / test files are worth flagging."""
    r = ec.execute("git ls-files --others --exclude-standard", timeout=60)
    files = [l.strip() for l in r["output"].splitlines() if l.strip()] if r["returncode"] == 0 else []
    return {"untracked": files, "untracked_test_or_hook": [f for f in files if is_test_path(f) or is_hook_path(f)]}


def diff_with_untracked(ec):
    """Like EpisodeContainer.diff() but new untracked files are included as additions (git add -N). Opt-in: the
    foundation harness still grades tracked files only (its `diff_source`)."""
    r = ec.execute(f"git add -A -N . >/dev/null 2>&1; git -c core.fileMode=false diff --no-color --no-ext-diff "
                   f"{ec.init_sha} >/tmp/_final_u.diff 2>/tmp/_final_u.err; echo RC=$?; cat /tmp/_final_u.diff; "
                   f"git reset -q >/dev/null 2>&1", timeout=120)
    head, _, body = r["output"].partition("\n")
    return body if head.strip() == "RC=0" else "", head.strip()


def _excluded(task, model_patch, version):
    if version == "v1":
        return sorted(set(task["test_files"])), [], []
    listed = set(structural_test_files(task))
    in_diff = _touched_both(model_patch)
    hooks = sorted(f for f in in_diff if is_hook_path(f))
    tests = sorted(f for f in in_diff if is_test_path(f) and f not in hooks)
    return sorted(listed | set(hooks) | set(tests)), hooks, tests


def grade(task, sif, mode="model", model_patch="", test_timeout=600, version=None, untracked=None):
    version = version or GRADER_VERSION
    if version not in ("v1", "v2"):
        raise ValueError(f"unknown grader version {version!r}")
    reason = ungradable_reason(task) if version == "v2" else None
    run_mode, patch = mode, model_patch or ""
    if version == "v2" and mode == "gold":
        run_mode, patch = "model", reverse_patch(task["bug_patch"])
    root = os.path.join(LOCAL, "eval", f"{task['instance_id'][-24:]}_{uuid.uuid4().hex[:6]}")
    excl, hooks, tests = _excluded(task, patch, version)
    if version == "v2":
        patch = filter_patch(patch, set(excl))      # excluded sections never reach git apply / GNU patch
    try:
        extract_testbed(sif, root)
        script = EVAL_SH.format(mode=run_mode, excludes=" ".join(f"--exclude={shlex.quote(f)}" for f in excl),
                                test_timeout=test_timeout, test_cmd=TEST_CMD.get(task.get("repo_key"), TEST_CMD["default"]))
        r = run_script(sif, root, script, {"bug.patch": task["bug_patch"], "model.patch": patch,
                                           "model_filtered.patch": filter_patch(patch, set(excl))},
                       timeout=test_timeout + 300)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    log = r["output"]
    st = parse_log(log)
    f2p, p2p = task["FAIL_TO_PASS"], task["PASS_TO_PASS"]
    f2p_ok = sum(st.get(t) in OK for t in f2p)
    p2p_ok = sum(st.get(t) in OK for t in p2p)
    m = re.search(r"@@TEST_END rc=(-?\d+)", log)
    out = {
        "mode": mode, "apply_ok": "@@APPLY_OK" in log, "bug_apply_fail": "@@BUG_APPLY_FAIL" in log,
        "test_rc": int(m.group(1)) if m else None, "timeout": r["timeout"] or (m is not None and m.group(1) == "137"),
        "n_parsed": len(st), "f2p_pass": f2p_ok, "f2p_total": len(f2p), "p2p_pass": p2p_ok, "p2p_total": len(p2p),
        "f2p_missing": sum(t not in st for t in f2p), "p2p_missing": sum(t not in st for t in p2p),
        "resolved": bool(f2p) and f2p_ok == len(f2p) and p2p_ok == len(p2p) and reason is None,
        "eval_s": r["secs"],
        "grader_version": version,
    }
    if version == "v2":
        in_diff = _touched_both(model_patch or "") if mode == "model" else set()
        out.update({"excluded_files": sorted(set(excl) & in_diff), "excluded_hook_files": [f for f in hooks if f in in_diff],
                    "excluded_test_files": [f for f in tests if f in in_diff], "ungradable": reason,
                    "gold_via": "model_diff" if mode == "gold" else None,
                    "untracked_test_or_hook_files": (sorted(f for f in untracked if is_test_path(f) or is_hook_path(f))
                                                     if untracked is not None else None)})
    out["log_tail"] = log[-3000:] if not out["resolved"] else log[-400:]
    return out


def parse_log(log):
    seg = log.split("@@TEST_START", 1)[-1]
    st = {}
    for line in seg.splitlines():
        m = STATUS_RE.match(line)
        if m:
            st[m.group(1)] = m.group(3)
    return st


if __name__ == "__main__":
    print(json.dumps({"grader_version": GRADER_VERSION, "test_cmd": TEST_CMD}))
