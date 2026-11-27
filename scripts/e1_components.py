"""E1, step 1: which part of a combined bug does each failing test detect?  No model call.

  python scripts/e1_components.py [--tasks data/c1/confirm.jsonl] [--out data/e1/components.jsonl] [--workers 16]

A task's ``bug_patch`` is cut into code-change COMPONENTS: its hunks, merged when they touch the same function
(top-level function or method, decorators included).  A removed line belongs to the function that holds it in the
healthy source, an added line to the function that holds it in the patched text.  A hunk that spans two functions
joins them.  A change outside any function (module or class level), a non-Python file, a new or deleted file, a
patch that does not match the source text, or a source that does not parse before or after the patch makes the task
ambiguous and it is left out.  These are individually test-detectable components of the change, not
a claim about how the benchmark planted its bugs.

Each state is then run in the task's container on the files that hold the failing tests: the healthy tree, the tree
with the whole bug, and the tree with one component alone.  A task is DECOMPOSABLE when
  - on the healthy tree every FAIL_TO_PASS test passes, and with the whole bug every one fails;
  - the components applied together give the same tree as the bug patch;
  - every component applies alone, compiles, lets every FAIL_TO_PASS test be collected and run in time, and makes
    at least one of them FAIL (a test failure, not a collection error);
  - in every state every FAIL_TO_PASS test either passes or FAILS: a setup error or a skip leaves the task out;
  - every FAIL_TO_PASS test fails under at least one single component (none fails only under the combination).
The record keeps, per component, the tests it makes fail, and which test files import one another.
"""
import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness_graph.swe.container import LOCAL, ensure_sif, extract_testbed, run_script  # noqa: E402
from harness_graph.swe.grading import OK, is_test_path, parse_log  # noqa: E402

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
STATE_TIMEOUT = 300
RUN_SH = """set -uo pipefail
cd /testbed
source /opt/miniconda3/bin/activate
conda activate testbed
state() {
  name=$1; shift
  git checkout -q -- .
  for p in "$@"; do
    git apply --whitespace=nowarn "/io/$p" || { echo "@@STATE $name APPLY_FAIL"; return; }
  done
  echo "@@STATE $name BEGIN"
  echo "@@TREE $name $(git diff --no-color --no-ext-diff | sha256sum | cut -c1-16)"
  for f in $(git diff --name-only | grep '\\.py$'); do python -m py_compile "$f" >/dev/null 2>&1 || echo "@@NOCOMPILE $name $f"; done
  timeout -s KILL %(timeout)d pytest --disable-warnings --color=no --tb=no --verbose -p no:cacheprovider %(files)s
  echo "@@STATE $name END rc=$?"
}
%(calls)s
"""


class Ambiguous(Exception):
    pass


def parse_patch(patch: str) -> list[dict]:
    """[{file, header, hunks: [{old_start, lines}]}] for a patch that only modifies existing files."""
    files, cur = [], None
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            m = re.match(r"diff --git a/(\S+) b/(\S+)", line)
            if not m or m.group(1) != m.group(2):
                raise Ambiguous("renamed_or_odd_file_header")
            cur = {"file": m.group(1), "hunks": []}
            files.append(cur)
        elif cur is None:
            continue
        elif line.startswith(("new file mode", "deleted file mode", "rename ", "Binary files", "GIT binary patch")):
            raise Ambiguous("new_deleted_or_binary_file")
        elif line.startswith(("index ", "--- ", "+++ ", "old mode", "new mode")) and not cur["hunks"]:
            continue
        elif HUNK.match(line):
            m = HUNK.match(line)
            # "@@ -a,0 +c,d @@" inserts after old line a, so the first old line the hunk could consume is a + 1
            cur["hunks"].append({"old_start": int(m.group(1)) + (m.group(2) == "0"), "lines": [line]})
        elif cur["hunks"] and (line[:1] in " +-\\" or line == ""):
            cur["hunks"][-1]["lines"].append(line if line else " ")
        else:
            raise Ambiguous("unparsed_patch_line")
    if not files or any(not f["hunks"] for f in files):
        raise Ambiguous("file_without_hunks")
    return files


def function_spans(source: str) -> list[tuple[int, int, str]]:
    """(first line, last line, qualified name) of every top-level function and method, decorators included."""
    out = []

    def walk(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                first = min([ch.lineno] + [d.lineno for d in ch.decorator_list])
                out.append((first, ch.end_lineno, prefix + ch.name))
            elif isinstance(ch, ast.ClassDef):
                walk(ch, prefix + ch.name + ".")
    walk(ast.parse(source), "")
    return out


def apply_hunks(source: str, hunks: list[dict]) -> tuple[str, list[list[tuple[str, int, str]]]]:
    """The patched text, and for every hunk its changed lines as (tag, line number, text): a removed line carries its
    number in the source, an added line its number in the patched text.  The source must match the patch exactly."""
    old, new, pos, rows = source.split("\n"), [], 0, []
    for h in sorted(hunks, key=lambda h: h["old_start"]):
        start = h["old_start"] - 1
        if start < pos or start > len(old):
            raise Ambiguous("patch_does_not_match_the_source_text")
        new += old[pos:start]
        pos, mine = start, []
        for line in h["lines"][1:]:
            tag, text = line[:1], line[1:]
            if tag == "\\":
                continue
            if tag in " -":
                if pos >= len(old) or old[pos].rstrip("\r") != text.rstrip("\r"):
                    raise Ambiguous("patch_does_not_match_the_source_text")
                if tag == " ":
                    new.append(old[pos])
                else:
                    mine.append(("-", pos + 1, text))
                pos += 1
            else:
                new.append(text)
                mine.append(("+", len(new), text))
        rows.append((h, mine))
    new += old[pos:]
    by_hunk = {id(h): mine for h, mine in rows}
    return "\n".join(new), [by_hunk[id(h)] for h in hunks]


def hunk_scopes(changed: list[tuple[str, int, str]], spans_before, spans_after) -> set[str]:
    """The functions a hunk changes.  A removed line belongs to the function that holds it in the source, an added
    line to the function that holds it in the patched text.  Blank lines are ignored; a changed line that sits in no
    top-level function or method (module level, class body) makes the task ambiguous."""
    scopes = set()
    for tag, line_no, text in changed:
        if not text.strip():
            continue
        hit = [name for a, b, name in (spans_before if tag == "-" else spans_after) if a <= line_no <= b]
        if not hit:
            raise Ambiguous("change_outside_any_function")
        scopes.add(hit[0])
    if not scopes:
        raise Ambiguous("hunk_changes_only_blank_lines")
    return scopes


def components(patch: str, read_source) -> list[dict]:
    """Components of a bug patch: [{scopes: [file::function], patch: text}]."""
    files = parse_patch(patch)
    items = []                                           # (file, hunk, scope keys)
    for f in files:
        if not f["file"].endswith(".py") or is_test_path(f["file"]):
            raise Ambiguous("non_python_or_test_file_in_bug_patch")
        try:
            src = read_source(f["file"])
            before = function_spans(src)
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
            raise Ambiguous("source_not_readable_or_not_parsed")
        patched, changed = apply_hunks(src, f["hunks"])
        try:
            after = function_spans(patched)
        except (SyntaxError, ValueError):
            raise Ambiguous("patched_source_not_parsed")
        for h, rows in zip(f["hunks"], changed):
            items.append((f["file"], h, {f"{f['file']}::{s}" for s in hunk_scopes(rows, before, after)}))
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(items)):
        for j in range(i):
            if items[i][2] & items[j][2]:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(items)):
        groups.setdefault(find(i), []).append(i)
    out = []
    for idx in sorted(groups.values(), key=lambda g: g[0]):
        text, last = [], None
        for i in idx:                                    # items are in patch order, so files stay grouped
            file, h, _ = items[i]
            if file != last:
                text += [f"diff --git a/{file} b/{file}", f"--- a/{file}", f"+++ b/{file}"]
                last = file
            text += h["lines"]
        out.append({"scopes": sorted(set().union(*(items[i][2] for i in idx))), "patch": "\n".join(text) + "\n"})
    return out


def import_units(files: list[str], read_source) -> list[list[str]]:
    """Test files grouped so that a file and any listed file it imports stay together."""
    stems = {f: Path(f).stem for f in files}
    parent = {f: f for f in files}

    def find(f):
        while parent[f] != f:
            parent[f] = parent[parent[f]]
            f = parent[f]
        return f
    for f in files:
        try:
            tree = ast.parse(read_source(f))
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
            raise Ambiguous("test_file_not_readable_or_not_parsed")
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {p for a in node.names for p in a.name.split(".")}
            elif isinstance(node, ast.ImportFrom):
                names |= set((node.module or "").split(".")) | {a.name for a in node.names}
        for g in files:
            if g != f and stems[g] in names:
                parent[find(f)] = find(g)
    units: dict[str, list[str]] = {}
    for f in files:
        units.setdefault(find(f), []).append(f)
    return sorted(sorted(u) for u in units.values())


def segments(log: str) -> dict[str, dict]:
    out = {}
    for m in re.finditer(r"@@STATE (\S+) (APPLY_FAIL|BEGIN)", log):
        name = m.group(1)
        if m.group(2) == "APPLY_FAIL":
            out[name] = {"applied": False}
            continue
        end = re.search(rf"@@STATE {re.escape(name)} END rc=(-?\d+)", log[m.end():])
        body = log[m.end(): m.end() + end.start()] if end else log[m.end():]
        tree = re.search(rf"@@TREE {re.escape(name)} (\w+)", body)
        out[name] = {"applied": True, "finished": bool(end) and end.group(1) != "137", "tree": tree.group(1) if tree else None,
                     "compiles": f"@@NOCOMPILE {name} " not in body, "status": parse_log(body)}
    return out


def one(task: dict, sif: str) -> dict:
    rec = {"instance_id": task["instance_id"], "repo_key": task["repo_key"], "strategy": task["strategy"],
           "n_f2p": len(task["FAIL_TO_PASS"]), "f2p_files": task["f2p_files"]}
    root = os.path.join(LOCAL, "e1", f"{task['instance_id'][-24:]}_{uuid.uuid4().hex[:6]}")
    try:
        if any(not is_test_path(f) for f in task["f2p_files"]):
            raise Ambiguous("failing_tests_in_source_files")
        if {t.split("::")[0] for t in task["FAIL_TO_PASS"]} != set(task["f2p_files"]):
            raise Ambiguous("test_ids_do_not_match_the_listed_files")
        extract_testbed(sif, root)
        read = lambda f: open(os.path.join(root, "testbed", f), encoding="utf-8").read()
        comps = components(task["bug_patch"], read)
        rec["units"] = import_units(task["f2p_files"], read)
        io = {"bug.patch": task["bug_patch"]} | {f"c{i}.patch": c["patch"] for i, c in enumerate(comps)}
        calls = ["state healthy", "state full bug.patch", "state joined " + " ".join(f"c{i}.patch" for i in range(len(comps)))]
        calls += [f"state c{i} c{i}.patch" for i in range(len(comps))]
        script = RUN_SH % {"timeout": STATE_TIMEOUT, "files": " ".join(task["f2p_files"]), "calls": "\n".join(calls)}
        r = run_script(sif, root, script, io, timeout=(len(calls) + 1) * (STATE_TIMEOUT + 60))
        seg = segments(r["output"])
        f2p = task["FAIL_TO_PASS"]
        ran = lambda s: s.get("applied") and s.get("finished") and all(t in s["status"] for t in f2p)
        for name in ["healthy", "full", "joined"] + [f"c{i}" for i in range(len(comps))]:
            s = seg.get(name) or {}
            if not ran(s):
                raise Ambiguous(f"state_{'component' if name.startswith('c') else name}_did_not_apply_or_run_every_test")
            if not s["compiles"]:
                raise Ambiguous("state_does_not_compile")
            if any(s["status"][t] not in OK and s["status"][t] != "FAILED" for t in f2p):
                raise Ambiguous("a_failing_test_errors_or_is_skipped_in_a_control_state")
        if any(seg["healthy"]["status"][t] not in OK for t in f2p):
            raise Ambiguous("a_failing_test_fails_on_the_healthy_tree")
        if any(seg["full"]["status"][t] in OK for t in f2p):
            raise Ambiguous("a_failing_test_passes_with_the_whole_bug")
        if seg["joined"]["tree"] != seg["full"]["tree"]:
            raise Ambiguous("components_do_not_rebuild_the_bug_patch")
        rec["components"] = []
        for i, c in enumerate(comps):
            failed = sorted(t for t in f2p if seg[f"c{i}"]["status"][t] == "FAILED")
            if not failed:
                raise Ambiguous("a_component_alone_fails_no_test")
            rec["components"].append({"scopes": c["scopes"], "detected_by": failed,
                                      "patch_sha": hashlib.sha256(c["patch"].encode()).hexdigest()[:16]})
        joint = sorted(set(f2p) - set().union(*(c["detected_by"] for c in rec["components"])))
        if joint:
            rec["joint_tests"] = len(joint)
            raise Ambiguous("a_test_fails_only_under_the_combination")
        rec["status"] = "decomposable"
    except Ambiguous as e:
        rec["status"], rec["reason"] = "excluded", str(e)
    except Exception as e:                               # container or file trouble: reported, rerun by resuming
        rec["status"], rec["reason"] = "error", f"{type(e).__name__}: {e}"[:400]
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return rec


def main(a):
    tasks = [json.loads(l) for l in open(a.tasks) if l.strip()]
    tasks = [t for t in tasks if len(t["f2p_files"]) >= 2]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        done = {r["instance_id"]: r for r in map(json.loads, open(out)) if r["status"] != "error"}
        out.write_text("".join(json.dumps(r) + "\n" for r in done.values()))
    todo = [t for t in tasks if t["instance_id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(tasks)} tasks with two or more failing-test files; {len(done)} done, {len(todo)} to run", flush=True)
    sifs = {img: ensure_sif(img) for img in sorted({t["image_name"] for t in todo})}
    with ThreadPoolExecutor(a.workers) as pool, open(out, "a") as f:
        for i, rec in enumerate(pool.map(lambda t: one(t, sifs[t["image_name"]]), todo), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{i}/{len(todo)} {rec['instance_id']} {rec['status']} {rec.get('reason', '')} "
                  f"components={len(rec.get('components', []))}", flush=True)
    recs = [json.loads(l) for l in open(out)]
    from collections import Counter
    print(json.dumps({"tasks": len(recs), "status": Counter(r["status"] for r in recs),
                      "excluded_by_reason": Counter(r["reason"] for r in recs if r["status"] == "excluded"),
                      "components_per_decomposable_task": Counter(len(r["components"]) for r in recs if r["status"] == "decomposable")},
                     indent=1, default=dict))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default=str(ROOT / "data" / "c1" / "confirm.jsonl"))
    p.add_argument("--out", default=str(ROOT / "data" / "e1" / "components.jsonl"))
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="only the first N remaining tasks (smoke run)")
    main(p.parse_args())
