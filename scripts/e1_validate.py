"""E1, step 3: check every candidate split in the containers, through the code the experiment itself uses.  No model.

  python scripts/e1_validate.py [--candidates data/e1/candidates.jsonl] [--out data/e1/validated.jsonl] [--workers 16]

For each candidate task the split is VALID when all of these hold:
  workspace, bug applied     no withheld file is present; a scan of the repository and of the rest of the container
                             finishes and finds no file with the name of a withheld file, no bytecode of one
                             (`name.pyc`, `__pycache__/name.*.pyc`) and no backup copy (`name.py.*`, `name.py~`);
                             every shown failing test is collected and FAILS (not an error, not skipped), both when
                             its files are run alone and in the reduced-suite command the check node runs
  workspace, healthy tree    every shown failing test passes; the reduced suite finishes (the check node's baseline)
                             and reports none of the shown failing tests; removing the withheld files adds no
                             failure or collection error to the suite of the complete healthy tree
  grader, reference fix      resolved; every shown and every withheld failing test passes
  grader, unchanged bug      not resolved; no shown and no withheld failing test passes or is missing
A scan hit is a file-name collision, which includes unrelated files that share the name (for example a dependency's
own `test_common.py`); such a task is left out all the same.  A task that fails any condition is left out and counted
with the reason.  Errors are rerun by starting the job again.
"""
import argparse
import asyncio
import json
import shlex
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness_graph.check import visible_failures  # noqa: E402
from harness_graph.swe.container import Workspace, ensure_sif  # noqa: E402
from harness_graph.swe.grading import OK, grade, parse_log  # noqa: E402

PYTEST = "python -m pytest --disable-warnings --color=no --tb=no --verbose -p no:cacheprovider "
SCAN_DONE = "@@SCAN_DONE"


def scan_command(withheld_files: list[str], roots=("/testbed", "/")) -> str:
    """One `find` per root (the repository is a separate mount, so it is named explicitly; `-xdev` keeps the scan of
    `/` off other mounts and `/proc`, `/sys`).  Prints every hit and the end marker.  The marker is printed only when
    every root exists, every traversal ran, and the only thing left unread is a directory that this user neither owns
    nor may enter (listed as @@SCAN_DENIED_DIR): the agent runs as the same user, so it cannot open a file in such a
    directory or change its permissions.  A directory that cannot be listed but is owned by the user or can be
    entered, and any other error, makes the scan incomplete."""
    names = sorted({Path(f).name for f in withheld_files})
    pats = []
    for n in names:
        stem = Path(n).stem
        pats += [n, n + ".*", n + "~", stem + ".pyc", stem + ".pyo", stem + ".*.pyc", stem + ".*.pyo"]
    expr = " -o ".join(f"-name {shlex.quote(x)}" for x in pats)
    prune = "\\( -path /proc -o -path /sys \\) -prune -o"
    steps = "\n".join(
        f"[ -d {shlex.quote(r)} ] || {{ echo '@@SCAN_ERR no such root' {shlex.quote(r)}; ok=0; }}\n"
        f"find {shlex.quote(r)} -xdev {prune} \\( {expr} \\) -print >> \"$o\" 2>> \"$e\"\n"
        f"rc=$?; [ $rc -le 1 ] || {{ echo \"@@SCAN_ERR find rc=$rc\" {shlex.quote(r)}; ok=0; }}\n"
        f"find {shlex.quote(r)} -xdev {prune} -type d ! -readable -print >> \"$d\" 2>/dev/null"
        for r in roots)
    return ("o=$(mktemp) && e=$(mktemp) && d=$(mktemp) || exit 3\nok=1\n" + steps + "\n"
            "other=$(grep -v 'Permission denied' \"$e\" | head -n 5)\n"
            "[ -z \"$other\" ] || { echo \"$other\" | sed 's/^/@@SCAN_ERR /'; ok=0; }\n"
            "sort -u \"$d\" > \"$d.u\"\n"
            "while IFS= read -r x; do\n"
            "  if [ -O \"$x\" ] || [ -x \"$x\" ]; then echo \"@@SCAN_ERR not listed but reachable: $x\"; ok=0\n"
            "  else echo \"@@SCAN_DENIED_DIR $x\"; fi\n"
            "done < \"$d.u\"\n"
            # every "Permission denied" of the traversal has to be one of those directories, nothing else
            "[ \"$(grep -c 'Permission denied' \"$e\")\" -le \"$(wc -l < \"$d.u\")\" ] || "
            "{ echo '@@SCAN_ERR unexplained permission errors:' \"$(grep 'Permission denied' \"$e\" | head -n 3)\"; ok=0; }\n"
            "sort -u \"$o\" || ok=0\n"
            "rm -f \"$o\" \"$e\" \"$d\" \"$d.u\"\n"
            f"[ $ok = 1 ] && echo {SCAN_DONE} || exit 4")


def scan_hits(out: dict) -> list[str] | None:
    """Paths found, or None when the scan timed out, exited with an error or did not print its end marker."""
    lines = [l for l in out["output"].split("\n") if l.strip()]
    if out.get("timed_out") or out.get("returncode", 0) != 0 or not lines or lines[-1] != SCAN_DONE:
        return None
    return [l for l in lines[:-1] if not l.startswith("@@SCAN_")]


def reported(test_id: str, failures: set[str]) -> bool:
    """Is this test among the ids a `pytest -rfE` summary lists?  (An id with a space is listed up to the space.)"""
    return test_id in failures or test_id.split()[0] in failures


def one(task: dict, sif: str) -> dict:
    rec = {"instance_id": task["instance_id"], "block": task["e1"]["block"], "problems": []}
    bad = rec["problems"].append
    shown_files = " ".join(map(shlex.quote, task["e1"]["shown_files"]))
    try:
        ws = Workspace(sif, tag="e1v")
        try:
            ws.setup(task, tests_visible=True, bug=True)
            present = ws.execute("ls -d " + " ".join(map(shlex.quote, task["withheld_files"])) + " 2>/dev/null", 60)["output"].split()
            if present:
                bad("withheld_file_present")
            scan = ws.execute(scan_command(task["withheld_files"]), 600)
            hits = scan_hits(scan)
            rec["scan"] = [l for l in scan["output"].split("\n") if l.startswith("@@SCAN_")][:8]
            if hits is None:
                bad("scan_for_withheld_files_did_not_finish")
            else:
                rec["collisions_found"] = hits[:10]
                if any(h.startswith("/testbed/") for h in hits):
                    bad("file_name_collision_in_the_repository")
                if any(not h.startswith("/testbed/") for h in hits):
                    bad("file_name_collision_elsewhere_in_the_container")
            st = parse_log(ws.execute(PYTEST + shown_files, 300)["output"])
            if any(t not in st for t in task["shown_f2p"]):
                bad("shown_test_not_collected_with_the_bug")
            elif any(st[t] in OK for t in task["shown_f2p"]):
                bad("shown_test_passes_with_the_bug")
            elif any(st[t] != "FAILED" for t in task["shown_f2p"]):
                bad("shown_test_errors_or_is_skipped_with_the_bug")
            seen = asyncio.run(visible_failures(ws))
            if seen is None:
                bad("reduced_suite_does_not_finish_with_the_bug")
            elif any(not reported(t, seen) for t in task["shown_f2p"]):
                bad("reduced_suite_does_not_report_a_shown_test_with_the_bug")
        finally:
            ws.close()
        ws = Workspace(sif, tag="e1v")
        try:
            ws.setup(task, tests_visible=True, bug=False)
            st = parse_log(ws.execute(PYTEST + shown_files, 300)["output"])
            if any(st.get(t) not in OK for t in task["shown_f2p"]):
                bad("shown_test_fails_on_the_healthy_reduced_tree")
            base = asyncio.run(visible_failures(ws))
            rec["baseline_failures"] = None if base is None else len(base)
            if base is None:
                bad("reduced_suite_does_not_finish_on_the_healthy_tree")
            elif any(reported(t, base) for t in task["shown_f2p"]):
                bad("reduced_suite_reports_a_shown_test_on_the_healthy_tree")
        finally:
            ws.close()
        if base is not None:
            ws = Workspace(sif, tag="e1v")
            try:
                ws.setup({**task, "withheld_files": []}, tests_visible=True, bug=False)     # the complete healthy tree
                whole = asyncio.run(visible_failures(ws))
                if whole is None:
                    bad("complete_suite_does_not_finish_on_the_healthy_tree")
                else:
                    rec["failures_added_by_removing_files"] = sorted(base - whole)[:10]
                    if base - whole:
                        bad("removing_the_withheld_files_adds_a_failure_or_collection_error")
            finally:
                ws.close()
        gold = grade(task, sif, "gold")
        s = gold.get("split") or {}
        if not (gold["resolved"] and s.get("shown_f2p_success") and s.get("withheld_success") and s.get("partial_evaluator_accepts")):
            bad("reference_fix_not_accepted_on_both_parts")
        empty = grade(task, sif, "empty")
        s = empty.get("split") or {}
        if empty["resolved"] or s["shown"]["pass"] or s["withheld"]["pass"] or s["shown"]["missing"] or s["withheld"]["missing"]:
            bad("unchanged_bug_passes_or_misses_a_failing_test")
        rec["status"] = "valid" if not rec["problems"] else "invalid"
    except Exception as e:
        rec["status"], rec["error"] = "error", f"{type(e).__name__}: {e}"[:400]
    return rec


def scan_only(task: dict, sif: str) -> dict:
    """The scan alone, on a fresh workspace with the bug applied (used to repeat the scan after its rule was corrected)."""
    rec = {"instance_id": task["instance_id"], "block": task["e1"]["block"]}
    try:
        ws = Workspace(sif, tag="e1s")
        try:
            ws.setup(task, tests_visible=True, bug=True)
            scan = ws.execute(scan_command(task["withheld_files"]), 600)
            hits = scan_hits(scan)
            rec["scan"] = [l for l in scan["output"].split("\n") if l.startswith("@@SCAN_")][:8]
            rec["hits"] = None if hits is None else hits[:10]
            rec["status"] = "scan_did_not_finish" if hits is None else "collision" if hits else "clean"
        finally:
            ws.close()
    except Exception as e:
        rec["status"], rec["error"] = "error", f"{type(e).__name__}: {e}"[:400]
    return rec


def main(a):
    tasks = [json.loads(l) for l in open(a.candidates) if l.strip()]
    if a.scan_only:
        valid = {r["instance_id"] for r in map(json.loads, open(a.out)) if r["status"] == "valid"}
        todo = [t for t in tasks if t["instance_id"] in valid]
        sifs = {img: ensure_sif(img) for img in sorted({t["image_name"] for t in todo})}
        with ThreadPoolExecutor(a.workers) as pool:
            recs = list(pool.map(lambda t: scan_only(t, sifs[t["image_name"]]), todo))
        Path(a.scan_only).write_text("".join(json.dumps(r) + "\n" for r in recs))
        denied = Counter(l for r in recs for l in r.get("scan", []) if l.startswith("@@SCAN_DENIED_DIR"))
        print(json.dumps({"valid_tasks_scanned": len(recs), "status": Counter(r["status"] for r in recs), "denied": denied,
                          "not_clean": [(r["instance_id"], r["status"], r.get("hits"), r.get("scan"), r.get("error")) for r in recs if r["status"] != "clean"][:20]},
                         indent=1, default=dict))
        return
    out = Path(a.out)
    done = {}
    if out.exists():
        done = {r["instance_id"]: r for r in map(json.loads, open(out)) if r["status"] != "error"}
        out.write_text("".join(json.dumps(r) + "\n" for r in done.values()))
    todo = [t for t in tasks if t["instance_id"] not in done][: a.limit or None]
    print(f"{len(tasks)} candidates; {len(done)} done, {len(todo)} to check", flush=True)
    sifs = {img: ensure_sif(img) for img in sorted({t["image_name"] for t in todo})}
    with ThreadPoolExecutor(a.workers) as pool, open(out, "a") as f:
        for i, rec in enumerate(pool.map(lambda t: one(t, sifs[t["image_name"]]), todo), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{i}/{len(todo)} {rec['instance_id']} {rec['status']} {rec['problems']} {rec.get('error', '')}", flush=True)
    recs = [json.loads(l) for l in open(out)]
    print(json.dumps({"checked": len(recs), "status_by_block": Counter(f"{r['block']}: {r['status']}" for r in recs),
                      "problems": Counter(p for r in recs for p in r["problems"])}, indent=1, default=dict))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", default=str(ROOT / "data" / "e1" / "candidates.jsonl"))
    p.add_argument("--out", default=str(ROOT / "data" / "e1" / "validated.jsonl"))
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--scan-only", default="", help="write the repeated scan of the valid tasks to this file and stop")
    main(p.parse_args())
