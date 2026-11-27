"""E1 and E2: components of a bug patch, the split rule, per-part grading counts, sampling and the read-outs."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e1_components as C  # noqa: E402
import e1_sample as SA  # noqa: E402
import e1_split as SP  # noqa: E402
from harness_graph.swe.grading import split_counts  # noqa: E402

SRC = """import os


def a(x):
    y = x + 1
    return y


class K:
    @property
    def b(self):
        return 2

    def c(self):
        return 3
"""
PATCH = """diff --git a/pkg/m.py b/pkg/m.py
index 111..222 100644
--- a/pkg/m.py
+++ b/pkg/m.py
@@ -3,5 +3,5 @@ import os
 
 def a(x):
-    y = x + 1
+    y = x - 1
     return y
 
@@ -9,5 +9,5 @@ class K:
 class K:
     @property
     def b(self):
-        return 2
+        return 20
 
@@ -14,2 +14,2 @@ class K:
     def c(self):
-        return 3
+        return 30
"""


def test_a_patch_is_cut_into_one_component_per_function_and_they_rebuild_it():
    comps = C.components(PATCH, lambda f: SRC)
    assert [c["scopes"] for c in comps] == [["pkg/m.py::a"], ["pkg/m.py::K.b"], ["pkg/m.py::K.c"]]
    assert all(c["patch"].startswith("diff --git a/pkg/m.py b/pkg/m.py") for c in comps)
    body = lambda text: [l for l in text.splitlines() if not l.startswith(("diff ", "index ", "--- ", "+++ "))]
    assert sum((body(c["patch"]) for c in comps), []) == body(PATCH)


def test_two_hunks_in_one_function_are_one_component_and_a_decorator_belongs_to_its_function():
    head, first = PATCH.split("@@ -9,5 +9,5 @@")[0], "@@ -10,2 +10,2 @@ class K:\n-    @property\n+    @staticmethod\n     def b(self):\n"
    two = head + first + "@@ -12,1 +12,1 @@ class K:\n-        return 2\n+        return 20\n"
    comps = C.components(two, lambda f: SRC)
    assert [c["scopes"] for c in comps] == [["pkg/m.py::a"], ["pkg/m.py::K.b"]]
    with pytest.raises(C.Ambiguous, match="patch_does_not_match_the_source_text"):          # overlapping hunks
        C.components(head + "@@ -9,5 +9,5 @@" + PATCH.split("@@ -9,5 +9,5 @@")[1].split("@@ -14,2")[0] + first, lambda f: SRC)


@pytest.mark.parametrize("patch,why", [
    (PATCH.replace("-    y = x + 1\n+    y = x - 1", "-    y = x + 1\n+    y = x - 1").replace("@@ -3,5 +3,5 @@ import os\n \n def a(x):\n-    y = x + 1\n+    y = x - 1\n     return y\n \n",
                                                                                         "@@ -1,2 +1,2 @@\n-import os\n+import sys\n \n"), "change_outside_any_function"),
    (PATCH.replace("a/pkg/m.py b/pkg/m.py", "a/pkg/m.txt b/pkg/m.txt"), "non_python_or_test_file_in_bug_patch"),
    (PATCH.replace("index 111..222 100644", "new file mode 100644"), "new_deleted_or_binary_file"),
])
def test_ambiguous_patches_are_refused(patch, why):
    with pytest.raises(C.Ambiguous, match=why):
        C.components(patch, lambda f: SRC)


def _one_hunk(lines, old_start, n_old, n_new):
    return f"diff --git a/pkg/m.py b/pkg/m.py\n--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -{old_start},{n_old} +{old_start},{n_new} @@\n" + lines


def test_a_line_inserted_after_a_function_belongs_to_what_holds_it_in_the_patched_text():
    # module level, directly after the last line of a(): not part of a()
    with pytest.raises(C.Ambiguous, match="change_outside_any_function"):
        C.components(_one_hunk("     return y\n+GLOBAL = 2\n", 6, 1, 2), lambda f: SRC)
    # the same position, indented: it continues a()
    assert [c["scopes"] for c in C.components(_one_hunk("     return y\n+    y = 0\n", 6, 1, 2), lambda f: SRC)] == [["pkg/m.py::a"]]
    # a statement added in the class body between two methods belongs to neither
    with pytest.raises(C.Ambiguous, match="change_outside_any_function"):
        C.components(_one_hunk("         return 2\n+    flag = True\n", 12, 1, 2), lambda f: SRC)
    # a new method is its own scope
    new = C.components(_one_hunk("         return 2\n+\n+    def d(self):\n+        return 4\n", 12, 1, 4), lambda f: SRC)
    assert [c["scopes"] for c in new] == [["pkg/m.py::K.d"]]
    # an insertion with no old line in the hunk ("-6,0"): after line 6
    assert [c["scopes"] for c in C.components(
        "diff --git a/pkg/m.py b/pkg/m.py\n--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -6,0 +7,1 @@\n+    y = 0\n", lambda f: SRC)] == [["pkg/m.py::a"]]


def test_a_patch_that_does_not_match_the_source_or_leaves_it_unparsable_is_refused():
    with pytest.raises(C.Ambiguous, match="patch_does_not_match_the_source_text"):
        C.components(PATCH, lambda f: SRC.replace("y = x + 1", "y = x + 2"))
    with pytest.raises(C.Ambiguous, match="patched_source_not_parsed"):
        C.components(_one_hunk("-    return y\n+    return (y\n", 6, 1, 1), lambda f: SRC)


def test_the_scan_finds_bytecode_and_backup_copies_of_a_withheld_file(tmp_path):
    import e1_validate as V
    (tmp_path / "tests" / "__pycache__").mkdir(parents=True)
    (tmp_path / "other").mkdir()
    planted = ["tests/__pycache__/test_b.cpython-310.pyc", "tests/__pycache__/test_b.cpython-310-pytest-8.3.4.pyc",
               "tests/test_b.pyc", "tests/test_b.py.orig", "tests/test_b.py~", "other/test_b.py"]
    for f in planted + ["tests/test_a.py", "tests/__pycache__/test_a.cpython-310.pyc", "tests/test_bb.py"]:
        (tmp_path / f).write_text("x")
    def run(files, roots=(str(tmp_path),)):
        r = subprocess.run(["bash", "-c", V.scan_command(files, roots=roots)], capture_output=True, text=True)
        return {"output": r.stdout, "returncode": r.returncode, "timed_out": False}
    hits = V.scan_hits(run(["tests/test_b.py"]))
    assert sorted(hits) == sorted(str(tmp_path / f) for f in planted)
    assert V.scan_hits(run(["tests/test_zz.py"])) == []
    assert V.scan_hits({"output": "/x/test_b.py\n", "timed_out": False}) is None          # no end marker: the scan did not finish
    assert V.scan_hits({"output": V.SCAN_DONE + "\n", "timed_out": True}) is None
    assert V.scan_hits({"output": V.SCAN_DONE + "\n", "returncode": 4, "timed_out": False}) is None
    # a root that does not exist is a failed scan, also when another root was scanned without trouble before or after it
    gone = str(tmp_path / "no_such_dir")
    for roots in ((gone,), (gone, str(tmp_path)), (str(tmp_path), gone)):
        out = run(["tests/test_b.py"], roots)
        assert V.scan_hits(out) is None and "@@SCAN_ERR" in out["output"] and V.SCAN_DONE not in out["output"]
    # a directory that cannot be listed but can be entered hides a readable file with a known name: incomplete scan
    import os
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    (hidden / "test_b.py").write_text("x")
    os.chmod(hidden, 0o311)
    try:
        if os.geteuid() != 0:
            out = run(["tests/test_b.py"])
            assert V.scan_hits(out) is None and "not listed but reachable" in out["output"]
            os.chmod(hidden, 0)                              # no entry either, but the user owns it and can change that
            out = run(["tests/test_b.py"])
            assert V.scan_hits(out) is None and "not listed but reachable" in out["output"]
    finally:
        os.chmod(hidden, 0o700)
    assert len(V.scan_hits(run(["tests/test_b.py"]))) == len(planted) + 1
    # a directory of another user that may not be entered is listed and tolerated
    if os.geteuid() != 0 and os.path.isdir("/root") and not os.access("/root", os.R_OK | os.X_OK) and os.stat("/root").st_uid != os.geteuid():
        out = run(["tests/test_b.py"], ("/root",))
        assert V.scan_hits(out) == [] and "@@SCAN_DENIED_DIR /root" in out["output"]


def test_workspace_setup_removes_bytecode_and_backups_with_a_hidden_file():
    from harness_graph.swe.container import SETUP_SH
    assert '"$d/__pycache__/$s".*.pyc' in SETUP_SH and '"$d/$s.pyc"' in SETUP_SH and '"$f".*' in SETUP_SH
    assert SETUP_SH.index("hide_files.txt") < SETUP_SH.index("git init")          # removed before the commit the agent can read


def test_test_files_that_import_one_another_stay_in_one_unit():
    src = {"tests/test_a.py": "from tests.test_b import helper\n", "tests/test_b.py": "import os\n", "tests/test_c.py": "x = 1\n"}
    assert C.import_units(sorted(src), src.__getitem__) == [["tests/test_a.py", "tests/test_b.py"], ["tests/test_c.py"]]


def _rec(detected):
    return {"instance_id": "t1", "units": [["tests/test_a.py"], ["tests/test_b.py"], ["tests/test_c.py"]],
            "components": [{"detected_by": d} for d in detected]}


TASK = {"FAIL_TO_PASS": ["tests/test_a.py::x", "tests/test_b.py::y", "tests/test_c.py::z"]}


def test_the_split_is_deterministic_shows_about_half_and_prefers_a_seed_where_shown_tests_detect_every_component():
    shown, withheld = SP.split("t1", _rec([])["units"], 0)
    assert len(shown) == 2 and len(withheld) == 1 and SP.split("t1", _rec([])["units"], 0) == (shown, withheld)
    e = SP.choose(_rec([["tests/test_a.py::x"], ["tests/test_b.py::y"]]), TASK)       # a seed that shows a and b exists
    assert e["block"] == "primary" and e["components_missed_by_shown_tests"] == 0
    assert set(e["shown_files"]) >= {"tests/test_a.py", "tests/test_b.py"} and e["withheld_f2p"] == ["tests/test_c.py::z"]
    assert not e["withheld_tests_detect_every_component"]
    e = SP.choose(_rec([["tests/test_a.py::x"], ["tests/test_b.py::y"], ["tests/test_c.py::z"]]), TASK)   # no split can show all
    assert e["block"] == "incomplete" and e["split_seed"] == 0 and e["components_missed_by_shown_tests"] == 1
    assert SP.choose({**_rec([["tests/test_a.py::x"]]), "units": [["tests/test_a.py", "tests/test_b.py", "tests/test_c.py"]]}, TASK) is None


def test_per_part_counts_and_the_success_definitions():
    task = {"FAIL_TO_PASS": ["tests/test_a.py::x", "tests/test_b.py::y"], "shown_f2p": ["tests/test_a.py::x"],
            "withheld_f2p": ["tests/test_b.py::y"], "withheld_files": ["tests/test_b.py"],
            "PASS_TO_PASS": ["tests/test_a.py::p", "tests/test_b.py::q"]}
    s = split_counts(task, {"tests/test_a.py::x": "PASSED", "tests/test_b.py::y": "FAILED", "tests/test_a.py::p": "PASSED", "tests/test_b.py::q": "PASSED"})
    assert s["shown_f2p_success"] and not s["withheld_f2p_success"] and not s["withheld_success"] and s["partial_evaluator_accepts"]
    # every failing test passes, a passing test inside the withheld file breaks: the primary outcome fails, but this is
    # not a rejection by the withheld failing tests, and success on the shown failing tests does not depend on it
    s = split_counts(task, {"tests/test_a.py::x": "PASSED", "tests/test_b.py::y": "PASSED", "tests/test_a.py::p": "PASSED", "tests/test_b.py::q": "FAILED"})
    assert s["shown_f2p_success"] and s["withheld_f2p_success"] and not s["withheld_success"] and s["partial_evaluator_accepts"]
    assert s["withheld"] == {"pass": 1, "total": 1, "missing": 0} and s["p2p_available"]["total"] == 1
    s = split_counts(task, {"tests/test_a.py::x": "PASSED", "tests/test_b.py::y": "PASSED", "tests/test_a.py::p": "FAILED", "tests/test_b.py::q": "PASSED"})
    assert s["shown_f2p_success"] and not s["withheld_success"] and not s["partial_evaluator_accepts"]
    s = split_counts(task, {"tests/test_a.py::x": "PASSED"})
    assert s["withheld"]["missing"] == 1 and not s["withheld_success"] and not s["withheld_f2p_success"]


def test_quotas_are_equal_and_a_short_repository_passes_its_shortfall_on():
    ids = {"r1": [f"r1-{i}" for i in range(10)], "r2": [f"r2-{i}" for i in range(2)], "r3": [f"r3-{i}" for i in range(10)]}
    got = SA.quota_sample(ids, 12)
    count = lambda r: sum(t.startswith(r) for t in got)
    assert len(got) == 12 and count("r2") == 2 and count("r1") == 5 and count("r3") == 5
    assert SA.quota_sample(ids, 12) == got and SA.quota_sample(ids, 100) == sorted(sum(ids.values(), []), key=SA.key)


P, K_ = "g-p0-fn-t0-vn-k0-c0-snon", "g-p0-fn-t0-vn-kcon2-c0-snon"


def _run(task, graph, seed, shown, withheld, p2p_in_withheld_file=True, **kw):
    """One stored run of a task with one shown and one withheld failing test and one passing test in each file."""
    n = lambda ok: {"pass": int(ok), "total": 1, "missing": 0}
    split = {"shown": n(shown), "withheld": n(withheld), "p2p": {"pass": 1 + int(p2p_in_withheld_file), "total": 2, "missing": 0},
             "p2p_available": n(True), "shown_f2p_success": shown, "withheld_f2p_success": withheld,
             "withheld_success": withheld and p2p_in_withheld_file, "partial_evaluator_accepts": shown}
    r = {"episode_id": f"m/{graph}/{task}/s{seed}", "task_id": task, "graph": graph, "seed": seed, "model": "qwen3.8-27b",
         "status": "ok", "tests": "visible", "budget": {"model_calls": 50, "wall_s": 3600}, "diff": "d",
         "resolved": shown and withheld and p2p_in_withheld_file,
         "grade": {"split": split}, "model_calls": 10, "prompt_tokens": 100, "completion_tokens": 10, "exit_reason": "end"}
    r.update(kw)
    return r


def _task(i):
    return {"instance_id": f"t{i}", "repo_key": f"r{i % 2}", "shown_f2p": ["tests/test_a.py::x"], "withheld_f2p": ["tests/test_b.py::y"],
            "withheld_files": ["tests/test_b.py"], "PASS_TO_PASS": ["tests/test_a.py::p", "tests/test_b.py::q"],
            "e1": {"block": "primary", "withheld_tests_detect_every_component": i == 0}}


def _e1(tmp_path, rows, n=4):
    tasks = [_task(i) for i in range(n)]
    (tmp_path / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    (tmp_path / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    subprocess.run([sys.executable, str(ROOT / "scripts/e1_analysis.py"), "--runs", str(tmp_path / "runs.jsonl"),
                    "--tasks", str(tmp_path / "tasks.jsonl"), "--out", str(tmp_path / "out.json")], check=True, capture_output=True)
    return json.loads((tmp_path / "out.json").read_text())


def test_the_e1_read_out_scores_the_withheld_part_and_the_rejection_rate(tmp_path):
    rows = []
    for i in range(4):
        for s in (0, 1):
            rows.append(_run(f"t{i}", P, s, shown=i < 2, withheld=i < 1))          # plain: t0 passes both, t1 only the shown part
            rows.append(_run(f"t{i}", K_, s, shown=True, withheld=i < 3))          # check: every shown part, withheld on three tasks
    out = _e1(tmp_path, rows)
    b = out["primary_block"]
    assert out["verdict"] == "complete" and b["every_withheld_failing_test_and_every_passing_test"]["check_minus_plain"]["mean"] == 0.5
    assert b["every_shown_failing_test"]["check_and_retry"]["mean"] == 1.0
    assert b["every_shown_failing_test"]["plain_loop"]["mean"] == 0.5
    rej = b["withheld_test_rejection_among_partial_evaluator_acceptances"]
    assert (rej["plain_loop"]["rejected"], rej["plain_loop"]["accepted"]) == (2, 4) and (rej["check_and_retry"]["rejected"], rej["check_and_retry"]["accepted"]) == (2, 8)
    assert out["primary"]["reading"] in ("positive transfer advantage supported", "inconclusive")
    assert "secondary_withheld_tests_detect_every_component" in out


def test_a_broken_passing_test_in_a_withheld_file_fails_the_primary_outcome_but_is_not_a_withheld_test_rejection(tmp_path):
    rows = [_run(f"t{i}", g, s, True, True, p2p_in_withheld_file=not (i == 0 and g == K_)) for i in range(4) for g in (P, K_) for s in (0, 1)]
    b = _e1(tmp_path, rows)["primary_block"]
    assert b["every_withheld_failing_test_and_every_passing_test"]["check_and_retry"]["mean"] == 0.75
    rej = b["withheld_test_rejection_among_partial_evaluator_acceptances"]["check_and_retry"]
    assert (rej["rejected"], rej["accepted"]) == (0, 8) and rej["with_passing_tests_of_withheld_files"]["rejected"] == 2


def test_stored_counts_that_disagree_with_the_task_stop_the_read_out(tmp_path):
    rows = [_run(f"t{i}", g, s, True, True) for i in range(4) for g in (P, K_) for s in (0, 1)]
    rows[0]["grade"]["split"]["withheld"]["total"] = 3                                  # not the task's one withheld test
    out = _e1(tmp_path, rows)
    assert out["verdict"] == "incomplete" and "withheld total" in out["stored_record_problems"][0]
    rows = [_run(f"t{i}", g, s, True, True) for i in range(4) for g in (P, K_) for s in (0, 1)]
    rows[0]["grade"]["split"]["withheld_success"] = False                               # a flag that its own counts contradict
    assert _e1(tmp_path, rows)["verdict"] == "incomplete"


def test_the_pilot_check_counts_missing_repeated_runs_and_unmeasured_baselines(tmp_path, capsys):
    import e1_pilot_check as PC
    rows = [_run(f"t{i}", g, s, True, True, events=[]) for i in range(2) for g in (P, K_) for s in (90, 91)]
    for r in rows:
        r["node_logs"] = [{"node": "check", "baseline": "measured:0", "summary": "pass"}] if r["graph"] == K_ else []
    (tmp_path / "tasks.jsonl").write_text("".join(json.dumps(_task(i)) + "\n" for i in range(2)))
    def check(rs):
        (tmp_path / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
        return PC.main(type("A", (), {"runs": str(tmp_path / "runs.jsonl"), "tasks": str(tmp_path / "tasks.jsonl")}))
    assert check(rows)["defects"] == 0
    assert check(rows[:-1])["counts"]["missing runs"] == 1 and check(rows + rows[:1])["counts"]["repeated runs"] == 1
    rows[-1]["node_logs"] = [{"node": "check", "summary": "pass_untested", "baseline": "unavailable"}]
    c = check(rows)
    assert c["counts"]["check visits without a measured baseline"] == 1 and c["defects"] == 2
    rows[-1]["node_logs"], rows[0]["events"] = [], [{"role": "observation", "content": "cat tests/test_b.py: No such file"}]
    assert check(rows)["n_such_runs"] == 1


def test_a_missing_or_foreign_run_and_an_ungraded_empty_diff(tmp_path):
    rows = [_run(f"t{i}", g, s, True, True) for i in range(4) for g in (P, K_) for s in (0, 1)]
    assert _e1(tmp_path, rows[:-1])["verdict"] == "incomplete"
    assert _e1(tmp_path, rows + [_run("t0", P, 7, True, True)])["verdict"] == "incomplete"
    rows[0] = _run("t0", P, 0, False, False, diff="", grade={}, resolved=False)       # empty diff: not graded, counts as passing nothing
    out = _e1(tmp_path, rows)
    assert out["verdict"] == "complete" and out["primary_block"]["cost_per_run"]["plain_loop"]["not_graded_empty_diff"] == 1


def test_the_e2_read_out_names_the_primary_contrast(tmp_path):
    arms = {"plain_loop": P, "full_gate": K_, "binary_gate": "e2-binary-gate", "forced_continuation": "e2-forced-continue"}
    rate = {"plain_loop": 1, "full_gate": 4, "binary_gate": 3, "forced_continuation": 2}       # tasks solved out of 4
    rows = []
    for name, g in arms.items():
        for i in range(4):
            for s in (4, 5):
                r = _run(f"t{i}", g, s, True, True, resolved=i < rate[name])
                r["node_logs"] = [{"node": "check", "outcome": "fail"}] if name != "plain_loop" else []
                rows.append(r)
    (tmp_path / "tasks.jsonl").write_text("".join(json.dumps({"instance_id": f"t{i}", "repo_key": "r"}) + "\n" for i in range(4)))
    (tmp_path / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    subprocess.run([sys.executable, str(ROOT / "scripts/e2_analysis.py"), "--runs", str(tmp_path / "runs.jsonl"),
                    "--tasks", str(tmp_path / "tasks.jsonl"), "--out", str(tmp_path / "out.json")], check=True, capture_output=True)
    out = json.loads((tmp_path / "out.json").read_text())
    c = out["contrasts"]
    assert out["verdict"] == "complete" and c["full_gate_minus_binary_gate"]["role"] == "primary" and c["full_gate_minus_binary_gate"]["mean"] == 0.25
    assert c["full_gate_minus_plain_loop"]["mean"] == 0.75 and out["per_run"]["plain_loop"]["runs_sent_back_at_least_once"] == 0
