"""Integration tests against a real SWE-smith image (need apptainer).  Skipped unless HG_TEST_TASKS points
to a task file; scripts/sge/selftest.sge sets it.  These double as the positive and negative controls."""
import asyncio
import json
import os
import re

import pytest

TASKS = os.environ.get("HG_TEST_TASKS")
pytestmark = pytest.mark.skipif(not TASKS, reason="set HG_TEST_TASKS to a SWE-smith task file")


@pytest.fixture(scope="module")
def env():
    from harness_graph.swe.container import Workspace, ensure_sif
    task = json.loads(open(TASKS).readline())
    sif = ensure_sif(task["image_name"])
    ws = Workspace(sif, tag="test")
    ws.setup(task)
    yield task, sif, ws
    ws.close()


def test_agent_view_hides_failing_tests_and_history(env):
    task, _, ws = env
    assert ws.execute("git log --oneline | wc -l")["output"].strip() == "1"
    for f in task["f2p_files"]:
        assert ws.execute(f"test -e {f}")["returncode"] != 0
    assert ws.execute("curl -s -m 5 https://example.com >/dev/null")["returncode"] != 0     # no network


def test_state_persists_between_commands_and_timeouts_are_enforced(env):
    _, _, ws = env
    ws.execute("echo hello > /tmp/note.txt")
    assert ws.execute("cat /tmp/note.txt")["output"].strip() == "hello"
    out = ws.execute("sleep 30", timeout=2)
    assert out["timed_out"] and out["secs"] < 20
    bad = ws.execute("echo a\x00b")                   # a null byte cannot be passed to a process
    assert bad["returncode"] != 0 and "null byte" in bad["output"] and not bad["timed_out"]


def test_check_and_grader_on_empty_gold_and_broken_patches(env):
    from harness_graph.check import run_check, visible_failures
    from harness_graph.swe.grading import grade, reverse_patch
    task, sif, ws = env
    baseline = asyncio.run(visible_failures(ws))
    assert baseline is not None
    ok, _, reason = asyncio.run(run_check(ws, baseline))
    assert not ok and reason == "empty_diff" and ws.diff() == ""
    assert grade(task, sif, "empty")["resolved"] is False

    gold = reverse_patch(task["bug_patch"])
    open(os.path.join(ws.root, "testbed", "_gold.patch"), "w").write(gold)
    assert ws.execute("git apply _gold.patch && rm _gold.patch")["returncode"] == 0
    ok, _, reason = asyncio.run(run_check(ws, baseline))
    assert ok and reason == "pass"
    g = grade(task, sif, "model", ws.diff())
    assert g["resolved"] and g["f2p_pass"] == g["f2p_total"] > 0

    changed = ws.execute(f"git diff --name-only {ws.init_sha}")["output"].split()[0]
    ws.execute(f"echo 'def broken(:' >> {changed}")
    ok, feedback, reason = asyncio.run(run_check(ws, baseline))
    assert not ok and reason == "compile_error" and "SyntaxError" in feedback
    ws.restore()
    assert ws.diff() == ""


def test_visible_tests_gate_fails_until_the_bug_is_fixed_and_ignores_test_edits(env):
    from harness_graph.check import run_check, visible_failures
    from harness_graph.swe.container import Workspace
    from harness_graph.swe.grading import reverse_patch
    task, sif, _ = env
    healthy, ws = Workspace(sif, tag="test"), Workspace(sif, tag="test")
    try:
        healthy.setup(task, tests_visible=True, bug=False)
        baseline = asyncio.run(visible_failures(healthy))
        assert baseline is not None and not set(task["FAIL_TO_PASS"]) & baseline
        ws.setup(task, tests_visible=True)
        for f in task["f2p_files"]:
            assert ws.execute(f"test -e {f}")["returncode"] == 0
        # a change that does not fix the bug, plus an attempt to silence the failing tests
        src = re.search(r"^diff --git a/(\S+)", task["bug_patch"], re.M).group(1)
        ws.execute(f"echo '# touched' >> {src} && echo 'def test_nothing(): pass' > {task['f2p_files'][0]}")
        ok, feedback, reason = asyncio.run(run_check(ws, baseline))
        assert not ok and reason.startswith("new_test_failures") and "project's suite" in feedback
        ws.restore()
        open(os.path.join(ws.root, "testbed", "_gold.patch"), "w").write(reverse_patch(task["bug_patch"]))
        assert ws.execute("git apply _gold.patch && rm _gold.patch")["returncode"] == 0
        ok, _, reason = asyncio.run(run_check(ws, baseline))
        assert ok and reason == "pass"
    finally:
        healthy.close(); ws.close()
