"""P1: the trusted runner's verdicts, the stored-patch rules and the analysis, on synthetic data (no container)."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from harness_graph import gencheck as G

ROOT = Path(__file__).resolve().parents[1]
ENV = {"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"}


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verdict_of(tmp_path, body):
    (tmp_path / "runner.py").write_text(G.RUNNER)
    (tmp_path / "check.py").write_text(body)
    (tmp_path / "helper.py").write_text("def boom():\n    assert False, 'inside a library'\n")
    r = subprocess.run([sys.executable, str(tmp_path / "runner.py"), str(tmp_path / "check.py"), str(tmp_path / "res.json")],
                       cwd=tmp_path, capture_output=True, text=True)
    res = json.loads((tmp_path / "res.json").read_text())
    assert r.returncode == {"fixed": 0, "problem_present": 1}.get(res["verdict"], 2)
    return res


@pytest.mark.parametrize("body, verdict, reason", [
    ("def check():\n    assert 1 + 1 == 2, 'ok'\n", "fixed", ""),
    ("def check():\n    x = 3\n    assert x == 2, f'expected 2, got {x}'\n", "problem_present", "assert_line_3"),
    ("def helper():\n    assert False, 'in a helper of the check'\ndef check():\n    helper()\n", "problem_present", "assert_line_2"),
    ("def check():\n    raise TypeError('boom')\n", "invalid", "exception:TypeError"),
    ("import not_a_module\ndef check():\n    pass\n", "invalid", "import_error:ModuleNotFoundError"),
    ("assert False, 'at import time'\ndef check():\n    pass\n", "invalid", "import_error:AssertionError"),
    ("x = 1\n", "invalid", "no_check_function"),
    ("import helper\ndef check():\n    helper.boom()\n", "invalid", "assertion_outside_check"),
    ("import sys\ndef check():\n    print('HG_CHECK: FIXED')\n    sys.exit(0)\n", "invalid", "exception:SystemExit"),
    ("def check():\n    try:\n        raise KeyError('k')\n    except KeyError as e:\n        assert False, f'raised {e!r}'\n",
     "problem_present", "assert_line_5"),
])
def test_runner_owns_the_verdict(tmp_path, body, verdict, reason):
    res = verdict_of(tmp_path, body)
    assert (res["verdict"], res["reason"]) == (verdict, reason)


def test_generator_versions(tmp_path):
    import hashlib
    man = json.loads((ROOT / "data/p1/manifest_G1.json").read_text())["sha256_text"]
    h = lambda t: hashlib.sha256(t.encode()).hexdigest()[:16]
    g1 = G.GENERATORS["G1"]
    assert h(g1["task"]) == man["task prompt"] and h(g1["runner"]) == man["trusted runner"]      # G1 is what was run
    body = "import sys\ndef check():\n    assert 'hg_check' in sys.modules, 'the check module is not registered'\n"
    (tmp_path / "check.py").write_text(body)
    for runner, want in ((G.RUNNER, 1), (G.RUNNER_V2, 0)):
        (tmp_path / "runner.py").write_text(runner)
        r = subprocess.run([sys.executable, str(tmp_path / "runner.py"), str(tmp_path / "check.py")], cwd=tmp_path, capture_output=True)
        assert r.returncode == want
    assert G.static_problem(G.NO_CHECK + ": no evidence", may_abstain=True) == "abstained"
    assert G.static_problem("\n" + G.NO_CHECK + ":  the issue gives no example \n\n", may_abstain=True) == "abstained"
    assert G.static_problem(G.NO_CHECK + ": no evidence") == "no_header"
    for malformed in (G.NO_CHECK, G.NO_CHECK + ":", G.NO_CHECK + ": x\ndef check():\n    pass\n", G.NO_CHECK + "ING: x"):
        assert G.static_problem(malformed, may_abstain=True) == "no_header"      # judged as a check, and found wanting
    assert G.runner_sha("G1") != G.runner_sha("G2")
    for k in ("problem", "artifact", "header", "runner", "cmd_timeout", "fence", "done", "no_check"):
        assert "{" + k + "}" in G.TASK_G2 or k == "no_check" and "{no_check}" in G.TASK_G2
    G.TASK_G2.format(problem="p", artifact="a", header="h", runner="r", cmd_timeout=1, fence="f", done="d", no_check="n", steps=15)


def test_static_rules():
    ok = G.HEADER + "\n# issue says x -> assert x\ndef check():\n    assert True\n"
    assert G.static_problem(ok) is None
    assert G.static_problem("  \n") == "no_script"
    assert G.static_problem("def check():\n    pass\n") == "no_header"
    assert G.static_problem(ok + "#" * G.MAX_ARTIFACT) == "too_long"
    for why in G.REASONS:                                       # every reason can be turned into feedback text
        assert G.feedback({"why": why, "runs": [{"reason": "exception:TypeError", "detail": "tb"}]})


def grade(resolved, **kw):
    g = {"grader_version": "v2", "apply_ok": True, "bug_apply_fail": False, "timeout": False, "ungradable": None, "test_rc": 1,
         "n_parsed": 10, "f2p_missing": 0, "p2p_missing": 0, "f2p_total": 2, "f2p_pass": 2 if resolved else 0, "p2p_total": 5,
         "p2p_pass": 5, "resolved": resolved}
    return {**g, **kw}


def ep(task, diff, resolved, tests="hidden", exit_reason="end", graph="g", seed=0, **kw):
    return {"task_id": task, "status": "ok", "diff": diff, "grade": grade(resolved, **kw), "tests": tests, "graph": graph,
            "seed": seed, "exit_reason": exit_reason}


def test_patch_pool_rules(tmp_path):
    P = load("p1_patches")
    assert P.kind_of(ep("t", "d", True)) == "correct" and P.kind_of(ep("t", "d", False)) == "wrong"
    for bad in ({"timeout": True}, {"apply_ok": False}, {"f2p_missing": 1}, {"p2p_missing": 2}, {"test_rc": None},
                {"ungradable": "x"}, {"grader_version": "v1"}, {"n_parsed": 0}):
        assert P.kind_of(ep("t", "d", False, **bad)) is None
    assert P.kind_of({**ep("t", "", False)}) is None and P.kind_of({**ep("t", "d", False), "status": "engine_error"}) is None
    hidden = [ep("t1", "w1", False, exit_reason="model_call_budget", graph="a"), ep("t1", "w1", False, exit_reason="end", graph="b"),
              ep("t1", "w2", False, exit_reason="model_call_budget"), ep("t1", "c1", True),
              ep("t1", "both", True), ep("t1", "both", False, graph="z")]
    visible = [ep("t1", f"v{i}", False, tests="visible") for i in range(5)] + [ep("t1", f"vc{i}", True, tests="visible") for i in range(5)] + \
              [ep("t1", "w1", False, tests="visible")]
    (tmp_path / "tasks.jsonl").write_text(json.dumps({"instance_id": "t1"}) + "\n" + json.dumps({"instance_id": "t2"}))
    (tmp_path / "h.jsonl").write_text("\n".join(map(json.dumps, hidden)))
    (tmp_path / "v.jsonl").write_text("\n".join(map(json.dumps, visible)))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/p1_patches.py"), "--tasks", str(tmp_path / "tasks.jsonl"), "--hidden",
                        str(tmp_path / "h.jsonl"), "--visible", str(tmp_path / "v.jsonl"), "--out", str(tmp_path / "p.jsonl")],
                       capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, r.stderr
    rows = [json.loads(l) for l in open(tmp_path / "p.jsonl")]
    pool = lambda name: sorted(x["diff"] for x in rows if x["pool"] == name)
    assert pool("primary") == ["w1"]                           # one agent-ended occurrence is enough, whatever the file order
    assert "w2" in pool("secondary") and len(pool("secondary")) == 3 and "w1" not in pool("secondary")
    assert "c1" in pool("correct") and len(pool("correct")) == 3
    assert all(x["diff"] != "both" for x in rows)              # graded both ways: dropped
    summary = json.loads((tmp_path / "p.summary.json").read_text())
    assert summary["dropped_same_diff_graded_both_ways"]["hidden"] == 1
    again = subprocess.run([sys.executable, str(ROOT / "scripts/p1_patches.py"), "--tasks", str(tmp_path / "tasks.jsonl"), "--hidden",
                            str(tmp_path / "h.jsonl"), "--visible", str(tmp_path / "v.jsonl"), "--out", str(tmp_path / "q.jsonl")],
                           capture_output=True, text=True, env=ENV)
    assert again.returncode == 0 and (tmp_path / "q.jsonl").read_text() == (tmp_path / "p.jsonl").read_text()


SHA = __import__("hashlib").sha256(b"x").hexdigest()[:16]


def synthetic(tmp_path, n_tasks=100, change=None):
    """Every task: one primary wrong patch, one accepted patch.  Seed 0 delivers a check, seed 1 does not.
    The check rejects the wrong patch on tasks 0-59 (behaviourally on 0-49, by crashing on 50-59), accepts the
    reference everywhere and the accepted patch everywhere."""
    tasks = [{"instance_id": f"t{i:03d}", "repo_key": f"r{i % 4}", "strategy": "func_basic" if i % 2 else "combine_file"} for i in range(n_tasks)]
    patches, gen, ev = [], [], []
    for i, t in enumerate(tasks):
        tid = t["instance_id"]
        patches += [{"task_id": tid, "patch_id": "w", "pool": "primary", "f2p_fail": 1, "diff": "w"},
                    {"task_id": tid, "patch_id": "c", "pool": "correct", "f2p_fail": 0, "diff": "c"}]
        for s in (0, 1):
            gen.append({"episode_id": f"m/G1/{tid}/s{s}", "generator": "G1", "task_id": tid, "seed": s, "status": "ok", "valid": s == 0,
                        "first_valid": s == 0, "model_calls": 10, "prompt_tokens": 100, "completion_tokens": 10, "exit_reason": "valid" if s == 0 else "invalid",
                        "validations": [{"valid": s == 0, "why": None if s == 0 else "fixed"}], "repo_changed": False, "artifact": "x"})
        run = lambda tree, pid, rep, verdict: {"episode_id": f"m/G1/{tid}/s0", "artifact_sha": SHA, "runner_sha": G.runner_sha("G1"), "task_id": tid, "seed": 0, "tree": tree,
                                               "patch_id": pid, "rep": rep, "status": "ok", "applied": True, "apply": "git",
                                               "verdict": verdict, "reason": ""}
        ev += [run("unrepaired", "", 0, "problem_present"), run("reference", "", 0, "fixed"), run("reference", "", 1, "fixed"),
               run("patch", "w", 0, "problem_present" if i < 50 else "invalid" if i < 60 else "fixed"), run("patch", "c", 0, "fixed")]
    if change:
        gen, ev = change(gen, ev)
    for name, rows in (("tasks", tasks), ("patches", patches), ("gen", gen), ("eval", ev)):
        (tmp_path / f"{name}.jsonl").write_text("\n".join(map(json.dumps, rows)))
    (tmp_path / "audit.jsonl").write_text("\n".join(json.dumps({"instance_id": f"t{i:03d}", "spec": "full" if i < 10 else "partial"}) for i in range(20)))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/p1_analysis.py"), "--gen", str(tmp_path / "gen.jsonl"), "--eval", str(tmp_path / "eval.jsonl"),
                        "--tasks", str(tmp_path / "tasks.jsonl"), "--patches", str(tmp_path / "patches.jsonl"), "--audit", str(tmp_path / "audit.jsonl"),
                        "--reps", "200", "--out", str(tmp_path / "out.json")], capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads((tmp_path / "out.json").read_text())


def test_analysis_measures_and_pass(tmp_path):
    out = synthetic(tmp_path)
    m = out["measures"]
    assert m["validity"]["mean"] == pytest.approx(0.5)
    assert m["detection_primary"]["mean"] == pytest.approx(0.30)          # 60 of 100 tasks, one of two seeds each
    assert m["detection_primary"]["behavioural"] == pytest.approx(0.25) and m["detection_primary"]["crash_or_timeout"] == pytest.approx(0.05)
    assert m["false_rejection_reference"]["mean"] == 0 and m["false_rejection_reference"]["n_tasks"] == 100
    assert m["false_rejection_accepted"]["mean"] == 0 and m["still_rejects_unrepaired_tree"]["mean"] == 1
    assert out["decision"]["verdict"] == "pass" and not out["stored_record_problems"]
    assert out["by_group"]["issue audit: full"]["detection_primary"]["n_tasks"] == 10
    assert out["generator_cost"]["why_first_validation_failed"] == {"fixed": 100}


def test_analysis_fail_and_inconclusive(tmp_path):
    def reject_reference(gen, ev):                              # three checks reject the healthy repository in both runs
        for r in ev:
            if r["tree"] == "reference" and r["task_id"] in ("t000", "t001", "t002"):
                r["verdict"] = "invalid"
        return gen, ev
    out = synthetic(tmp_path, change=reject_reference)
    assert out["measures"]["false_rejection_reference"]["mean"] == pytest.approx(0.03)
    assert out["measures"]["false_rejection_reference_per_attempt"]["mean"] == pytest.approx(0.015)
    assert out["decision"]["verdict"] == "fail"

    def lose_a_run(gen, ev):
        return gen, [r for r in ev if not (r["task_id"] == "t005" and r["tree"] == "patch" and r["patch_id"] == "c")]
    out = synthetic(tmp_path, change=lose_a_run)
    assert out["decision"]["verdict"] == "inconclusive" and out["stored_record_problems"]

    out = synthetic(tmp_path, n_tasks=60)                       # thresholds met, but too few tasks behind the conditional rates
    assert out["decision"]["verdict"] == "inconclusive"

    def detect_less(gen, ev):
        for r in ev:
            if r["tree"] == "patch" and r["patch_id"] == "w" and r["task_id"] < "t010":
                r["verdict"] = "fixed"
        return gen, ev
    out = synthetic(tmp_path, change=detect_less)
    assert out["measures"]["detection_primary"]["mean"] == pytest.approx(0.25) and out["decision"]["verdict"] == "fail"


def test_patch_that_does_not_apply_is_left_out_and_counted(tmp_path):
    def break_apply(gen, ev):
        for r in ev:
            if r["tree"] == "patch" and r["patch_id"] == "w" and r["task_id"] == "t099":
                r.update(applied=False, apply="error: patch failed")
                r.pop("verdict")
        return gen, ev
    out = synthetic(tmp_path, change=break_apply)
    assert out["measures"]["detection_primary"]["n_tasks"] == 99
    assert sum(out["measures"]["patches_left_out_not_applied"].values()) == 1
    assert out["decision"]["verdict"] == "inconclusive"         # an unexplained exclusion blocks a verdict


def test_records_that_do_not_belong_block_a_verdict(tmp_path):
    def foreign(gen, ev):                                       # a run for a seed that delivered no check
        return gen, ev + [dict(ev[0], seed=1, episode_id="m/G1/t000/s1")]
    assert synthetic(tmp_path, change=foreign)["decision"]["verdict"] == "inconclusive"

    def other_check(gen, ev):                                   # a run made with a different check file
        ev[3]["artifact_sha"] = "0" * 16
        return gen, ev
    out = synthetic(tmp_path, change=other_check)
    assert out["decision"]["verdict"] == "inconclusive" and "another check" in out["stored_record_problems"][0]

    def other_runner(gen, ev):                                  # one run made with the other runner
        ev[7]["runner_sha"] = G.runner_sha("G2")
        return gen, ev
    out = synthetic(tmp_path, change=other_runner)
    assert out["decision"]["verdict"] == "inconclusive" and "expected runner" in out["stored_record_problems"][0]

    def lose_generator_record(gen, ev):
        return [g for g in gen if not (g["task_id"] == "t050" and g["seed"] == 1)], ev
    out = synthetic(tmp_path, change=lose_generator_record)
    assert out["decision"]["verdict"] == "inconclusive"

    def low_detection_and_a_missing_run(gen, ev):               # low detection from partial data is not a "fail"
        ev = [r for r in ev if not (r["tree"] == "patch" and r["patch_id"] == "w" and r["task_id"] < "t030")]
        return gen, ev
    assert synthetic(tmp_path, change=low_detection_and_a_missing_run)["decision"]["verdict"] == "inconclusive"
