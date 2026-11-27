"""A1 read-out: gains, losses and the completeness rule (scripts/a1_analysis.py)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRAPH = "g-p0-fn-t0-vn-kcon2-c0-snon"


def rec(task, seed, at50, end, calls, terminal=False, **kw):
    r = {"episode_id": f"m/{GRAPH}/{task}/s{seed}", "graph": GRAPH, "task_id": task, "seed": seed, "status": "ok",
         "model": "qwen3.8-27b", "exec_s": 9.0,
         "tests": "visible", "resolved": end, "exit_reason": "end" if end else "model_call_budget", "model_calls": calls,
         "prompt_tokens": 1000 * calls, "completion_tokens": 10 * calls, "max_prompt_tokens": 2000, "latency_s": 10.0,
         "budget": {"model_calls": 100, "wall_s": 3600, "snapshot_at": 50},
         "snapshot": {"at": 50, "resolved": at50, "terminal": terminal, "model_calls": min(calls, 50),
                      "prompt_tokens": 1000 * min(calls, 50), "completion_tokens": 10 * min(calls, 50)}}
    if terminal:
        r["snapshot"]["resolved"] = end
    r.update(kw)
    return r


def run(tmp_path, rows, tasks=("a", "b")):
    (tmp_path / "tasks.jsonl").write_text("".join(json.dumps({"instance_id": t, "repo_key": "r"}) + "\n" for t in tasks))
    (tmp_path / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    subprocess.run([sys.executable, str(ROOT / "scripts/a1_analysis.py"), "--runs", str(tmp_path / "runs.jsonl"),
                    "--tasks", str(tmp_path / "tasks.jsonl"), "--out", str(tmp_path / "out.json")], check=True, capture_output=True)
    return json.loads((tmp_path / "out.json").read_text())


def test_gains_and_losses_are_counted_run_by_run(tmp_path):
    rows = [rec("a", 2, False, True, 70), rec("a", 3, False, False, 100),       # one gain, one still unsolved
            rec("b", 2, True, True, 20, terminal=True), rec("b", 3, True, False, 90)]   # early finish, one loss
    out = run(tmp_path, rows)
    assert out["verdict"] == "complete" and out["scored_runs"] == 4
    assert out["repair_of_the_state_after_call_50"]["mean"] == 0.5 and out["repair_at_the_end_of_the_100_call_run"]["mean"] == 0.5
    assert out["primary_end_minus_state_after_call_50"]["mean"] == 0.0
    assert out["runs_by_outcome"]["unsolved after call 50 -> solved at the end"] == 1
    assert out["runs_by_outcome"]["solved after call 50 -> unsolved at the end"] == 1
    reached = out["runs_that_reached_the_boundary_after_call_50"]
    assert reached["n"] == 3 and reached["unsolved_after_call_50"] == 2 and reached["of_those_solved_at_the_end"] == 1
    assert reached["with_at_least_one_further_model_call"] == 3 and out["engine_error_records_rerun"] == 0
    assert out["unsolved_after_call_50_finishing_solved_by_block"]["61-70"]["finish_solved"] == 1
    assert out["unsolved_after_call_50_all_runs"] == {"n": 2, "run_had_already_ended": 0}
    assert out["tokens_all_runs"]["up_to_the_state_after_call_50"] == 1010 * 170 and out["tokens_all_runs"]["at_the_end"] == 1010 * 280
    assert out["grades_that_timed_out"] == {"state_after_call_50": [], "end": []}


def test_grader_timeouts_and_runs_that_ended_unsolved_before_the_boundary_are_reported(tmp_path):
    slow = rec("a", 2, False, False, 100, grade={"timeout": False})
    slow["snapshot"]["grade"] = {"timeout": True}
    rows = [slow, rec("a", 3, False, False, 30, terminal=True, exit_reason="visit_cap"),
            rec("b", 2, True, True, 20, terminal=True), rec("b", 3, True, True, 20, terminal=True)]
    out = run(tmp_path, rows)
    assert out["unsolved_after_call_50_all_runs"] == {"n": 2, "run_had_already_ended": 1}
    assert out["grades_that_timed_out"] == {"state_after_call_50": [slow["episode_id"]], "end": []}


def test_a_missing_run_or_a_wrong_budget_makes_the_result_incomplete(tmp_path):
    rows = [rec("a", 2, False, True, 70), rec("a", 3, False, False, 100), rec("b", 2, True, True, 20, terminal=True)]
    assert run(tmp_path, rows)["verdict"] == "incomplete"
    rows.append(rec("b", 3, True, True, 20, terminal=True, budget={"model_calls": 50, "wall_s": 3600}))
    assert run(tmp_path, rows)["verdict"] == "incomplete"


def test_an_engine_error_does_not_count_and_does_not_block_its_rerun(tmp_path):
    rows = [rec("a", 2, False, True, 70), rec("a", 2, False, False, 0, status="engine_error"), rec("a", 3, False, False, 100),
            rec("b", 2, True, True, 20, terminal=True), rec("b", 3, True, True, 30, terminal=True)]
    assert run(tmp_path, rows)["verdict"] == "complete"


def test_a_run_stopped_at_the_boundary_stays_in_and_is_counted_apart(tmp_path):
    rows = [rec("a", 2, False, False, 50, exit_reason="context_exceeded"), rec("a", 3, False, True, 60),
            rec("b", 2, True, True, 20, terminal=True), rec("b", 3, True, True, 30, terminal=True)]
    out = run(tmp_path, rows)
    reached = out["runs_that_reached_the_boundary_after_call_50"]
    assert out["verdict"] == "complete" and reached["n"] == 2 and reached["with_at_least_one_further_model_call"] == 1
    assert reached["stopped_at_the_boundary_by"] == {"context_exceeded": 1}


def test_a_snapshot_that_contradicts_its_run_or_a_foreign_model_is_a_problem(tmp_path):
    good = [rec("a", 2, False, True, 70), rec("a", 3, False, False, 100), rec("b", 2, True, True, 20, terminal=True)]
    bad = rec("b", 3, True, True, 30, terminal=True)
    bad["snapshot"]["resolved"] = False                       # a terminal snapshot must equal the end
    assert run(tmp_path, good + [bad])["verdict"] == "incomplete"
    assert run(tmp_path, good + [rec("b", 3, True, True, 30, terminal=True, model="other")])["verdict"] == "incomplete"

