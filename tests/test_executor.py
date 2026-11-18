"""Executor tests with a fake workspace and a scripted model (no container, no GPU)."""
import asyncio
import json
from pathlib import Path

import pytest

from harness_graph import END, Budget, Graph, MockLLM, load_battery, run_episode
from harness_graph import prompts as P
from harness_graph.check import parse_failures

ROOT = Path(__file__).resolve().parents[1]
TASK = {"instance_id": "repo.bug1", "problem_statement": "add() returns the wrong sum."}


class FakeWS:
    """Stands in for swe.container.Workspace: `EDIT` changes the tree, everything else is canned."""
    init_sha = "abc123"

    def __init__(self, repro_rc=None, failures=(), breaks_tests=(), hangs_after_edit=False):
        self.changed, self.restored, self.commands = False, 0, []
        self.hangs_after_edit = hangs_after_edit
        self.repro_rc, self.failures, self.breaks_tests = repro_rc, set(failures), set(breaks_tests)

    def execute(self, command, timeout=None):
        self.commands.append(command)
        if command.startswith("git diff --name-only"):
            out = "pkg/a.py\n" if self.changed else ""
        elif "py_compile" in command:
            out = ""
        elif command.startswith("[ -f /tmp/repro.py ]"):
            out = "" if self.repro_rc is None else f"RC={self.repro_rc}\nAssertionError"
        elif "pytest" in command and "-rfE" in command and self.hangs_after_edit and self.changed:
            out = "collecting ...\n"                      # no summary line: the suite did not finish
        elif "pytest" in command and "-rfE" in command:
            failing = self.failures | (self.breaks_tests if self.changed else set())
            out = "".join(f"FAILED {t} - assert\n" for t in sorted(failing)) + "3 passed in 0.10s\n"
        elif command.startswith("EDIT"):
            self.changed, out = True, "edited"
            if "FIXREPRO" in command:
                self.repro_rc = 0
        elif command.startswith("BOOM"):
            raise OSError("container runtime died")
        else:
            out = "ok"
        return {"output": out, "returncode": 0, "timed_out": False, "secs": 0.0}

    def diff(self):
        return "diff --git a/pkg/a.py b/pkg/a.py\n+fix\n" if self.changed else ""

    def restore(self):
        self.changed, self.restored = False, self.restored + 1


def act(cmd, thought="ok"):
    return f"THOUGHT: {thought}\n\n```{P.FENCE}\n{cmd}\n```"


DONE = act(P.DONE_COMMAND)


def script(*replies):
    """A model that answers with the given replies in order, then keeps saying it is done."""
    it = iter(replies)
    return MockLLM(lambda messages, seed: next(it, DONE))


def graph(nodes, edges, entry="solve"):
    return Graph.from_dict({"name": "t", "entry": entry, "nodes": nodes, "edges": edges})


REACT = graph({"solve": {"type": "agent"}}, [["solve", END]])
CHECKED = graph({"solve": {"type": "agent", "max_visits": 3}, "check": {"type": "check", "max_visits": 3}},
                [["solve", "check"], ["check", END, "pass"], ["check", "solve", "fail"]])


def go(g, ws, llm, baseline=frozenset(), **kw):
    graded = []

    async def grade(diff):
        graded.append(diff)
        return {"resolved": True}

    async def base():
        return set(baseline)
    ep = asyncio.run(run_episode(g, TASK, ws, llm, seed=0, baseline=base, grade=grade, **kw))
    return ep, graded


def test_plain_loop_runs_commands_until_the_model_finishes():
    ws = FakeWS()
    ep, graded = go(REACT, ws, script(act("ls"), act("EDIT pkg/a.py")))
    assert ep.status == "ok" and ep.exit_reason == "end" and ep.agent_done and ep.resolved
    assert ws.commands[:2] == ["ls", "EDIT pkg/a.py"] and graded == [ws.diff()]
    assert ep.model_calls == 3 and ep.node_logs[0]["commands"] == 2


def test_empty_diff_is_not_graded_as_a_solution():
    ep, graded = go(REACT, FakeWS(), script(act("ls")))
    assert graded == [] and not ep.resolved and ep.agent_done


def test_failed_check_sends_the_agent_back_with_feedback():
    llm = script(act("ls"), DONE, act("EDIT pkg/a.py"))
    ep, _ = go(CHECKED, FakeWS(), llm)
    assert ep.path == ["solve", "check", "solve", "check"]
    assert [l["outcome"] for l in ep.node_logs] == ["done", "fail", "done", "pass"]
    assert ep.node_logs[1]["summary"] == "empty_diff"
    assert "No tracked source file has been changed" in llm.calls[2][-1]["content"]


def test_finish_before_any_command_is_rejected_not_accepted():
    llm = script(DONE, act("EDIT pkg/a.py"))
    ep, _ = go(REACT, FakeWS(), llm)
    assert ep.resolved and ep.model_calls == 3 and "before running any command" in llm.calls[1][-1]["content"]
    ep, _ = go(REACT, FakeWS(), MockLLM(lambda m, s: DONE))
    assert ep.exit_reason == "format_errors" and not ep.agent_done


def test_check_flags_only_failures_that_are_new():
    old = {"tests/test_a.py::test_old"}
    ep, _ = go(CHECKED, FakeWS(failures=old), script(act("EDIT pkg/a.py")), baseline=old)
    assert ep.node_logs[-1]["outcome"] == "pass"
    ep, _ = go(CHECKED, FakeWS(failures=old, breaks_tests={"tests/test_a.py::test_new"}),
               script(act("EDIT pkg/a.py")), baseline=old)
    assert ep.node_logs[1]["summary"] == "new_test_failures:1" and ep.exit_reason == "visit_cap"


def test_check_reruns_the_agents_own_reproduction_script():
    llm = script(act("EDIT pkg/a.py"), DONE, act("EDIT FIXREPRO"))
    ep, _ = go(CHECKED, FakeWS(repro_rc=1), llm)
    assert [l["summary"] for l in ep.node_logs if l["type"] == "check"] == ["repro_fails", "pass"]
    assert "/tmp/repro.py still fails" in llm.calls[2][-1]["content"]


def test_reset_restores_the_repository_and_keeps_only_the_lesson():
    g = graph({"solve": {"type": "agent", "max_visits": 2}, "check": {"type": "check", "max_visits": 2},
               "reflect": {"type": "reflect"}, "reset": {"type": "reset"}},
              [["solve", "check"], ["check", END, "pass"], ["check", "reflect", "fail"], ["reflect", "reset"],
               ["reset", "solve"]])
    ws = FakeWS(breaks_tests={"tests/test_a.py::test_x"})
    llm = script(act("EDIT pkg/a.py"), DONE, "I edited the wrong branch of add().", DONE)
    ep, _ = go(g, ws, llm)
    assert ep.path == ["solve", "check", "reflect", "reset", "solve", "check"] and ws.restored == 1
    fresh = llm.calls[3]
    assert len(fresh) == 2 and "I edited the wrong branch of add()." in fresh[1]["content"]
    assert "EDIT pkg/a.py" not in json.dumps(fresh)


def test_compaction_replaces_the_conversation_when_the_prompt_grows():
    g = graph({"solve": {"type": "agent", "params": {"compact_at": 600}, "max_visits": 4},
               "compact": {"type": "compact", "max_visits": 3}},
              [["solve", END, "done"], ["solve", "compact", "context_high"], ["compact", "solve"]])
    llm = script(act("ls " + "x" * 3000), act("EDIT pkg/a.py"), "cause is in pkg/a.py; fixed add()", DONE)
    ep, _ = go(g, FakeWS(), llm)
    assert ep.path == ["solve", "compact", "solve"] and ep.node_logs[0]["outcome"] == "context_high"
    after = llm.calls[-1]
    assert len(after) == 2 and "cause is in pkg/a.py" in after[1]["content"] and "x" * 100 not in json.dumps(after)
    assert "diff --git a/pkg/a.py" in after[1]["content"]


def test_model_call_budget_stops_the_episode_and_still_grades_the_diff():
    llm = MockLLM(lambda m, s: act("EDIT pkg/a.py"))
    ep, graded = go(CHECKED, FakeWS(), llm, budget=Budget(model_calls=5))
    assert ep.exit_reason == "model_call_budget" and ep.model_calls == 5 and not ep.agent_done
    assert len(graded) == 1 and ep.node_logs[-1]["outcome"] == "stopped:model_call_budget"


def test_plan_and_reflection_calls_count_against_the_same_budget():
    g = graph({"plan": {"type": "plan"}, "solve": {"type": "agent"}}, [["plan", "solve"], ["solve", END]], "plan")
    llm = MockLLM(lambda m, s: "1. read pkg/a.py" if "write a short plan" in m[-1]["content"] else act("ls"))
    ep, _ = go(g, FakeWS(), llm, budget=Budget(model_calls=4))
    assert [l["model_calls"] for l in ep.node_logs] == [1, 3] and ep.exit_reason == "model_call_budget"
    assert "1. read pkg/a.py" in json.dumps(llm.calls[1]) and P.PLAN_FOLLOW in llm.calls[1][-1]["content"]


def test_segment_step_cap_moves_on_to_the_next_phase():
    g = graph({"reproduce": {"type": "agent", "params": {"role": "reproduce", "max_steps": 2}},
               "fix": {"type": "agent", "params": {"role": "fix"}}},
              [["reproduce", "fix"], ["fix", END]], "reproduce")
    llm = script(act("ls"), act("ls"), act("EDIT pkg/a.py"))
    ep, _ = go(g, FakeWS(), llm)
    assert [l["outcome"] for l in ep.node_logs] == ["step_cap", "done"]
    assert "Phase 1 of 2" in llm.calls[0][-1]["content"] and "Phase 2 of 2" in llm.calls[2][-1]["content"]


def test_stall_fires_once_when_no_edit_was_made_and_never_after_an_edit():
    g = graph({"solve": {"type": "agent", "params": {"edit_by": 2}, "max_visits": 2}, "reflect": {"type": "reflect"},
               "reset": {"type": "reset"}},
              [["solve", END, "done"], ["solve", "reflect", "stalled"], ["reflect", "reset"], ["reset", "solve"]])
    llm = script(act("ls"), act("ls"), "The bug is in add().", act("ls"), act("ls"), act("EDIT pkg/a.py"))
    ep, _ = go(g, FakeWS(), llm)
    assert ep.path == ["solve", "reflect", "reset", "solve"] and ep.resolved
    assert [l["outcome"] for l in ep.node_logs] == ["stalled", "next", "next", "done"]
    assert "no source file has been changed" in llm.calls[2][-1]["content"]
    assert "The bug is in add()." in llm.calls[3][-1]["content"] and len(llm.calls[3]) == 2
    ep, _ = go(g, FakeWS(), script(act("EDIT pkg/a.py"), act("ls")))
    assert ep.path == ["solve"]


def test_one_malformed_reply_is_corrected_three_in_a_row_end_the_episode():
    ep, _ = go(REACT, FakeWS(), script("I will just talk.", act("EDIT pkg/a.py")))
    assert ep.exit_reason == "end" and ep.resolved
    ep, _ = go(REACT, FakeWS(), MockLLM(lambda m, s: "no command here"))
    assert ep.exit_reason == "format_errors" and ep.model_calls == 3


def test_engine_fault_is_not_scored_as_a_task_failure():
    ep, graded = go(REACT, FakeWS(), script(act("BOOM")))
    assert ep.status == "engine_error" and "container runtime died" in ep.error and graded == []


def test_node_logs_account_for_every_model_call_and_token():
    ep, _ = go(CHECKED, FakeWS(), script(act("ls"), DONE, act("EDIT pkg/a.py")))
    assert [l["node"] for l in ep.node_logs] == ep.path
    assert ep.model_calls == sum(l["model_calls"] for l in ep.node_logs) == 4
    assert ep.prompt_tokens == sum(l["prompt_tokens"] for l in ep.node_logs) > 0
    assert [l["visit"] for l in ep.node_logs] == [1, 1, 2, 2]
    assert {e["role"] for e in ep.events} == {"harness", "model", "command", "observation"}


def test_model_seeds_are_reproducible_and_differ_across_seeds():
    def seeds_of(seed):
        seen = []
        llm = MockLLM(lambda m, s: seen.append(s) or (act("EDIT x") if len(seen) < 3 else DONE))
        asyncio.run(run_episode(REACT, TASK, FakeWS(), llm, seed=seed))
        return seen
    assert seeds_of(0) == seeds_of(0) and len(set(seeds_of(0))) == 3 and not set(seeds_of(0)) & set(seeds_of(1))


@pytest.mark.parametrize("g", load_battery(ROOT / "graphs"), ids=lambda g: g.name)
def test_every_battery_graph_finishes_with_a_competent_model(g):
    ws = FakeWS()

    def fn(messages, seed):
        if messages[-1]["content"].rstrip().endswith("Do not write a command in this response."):
            return "notes"                       # plan, reflection, summary
        return DONE if ws.changed else act("EDIT pkg/a.py")
    ep, _ = go(g, ws, MockLLM(fn))
    assert ep.status == "ok", ep.error
    assert ep.exit_reason == "end" and ep.resolved, (ep.exit_reason, ep.path)


def test_parse_failures_and_observation_truncation():
    log = "FAILED tests/a.py::t1 - assert 1\nERROR tests/b.py - ImportError\n2 failed, 1 error, 5 passed in 1.2s\n"
    assert parse_failures(log) == {"tests/a.py::t1", "tests/b.py"}
    assert parse_failures("Traceback (most recent call last): boom") is None
    assert parse_failures("7 passed in 0.3s\n") == set()
    obs = P.observation({"output": "a" * 30000, "returncode": 0, "timed_out": False})
    assert len(obs) < 11000 and "<elided_chars>20000</elided_chars>" in obs
    assert "timed out" in P.observation({"output": "", "returncode": 137, "timed_out": True})


def test_checker_tool_runs_the_same_check_at_the_agents_request_without_enforcing_it():
    g = graph({"solve": {"type": "agent", "params": {"checker": True}}}, [["solve", END]])
    llm = script(act(P.CHECK_COMMAND), act("EDIT pkg/a.py"), act(P.CHECK_COMMAND))
    ep, _ = go(g, FakeWS(), llm)
    assert P.CHECK_COMMAND in llm.calls[0][-1]["content"]                        # the instruction is in the prompt
    assert "No tracked source file has been changed" in llm.calls[1][-1]["content"]
    assert llm.calls[3][-1]["content"] == P.CHECK_PASSED
    assert ep.path == ["solve"] and ep.agent_done                                # nothing forces the check
    ep, _ = go(REACT, FakeWS(), script(act(P.CHECK_COMMAND), act("EDIT pkg/a.py")))
    assert P.CHECK_COMMAND not in ep.events[0]["content"]                        # without the parameter it is a shell command


def test_a_suite_that_does_not_finish_fails_the_check_instead_of_passing():
    ep, _ = go(CHECKED, FakeWS(hangs_after_edit=True), script(act("EDIT pkg/a.py")))
    checks = [l for l in ep.node_logs if l["type"] == "check"]
    assert checks[0]["outcome"] == "fail" and checks[0]["summary"] == "tests_did_not_finish"


def test_dollar_budget_ends_the_episode_and_the_work_is_still_graded():
    from harness_graph.llm import Completion, LLM

    class Paid(LLM):
        def __init__(self):
            self.replies = iter([act("EDIT pkg/a.py"), act("ls"), act("ls"), act("ls")])

        async def chat(self, messages, *, max_tokens=4096, seed=None, affinity=0):
            return Completion(next(self.replies, DONE), 10, 10, cost_usd=0.4)
    ep, graded = go(REACT, FakeWS(), Paid(), budget=Budget(usd=1.0))
    assert ep.status == "ok" and ep.exit_reason == "usd_budget"
    assert ep.model_calls == 3 and graded                 # three calls reach $1.20; the edit made before is graded
