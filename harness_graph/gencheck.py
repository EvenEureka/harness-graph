"""A check the harness writes for itself: generator graph, trusted runner, and how a check is run on a tree.

The generator is a graph with two nodes.
  write_check     an agent conversation of its own (not the solver's): reads the issue and the repository with the
                  defect, and writes one file, /tmp/hg_check.py, that defines ``check()`` with plain asserts.
  validate_check  a program: copies that file, and nothing else, into a fresh workspace with the same defect and
                  runs it there with the trusted runner, twice.
Edges: write_check -> validate_check when the agent finishes or reaches its step cap (the file is the only thing
passed); validate_check -> END when both runs say the problem is present; validate_check -> write_check otherwise,
at most once, with the reason added to the conversation.

The runner, not the generated file, decides the verdict:
  fixed            check() returned;
  problem_present  an AssertionError was raised by code in the check file (an ``assert`` statement there, or an
                   explicit ``raise AssertionError``); this is a mechanical rule, not a judgment that the
                   assertion is the right one;
  invalid          anything else (import error, any other exception, an assertion raised elsewhere, no check()).
The gold fix, the hidden grading tests and stored patches are never given to the generator or used to pick a check.

Every workspace of this module (generation, validation, evaluation) is built by ``setup``: the usual hidden-test
workspace, with compiled ``.pyc`` files under the repository removed before the initial commit, because bytecode
compiled when the image was built can hold the healthy implementation.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import traceback
import uuid

from . import prompts as P
from .executor import ACTION, stable_int
from .llm import ContextExceeded
from .swe.container import Workspace
from .swe.grading import _excluded, filter_patch

ARTIFACT, RUNNER_PATH, RESULT_PATH = "/tmp/hg_check.py", "/tmp/hg_run_check.py", "/tmp/hg_result.json"
HEADER = "# ISSUE -> ASSERTIONS"
MAX_ARTIFACT = 30_000            # characters
RUN_TIMEOUT = 120                # seconds for one run of a check
STEPS_PER_VISIT, MODEL_CALLS, MAX_FORMAT_ERRORS = 15, 30, 3
VALIDATION_RUNS = 2

RUNNER = r'''import importlib.util, json, os, sys, traceback
ART = sys.argv[1] if len(sys.argv) > 1 else "/tmp/hg_check.py"
OUT = sys.argv[2] if len(sys.argv) > 2 else ""
sys.path.insert(0, os.getcwd())


def finish(verdict, reason="", detail=""):
    res = {"verdict": verdict, "reason": reason, "detail": detail[-2000:]}
    if OUT:
        with open(OUT, "w") as f:
            json.dump(res, f)
    print("HG_CHECK: " + verdict.upper() + (" - " + reason if reason else ""))
    if detail:
        print(detail[-2000:])
    sys.stdout.flush()
    os._exit({"fixed": 0, "problem_present": 1}.get(verdict, 2))


try:
    spec = importlib.util.spec_from_file_location("hg_check", ART)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
except BaseException as e:
    finish("invalid", "import_error:" + type(e).__name__, traceback.format_exc())
if not callable(getattr(mod, "check", None)):
    finish("invalid", "no_check_function")
try:
    mod.check()
except AssertionError as e:
    last = traceback.extract_tb(e.__traceback__)[-1]
    if os.path.abspath(last.filename) == os.path.abspath(ART):
        finish("problem_present", "assert_line_%d" % last.lineno, "line %d: %s\n%s" % (last.lineno, last.line, e))
    finish("invalid", "assertion_outside_check", traceback.format_exc())
except BaseException as e:
    finish("invalid", "exception:" + type(e).__name__, traceback.format_exc())
finish("fixed")
'''

TASK = """<issue>
{problem}
</issue>

The repository in /testbed has the problem described above. Your job is NOT to fix it. Your job is to write a
check: a Python file {artifact} that tells whether the problem is still there. Later, a separate program runs
your check against changed versions of this repository to decide whether a proposed fix can be accepted.

What the file must contain:
- One function, check(), with no arguments. It exercises the behaviour the issue reports, through the public
  interface of the package where possible, and states what must be true with plain assert statements written in
  {artifact} itself, each with a message: assert <condition>, "<what was expected, what was observed>".
- While the problem is present, at least one of your assert statements must fail. Once the problem is fixed,
  check() must return without raising anything.
- If the problem shows up as an exception, catch it inside check() and turn it into a failed assert, for example
  `try: ... except TypeError as e: assert False, f"raised {{e!r}}"`. An exception that leaves check() and is not a
  failed assert of yours makes the check invalid. So does an import error, and so do unittest or pytest helpers.
- Assert only what the issue says, or what clearly follows from it together with the surrounding code and
  documentation. A different but equally correct fix must pass too: do not assert the exact wording of error
  messages, names of internal helpers, or anything the issue does not require.
- The check must give the same answer every time, need no network, finish within 60 seconds, and use nothing
  but the repository and the standard library. It must not read other files in /tmp.
- The file begins with a comment block headed `{header}` that lists, one line each, a behaviour reported in the
  issue and the assert that tests it.

You can try the check at any time with
  python {runner}
It runs check() the way the later program will and prints PROBLEM_PRESENT (one of your asserts failed), FIXED
(check() returned) or INVALID with the reason. On this repository the right answer is PROBLEM_PRESENT.

Rules of the environment:
- You issue one command, it runs in a new subshell in /testbed, and you see its output. Directory and
  environment-variable changes do not persist between commands; files do.
- The project's Python environment (python, pytest) is first on PATH with the package installed in editable mode.
- There is no network access. Each command is killed after {cmd_timeout} seconds. Use non-interactive commands only.
- Do not change any file of the repository. Only {artifact} is kept; your check is run on a clean copy of the
  repository, without anything else you created.
- Start by reading the code the issue is about. When the check is written and the command above prints
  PROBLEM_PRESENT, end by sending this command alone in the block (it is not a first step):

```{fence}
{done}
```"""

# G2 (written after the G1 result, from reading why G1's checks rejected the healthy repository):
#   - the runner registers the check module in sys.modules (G1's did not; libraries that look their caller up there failed);
#   - the issue's own examples, unchanged; expected values corroborated by the repository's tests and documentation;
#   - the generator may decline to write a check.
RUNNER_V2 = RUNNER.replace("    mod = importlib.util.module_from_spec(spec)\n",
                           "    mod = importlib.util.module_from_spec(spec)\n    sys.modules[\"hg_check\"] = mod\n")
assert RUNNER_V2 != RUNNER
NO_CHECK = "# NO CHECK"

TASK_G2 = """<issue>
{problem}
</issue>

The repository in /testbed has the problem described above. Your job is NOT to fix it. Your job is to write a
check: a Python file {artifact} that tells whether the problem is still there. Later, a separate program runs
your check against changed versions of this repository to decide whether a proposed fix can be accepted. A check
that rejects a correct fix does harm, so every assert must be one you can support.

What the file must contain:
- One function, check(), with no arguments. It states what must be true with plain assert statements written in
  {artifact} itself, each with a message: assert <condition>, "<what was expected, what was observed>".
- While the problem is present, at least one of your assert statements must fail. Once the problem is fixed,
  check() must return without raising anything.
- If the problem shows up as an exception, catch it inside check() and turn it into a failed assert, for example
  `try: ... except TypeError as e: assert False, f"raised {{e!r}}"`. An exception that leaves check() and is not a
  failed assert of yours makes the check invalid. So does an import error, and so do unittest or pytest helpers.
- The check must give the same answer every time, need no network, finish within 60 seconds, and use nothing
  but the repository and the standard library. It must not read other files in /tmp.

Which behaviour to assert:
- Use the examples the issue gives, exactly: the same calls, the same argument values, the same quoting and
  escaping. Do not build other scenarios of your own. A shell command in the issue is run as that command, with
  the argument list a shell would pass (no extra quote characters inside the arguments).
- Issue texts are sometimes wrong about the exact expected result. Before you assert an expected value, look for
  what the repository itself says should happen: its remaining tests, docstrings and documentation (search for
  the function and for similar expected values). Where they state the expected result, follow them, also when
  they differ from the issue in details such as quoting, spacing, capitalisation or the type of the result.
- The present output of the code is not evidence of what is correct: this code has the problem.
- If the issue reports an unwanted exception or crash, it is enough to assert that the call from the issue no
  longer raises that exception. Add an assert on the returned value only where the repository's tests or
  documentation support it.
- Do not assert the exact wording of error messages, names of internal helpers, or anything that neither the
  issue nor the repository requires. A different but equally correct fix must pass too.
- The file begins with a comment block headed `{header}`: one line per assert, naming the behaviour from the
  issue and the evidence for the expected result (file and line of a test, docstring or document, or "issue only").
- If you cannot support any assert that fails on this repository, do not guess: write {artifact} with the single
  line `{no_check}: <reason>` and finish.

You can try the check at any time with
  python {runner}
It runs check() the way the later program will and prints PROBLEM_PRESENT (one of your asserts failed), FIXED
(check() returned) or INVALID with the reason. On this repository the right answer is PROBLEM_PRESENT.

Rules of the environment:
- You issue one command, it runs in a new subshell in /testbed, and you see its output. Directory and
  environment-variable changes do not persist between commands; files do.
- The project's Python environment (python, pytest) is first on PATH with the package installed in editable mode.
- There is no network access. Each command is killed after {cmd_timeout} seconds. Use non-interactive commands only.
- Do not change any file of the repository. Only {artifact} is kept; your check is run on a clean copy of the
  repository, without anything else you created.
- You have at most {steps} responses, the finish response included. Read the code the issue is about and the
  tests and documentation that cover it, but write a first version of {artifact} by about your tenth response
  and correct it afterwards: exploration is lost if no file exists when the responses run out.
- When the check is written and the command above prints PROBLEM_PRESENT, end by sending this command alone in
  the block (it is not a first step):

```{fence}
{done}
```"""

def runner_sha(version: str) -> str:
    return hashlib.sha256(GENERATORS[version]["runner"].encode()).hexdigest()[:16]


GENERATORS = {"G1": {"task": TASK, "runner": RUNNER, "may_abstain": False},
              "G2": {"task": TASK_G2, "runner": RUNNER_V2, "may_abstain": True}}

INVALID_FEEDBACK = """Your check was run on a clean copy of the repository and was not accepted: {reason}

{detail}

Correct {artifact} so that `python {runner}` prints PROBLEM_PRESENT on this repository because one of your own
assert statements fails, then finish again with the finish command."""

REASONS = {
    "no_script": "the file {artifact} does not exist or is empty.",
    "too_long": "the file is longer than {n} characters.",
    "no_header": "the file does not contain the comment block headed `{header}`.",
    "fixed": "check() returned without a failed assert, although the problem is present in this repository.",
    "invalid": "the check is invalid ({why}).",
    "timeout": "the check did not finish within {t} seconds.",
    "unstable": "two runs of the check gave different answers.",
    "abstained": "no check was written.",
}


def setup(ws: Workspace, task: dict, bug: bool = True) -> None:
    """Hidden-test workspace without pre-existing bytecode; the initial commit is rewritten to the cleaned tree."""
    ws.setup(task, tests_visible=False, bug=bug)
    r = ws.execute("find /testbed -name '*.pyc' -delete; find /testbed -depth -type d -name __pycache__ -empty -delete; "
                   "git add -A && git -c user.name=agent -c user.email=agent@localhost commit -q --amend --no-edit --allow-empty "
                   "&& git rev-parse HEAD", 120)
    sha = r["output"].strip().splitlines()[-1] if r["output"].strip() else ""
    if r["returncode"] != 0 or len(sha) != 40:
        raise RuntimeError(f"could not clean the workspace: {r['output'][-300:]}")
    ws.init_sha = sha


def put(ws: Workspace, path: str, text: str) -> None:
    """Place a file inside the container: written on the host into the bound repository directory, then moved."""
    name = f".hg_put_{uuid.uuid4().hex[:8]}"
    with open(os.path.join(ws.root, "testbed", name), "w") as f:
        f.write(text)
    r = ws.execute(f"mv /testbed/{name} {path}", 60)
    if r["returncode"] != 0:
        raise RuntimeError(f"could not place {path}: {r['output'][-300:]}")


def run_artifact(ws: Workspace, artifact: str, runner: str, timeout: int = RUN_TIMEOUT) -> dict:
    """Run one check in a workspace with a fresh copy of the trusted runner.  -> {verdict, reason, detail, secs}."""
    put(ws, RUNNER_PATH, runner)
    put(ws, ARTIFACT, artifact)
    ws.execute(f"rm -f {RESULT_PATH}", 60)
    run = ws.execute(f"python {RUNNER_PATH} {ARTIFACT} {RESULT_PATH} > /tmp/hg_run.out 2>&1; echo RC=$?", timeout)
    got = ws.execute(f"cat {RESULT_PATH} 2>/dev/null", 60)["output"].strip()
    if run["timed_out"]:
        return {"verdict": "invalid", "reason": "timeout", "detail": "", "secs": run["secs"]}
    try:
        res = json.loads(got)
        assert res["verdict"] in ("fixed", "problem_present", "invalid")
    except Exception:
        tail = ws.execute("tail -c 1500 /tmp/hg_run.out 2>/dev/null", 60)["output"]
        return {"verdict": "invalid", "reason": "no_result", "detail": tail, "secs": run["secs"]}
    return {"verdict": res["verdict"], "reason": res.get("reason", ""), "detail": res.get("detail", ""), "secs": run["secs"]}


def static_problem(artifact: str, may_abstain: bool = False) -> str | None:
    if not artifact.strip():
        return "no_script"
    lines = [l for l in artifact.splitlines() if l.strip()]
    if may_abstain and len(lines) == 1 and re.fullmatch(re.escape(NO_CHECK) + r":\s*\S.*", lines[0].strip()):
        return "abstained"                     # exactly one line "# NO CHECK: <reason>"; anything else is judged as a check
    if len(artifact) > MAX_ARTIFACT:
        return "too_long"
    if HEADER not in artifact:
        return "no_header"
    return None


def validate(task: dict, sif: str, artifact: str, version: str = "G1", runs: int = VALIDATION_RUNS) -> dict:
    """The deployable acceptance rule: in fresh workspaces with the defect, every run says problem_present."""
    gen = GENERATORS[version]
    bad = static_problem(artifact, gen["may_abstain"])
    if bad:
        return {"valid": False, "why": bad, "runs": []}
    out = []
    for _ in range(runs):                      # a fresh workspace per run: one run cannot leave state for the next
        ws = Workspace(sif, tag="val")
        try:
            setup(ws, task, bug=True)
            out.append(run_artifact(ws, artifact, gen["runner"]))
        finally:
            ws.close()
    verdicts = [o["verdict"] for o in out]
    if all(v == "problem_present" for v in verdicts):
        why = None
    elif len(set(verdicts)) > 1:
        why = "unstable"
    elif out[0]["reason"] == "timeout":
        why = "timeout"
    else:
        why = verdicts[0]                      # "fixed" or "invalid"
    return {"valid": why is None, "why": why, "runs": out}


def feedback(v: dict) -> str:
    first = v["runs"][0] if v["runs"] else {"reason": "", "detail": ""}
    reason = REASONS[v["why"]].format(artifact=ARTIFACT, n=MAX_ARTIFACT, header=HEADER, why=first["reason"], t=RUN_TIMEOUT)
    return INVALID_FEEDBACK.format(reason=reason, detail=first["detail"][-1500:], artifact=ARTIFACT, runner=RUNNER_PATH)


def apply_patch(ws: Workspace, task: dict, patch: str) -> tuple[bool, str]:
    """Apply a stored model patch as the grader does: test and test-runner files left out, git apply, then GNU patch."""
    excl, _, _ = _excluded(task, patch, "v2")
    body = filter_patch(patch, set(excl))
    if not body.strip():
        return False, "empty_after_filter"
    put(ws, "/tmp/hg_model.patch", body)
    r = ws.execute(f"(git apply --whitespace=nowarn /tmp/hg_model.patch 2>/tmp/hg_apply.err && echo APPLIED_GIT || "
                   f"(git checkout -q -f {ws.init_sha} -- . && patch --batch --fuzz=5 -p1 -i /tmp/hg_model.patch "
                   f">/tmp/hg_apply.err 2>&1 && echo APPLIED_PATCH || echo APPLY_FAILED)); tail -c 300 /tmp/hg_apply.err", 120)
    out = r["output"]
    return ("APPLIED_GIT" in out or "APPLIED_PATCH" in out), ("git" if "APPLIED_GIT" in out else
                                                               "gnu_patch" if "APPLIED_PATCH" in out else out[-300:])


def run_on_tree(task: dict, sif: str, artifact: str, tree: str, patch: str = "", runner: str = RUNNER) -> dict:
    """tree: 'unrepaired' (defect, no patch), 'reference' (healthy repository) or 'patch' (defect + a stored patch)."""
    ws = Workspace(sif, tag="tree")
    try:
        setup(ws, task, bug=tree != "reference")
        if tree == "patch":
            ok, how = apply_patch(ws, task, patch)
            if not ok:
                return {"applied": False, "apply": how}
            return {"applied": True, "apply": how, **run_artifact(ws, artifact, runner)}
        return {"applied": True, "apply": "", **run_artifact(ws, artifact, runner)}
    finally:
        ws.close()


async def generate(task: dict, sif: str, llm, seed: int, model: str, version: str = "G1", cmd_timeout: int = 180,
                   max_tokens: int = 4096) -> dict:
    """One generator episode -> a record with the final check, its validations and the transcript."""
    gen = GENERATORS[version]
    eid = f"{model}/{version}/{task['instance_id']}/s{seed}"
    rec = {"episode_id": eid, "generator": version, "task_id": task["instance_id"], "seed": seed, "model": model,
           "status": "ok", "exit_reason": "", "path": [], "node_logs": [], "events": [], "validations": [],
           "artifact": "", "valid": False, "first_valid": False, "repo_changed": False, "model_calls": 0,
           "prompt_tokens": 0, "completion_tokens": 0, "error": ""}
    t0 = time.perf_counter()
    ws = Workspace(sif, tag="gen", cmd_timeout=cmd_timeout)
    messages = [{"role": "system", "content": P.SYSTEM},
                {"role": "user", "content": gen["task"].format(problem=task["problem_statement"], artifact=ARTIFACT, header=HEADER,
                                                               runner=RUNNER_PATH, cmd_timeout=cmd_timeout, fence=P.FENCE,
                                                               done=P.DONE_COMMAND, no_check=NO_CHECK, steps=STEPS_PER_VISIT)}]
    ex = lambda cmd, t=None: asyncio.to_thread(ws.execute, cmd, t or cmd_timeout)

    def say(text):
        if messages[-1]["role"] == "user":
            messages[-1]["content"] += "\n\n" + text
        else:
            messages.append({"role": "user", "content": text})
        rec["events"].append({"role": "harness", "content": text[:6000]})

    async def write_check(visit: int) -> str:
        log = {"node": "write_check", "visit": visit, "model_calls": 0, "commands": 0, "prompt_tokens": 0, "completion_tokens": 0}
        rec["node_logs"].append(log)
        rec["path"].append("write_check")
        steps = bad = 0
        worked = visit > 1
        while steps < STEPS_PER_VISIT:
            if rec["model_calls"] >= MODEL_CALLS:
                log["outcome"] = "model_call_budget"
                return "model_call_budget"
            rec["model_calls"] += 1
            try:
                r = await llm.chat(messages, max_tokens=max_tokens, seed=stable_int(eid, rec["model_calls"]),
                                   affinity=stable_int(eid))
            except ContextExceeded:
                log["outcome"] = "context_exceeded"
                return "context_exceeded"
            steps += 1
            for k in ("prompt_tokens", "completion_tokens"):
                log[k] += getattr(r, k)
                rec[k] += getattr(r, k)
            log["model_calls"] += 1
            messages.append({"role": "assistant", "content": r.text})
            rec["events"].append({"role": "model", "content": r.text[:6000], "truncated": r.truncated})
            actions = ACTION.findall(r.text)
            command = actions[0].strip() if len(actions) == 1 else ""
            if len(actions) != 1 or (command.startswith(P.DONE_COMMAND) and not worked):
                bad += 1
                if bad >= MAX_FORMAT_ERRORS:
                    log["outcome"] = "format_errors"
                    return "format_errors"
                say(P.TRUNCATED if r.truncated else P.FORMAT_ERROR.format(n=len(actions)) if len(actions) != 1 else P.EARLY_FINISH)
                continue
            bad = 0
            if command.startswith(P.DONE_COMMAND):
                log["outcome"] = "done"
                return "done"
            out = await ex(command)
            log["commands"] += 1
            worked = True
            obs = P.observation(out)
            messages.append({"role": "user", "content": obs})
            rec["events"].append({"role": "command", "content": command[:3000]})
            rec["events"].append({"role": "observation", "content": obs[:3000], "rc": out["returncode"]})
        log["outcome"] = "step_cap"
        return "step_cap"

    async def validate_check(visit: int) -> bool:
        rec["path"].append("validate_check")
        art = (await ex(f"head -c {MAX_ARTIFACT + 1} {ARTIFACT} 2>/dev/null", 60))["output"]
        rec["repo_changed"] = rec["repo_changed"] or bool((await asyncio.to_thread(ws.diff)).strip())
        v = await asyncio.to_thread(validate, task, sif, art, version)
        rec["artifact"] = art
        rec["validations"].append({"visit": visit, **v})
        rec["node_logs"].append({"node": "validate_check", "visit": visit, "outcome": "valid" if v["valid"] else "invalid:" + v["why"]})
        return v["valid"]

    try:
        await asyncio.to_thread(setup, ws, task)
        await asyncio.to_thread(put, ws, RUNNER_PATH, gen["runner"])
        for visit in (1, 2):
            how = await write_check(visit)
            ok = await validate_check(visit)
            if visit == 1:
                rec["first_valid"] = ok
            abstained = rec["validations"][-1]["why"] == "abstained"         # a declined check is final: no retry
            if ok or visit == 2 or abstained or how in ("model_call_budget", "context_exceeded", "format_errors"):
                rec["valid"] = ok
                rec["exit_reason"] = "valid" if ok else "abstained" if abstained else how if how not in ("done", "step_cap") else "invalid"
                break
            await asyncio.to_thread(ws.restore)                 # the repository is put back before the retry
            say(feedback(rec["validations"][-1]))
    except Exception as e:
        rec["status"], rec["error"] = "engine_error", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
    finally:
        await asyncio.to_thread(ws.close)
    rec["latency_s"] = round(time.perf_counter() - t0, 3)
    return rec
