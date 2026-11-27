"""Prompt text.  The action format (one fenced shell command per turn) follows mini-swe-agent's text interface."""

FENCE = "mswea_bash_command"
DONE_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

SYSTEM = f"""You are a software engineer who solves programming tasks by running shell commands, one per turn.
Every response must contain exactly ONE code block with ONE command (commands may be joined with && or ||).
Write a short THOUGHT before the command.

<format_example>
THOUGHT: why this command is the right next step

```{FENCE}
your_command_here
```
</format_example>

Responses that do not follow this format are rejected."""

TASK = """<pr_description>
{problem}
</pr_description>

Your job is to change the non-test source files of the repository in /testbed so that the problem described
above is fixed, in a way that is general and consistent with the codebase.

Rules of the environment:
- You issue one command, it runs in a new subshell in /testbed, and you see its output. Directory and
  environment-variable changes do not persist between commands; files do.
- The project's Python environment (python, pytest) is first on PATH with the package installed in editable mode.
- There is no network access. Each command is killed after {cmd_timeout} seconds. Use non-interactive commands only.
- Do not modify tests or configuration files (pyproject.toml, setup.cfg, ...). Scratch files belong in /tmp.
- Your final answer is the state of the repository: whatever source changes exist when you finish are submitted.
- Start by inspecting the repository. Only after you have made your changes and confirmed them, end the task by
  sending this command alone in the block (it is not a first step):

```{fence}
{done}
```"""

ROLE = {
    "solve": """Solve the task now. A workflow that usually works: find and read the relevant code, reproduce the
problem with a small script, edit the source, rerun the script to confirm the fix, then finish.""",
    "reproduce": """Phase 1 of 2: reproduce, do not fix yet.
Find the code responsible and write a script /tmp/repro.py that exits with status 1 while the problem is present
and status 0 once it is fixed. Run it and confirm that it exits with 1 now. Do not edit any source file in this
phase. When the script is in place and fails as intended, finish the phase with the finish command.""",
    "fix": """Phase 2 of 2: fix.
Edit the source so that the problem is fixed. If /tmp/repro.py exists, it must exit with status 0 afterwards.
Then finish with the finish command.""",
    "verify": """Review phase: do not finish yet.
Re-read the problem description and list every behaviour it reports as wrong. For each one, run a command that
shows whether it is correct now. Then run the existing tests that cover the files you changed, and read the code
around your change for the same kind of defect. Fix whatever is still wrong. When every point is confirmed,
finish with the finish command.""",
}

CHECK_COMMAND = "hg-check"

CHECKER = f"""A checker is available. Sending this command alone in the block runs the project's test suite against
your current change and reports what still fails:

```{FENCE}
{CHECK_COMMAND}
```

Run it before you finish, and fix what it reports."""

CHECK_PASSED = "The check passed: nothing fails that passes on the healthy code base."

CONTINUE = "Continue working on the task. When it is fixed, finish again with the finish command."

PLAN = """Tracked files in the repository (first {n}):
{listing}

Before running anything, write a short plan, at most 8 lines: (1) which files most likely contain the cause,
(2) how you will reproduce the problem, (3) how you will verify the fix. Do not write a command in this response."""

PLAN_FOLLOW = "Follow your plan, and revise it when the evidence contradicts it."

STALLED = """You have run {n} commands and no source file has been changed yet. Stop exploring: name the most
likely cause and make the edit."""

CHECK_FAILED = "An automatic check of your work failed.\n\n{feedback}"
# control variants of the check node (E2): the same opening sentence without any detail, and a continuation
# request that mentions no check
CHECK_FAILED_BINARY = "An automatic check of your work failed."
FORCED_CONTINUE = "Continue reviewing and repairing the task."

REFLECT = """Before acting again, diagnose the failure in at most 6 sentences: what in your approach or your change
was wrong, and what you will do differently. Do not write a command in this response."""

RESET = """A previous attempt at this task was stopped and discarded. The repository is back in its
original state; scratch files in /tmp are still there.

What went wrong in that attempt:
{lesson}"""

COMPACT = """The conversation is getting long and will be replaced by your summary. Write that summary now, for
yourself: (1) what you learned about the cause, with file paths and function names, (2) what you changed and why,
(3) what remains to be done. Be specific and complete; at most 25 lines. Do not write a command in this response."""

COMPACTED = """Summary of your work so far:
{summary}

Current diff of the repository:
{diff}"""

EARLY_FINISH = """You sent the finish command before running any command, so nothing has been done yet. Start working
on the task: inspect the repository with a command."""

FORMAT_ERROR = f"""Your response was rejected: it must contain exactly one ```{FENCE} block with one command
(found {{n}}). Respond again in the required format. To finish, use the finish command from the first message."""

TRUNCATED = """Your response reached the output limit before a complete command was produced. Respond more
briefly, with exactly one command in the required format."""


def observation(out: dict, limit: int = 10000) -> str:
    text, head = out["output"], ""
    if out.get("timed_out"):
        head = "<exception>The command timed out and was killed.</exception>\n"
    head += f"<returncode>{out['returncode']}</returncode>\n"
    if len(text) <= limit:
        return head + f"<output>\n{text}</output>"
    half = limit // 2
    return (head + "<warning>The output was too long; only its head and tail are shown. Use a more selective "
            "command (head, tail, sed -n, grep) next time.</warning>\n"
            f"<output_head>\n{text[:half]}\n</output_head>\n<elided_chars>{len(text) - limit}</elided_chars>\n"
            f"<output_tail>\n{text[-half:]}\n</output_tail>")
