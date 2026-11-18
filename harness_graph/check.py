"""The ``check`` gate: what a harness can verify about the working tree without the hidden tests.

Four tests, cheapest first; the first that fails produces the feedback shown to the agent:
  1. the diff is not empty;
  2. every changed Python file compiles;
  3. the agent's own reproduction script (/tmp/repro.py), if it wrote one, exits with status 0;
  4. the visible test suite finishes and has no failure outside a baseline (a suite that does not finish is a
     failed check; if no baseline could be measured the suite is skipped and the reason says so).  With hidden failing tests the baseline is
     what failed before the agent started, so the test only catches regressions.  With visible tests
     (``ws.tests_visible``) the baseline is what fails on the healthy repository, so tests the bug broke
     count until they pass; the agent's edits to test files and test-runner configuration are reverted first.
"""
from __future__ import annotations

import asyncio
import re
import shlex

TEST_CMD = "python -m pytest -q --disable-warnings --color=no --tb=no -p no:cacheprovider -rfE 2>&1 | tail -n 300"
_SUMMARY = re.compile(r"\b\d+ (passed|failed|errors?|skipped|deselected|xfailed)\b")
_ROW = re.compile(r"^(FAILED|ERROR) (\S+)", re.M)
REPRO = "/tmp/repro.py"


def parse_failures(output: str) -> set[str] | None:
    """Ids of failing tests in a ``pytest -q -rfE`` log, or None if pytest did not finish."""
    if not _SUMMARY.search(output[-600:]):
        return None
    return {m.group(2) for m in _ROW.finditer(output)}


async def visible_failures(ws, timeout: int = 300) -> set[str] | None:
    out = await asyncio.to_thread(ws.execute, TEST_CMD, timeout)
    return None if out["timed_out"] else parse_failures(out["output"])


async def run_check(ws, baseline: set[str] | None, test_timeout: int = 300) -> tuple[bool, str, str]:
    """Returns (passed, feedback for the agent, short reason for the log)."""
    ex = lambda cmd, t=120: asyncio.to_thread(ws.execute, cmd, t)
    changed = (await ex(f"git diff --name-only {ws.init_sha}"))["output"].split()
    if getattr(ws, "tests_visible", False):
        from .swe.grading import is_hook_path, is_test_path
        fixed = [f for f in changed if is_test_path(f) or is_hook_path(f)]
        if fixed:
            await ex(f"git checkout -q {ws.init_sha} -- " + " ".join(map(shlex.quote, fixed)))
            changed = [f for f in changed if f not in fixed]
    if not changed:
        return False, ("No tracked source file has been changed, so nothing would be submitted. "
                       "Edit the source to fix the problem."), "empty_diff"
    py = [f for f in changed if f.endswith(".py")]
    if py:
        r = await ex("for f in " + " ".join(map(shlex.quote, py)) +
                     "; do [ -f \"$f\" ] && python -m py_compile \"$f\" 2>&1; done | tail -n 20")
        if r["output"].strip():
            return False, "A changed file does not compile:\n" + r["output"].strip()[-1500:], "compile_error"
    r = await ex(f"[ -f {REPRO} ] && {{ python {REPRO} > /tmp/_repro.out 2>&1; echo \"RC=$?\"; tail -n 30 /tmp/_repro.out; }}")
    if r["output"].startswith("RC=") and not r["output"].startswith("RC=0"):
        return False, (f"Your reproduction script {REPRO} still fails (the first line is its exit status):\n"
                       + r["output"].strip()[-1500:]), "repro_fails"
    if baseline is None:
        return True, "", "pass_untested"         # no baseline could be measured, so the suite was not consulted
    now = await visible_failures(ws, test_timeout)
    if now is None:                              # the suite finished on the baseline but not on this working tree
        return False, ("The project's test suite did not finish on your working tree (it timed out or stopped "
                       "before printing a summary). Find what makes it hang or crash and fix it."), "tests_did_not_finish"
    if True:
        new = sorted(now - baseline)
        if new:
            ids = " ".join(shlex.quote(t) for t in new[:3])
            detail = await ex(f"python -m pytest -q --disable-warnings --color=no --tb=short -p no:cacheprovider "
                              f"{ids} 2>&1 | tail -n 60", 180)
            what = ("test(s) of the project's suite fail" if getattr(ws, "tests_visible", False)
                    else "test(s) that passed before your change now fail")
            return False, (f"{len(new)} {what}, for example:\n"
                           + "\n".join(new[:8]) + "\n\n" + detail["output"].strip()[-2500:]), f"new_test_failures:{len(new)}"
    return True, "", "pass"
