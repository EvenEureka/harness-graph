"""Graph executor: walks one harness graph on one task and logs every node visit.

All graphs spend from the same per-episode budget (model calls and wall clock), so a harness that
retries cannot win just by using more compute.  When the budget runs out the episode stops where it
is and the current state of the repository is what gets graded.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Awaitable, Callable

from . import prompts as P
from .check import restore_protected, run_check
from .graph import END, Graph, Node
from .llm import LLM, ContextExceeded

ACTION = re.compile(r"```" + P.FENCE + r"[ \t]*\n(.*?)\n?```", re.S)


def stable_int(*parts) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


@dataclass
class Budget:
    model_calls: int = 50            # every model call counts: agent steps, plans, reflections, summaries
    wall_s: float = 3600.0
    cmd_timeout_s: int = 180
    max_tokens: int = 4096
    max_format_errors: int = 3       # consecutive
    usd: float | None = None         # spend allowed for this episode (hosted APIs); the episode ends when it is used up
    snapshot_at: int | None = None   # keep the diff as it stands once this many model calls and their actions are done:
                                     # the state an episode with that call budget would end in; the agent is not told


def budget_meta(b: Budget) -> dict:
    """What a stored record says about the budget it ran under."""
    meta = {"model_calls": b.model_calls, "wall_s": b.wall_s}
    if b.snapshot_at is not None:
        meta["snapshot_at"] = b.snapshot_at
    return meta


class EpisodeStop(Exception):
    """The episode ends here; the reason goes to the log and the repository is graded as it is."""


@dataclass
class NodeLog:
    node: str
    type: str
    visit: int
    outcome: str = ""
    model_calls: int = 0
    commands: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    truncated: int = 0
    latency_s: float = 0.0
    summary: str = ""
    baseline: str = ""               # check node: "measured:<failing tests on the unchanged repository>" or "unavailable"


@dataclass
class Episode:
    episode_id: str
    graph: str
    task_id: str
    seed: int
    model: str
    status: str = "ok"               # ok | engine_error (engine errors are never counted as task failures)
    exit_reason: str = ""            # end | visit_cap | model_call_budget | wall_time | usd_budget | context_exceeded | format_errors
    resolved: bool = False
    agent_done: bool = False         # the model declared the task finished at least once
    grade: dict = field(default_factory=dict)
    diff: str = ""
    path: list = field(default_factory=list)
    node_logs: list = field(default_factory=list)
    events: list = field(default_factory=list)       # transcript: every model output and every observation
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0           # prompt tokens served from the provider's prompt cache
    reasoning_tokens: int = 0        # completion tokens spent on hidden reasoning
    cost_usd: float = 0.0            # 0 for a local model
    max_prompt_tokens: int = 0
    latency_s: float = 0.0
    exec_s: float = 0.0              # time until the graph stopped, before the final diff is read and graded
    error: str = ""
    snapshot: dict = field(default_factory=dict)     # only with Budget.snapshot_at: the state at that call, graded like the end

    def to_dict(self) -> dict:
        return asdict(self)


def episode_id(graph: str, task_id: str, seed: int, model: str) -> str:
    return f"{model}/{graph}/{task_id}/s{seed}"


class Context:
    """What node handlers share: the workspace, the conversation, and metered model calls."""

    def __init__(self, task, ws, llm, budget, eid, baseline):
        self.task, self.ws, self.llm, self.budget, self.eid = task, ws, llm, budget, eid
        self.baseline = baseline               # async () -> set of failing visible tests before any change
        self.head = [{"role": "system", "content": P.SYSTEM},
                     {"role": "user", "content": P.TASK.format(problem=task["problem_statement"], fence=P.FENCE,
                                                               cmd_timeout=budget.cmd_timeout_s, done=P.DONE_COMMAND)}]
        self.messages = [dict(m) for m in self.head]
        self.state: dict = {}
        self.calls, self.t0 = 0, time.perf_counter()
        self.spent_usd = 0.0
        self.max_prompt, self.last_prompt = 0, 0
        self.agent_done = False
        self.events: list = []
        self.log: NodeLog | None = None
        self.tokens = [0, 0]                   # prompt and completion tokens so far
        self.snapshot: dict | None = None

    def say(self, text: str) -> None:
        """Add harness text to the conversation (merged into a trailing user turn, so roles alternate)."""
        if self.messages[-1]["role"] == "user":
            self.messages[-1]["content"] += "\n\n" + text
        else:
            self.messages.append({"role": "user", "content": text})
        self.events.append({"node": self.log.node, "role": "harness", "content": text[:6000]})

    async def ask(self, max_tokens: int | None = None) -> str:
        if self.snapshot is None and self.calls == self.budget.snapshot_at:
            peek = getattr(self.ws, "peek_diff", self.ws.diff)      # leaves nothing behind for the agent to find
            self.snapshot = {"diff": await asyncio.to_thread(peek), "model_calls": self.calls, "terminal": False,
                             "prompt_tokens": self.tokens[0], "completion_tokens": self.tokens[1],
                             "elapsed_s": round(time.perf_counter() - self.t0, 3)}
        if self.calls >= self.budget.model_calls:
            raise EpisodeStop("model_call_budget")
        if time.perf_counter() - self.t0 > self.budget.wall_s:
            raise EpisodeStop("wall_time")
        if self.budget.usd is not None and self.spent_usd >= self.budget.usd:
            raise EpisodeStop("usd_budget")
        self.calls += 1
        try:
            r = await self.llm.chat(self.messages, max_tokens=max_tokens or self.budget.max_tokens,
                                    seed=stable_int(self.eid, self.calls), affinity=stable_int(self.eid))
        except ContextExceeded:
            raise EpisodeStop("context_exceeded") from None
        self.log.model_calls += 1
        self.log.prompt_tokens += r.prompt_tokens
        self.log.completion_tokens += r.completion_tokens
        self.tokens[0] += r.prompt_tokens
        self.tokens[1] += r.completion_tokens
        self.log.cached_tokens += r.cached_tokens
        self.log.reasoning_tokens += r.reasoning_tokens
        self.log.cost_usd = round(self.log.cost_usd + r.cost_usd, 6)
        self.spent_usd += r.cost_usd
        self.log.truncated += int(r.truncated)
        self.last_prompt = r.prompt_tokens
        self.max_prompt = max(self.max_prompt, r.prompt_tokens)
        self.messages.append({"role": "assistant", "content": r.text})
        self.events.append({"node": self.log.node, "role": "model", "content": r.text[:6000], "truncated": r.truncated})
        self._truncated = r.truncated
        return r.text

    async def run(self, command: str, timeout: int | None = None) -> dict:
        out = await asyncio.to_thread(self.ws.execute, command, timeout or self.budget.cmd_timeout_s)
        self.log.commands += 1
        return out


# ---- node handlers: (ctx, node, visit) -> (outcome, summary) ----------------

async def _agent(ctx: Context, node: Node, visit: int):
    role = node.params.get("role", "solve")
    checker = bool(node.params.get("checker"))
    ctx.say((P.ROLE[role] + ("\n\n" + P.CHECKER if checker else "")) if visit == 1 else P.CONTINUE)
    cap, compact_at = node.params.get("max_steps"), node.params.get("compact_at")
    edit_by = node.params.get("edit_by")
    steps = bad = 0
    while True:
        if cap is not None and steps >= cap:
            return "step_cap", f"{steps} steps"
        text = await ctx.ask()
        steps += 1
        actions = ACTION.findall(text)
        if len(actions) != 1:
            bad += 1
            if bad >= ctx.budget.max_format_errors:
                raise EpisodeStop("format_errors")
            ctx.say(P.TRUNCATED if ctx._truncated else P.FORMAT_ERROR.format(n=len(actions)))
            continue
        command = actions[0].strip()
        if command.startswith(P.DONE_COMMAND) and not ctx.state.get("work"):
            bad += 1                                   # finishing before any command was run is not an answer
            if bad >= ctx.budget.max_format_errors:
                raise EpisodeStop("format_errors")
            ctx.say(P.EARLY_FINISH)
            continue
        bad = 0
        if command.startswith(P.DONE_COMMAND):
            ctx.agent_done = True
            ctx.state["work"] = True
            return "done", f"{steps} steps"
        if checker and command == P.CHECK_COMMAND:     # the same checker the check node runs, at the agent's request
            ok, feedback, reason = await run_check(ctx.ws, await ctx.baseline())
            ctx.log.commands += 1
            ctx.state["checker_calls"] = ctx.state.get("checker_calls", 0) + 1
            out = {"returncode": 0 if ok else 1}
            obs = P.CHECK_PASSED if ok else P.CHECK_FAILED.format(feedback=feedback)
        else:
            out = await ctx.run(command)
            obs = P.observation(out)
        ctx.messages.append({"role": "user", "content": obs})
        ctx.events.append({"node": node.name, "role": "command", "content": command[:3000]})
        ctx.events.append({"node": node.name, "role": "observation", "content": obs[:3000], "rc": out["returncode"]})
        ctx.state["work"] = True
        if edit_by is not None and steps == edit_by and not ctx.state.get("stalled"):
            if not (await asyncio.to_thread(ctx.ws.diff)).strip():      # fires at most once per episode
                ctx.state["stalled"] = True
                ctx.state["feedback"] = P.STALLED.format(n=steps)
                ctx.say(ctx.state["feedback"])
                return "stalled", f"{steps} steps, no edit"
        if compact_at is not None and ctx.last_prompt > compact_at:
            return "context_high", f"{steps} steps, prompt {ctx.last_prompt} tokens"


async def _plan(ctx: Context, node: Node, visit: int):
    listing = (await ctx.run("git ls-files | head -n 150"))["output"].strip()
    ctx.say(P.PLAN.format(n=150, listing=listing))
    ctx.state["plan"] = await ctx.ask(max_tokens=600)
    ctx.say(P.PLAN_FOLLOW)
    return "next", f"plan {len(ctx.state['plan'])} chars"


async def _check(ctx: Context, node: Node, visit: int):
    if node.params.get("mode") == "forced":            # control: no check is run and no outcome is consulted
        await restore_protected(ctx.ws)
        ctx.log.commands += 1
        if visit > FORCED_RETURNS:
            return "pass", "forced:submit"
        ctx.state["feedback"] = P.FORCED_CONTINUE
        ctx.say(P.FORCED_CONTINUE)
        return "fail", f"forced:continue:{visit}"
    base = await ctx.baseline()
    ctx.log.baseline = "unavailable" if base is None else f"measured:{len(base)}"
    if base is None and node.params.get("baseline") == "required":
        # the suite could not be measured on the unchanged repository: an infrastructure error (the run is repeated),
        # never a check that passes without having consulted the tests
        raise RuntimeError("check baseline unavailable")
    ok, feedback, reason = await run_check(ctx.ws, base)
    ctx.log.commands += 1
    if ok:
        return "pass", reason
    if node.params.get("feedback") == "binary":        # control: the same check, none of its detail reaches the agent
        ctx.state["feedback"] = P.CHECK_FAILED_BINARY
        ctx.say(P.CHECK_FAILED_BINARY)
        return "fail", reason
    ctx.state["feedback"] = feedback
    ctx.say(P.CHECK_FAILED.format(feedback=feedback))
    return "fail", reason


async def _reflect(ctx: Context, node: Node, visit: int):
    ctx.say(P.REFLECT)
    ctx.state["lesson"] = await ctx.ask(max_tokens=500)
    return "next", f"lesson {len(ctx.state['lesson'])} chars"


async def _reset(ctx: Context, node: Node, visit: int):
    await asyncio.to_thread(ctx.ws.restore)
    ctx.messages = [dict(m) for m in ctx.head]
    ctx.say(P.RESET.format(lesson=ctx.state.get("lesson") or ctx.state["feedback"]))
    return "next", "repository restored, conversation cleared"


async def _compact(ctx: Context, node: Node, visit: int):
    before = ctx.last_prompt
    ctx.say(P.COMPACT)
    summary = await ctx.ask(max_tokens=1200)
    diff = await asyncio.to_thread(ctx.ws.diff)
    ctx.messages = [dict(m) for m in ctx.head]
    ctx.say(P.COMPACTED.format(summary=summary, diff=diff[:8000] or "(no changes yet)"))
    return "next", f"prompt was {before} tokens"


FORCED_RETURNS = 2            # the forced-continuation control sends the agent back twice, as two failed checks would


HANDLERS = {"agent": _agent, "plan": _plan, "check": _check, "reflect": _reflect, "reset": _reset, "compact": _compact}


async def run_episode(graph: Graph, task: dict, ws, llm: LLM, seed: int, model: str = "mock",
                      budget: Budget | None = None,
                      baseline: Callable[[], Awaitable[set | None]] | None = None,
                      grade: Callable[[str], Awaitable[dict]] | None = None) -> Episode:
    """``ws`` is a ready workspace (see swe.container.Workspace); ``grade(diff)`` scores the final diff."""
    budget = budget or Budget()
    eid = episode_id(graph.name, task["instance_id"], seed, model)
    ep = Episode(eid, graph.name, task["instance_id"], seed, model)

    async def no_baseline():
        return None
    ctx = Context(task, ws, llm, budget, eid, baseline or no_baseline)
    visits: dict[str, int] = {}
    log = None
    try:
        name = graph.entry
        try:
            while name != END:
                node = graph.nodes[name]
                if visits.get(name, 0) >= node.max_visits:
                    ep.exit_reason = "visit_cap"
                    break
                visits[name] = visits.get(name, 0) + 1
                log = NodeLog(name, node.type, visits[name])
                ctx.log = log
                t = time.perf_counter()
                try:
                    log.outcome, log.summary = await HANDLERS[node.type](ctx, node, visits[name])
                finally:
                    log.latency_s = round(time.perf_counter() - t, 3)
                    ep.node_logs.append(asdict(log))
                    ep.path.append(name)
                name = graph.next_node(name, log.outcome)
            else:
                ep.exit_reason = "end"
        except EpisodeStop as stop:
            ep.exit_reason = str(stop)
            ep.node_logs[-1]["outcome"] = "stopped:" + str(stop)
        ep.exec_s = round(time.perf_counter() - ctx.t0, 3)
        ep.diff = await asyncio.to_thread(ws.diff)
        if grade is not None and ep.diff.strip():
            ep.grade = await grade(ep.diff)
            ep.resolved = bool(ep.grade.get("resolved"))
        if budget.snapshot_at is not None:
            # an episode that ended before the snapshot call contributes its final state; the same diff is not graded twice
            snap = ctx.snapshot or {"diff": ep.diff, "model_calls": ctx.calls, "terminal": True, "prompt_tokens": ctx.tokens[0],
                                    "completion_tokens": ctx.tokens[1], "elapsed_s": round(time.perf_counter() - ctx.t0, 3)}
            if snap["diff"] == ep.diff:
                snap["grade"] = ep.grade
            else:
                snap["grade"] = await grade(snap["diff"]) if grade is not None and snap["diff"].strip() else {}
            snap["resolved"] = bool(snap["grade"].get("resolved"))
            ep.snapshot = {"at": budget.snapshot_at, **snap}
    except Exception as e:
        ep.status, ep.error = "engine_error", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
    ep.agent_done, ep.events = ctx.agent_done, ctx.events
    ep.model_calls = sum(l["model_calls"] for l in ep.node_logs)
    ep.prompt_tokens = sum(l["prompt_tokens"] for l in ep.node_logs)
    ep.completion_tokens = sum(l["completion_tokens"] for l in ep.node_logs)
    ep.cached_tokens = sum(l["cached_tokens"] for l in ep.node_logs)
    ep.reasoning_tokens = sum(l["reasoning_tokens"] for l in ep.node_logs)
    ep.cost_usd = round(sum(l["cost_usd"] for l in ep.node_logs), 6)
    ep.max_prompt_tokens = ctx.max_prompt
    ep.latency_s = round(time.perf_counter() - ctx.t0, 3)
    return ep
