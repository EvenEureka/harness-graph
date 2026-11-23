"""Run graphs x tasks x seeds on SWE-smith tasks against an OpenAI-compatible endpoint.

Each episode gets its own container workspace, runs one harness graph, and is graded in a clean
container right after.  Resumable: episodes already in the output file with status ok are skipped.

  python scripts/run_grid.py --tasks data/tasks.jsonl --model qwen3.8-27b \
      --base-url http://127.0.0.1:8000/v1 --no-thinking --out results/raw/qwen3.8-27b.jsonl
"""
import argparse
import asyncio
import concurrent.futures
import json
import os
import time
from pathlib import Path

from harness_graph import Budget, OpenAIChat, load_battery, run_episode
from harness_graph.llm import Prices
from harness_graph.check import visible_failures
from harness_graph.executor import budget_meta, episode_id
from harness_graph.swe.container import Workspace, ensure_sif
from harness_graph.swe.grading import grade, ungradable_reason

ROOT = Path(__file__).resolve().parents[1]


def done_ids(path: Path) -> set[str]:
    out = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue                       # a line cut off by a kill
                if e.get("status") == "ok":
                    out.add(e["episode_id"])
    return out


def load_tasks(path: str, per_repo: int = 0) -> list[dict]:
    tasks = [json.loads(l) for l in open(path) if l.strip()]
    tasks = [t for t in tasks if not ungradable_reason(t)]
    if per_repo:
        seen: dict[str, int] = {}
        keep = []
        for t in tasks:
            seen[t["repo_key"]] = seen.get(t["repo_key"], 0) + 1
            if seen[t["repo_key"]] <= per_repo:
                keep.append(t)
        tasks = keep
    return tasks


async def main(a):
    asyncio.get_running_loop().set_default_executor(concurrent.futures.ThreadPoolExecutor(4 * a.workers + 8))
    graphs = load_battery(a.graphs)
    if a.only:
        graphs = [g for g in graphs if g.name in a.only.split(":")]
    tasks = load_tasks(a.tasks, a.per_repo)
    sifs = {img: ensure_sif(img) for img in sorted({t["image_name"] for t in tasks})}
    if a.api_key_env:                                   # hosted API: reasoning-model request shape, hard USD cap
        if a.max_usd is None:
            raise SystemExit("--max-usd is required with --api-key-env")
        llm = OpenAIChat(a.model, a.base_url, api_key=os.environ[a.api_key_env], concurrency=a.workers,
                         temperature=None, top_p=None, reasoning_effort=a.reasoning_effort, token_headroom=4096,
                         prices=Prices(*a.prices), max_usd=a.max_usd)
    else:                                               # local vLLM server
        extra = {"top_k": 20}
        if a.no_thinking:
            extra["chat_template_kwargs"] = {"enable_thinking": False}
        # with thinking on, the reasoning is part of the completion: leave room for it beside the answer
        llm = OpenAIChat(a.model, a.base_url, concurrency=a.workers, extra_body=extra,
                         token_headroom=0 if a.no_thinking else a.thinking_headroom)
    budget = Budget(model_calls=a.model_calls, wall_s=a.wall_s, snapshot_at=a.snapshot_at)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    have = done_ids(out)
    seeds = [int(s) for s in a.seeds.split(":")]
    # seeds outermost, graphs interleaved: a partial run is balanced across graphs, and load is mixed in time
    todo = [(g, t, s) for s in seeds for t in tasks for g in graphs
            if episode_id(g.name, t["instance_id"], s, a.model) not in have]
    print(f"{len(graphs)} graphs x {len(tasks)} tasks x {len(seeds)} seeds; {len(have)} done, {len(todo)} to run",
          flush=True)

    base_cache: dict[str, set | None] = {}
    base_locks: dict[str, asyncio.Lock] = {}

    def baseline_for(task):
        async def get():
            """Failing visible tests before any change, measured once per task in a separate workspace."""
            tid = task["instance_id"]
            async with base_locks.setdefault(tid, asyncio.Lock()):
                if tid not in base_cache:
                    ws = Workspace(sifs[task["image_name"]], tag="base")
                    try:
                        # visible tests: what fails on the healthy repository; hidden: what fails before the agent acts
                        await asyncio.to_thread(ws.setup, task, a.visible_tests, not a.visible_tests)
                        base_cache[tid] = await visible_failures(ws)
                    finally:
                        ws.close()
            return base_cache[tid]
        return get

    gate = asyncio.Semaphore(a.workers)
    t0, n, errs, solved = time.time(), 0, 0, 0
    job = os.environ.get("JOB_ID", "x")
    with open(out, "a") as f:
        async def one(g, t, s):
            nonlocal n, errs, solved
            async with gate:
                if llm.capped:                         # USD cap reached: leave the rest for a later resume
                    return
                sif = sifs[t["image_name"]]
                ws = Workspace(sif, tag=job, cmd_timeout=budget.cmd_timeout_s)
                try:
                    await asyncio.to_thread(ws.setup, t, a.visible_tests)

                    async def grade_fn(diff):
                        g_ = await asyncio.to_thread(grade, t, sif, "model", diff)
                        g_["log_tail"] = g_["log_tail"][-600:]
                        return g_
                    ep = await run_episode(g, t, ws, llm, s, model=a.model, budget=budget,
                                           baseline=baseline_for(t), grade=grade_fn)
                    rec = ep.to_dict()
                except Exception as e:  # workspace could not be built: an engine error, not a task failure
                    rec = {"episode_id": episode_id(g.name, t["instance_id"], s, a.model), "graph": g.name,
                           "task_id": t["instance_id"], "seed": s, "model": a.model, "status": "engine_error",
                           "error": f"setup: {type(e).__name__}: {e}"[:1500], "resolved": False}
                finally:
                    await asyncio.to_thread(ws.close)
            rec["task"] = {k: t.get(k) for k in ("repo_key", "strategy", "gold_files")}
            rec["tests"] = "visible" if a.visible_tests else "hidden"
            rec["budget"] = budget_meta(budget)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            errs += rec["status"] != "ok"
            solved += bool(rec.get("resolved"))
            print(f"[{time.strftime('%H:%M:%S')}] {n}/{len(todo)} {rec['episode_id']} status={rec['status']} "
                  f"exit={rec.get('exit_reason')} resolved={rec.get('resolved')} calls={rec.get('model_calls')} "
                  f"path={'>'.join(rec.get('path', []))} t={rec.get('latency_s')}s "
                  f"usd={rec.get('cost_usd', 0):.3f} total=${llm.spent:.2f}", flush=True)
        await asyncio.gather(*[one(*x) for x in todo])
    wall = time.time() - t0
    print(f"DONE {n} episodes, {solved} resolved, {errs} engine errors, {wall:.0f}s, "
          f"{n / max(wall, 1) * 3600:.0f} episodes/h, spent ${llm.spent:.2f}"
          + (" (USD CAP REACHED, run again with a higher cap to resume)" if llm.capped else ""), flush=True)
    if llm.capped:
        return
    if n and errs / n > a.max_error_rate:
        raise SystemExit(f"engine error rate {errs / n:.1%} exceeds {a.max_error_rate:.1%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True)
    p.add_argument("--graphs", default=str(ROOT / "graphs"))
    p.add_argument("--only", default="", help="colon-separated graph names")
    p.add_argument("--seeds", default="0", help="colon-separated seed ids, e.g. 0:1:2")
    p.add_argument("--per-repo", type=int, default=0, help="first N tasks of each repository only")
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="comma-separated for several replicas")
    p.add_argument("--workers", type=int, default=12, help="episodes in flight")
    p.add_argument("--model-calls", type=int, default=50)
    p.add_argument("--wall-s", type=float, default=3600)
    p.add_argument("--snapshot-at", type=int, default=None,
                   help="also grade the diff as it stands after this many model calls (the agent is not told)")
    p.add_argument("--no-thinking", action="store_true", help="send enable_thinking=false (Qwen chat template)")
    p.add_argument("--thinking-headroom", type=int, default=8192, help="extra output tokens when thinking is on")
    p.add_argument("--api-key-env", default="", help="name of the env var holding the key of a hosted API")
    p.add_argument("--reasoning-effort", default="low")
    p.add_argument("--prices", type=float, nargs=3, default=[0, 0, 0], metavar=("IN", "CACHED_IN", "OUT"),
                   help="USD per million tokens")
    p.add_argument("--max-usd", type=float, default=None, help="hard spend cap for this invocation")
    p.add_argument("--visible-tests", action="store_true",
                   help="leave the files with the failing tests in the workspace (write to a separate --out file)")
    p.add_argument("--max-error-rate", type=float, default=0.05)
    p.add_argument("--out", required=True)
    asyncio.run(main(p.parse_args()))
