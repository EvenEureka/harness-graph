"""Runs episodes on SWE-smith tasks for callers that decide what to run as they go (the live search).

``EpisodeRunner.run(graph, task, seed)`` builds a container workspace, runs one harness graph, grades the result
in a clean container and appends the record to the output file.  ``LivePull`` wraps a runner as the ``pull``
object the search methods expect: ``many(pairs)`` runs a batch of (graph name, task id) concurrently, each on a
rollout index not used before for that pair, and answers from the output file when the episode is already there,
so a search restarted with the same seed continues where it stopped.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
from collections import defaultdict
from pathlib import Path

from .analysis import episode_row
from .check import visible_failures
from .executor import Budget, episode_id, run_episode
from .search import Pull
from .swe.container import Workspace, ensure_sif
from .swe.grading import grade


class EpisodeRunner:
    def __init__(self, tasks: list[dict], llm, model: str, out: str | Path, budget: Budget | None = None,
                 visible_tests: bool = False, workers: int = 12):
        self.tasks = {t["instance_id"]: t for t in tasks}
        self.llm, self.model, self.visible = llm, model, visible_tests
        self.budget = budget or Budget()
        self.out = Path(out)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.sifs = {img: ensure_sif(img) for img in sorted({t["image_name"] for t in tasks})}
        self.gate = asyncio.Semaphore(workers)
        self.base_cache: dict[str, set | None] = {}
        self.base_locks: dict[str, asyncio.Lock] = {}
        self.job = os.environ.get("JOB_ID", "x")
        self.done: dict[str, dict] = {}
        if self.out.exists():
            for line in open(self.out):
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("status") == "ok":
                    self.done[e["episode_id"]] = episode_row(e)

    def _baseline(self, task):
        async def get():
            tid = task["instance_id"]
            async with self.base_locks.setdefault(tid, asyncio.Lock()):
                if tid not in self.base_cache:
                    ws = Workspace(self.sifs[task["image_name"]], tag="base")
                    try:
                        await asyncio.to_thread(ws.setup, task, self.visible, not self.visible)
                        self.base_cache[tid] = await visible_failures(ws)
                    finally:
                        ws.close()
            return self.base_cache[tid]
        return get

    async def run(self, graph, task_id: str, seed: int) -> dict:
        """One episode -> its summary row (the fields of ``analysis.episode_row``)."""
        t = self.tasks[task_id]
        eid = episode_id(graph.name, task_id, seed, self.model)
        if eid in self.done:
            return self.done[eid]
        async with self.gate:
            sif = self.sifs[t["image_name"]]
            ws = Workspace(sif, tag=self.job, cmd_timeout=self.budget.cmd_timeout_s)
            try:
                await asyncio.to_thread(ws.setup, t, self.visible)

                async def grade_fn(diff):
                    g = await asyncio.to_thread(grade, t, sif, "model", diff)
                    g["log_tail"] = g["log_tail"][-600:]
                    return g
                rec = (await run_episode(graph, t, ws, self.llm, seed, model=self.model, budget=self.budget,
                                         baseline=self._baseline(t), grade=grade_fn)).to_dict()
            except Exception as e:
                rec = {"episode_id": eid, "graph": graph.name, "task_id": task_id, "seed": seed, "model": self.model,
                       "status": "engine_error", "error": f"setup: {type(e).__name__}: {e}"[:1500], "resolved": False}
            finally:
                await asyncio.to_thread(ws.close)
        rec["task"] = {k: t.get(k) for k in ("repo_key", "strategy", "gold_files")}
        rec["tests"] = "visible" if self.visible else "hidden"
        rec["budget"] = {"model_calls": self.budget.model_calls, "wall_s": self.budget.wall_s}
        with open(self.out, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        row = episode_row(rec)
        if rec["status"] == "ok":
            self.done[eid] = row
        return row


class LivePull:
    """``pull`` for the search methods, backed by real episodes.  An engine error is retried once on a new rollout
    index and otherwise scored as a failure of the search step, never silently dropped."""

    def __init__(self, runner: EpisodeRunner, graphs: dict, reward=lambda row: float(bool(row["resolved"])),
                 log=print):
        self.runner, self.graphs, self.reward, self.log = runner, graphs, reward, log
        self.next_seed: dict[tuple[str, str], int] = defaultdict(int)
        self.engine_errors = 0
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.n = 0

    async def _one(self, arm: str, task: str, seed: int) -> Pull:
        row = await self.runner.run(self.graphs[arm], task, seed)
        if row["status"] != "ok":
            self.engine_errors += 1
            row = await self.runner.run(self.graphs[arm], task, seed + 1000)
        self.n += 1
        self.log(f"[pull {self.n}] {arm} {task} s{seed} status={row['status']} resolved={row['resolved']} "
                 f"calls={row['model_calls']} exit={row['exit_reason']}")
        return Pull(self.reward(row) if row["status"] == "ok" else 0.0, row)

    def many(self, pairs: list[tuple[str, str]]) -> list[Pull]:
        jobs = []
        for arm, task in pairs:
            s = self.next_seed[(arm, task)]
            self.next_seed[(arm, task)] += 1
            jobs.append(self._one(arm, task, s))

        async def gather():
            return await asyncio.gather(*jobs)
        return asyncio.run_coroutine_threadsafe(gather(), self.loop).result()

    def __call__(self, arm: str, task: str) -> Pull:
        return self.many([(arm, task)])[0]
