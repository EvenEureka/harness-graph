"""Live budgeted search over harness graphs on SWE-smith tasks.

  python scripts/run_search.py --tasks data/search_tasks.jsonl --panel sample:16:0 --method successive_halving \
      --budget 512 --parallel 12 --seed 0 --model qwen3.8-27b --base-url http://127.0.0.1:8000/v1 --no-thinking \
      --visible-tests --out results/search/run0

--panel   hand            the 17 hand-written graphs
          sample:N:SEED   h01, h04 and N-2 graphs drawn at random from the grammar
          FILE.json       a list of design-variable dicts
--method  uniform | successive_halving | thompson | linear_thompson | local_random | local_guided
          (the two local methods ignore the panel and move through the whole grammar from the plain loop)

Writes OUT/episodes.jsonl (every episode; a rerun with the same arguments resumes from it) and OUT/result.json.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from harness_graph import Budget, OpenAIChat
from harness_graph.grammar import HAND_WRITTEN, Config, all_configs, canonical, compile_config, features, neighbours
from harness_graph.llm import Prices
from harness_graph.runner import EpisodeRunner, LivePull
from harness_graph.search import METHODS, local_search, remedy_proposer


def panel(spec: str) -> list[Config]:
    if spec == "hand":
        return list(HAND_WRITTEN.values())
    if spec.startswith("sample:"):
        _, n, seed = spec.split(":")
        rng = np.random.default_rng(int(seed))
        fixed = [Config(), Config(gate=True)]
        rest = [c for c in all_configs() if c not in fixed]
        return fixed + [rest[i] for i in rng.choice(len(rest), size=int(n) - 2, replace=False)]
    return [canonical(Config(**d)) for d in json.loads(Path(spec).read_text())]


class Graphs(dict):
    """name -> compiled graph, filled on demand (the local methods name graphs outside the panel)."""

    def __init__(self):
        super().__init__()
        self.config: dict[str, Config] = {}

    def add(self, c: Config) -> str:
        c = canonical(c)
        if c.name not in self:
            self[c.name] = compile_config(c)
            self.config[c.name] = c
        return c.name


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(l) for l in open(a.tasks) if l.strip()]
    if a.api_key_env:
        if a.max_usd is None:
            raise SystemExit("--max-usd is required with --api-key-env")
        llm = OpenAIChat(a.model, a.base_url, api_key=os.environ[a.api_key_env], concurrency=a.parallel,
                         temperature=None, top_p=None, reasoning_effort=a.reasoning_effort, token_headroom=4096,
                         prices=Prices(*a.prices), max_usd=a.max_usd)
    else:
        extra = {"top_k": 20}
        if a.no_thinking:
            extra["chat_template_kwargs"] = {"enable_thinking": False}
        llm = OpenAIChat(a.model, a.base_url, concurrency=a.parallel, extra_body=extra,
                         token_headroom=0 if a.no_thinking else a.thinking_headroom)
    runner = EpisodeRunner(tasks, llm, a.model, out / "episodes.jsonl", Budget(model_calls=a.model_calls, wall_s=a.wall_s),
                           visible_tests=a.visible_tests, workers=a.parallel)
    graphs = Graphs()
    reward = lambda row: float(bool(row["resolved"])) - a.cost_weight * row["model_calls"] / a.model_calls
    pull = LivePull(runner, graphs, reward, log=lambda s: print(s, flush=True))
    rng = np.random.default_rng(a.seed)
    task_ids = [t["instance_id"] for t in tasks]
    checkpoints = [int(b) for b in a.checkpoints.split(":")] if a.checkpoints else []
    if a.method.startswith("local_"):
        start = graphs.add(Config())
        nb = lambda name: [graphs.add(n) for n in neighbours(graphs.config[name])]
        propose = remedy_proposer(lambda n: graphs.config[n], graphs.add) if a.method == "local_guided" else None
        res = local_search(start, task_ids, pull, a.budget, rng, nb, propose, batch=a.batch, checkpoints=checkpoints)
    else:
        arms = [graphs.add(c) for c in panel(a.panel)]
        kw = {"features": {n: features(graphs.config[n]) for n in arms}} if a.method == "linear_thompson" else {}
        res = METHODS[a.method](arms, task_ids, pull, a.budget, rng, checkpoints=checkpoints, parallel=a.parallel, **kw)
    rec = {"args": vars(a), "method": res.method, "recommendation": res.recommendation,
           "recommendation_config": graphs.config[res.recommendation].__dict__,
           "recommendation_graph": graphs[res.recommendation].to_dict(),
           "checkpoints": res.checkpoints, "spent": res.spent, "engine_errors": pull.engine_errors,
           "pulls": res.pulls, "means": res.means, "trace": res.trace,
           "configs": {n: c.__dict__ for n, c in graphs.config.items()}}
    (out / "result.json").write_text(json.dumps(rec, indent=1))
    print(f"DONE method={res.method} spent={res.spent} recommendation={res.recommendation} "
          f"mean={res.means.get(res.recommendation)} engine_errors={pull.engine_errors}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True)
    p.add_argument("--panel", default="hand")
    p.add_argument("--method", required=True, choices=sorted(METHODS) + ["local_random", "local_guided"])
    p.add_argument("--budget", type=int, required=True, help="episodes the search may spend")
    p.add_argument("--checkpoints", default="")
    p.add_argument("--batch", type=int, default=16, help="tasks per paired comparison (local methods)")
    p.add_argument("--parallel", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cost-weight", type=float, default=0.0)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model-calls", type=int, default=50)
    p.add_argument("--wall-s", type=float, default=3600)
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--thinking-headroom", type=int, default=8192)
    p.add_argument("--api-key-env", default="")
    p.add_argument("--reasoning-effort", default="low")
    p.add_argument("--prices", type=float, nargs=3, default=[0, 0, 0])
    p.add_argument("--max-usd", type=float, default=None)
    p.add_argument("--visible-tests", action="store_true")
    p.add_argument("--out", required=True)
    main(p.parse_args())
