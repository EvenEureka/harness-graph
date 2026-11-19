"""Stage 4, step 1: the graphs that go to the selection tasks.  No model call.

Candidates (frozen in refine-logs/EXPERIMENT_PLAN.md):
  - one standalone replay run per panel method on all search tasks of the panel table (budget 512, seed 0);
  - the recommendation of every live whole-grammar search;
  - the plain loop (h01) and check-and-retry (h04).
Identical graphs are merged.  Writes OUT/<graph>.yaml for the grid runner and OUT/candidates.json (which source
named which graph, and what each search saw of its own recommendation).

  python scripts/select_candidates.py results/raw/search_table/a/visible/qwen3.8-27b.jsonl \
      results/raw/search_table/b/visible/qwen3.8-27b.jsonl --panel graphs_panel/panel.json \
      --live results/search/local_*/result.json --out graphs_select
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from harness_graph import analysis as A
from harness_graph.grammar import Config, canonical, compile_config, features
from harness_graph.search import METHODS, Replay


def main(a):
    df = A.scored(A.load_episodes(*a.files))
    rows = df[df.tests == "visible"].to_dict("records")
    panel = {n: canonical(Config(**d)) for n, d in json.loads(Path(a.panel).read_text())["configs"].items()}
    arms, tasks = Replay(rows, np.random.default_rng(0)).arms, Replay(rows, np.random.default_rng(0)).tasks
    if set(arms) != set(panel):
        raise SystemExit(f"table graphs and panel differ: {sorted(set(arms) ^ set(panel))}")
    want = {json.loads(l)["instance_id"] for l in open(a.tasks) if l.strip()}
    if set(tasks) != want:
        raise SystemExit(f"tasks with every graph differ from {a.tasks}: {len(set(tasks) ^ want)} differ")

    config: dict[str, Config] = {}
    sources: dict[str, list[str]] = defaultdict(list)

    def add(c: Config, source: str):
        c = canonical(c)
        config[c.name] = c
        sources[c.name].append(source)

    add(Config(), "h01 plain loop")
    add(Config(gate=True), "h04 check-and-retry")
    panel_runs = {}
    for name, fn in METHODS.items():
        rng = np.random.default_rng(a.seed)
        rp = Replay(rows, rng)
        kw = {"features": {g: features(panel[g]) for g in arms}} if name == "linear_thompson" else {}
        res = fn(arms, tasks, rp.pull, a.budget, rng, **kw)
        rec = res.recommendation
        panel_runs[name] = {"recommendation": rec, "pulls_of_recommendation": res.pulls[rec],
                            "mean_seen_by_search": res.means[rec], "spent": res.spent,
                            "reused_share": rp.reused / max(rp.pulls, 1), "table_mean": rp.truth()[rec]}
        add(panel[rec], f"panel {name}")
    live = {}
    for path in sorted(a.live):
        r = json.loads(Path(path).read_text())
        run = Path(path).parent.name
        if r["spent"] != a.budget:
            raise SystemExit(f"{run}: spent {r['spent']}, expected {a.budget}")
        rec = r["recommendation"]
        live[run] = {"method": r["method"], "seed": r["args"]["seed"], "recommendation": rec,
                     "pulls_of_recommendation": r["pulls"][rec], "mean_seen_by_search": r["means"][rec],
                     "steps": len(r["trace"]), "accepted": sum(t["accepted"] for t in r["trace"]),
                     "graphs_run": len(r["pulls"]), "engine_errors": r["engine_errors"]}
        add(Config(**r["recommendation_config"]), f"live {run}")
    seeds_of = defaultdict(list)
    for r in live.values():
        seeds_of[r["method"]].append(r["seed"])
    want_seeds = [int(s) for s in a.live_seeds.split(":")]
    if set(seeds_of) != {"local_search_guided", "local_search_random"} or any(sorted(v) != want_seeds for v in seeds_of.values()):
        raise SystemExit(f"live results must be seeds {want_seeds} of both local methods; found {dict(seeds_of)}")
    if len(config) > a.max_graphs:
        raise SystemExit(f"{len(config)} candidates exceed the frozen maximum of {a.max_graphs}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.yaml"):
        f.unlink()
    for n, c in config.items():
        (out / f"{n}.yaml").write_text(yaml.safe_dump(compile_config(c).to_dict(), sort_keys=False, allow_unicode=True))
    (out / "candidates.json").write_text(json.dumps(
        {"budget": a.budget, "seed": a.seed, "table_files": a.files, "tasks": a.tasks, "n_table_tasks": len(tasks),
         "configs": {n: c.__dict__ for n, c in config.items()}, "sources": dict(sources),
         "panel_runs": panel_runs, "live": live}, indent=1))
    print(f"{len(config)} candidate graphs -> {out}")
    for n in config:
        print(f"  {n:34s} {len(compile_config(config[n]).nodes):2d} nodes  <- {'; '.join(sources[n])}")
    print("panel runs:", json.dumps(panel_runs, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--panel", required=True)
    p.add_argument("--live", nargs="+", required=True)
    p.add_argument("--budget", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tasks", default="data/v2/search.jsonl", help="the search tasks the table must cover exactly")
    p.add_argument("--live-seeds", default="0:1:2:3:4")
    p.add_argument("--max-graphs", type=int, default=16)
    p.add_argument("--out", required=True)
    main(p.parse_args())
