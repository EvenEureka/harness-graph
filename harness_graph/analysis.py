"""Analysis of agent episodes: per-graph resolve rate, paired differences, gate behaviour, routing.

The unit of resampling is the task.  Seeds of the same (graph, task) are averaged first, so a
confidence interval reflects which tasks were drawn, not how many seeds were run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE = "h01_react"


def episode_row(e: dict) -> dict:
    """Summary of one episode record (transcript dropped)."""
    logs = e.get("node_logs", [])
    checks = [l for l in logs if l["type"] == "check" and l["outcome"] in ("pass", "fail")]
    gold = (e.get("task") or {}).get("gold_files") or 0
    n_gold = gold if isinstance(gold, int) else len(gold)
    return {
        "episode_id": e["episode_id"], "model": e["model"], "graph": e["graph"], "task_id": e["task_id"],
        "seed": e["seed"], "tests": e.get("tests", "hidden"), "status": e["status"], "exit_reason": e.get("exit_reason", ""),
        "resolved": bool(e.get("resolved")), "agent_done": bool(e.get("agent_done")),
        "empty_diff": not (e.get("diff") or "").strip(),
        "repo": (e.get("task") or {}).get("repo_key", ""),
        "multi_file": n_gold > 1,
        "model_calls": e.get("model_calls", 0),
        "tokens": e.get("prompt_tokens", 0) + e.get("completion_tokens", 0),
        "cost_usd": e.get("cost_usd", 0.0), "cached_tokens": e.get("cached_tokens", 0),
        "max_prompt_tokens": e.get("max_prompt_tokens", 0), "latency_s": e.get("latency_s", 0.0),
        "n_checks": len(checks),
        "first_check": checks[0]["outcome"] if checks else None,
        "last_check": checks[-1]["outcome"] if checks else None,
        "check_reasons": [c["summary"].split(":")[0] for c in checks if c["outcome"] == "fail"],
        "n_compact": sum(l["type"] == "compact" for l in logs),
        "n_reset": sum(l["type"] == "reset" for l in logs),
        "path_len": len(e.get("path", [])),
    }


def load_episodes(*paths: str | Path) -> pd.DataFrame:
    """One row per episode (transcripts dropped).  Engine errors are kept and flagged, never scored."""
    rows = []
    for path in paths:
        with open(path) as f:
            for line in f:
                try:
                    rows.append(episode_row(json.loads(line)))
                except json.JSONDecodeError:
                    continue
    df = pd.DataFrame(rows)
    return df.drop_duplicates(["episode_id", "tests"], keep="last").reset_index(drop=True)


def scored(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.status == "ok"]


def task_matrix(df: pd.DataFrame, col: str = "resolved") -> pd.DataFrame:
    """tasks x graphs, mean over seeds; tasks missing any graph are dropped so comparisons stay paired."""
    m = scored(df).pivot_table(index="task_id", columns="graph", values=col, aggfunc="mean")
    return m.dropna()


def boot(values: np.ndarray, rng, reps: int = 4000) -> tuple[float, float]:
    """Percentile bootstrap over rows (tasks) of a 1-D array of per-task values."""
    idx = rng.integers(0, len(values), (reps, len(values)))
    lo, hi = np.percentile(values[idx].mean(1), [2.5, 97.5])
    return float(lo), float(hi)


def graph_table(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Resolve rate per graph with a task-bootstrap CI, the paired difference to the plain loop, and cost."""
    rng = np.random.default_rng(seed)
    m = task_matrix(df)
    ok = scored(df)
    cost = ok.groupby("graph").agg(calls=("model_calls", "mean"), tokens=("tokens", "mean"),
                                   latency_s=("latency_s", "mean"))
    rows = []
    for g in m.columns:
        lo, hi = boot(m[g].to_numpy(), rng)
        row = {"graph": g, "n_tasks": len(m), "resolved": m[g].mean(), "lo": lo, "hi": hi}
        if BASELINE in m.columns:
            d = (m[g] - m[BASELINE]).to_numpy()
            dlo, dhi = boot(d, rng) if g != BASELINE else (0.0, 0.0)
            row.update(delta=d.mean(), dlo=dlo, dhi=dhi,
                       wins=int((d > 0).sum()), losses=int((d < 0).sum()))
        row.update(cost.loc[g].to_dict())
        rows.append(row)
    return pd.DataFrame(rows).set_index("graph")


def cost_ratio(df: pd.DataFrame, graph: str, col: str = "tokens", seed: int = 0) -> dict:
    """Total ``col`` of ``graph`` divided by that of the plain loop on the same tasks, with a task-bootstrap CI."""
    rng = np.random.default_rng(seed)
    m = task_matrix(df, col)
    a, b = m[graph].to_numpy(), m[BASELINE].to_numpy()
    idx = rng.integers(0, len(m), (4000, len(m)))
    r = a[idx].sum(1) / b[idx].sum(1)
    return {"graph": graph, "col": col, "ratio": float(a.sum() / b.sum()),
            "lo": float(np.percentile(r, 2.5)), "hi": float(np.percentile(r, 97.5))}


def failure_modes(df: pd.DataFrame) -> pd.DataFrame:
    """Share of episodes by how they ended, per graph and bug type (single- or multi-file)."""
    ok = scored(df).copy()
    ok["mode"] = np.select(
        [ok.resolved, ok.empty_diff, ok.exit_reason == "end"],
        ["resolved", "no_edit", "wrong_patch_submitted"], "cut_off_with_edit")
    return pd.crosstab([ok.graph, ok.multi_file], ok["mode"], normalize="index")


def slice_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Resolve rate per graph within each level of a task attribute (e.g. multi_file, repo)."""
    ok = scored(df)
    per_task = ok.groupby(["graph", by, "task_id"]).resolved.mean().reset_index()
    return per_task.pivot_table(index="graph", columns=by, values="resolved", aggfunc="mean")


def seed_noise(df: pd.DataFrame) -> dict:
    """How much two runs of the *same* graph differ: the yardstick for any between-graph difference."""
    ok = scored(df)
    seeds = sorted(ok.seed.unique())
    if len(seeds) < 2:
        return {}
    a, b = seeds[:2]
    m = ok[ok.seed.isin([a, b])].pivot_table(index=["graph", "task_id"], columns="seed", values="resolved").dropna()
    per_graph = (m[a] - m[b]).groupby("graph").mean()
    return {"flip_rate": float((m[a] != m[b]).mean()),
            "abs_rate_gap_mean": float(per_graph.abs().mean()), "abs_rate_gap_max": float(per_graph.abs().max())}


def exit_table(df: pd.DataFrame) -> pd.DataFrame:
    ok = scored(df)
    t = pd.crosstab(ok.graph, ok.exit_reason, normalize="index")
    t["empty_diff"] = ok.groupby("graph").empty_diff.mean()
    t["agent_done"] = ok.groupby("graph").agent_done.mean()
    return t


def gate_table(df: pd.DataFrame) -> pd.DataFrame:
    """What the check gate did: how often it fired, why, and whether its verdict tracks the hidden tests."""
    ok = scored(df)
    ok = ok[ok.n_checks > 0]
    rows = []
    for g, d in ok.groupby("graph"):
        first_fail = d[d.first_check == "fail"]
        first_pass = d[d.first_check == "pass"]
        reasons = pd.Series([r for rs in d.check_reasons for r in rs]).value_counts().to_dict()
        rows.append({"graph": g, "episodes": len(d), "first_check_fail": len(first_fail) / len(d),
                     "resolved_if_first_pass": first_pass.resolved.mean() if len(first_pass) else np.nan,
                     "resolved_if_first_fail": first_fail.resolved.mean() if len(first_fail) else np.nan,
                     "n_first_fail": len(first_fail), "fail_reasons": reasons})
    return pd.DataFrame(rows).set_index("graph") if rows else pd.DataFrame()


def contrast(df: pd.DataFrame, a: str, b: str, margin: float = 0.05, seed: int = 0) -> dict:
    """Resolve rate of graph ``a`` minus graph ``b`` on the tasks both were scored on (seeds averaged per task)."""
    rng = np.random.default_rng(seed)
    ok = scored(df)
    m = ok[ok.graph.isin([a, b])].pivot_table(index="task_id", columns="graph", values="resolved", aggfunc="mean").dropna()
    d = (m[a] - m[b]).to_numpy()
    lo, hi = boot(d, rng)
    verdict = "a_higher" if lo > 0 else "b_higher" if hi < 0 else "equivalent" if -margin < lo and hi < margin else "inconclusive"
    return {"a": a, "b": b, "n_tasks": len(m), "rate_a": float(m[a].mean()), "rate_b": float(m[b].mean()),
            "delta": float(d.mean()), "lo": lo, "hi": hi, "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
            "engine_errors": int((df[df.graph.isin([a, b])].status != "ok").sum()), "verdict": verdict}


def conditions(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Hidden vs visible tests on the tasks and graphs run under both.

    ``gain`` is what leaving the tests visible adds to a graph's resolve rate; ``interaction`` is how much more
    it adds than it adds to the plain loop (difference of differences, paired by task).
    """
    rng = np.random.default_rng(seed)
    ok = scored(df)
    both = set(ok[ok.tests == "hidden"].graph) & set(ok[ok.tests == "visible"].graph)
    df = df[df.graph.isin(both)]
    h, v = task_matrix(df[df.tests == "hidden"]), task_matrix(df[df.tests == "visible"])
    tasks, graphs = h.index.intersection(v.index), [g for g in h.columns if g in v.columns]
    h, v = h.loc[tasks, graphs], v.loc[tasks, graphs]
    rows = []
    for g in graphs:
        gain = (v[g] - h[g]).to_numpy()
        lo, hi = boot(gain, rng)
        row = {"graph": g, "n_tasks": len(tasks), "hidden": h[g].mean(), "visible": v[g].mean(),
               "gain": gain.mean(), "gain_lo": lo, "gain_hi": hi}
        if BASELINE in graphs and g != BASELINE:
            dv = (v[g] - v[BASELINE]).to_numpy()
            inter = dv - (h[g] - h[BASELINE]).to_numpy()
            (vlo, vhi), (ilo, ihi) = boot(dv, rng), boot(inter, rng)
            row.update(visible_delta=dv.mean(), visible_dlo=vlo, visible_dhi=vhi,
                       interaction=inter.mean(), inter_lo=ilo, inter_hi=ihi)
        rows.append(row)
    return pd.DataFrame(rows).set_index("graph")


def oracle(df: pd.DataFrame) -> dict:
    """Head-room for adaptivity: best single graph vs. the best graph chosen per task with hindsight."""
    m = task_matrix(df)
    return {"best_single": m.mean().idxmax(), "best_single_rate": float(m.mean().max()),
            "baseline_rate": float(m[BASELINE].mean()) if BASELINE in m else None,
            "per_task_oracle_rate": float(m.max(1).mean()),
            "never_solved": float((m.max(1) == 0).mean()), "always_solved": float((m.min(1) == 1).mean())}


def routing(df: pd.DataFrame, key: str, splits: int = 500, seed: int = 0) -> dict:
    """Pick the best graph per level of ``key`` on half of the tasks, score it on the other half.

    Compared with the single best graph picked on the same half.  A positive ``gain`` means task-type
    routing helps on held-out tasks; hindsight routing (same tasks for picking and scoring) is reported
    only as an upper bound.
    """
    rng = np.random.default_rng(seed)
    m = task_matrix(df)
    level = scored(df).drop_duplicates("task_id").set_index("task_id")[key].reindex(m.index)
    tasks = np.arange(len(m))
    vals, lv = m.to_numpy(), level.to_numpy()
    single, routed = [], []
    for _ in range(splits):
        perm = rng.permutation(tasks)
        tr, te = perm[: len(perm) // 2], perm[len(perm) // 2:]
        g0 = vals[tr].mean(0).argmax()
        single.append(vals[te, g0].mean())
        got = np.empty(len(te))
        for i, t in enumerate(te):
            same = tr[lv[tr] == lv[t]]
            got[i] = vals[t, vals[same].mean(0).argmax() if len(same) else g0]
        routed.append(got.mean())
    single, routed = np.array(single), np.array(routed)
    hind = float(np.mean([vals[lv == l].mean(0).max() * (lv == l).mean() for l in np.unique(lv)]) * len(np.unique(lv)))
    gain = routed - single
    return {"key": key, "levels": int(len(np.unique(lv))), "single_heldout": float(single.mean()),
            "routed_heldout": float(routed.mean()), "gain": float(gain.mean()),
            "gain_lo": float(np.percentile(gain, 2.5)), "gain_hi": float(np.percentile(gain, 97.5)),
            "routed_hindsight": hind}
