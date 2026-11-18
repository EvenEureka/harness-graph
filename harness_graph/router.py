"""Choosing a graph before the run, from what can be observed before the run.

Three policies over a fixed portfolio of graphs, all fitted on training tasks only:

  FixedBest      the one graph with the highest mean reward (no adaptation; the comparator)
  RuleRouter     the best graph per level of one observable condition (e.g. are executable tests available)
  FeatureRouter  one success model per graph on task features; picks the graph with the highest prediction

``evaluate`` scores a fitted policy on other tasks, and ``crossfit`` repeats fit/evaluate over task splits so
that the policy and its comparator are never scored on the tasks that chose them.

An observation is a dict made by ``observe`` from the task as the agent receives it plus the run condition.
Nothing derived from the gold patch or from the outcome may appear in it.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

_CODE = re.compile(r"```|^\s{4}\S", re.M)
_TRACE = re.compile(r"Traceback \(most recent call last\)|Error:|Exception")


def observe(task: dict, tests_visible: bool, model: str = "") -> dict:
    """Pre-run observables of one episode."""
    text = task.get("problem_statement", "") or ""
    return {
        "tests_visible": bool(tests_visible),
        "model": model,
        "repo": task.get("repo_key", ""),
        "issue_words": len(text.split()),
        "issue_has_code": bool(_CODE.search(text)),
        "issue_has_traceback": bool(_TRACE.search(text)),
        "issue_names_file": bool(re.search(r"\b[\w/]+\.py\b", text)),
    }


NUMERIC = ("tests_visible", "issue_words", "issue_has_code", "issue_has_traceback", "issue_names_file")


def _vec(obs: dict) -> list[float]:
    return [np.log1p(obs[k]) if k == "issue_words" else float(obs[k]) for k in NUMERIC]


class Rewards:
    """Mean reward per (unit, graph), where a unit is one task under one condition; seeds are averaged."""

    def __init__(self, rows: list[dict], obs: dict[str, dict], reward=lambda r: float(bool(r["resolved"])),
                 portfolio: list[str] | None = None, allow_incomplete: bool = False):
        """``portfolio`` fixes the graphs (rows of other graphs are ignored).  A unit that lacks one of the graphs
        is an error unless ``allow_incomplete``; the units left out are then listed in ``dropped``."""
        acc: dict[tuple[str, str], list[float]] = defaultdict(list)
        for r in rows:
            if portfolio is None or r["graph"] in portfolio:
                acc[(r["unit"], r["graph"])].append(reward(r))
        self.graphs = sorted(portfolio) if portfolio is not None else sorted({g for _, g in acc})
        have = defaultdict(set)
        for u, g in acc:
            have[u].add(g)
        self.units = sorted(u for u, gs in have.items() if len(gs) == len(self.graphs))
        self.dropped = sorted(u for u, gs in have.items() if len(gs) != len(self.graphs))
        if self.dropped and not allow_incomplete:
            raise ValueError(f"{len(self.dropped)} units lack a graph of the portfolio, e.g. {self.dropped[:3]}")
        self.mean = {k: float(np.mean(v)) for k, v in acc.items()}
        self.obs = obs

    def of(self, unit: str, graph: str) -> float:
        return self.mean[(unit, graph)]

    def value(self, graph: str, units) -> float:
        return float(np.mean([self.of(u, graph) for u in units]))


class FixedBest:
    name = "fixed_best"

    def fit(self, R: Rewards, units):
        self.graph = max(R.graphs, key=lambda g: (R.value(g, units), -R.graphs.index(g)))
        return self

    def choose(self, obs: dict) -> str:
        return self.graph


class RuleRouter:
    def __init__(self, key: str = "tests_visible"):
        self.key, self.name = key, f"rule[{key}]"

    def fit(self, R: Rewards, units):
        groups = defaultdict(list)
        for u in units:
            groups[R.obs[u][self.key]].append(u)
        self.default = FixedBest().fit(R, units).graph
        self.table = {lvl: FixedBest().fit(R, us).graph for lvl, us in groups.items()}
        return self

    def choose(self, obs: dict) -> str:
        return self.table.get(obs[self.key], self.default)


class FeatureRouter:
    """Ridge regression of reward on task features, one model per graph."""
    name = "feature_router"

    def __init__(self, ridge: float = 3.0):
        self.ridge = ridge

    def fit(self, R: Rewards, units):
        X = np.array([_vec(R.obs[u]) for u in units])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        P = self.ridge * np.eye(Z.shape[1])
        P[0, 0] = 0.0                                    # the intercept is not penalised
        self.w = {g: np.linalg.solve(Z.T @ Z + P, Z.T @ np.array([R.of(u, g) for u in units])) for g in R.graphs}
        self.order = list(R.graphs)
        return self

    def choose(self, obs: dict) -> str:
        z = np.r_[1.0, (np.array(_vec(obs)) - self.mu) / self.sd]
        return max(self.order, key=lambda g: float(z @ self.w[g]))


def evaluate(policy, R: Rewards, units) -> dict:
    picks = [policy.choose(R.obs[u]) for u in units]
    share = {g: picks.count(g) / len(picks) for g in set(picks)}
    return {"value": float(np.mean([R.of(u, g) for u, g in zip(units, picks)])), "picks": share}


def oracle(R: Rewards, units) -> float:
    """Best graph per unit with hindsight.  An upper bound that also contains noise: it is not attainable."""
    return float(np.mean([max(R.of(u, g) for g in R.graphs) for u in units]))


def crossfit(R: Rewards, policies: dict, group_of=lambda u: u, splits: int = 200, train_share: float = 0.5,
             seed: int = 0) -> dict:
    """Repeated random splits by ``group_of(unit)`` (a task, so its conditions never straddle the split).
    Returns the mean held-out value of every policy and of its difference to FixedBest, with a 95% interval
    over splits (variation from which tasks train the policy and which tasks score it)."""
    rng = np.random.default_rng(seed)
    groups = sorted({group_of(u) for u in R.units})
    vals = defaultdict(list)
    for _ in range(splits):
        tr = set(rng.choice(groups, size=int(round(train_share * len(groups))), replace=False))
        train = [u for u in R.units if group_of(u) in tr]
        test = [u for u in R.units if group_of(u) not in tr]
        base = evaluate(FixedBest().fit(R, train), R, test)["value"]
        vals["fixed_best"].append(base)
        vals["oracle"].append(oracle(R, test))
        for name, make in policies.items():
            v = evaluate(make().fit(R, train), R, test)["value"]
            vals[name].append(v)
            vals[name + " - fixed_best"].append(v - base)
    return {k: {"mean": float(np.mean(v)), "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}
            for k, v in vals.items()}


# ---- fit once, freeze, score elsewhere ---------------------------------------------------------------------------

def freeze(policy) -> dict:
    """A fitted policy as plain data, to be committed before the tasks that will score it are run."""
    if isinstance(policy, FixedBest):
        return {"kind": "fixed_best", "graph": policy.graph}
    if isinstance(policy, RuleRouter):
        return {"kind": "rule", "key": policy.key, "default": policy.default,
                "table": [[lvl, g] for lvl, g in sorted(policy.table.items(), key=lambda kv: str(kv[0]))]}
    if isinstance(policy, FeatureRouter):
        return {"kind": "feature", "ridge": policy.ridge, "features": list(NUMERIC), "mu": policy.mu.tolist(),
                "sd": policy.sd.tolist(), "order": policy.order, "w": {g: w.tolist() for g, w in policy.w.items()}}
    raise TypeError(type(policy).__name__)


def thaw(d: dict):
    if d["kind"] == "fixed_best":
        p = FixedBest()
        p.graph = d["graph"]
    elif d["kind"] == "rule":
        p = RuleRouter(d["key"])
        p.default, p.table = d["default"], {lvl: g for lvl, g in d["table"]}
    elif d["kind"] == "feature":
        if d["features"] != list(NUMERIC):
            raise ValueError("the frozen feature router was fitted on a different feature list")
        p = FeatureRouter(d["ridge"])
        p.mu, p.sd, p.order = np.array(d["mu"]), np.array(d["sd"]), list(d["order"])
        p.w = {g: np.array(w) for g, w in d["w"].items()}
    else:
        raise ValueError(d["kind"])
    return p


def held_out(policies: dict, R: Rewards, group_of=lambda u: u, stratum_of=lambda g: "", reps: int = 4000,
             seed: int = 0) -> dict:
    """Score frozen policies on units they were not fitted on.

    Returns the value of every policy and every pairwise difference "a - b" (a after b in ``policies``), each with
    a 95% percentile interval from a paired bootstrap: groups (tasks) are redrawn with replacement inside each
    stratum (repository), so every replicate has the repository mix of the data and the same tasks for all policies.
    """
    rng = np.random.default_rng(seed)
    groups = sorted({group_of(u) for u in R.units})
    at = {g: i for i, g in enumerate(groups)}
    count = np.zeros(len(groups))
    total = {name: np.zeros(len(groups)) for name in policies}
    picks = {name: defaultdict(int) for name in policies}
    for u in R.units:
        i = at[group_of(u)]
        count[i] += 1
        for name, p in policies.items():
            g = p.choose(R.obs[u])
            total[name][i] += R.of(u, g)
            picks[name][g] += 1
    strata = defaultdict(list)
    for g, i in at.items():
        strata[stratum_of(g)].append(i)
    idx = np.concatenate([rng.choice(ix, size=(reps, len(ix))) for ix in map(np.array, strata.values())], axis=1)
    n = count[idx].sum(1)
    point = {name: float(t.sum() / count.sum()) for name, t in total.items()}
    draws = {name: t[idx].sum(1) / n for name, t in total.items()}
    out = {}
    for name in policies:
        lo, hi = np.percentile(draws[name], [2.5, 97.5])
        out[name] = {"mean": point[name], "lo": float(lo), "hi": float(hi),
                     "picks": {g: c / len(R.units) for g, c in picks[name].items()}}
    names = list(policies)
    for j, a in enumerate(names):
        for b in names[:j]:
            lo, hi = np.percentile(draws[a] - draws[b], [2.5, 97.5])
            out[f"{a} - {b}"] = {"mean": point[a] - point[b], "lo": float(lo), "hi": float(hi)}
    out["oracle"] = {"mean": oracle(R, R.units)}
    out["n_units"], out["n_groups"], out["n_strata"] = len(R.units), len(groups), len(strata)
    return out
