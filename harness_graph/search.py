"""Budgeted search over harness graphs.

Every method gets the same thing: a list of candidate graphs (arms), a pool of search tasks, and a number of
episodes ``budget`` it may spend through ``pull(arm, task) -> Pull``.  It returns a ``SearchResult`` whose
``recommendation`` is one arm; the value of a search is the held-out score of that arm, never the reward it
collected on the way.  ``checkpoints`` records what the method would have recommended had it been stopped
earlier, so methods can be compared along the whole budget axis from one run.

Methods
  uniform             every arm in turn, on tasks drawn at random; the reference every other method must beat
  successive_halving  rounds on shared task batches; the worse half is dropped after each round
  thompson            Beta posterior per arm
  linear_thompson     Bayesian linear model on graph features, so evidence is shared between similar graphs
  local_search        start from one graph and move through one-variable changes; the next change is either
                      drawn at random or proposed from what went wrong in the failed episodes (``guided``)
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


@dataclass
class Pull:
    reward: float                    # objective value of one episode (success, or success minus a cost term)
    info: dict = field(default_factory=dict)      # episode summary used for diagnosis (see ``diagnose``)


@dataclass
class SearchResult:
    method: str
    recommendation: str
    pulls: dict                      # arm -> number of episodes
    means: dict                      # arm -> mean reward observed
    checkpoints: dict                # budget -> arm recommended at that point
    spent: int
    trace: list = field(default_factory=list)     # method-specific log (rounds, accepted moves, ...)


def _many(pull, pairs: list[tuple[str, str]]) -> list[Pull]:
    """Run several episodes; a ``pull`` object with a ``many`` method runs them concurrently (live search)."""
    if hasattr(pull, "many"):
        return pull.many(pairs)
    return [pull(a, t) for a, t in pairs]


class _Stats:
    def __init__(self, arms, checkpoints, rng):
        self.arms, self.rng = list(arms), rng
        self.n = {a: 0 for a in self.arms}
        self.s = {a: 0.0 for a in self.arms}
        self.spent = 0
        self.want = sorted(checkpoints or [])
        self.at: dict[int, str] = {}

    def add(self, arm, reward, recommend: Callable[[], str]):
        self.n[arm] += 1
        self.s[arm] += reward
        self.spent += 1
        while self.want and self.spent >= self.want[0]:
            self.at[self.want.pop(0)] = recommend()

    def mean(self, a):
        return self.s[a] / self.n[a] if self.n[a] else float("-inf")

    def best(self, among=None, min_pulls: int = 1) -> str:
        """Highest observed mean, ties broken at random, arms never pulled excluded."""
        pool = [a for a in (among or self.arms) if self.n[a] >= min_pulls] or list(among or self.arms)
        top = max(self.mean(a) for a in pool)
        return str(self.rng.choice([a for a in pool if self.mean(a) == top]))

    def result(self, method, rec, trace=None) -> SearchResult:
        return SearchResult(method, rec, dict(self.n), {a: self.mean(a) for a in self.arms if self.n[a]},
                            dict(self.at), self.spent, trace or [])


def uniform(arms: Sequence[str], tasks: Sequence[str], pull, budget: int, rng, checkpoints=(),
            parallel: int = 1) -> SearchResult:
    st = _Stats(arms, checkpoints, rng)
    order: list[str] = []
    while st.spent < budget:
        pairs = []
        for _ in range(min(parallel, budget - st.spent)):
            if not order:
                order = list(rng.permutation(st.arms))
            pairs.append((order.pop(), str(rng.choice(tasks))))
        for (a, _), p in zip(pairs, _many(pull, pairs)):
            st.add(a, p.reward, st.best)
    return st.result("uniform", st.best())


def successive_halving(arms, tasks, pull, budget: int, rng, checkpoints=(), eta: int = 2,
                       parallel: int = 1) -> SearchResult:
    """Each round spends an equal share of the budget; survivors of a round are compared on the same tasks."""
    st = _Stats(arms, checkpoints, rng)
    alive = list(rng.permutation(st.arms))
    rounds = max(1, math.ceil(math.log(len(alive), eta)))
    trace = []
    for r in range(rounds):
        per_arm = (budget - st.spent) // ((rounds - r) * len(alive))
        if per_arm < 1:
            break
        batch = [str(t) for t in rng.choice(tasks, size=per_arm, replace=per_arm > len(tasks))]
        pairs = [(a, t) for t in batch for a in alive]
        for (a, _), p in zip(pairs, _many(pull, pairs)):
            st.add(a, p.reward, lambda: st.best(alive))
        keep = max(1, math.ceil(len(alive) / eta))
        alive = sorted(alive, key=lambda a: (st.mean(a), rng.random()), reverse=True)[:keep]
        trace.append({"round": r, "tasks_per_arm": per_arm, "kept": list(alive)})
        if len(alive) == 1:
            break
    while st.spent < budget and len(alive) > 1:          # left-over episodes go to the finalists
        t = str(rng.choice(tasks))
        pairs = [(a, t) for a in alive][: budget - st.spent]
        for (a, _), p in zip(pairs, _many(pull, pairs)):
            st.add(a, p.reward, lambda: st.best(alive))
    return st.result("successive_halving", st.best(alive), trace)


def thompson(arms, tasks, pull, budget: int, rng, checkpoints=(), parallel: int = 1) -> SearchResult:
    """With ``parallel`` > 1 the posterior is sampled that many times before it is updated."""
    st = _Stats(arms, checkpoints, rng)

    def recommend():                                     # posterior mean under Beta(1, 1): shrinks rarely pulled arms
        score = {a: (st.s[a] + 1) / (st.n[a] + 2) for a in st.arms}
        top = max(score.values())
        return str(rng.choice([a for a in st.arms if score[a] == top]))
    while st.spent < budget:
        pairs = []
        for _ in range(min(parallel, budget - st.spent)):
            draw = {a: rng.beta(1 + max(st.s[a], 0), 1 + max(st.n[a] - st.s[a], 0)) for a in st.arms}
            pairs.append((max(draw, key=draw.get), str(rng.choice(tasks))))
        for (a, _), p in zip(pairs, _many(pull, pairs)):
            st.add(a, min(max(p.reward, 0.0), 1.0), recommend)
    return st.result("thompson", recommend())


def linear_thompson(arms, tasks, pull, budget: int, rng, features: dict, checkpoints=(), ridge: float = 1.0,
                    noise: float = 0.25, prior_weight: float = 4.0, parallel: int = 1) -> SearchResult:
    """Thompson sampling on reward ~ features . theta.  The recommendation mixes each arm's own mean with the
    model's prediction (worth ``prior_weight`` episodes), so a wrong linear model cannot override plenty of data."""
    st = _Stats(arms, checkpoints, rng)
    X = np.array([features[a] for a in st.arms], dtype=float)
    d = X.shape[1]
    A, b = ridge * np.eye(d), np.zeros(d)
    idx = {a: i for i, a in enumerate(st.arms)}

    def recommend():
        pred = X @ np.linalg.solve(A, b)
        score = [(st.s[a] + prior_weight * pred[idx[a]]) / (st.n[a] + prior_weight) for a in st.arms]
        return st.arms[int(np.argmax(score))]
    while st.spent < budget:
        cov = np.linalg.inv(A)
        pairs = []
        for _ in range(min(parallel, budget - st.spent)):
            scores = X @ rng.multivariate_normal(cov @ b, noise * cov)
            pairs.append((st.arms[int(rng.choice(np.flatnonzero(scores == scores.max())))], str(rng.choice(tasks))))
        for (a, _), p in zip(pairs, _many(pull, pairs)):
            A += np.outer(X[idx[a]], X[idx[a]])
            b += p.reward * X[idx[a]]
            st.add(a, p.reward, recommend)
    return st.result("linear_thompson", recommend())


# ---- search guided by what went wrong ---------------------------------------------------------------------------

SIGNALS = ("submitted_wrong", "ended_on_failed_check", "no_edit", "out_of_budget", "context")


def diagnose(info: dict) -> str | None:
    """One coarse reason for a failed episode, from fields every episode record has.  None for a success."""
    if info.get("resolved"):
        return None
    if info.get("exit_reason") == "context_exceeded" or info.get("max_prompt_tokens", 0) > 48000:
        return "context"
    if info.get("empty_diff"):
        return "no_edit"
    if info.get("exit_reason") in ("model_call_budget", "wall_time", "usd_budget"):
        return "out_of_budget"
    if info.get("last_check") == "fail":
        return "ended_on_failed_check"
    return "submitted_wrong"


# design-variable changes worth trying for each reason, most direct first
REMEDIES: dict[str, list[dict]] = {
    "submitted_wrong": [{"gate": True}, {"checker_tool": True}, {"verify": "same"}, {"first": "reproduce"}],
    "ended_on_failed_check": [{"gate": True}, {"retries": 3}, {"on_fail": "reflect"}, {"on_fail": "reset"}],
    "no_edit": [{"stall": "nudge"}, {"stall": "reflect"}, {"plan": True}, {"stall": "reset"}],
    "out_of_budget": [{"compact_at": 16000}, {"verify": "none"}, {"plan": False}, {"first": "none"}],
    "context": [{"compact_at": 16000}, {"compact_at": 32000}, {"stall": "compact"}],
}


def local_search(start: str, tasks, pull, budget: int, rng, neighbours: Callable[[str], list[str]],
                 propose: Callable[[str, Counter], list[str]] | None = None, batch: int = 16,
                 checkpoints=()) -> SearchResult:
    """Hill climbing with paired comparisons.  The current graph and a candidate are run on the same ``batch`` of
    tasks; the candidate replaces the current graph if its mean is higher.  ``propose(arm, reasons)`` orders the
    candidates from the failure reasons of the current graph; without it the next candidate is a random neighbour.
    Both variants spend the same budget, so their difference is the value of the diagnosis."""
    arms_seen = [start]
    st = _Stats([start], checkpoints, rng)
    cur, tried, trace = start, {start}, []
    by_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    reasons: dict[str, Counter] = defaultdict(Counter)

    def record(arm, t, p):
        if arm not in st.n:
            st.arms.append(arm); st.n[arm] = 0; st.s[arm] = 0.0; arms_seen.append(arm)
        by_task[arm][t].append(p.reward)
        why = diagnose(p.info) if p.info else None
        if why:
            reasons[arm][why] += 1
        st.add(arm, p.reward, lambda: cur)

    while st.spent + 2 <= budget:
        options = [n for n in neighbours(cur) if n not in tried]
        if not options:
            break
        if propose is not None:
            ranked = [n for n in propose(cur, reasons[cur]) if n in options]
            cand = ranked[0] if ranked else str(rng.choice(options))
        else:
            cand = str(rng.choice(options))
        tried.add(cand)
        n = min(batch, (budget - st.spent) // 2)
        ts = [str(t) for t in rng.choice(tasks, size=n, replace=n > len(tasks))]
        pairs = [(arm, t) for t in ts for arm in (cur, cand)]
        got = {cur: [], cand: []}
        for (arm, t), p in zip(pairs, _many(pull, pairs)):
            record(arm, t, p)
            got[arm].append(p.reward)
        diff = float(np.mean(got[cand]) - np.mean(got[cur]))        # this batch only: same tasks, fresh episodes
        trace.append({"from": cur, "candidate": cand, "paired_diff": diff, "accepted": diff > 0,
                      "reasons": dict(reasons[cur])})
        if diff > 0:
            cur, tried = cand, tried | {cur}
    return st.result("local_search_guided" if propose else "local_search_random", cur, trace)


def remedy_proposer(config_of: Callable[[str], object], name_of: Callable[[object], str]):
    """``propose`` for ``local_search`` over grammar configs: apply the remedies of the commonest failure reasons."""
    from dataclasses import replace

    from .grammar import canonical

    def propose(arm: str, why: Counter) -> list[str]:
        c, out = config_of(arm), []
        for reason, _ in why.most_common():
            for change in REMEDIES.get(reason, []):
                try:
                    n = name_of(canonical(replace(c, **change)))
                except ValueError:
                    continue
                if n != arm and n not in out:
                    out.append(n)
        return out
    return propose


METHODS = {"uniform": uniform, "successive_halving": successive_halving, "thompson": thompson,
           "linear_thompson": linear_thompson}


# ---- offline replay ---------------------------------------------------------------------------------------------

class Replay:
    """Answers pulls from episodes that were already run: a stored episode of (graph, task) drawn at random.
    Lets search code be tested, and methods be compared, without a model call."""

    def __init__(self, rows: list[dict], rng, cost_weight: float = 0.0, cost_key: str = "model_calls",
                 cost_scale: float = 50.0):
        self.rng = rng
        self.table: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in rows:
            self.table[(r["graph"], r["task_id"])].append(r)
        self.arms = sorted({g for g, _ in self.table})
        tasks = defaultdict(set)
        for g, t in self.table:
            tasks[t].add(g)
        self.tasks = sorted(t for t, gs in tasks.items() if len(gs) == len(self.arms))    # tasks every arm has
        self.w, self.key, self.scale = cost_weight, cost_key, cost_scale
        self.used: dict = {}
        self.order: dict = {}
        self.pulls = self.reused = 0

    def reward(self, r: dict) -> float:
        return float(bool(r["resolved"])) - self.w * float(r.get(self.key, 0)) / self.scale

    def pull(self, arm: str, task: str) -> Pull:
        """A stored episode of (arm, task) not handed out before; once all are used they are reused in order and
        the pull is counted in ``reused`` (a budget beyond the table is a budget of table draws, not of episodes)."""
        rows = self.table[(arm, task)]
        k = self.used.get((arm, task), 0)
        self.used[(arm, task)] = k + 1
        self.pulls += 1
        if k >= len(rows):
            self.reused += 1
        if k == 0:
            self.order[(arm, task)] = list(self.rng.permutation(len(rows)))
        r = rows[self.order[(arm, task)][k % len(rows)]]
        return Pull(self.reward(r), r)

    def truth(self, tasks=None) -> dict[str, float]:
        """Mean reward of every arm over all stored episodes of ``tasks`` (seeds averaged within a task)."""
        tasks = tasks or self.tasks
        return {a: float(np.mean([np.mean([self.reward(r) for r in self.table[(a, t)]]) for t in tasks]))
                for a in self.arms}
