import numpy as np
import pytest

from harness_graph.grammar import Config, canonical, features, neighbours
from harness_graph.search import (METHODS, Pull, Replay, diagnose, linear_thompson, local_search, remedy_proposer,
                                  successive_halving, uniform)

TASKS = [f"t{i}" for i in range(80)]


def bandit(p, rng, counter=None):
    def pull(arm, task):
        if counter is not None:
            counter.append((arm, task))
        return Pull(float(rng.random() < p[arm]))
    return pull


@pytest.mark.parametrize("method", sorted(METHODS))
def test_methods_spend_exactly_the_budget_and_find_a_clear_winner(method):
    p = {f"a{i}": 0.3 for i in range(8)} | {"best": 0.9}
    wins = 0
    for s in range(20):
        rng, calls = np.random.default_rng(s), []
        kw = {"features": {a: [1.0, float(a == "best")] for a in p}} if method == "linear_thompson" else {}
        r = METHODS[method](list(p), TASKS, bandit(p, rng, calls), 400, rng, checkpoints=[100, 400], **kw)
        assert r.spent == len(calls) == 400 and sum(r.pulls.values()) == 400
        assert set(r.checkpoints) == {100, 400} and r.checkpoints[400] == r.recommendation
        wins += r.recommendation == "best"
    assert wins >= 17


def test_successive_halving_compares_survivors_on_shared_tasks():
    rng, calls = np.random.default_rng(0), []
    p = {f"a{i}": 0.5 for i in range(4)}
    r = successive_halving(list(p), TASKS, bandit(p, rng, calls), 64, rng)
    first_round = calls[: 4 * r.trace[0]["tasks_per_arm"]]
    by_arm = {a: [t for x, t in first_round if x == a] for a in p}
    assert len({tuple(v) for v in by_arm.values()}) == 1


def test_linear_thompson_shares_evidence_between_similar_arms():
    """32 arms from five binary factors, three of which add up.  With three pulls per arm on average the best
    corner (4 arms) is found more often by the feature model than by uniform allocation."""
    arms = {f"{i:05b}": [1.0] + [float(b) for b in f"{i:05b}"] for i in range(32)}
    p = {a: 0.15 + 0.22 * sum(x[1:4]) for a, x in arms.items()}
    good = {a for a in p if p[a] > 0.8}
    assert len(good) == 4
    hit = {"lin": 0, "uni": 0}
    for s in range(60):
        rng = np.random.default_rng(s)
        hit["lin"] += linear_thompson(list(arms), TASKS, bandit(p, rng), 96, rng, arms).recommendation in good
        rng = np.random.default_rng(s)
        hit["uni"] += uniform(list(arms), TASKS, bandit(p, rng), 96, rng).recommendation in good
    assert hit["lin"] > hit["uni"] and hit["lin"] >= 45


def test_diagnose_reads_the_episode_record():
    assert diagnose({"resolved": True}) is None
    assert diagnose({"resolved": False, "empty_diff": True}) == "no_edit"
    assert diagnose({"resolved": False, "exit_reason": "model_call_budget"}) == "out_of_budget"
    assert diagnose({"resolved": False, "last_check": "fail"}) == "ended_on_failed_check"
    assert diagnose({"resolved": False, "exit_reason": "end"}) == "submitted_wrong"


def test_guided_local_search_reaches_the_gate_sooner_than_random_moves():
    """Success depends on the gate only, and failures say 'submitted wrong': the remedy list tries the gate first."""
    by_name = {}

    def name_of(c):
        by_name[c.name] = canonical(c)
        return c.name
    start = name_of(Config())

    def nb(arm):
        return [name_of(n) for n in neighbours(by_name[arm])]

    def make_pull(rng):
        def pull(arm, task):
            ok = rng.random() < (0.85 if by_name[arm].gate else 0.35)
            return Pull(float(ok), {"resolved": ok, "exit_reason": "end"})
        return pull
    propose = remedy_proposer(lambda a: by_name[a], name_of)
    got = {"guided": 0, "random": 0}
    for s in range(20):
        rng = np.random.default_rng(s)
        r = local_search(start, TASKS, make_pull(rng), 96, rng, nb, propose)
        assert r.spent <= 96
        got["guided"] += by_name[r.recommendation].gate
        rng = np.random.default_rng(s)
        got["random"] += by_name[local_search(start, TASKS, make_pull(rng), 96, rng, nb).recommendation].gate
    assert got["guided"] >= 17 and got["guided"] > got["random"]


def test_replay_answers_from_stored_episodes_and_keeps_only_complete_tasks():
    rows = [{"graph": g, "task_id": t, "resolved": g == "b", "model_calls": 10}
            for g in "ab" for t in ("t1", "t2")] + [{"graph": "a", "task_id": "t3", "resolved": True, "model_calls": 5}]
    rp = Replay(rows, np.random.default_rng(0))
    assert rp.arms == ["a", "b"] and rp.tasks == ["t1", "t2"]
    assert rp.pull("b", "t1").reward == 1.0 and rp.truth() == {"a": 0.0, "b": 1.0}
    assert rp.reused == 0
    rp.pull("b", "t1")
    assert rp.reused == 1 and rp.pulls == 2                 # the only stored episode of that cell was handed out again
    assert Replay(rows, np.random.default_rng(0), cost_weight=1.0).pull("b", "t1").reward == pytest.approx(0.8)


def test_live_pull_runs_batches_concurrently_and_never_reuses_a_rollout():
    import asyncio

    from harness_graph.runner import LivePull

    class StubRunner:
        def __init__(self):
            self.seen, self.active, self.peak = [], 0, 0

        async def run(self, graph, task, seed):
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            self.seen.append((graph, task, seed))
            return {"status": "ok", "resolved": graph == "good", "model_calls": 3, "exit_reason": "end"}
    stub = StubRunner()
    pull = LivePull(stub, {"good": "good", "bad": "bad"}, log=lambda s: None)
    rng = np.random.default_rng(0)
    r = successive_halving(["good", "bad"], TASKS[:4], pull, 24, rng, parallel=8)
    assert r.recommendation == "good" and r.spent == 24
    assert stub.peak > 1 and len(set(stub.seen)) == len(stub.seen) == 24
