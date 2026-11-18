import numpy as np

from harness_graph.router import FeatureRouter, FixedBest, Rewards, RuleRouter, crossfit, evaluate, observe, oracle


def world(n=120, seed=0, flip=True):
    """Two graphs.  'gate' wins when tests are visible and (if flip) loses slightly when they are hidden."""
    rng = np.random.default_rng(seed)
    rows, obs = [], {}
    for i in range(n):
        for vis in (False, True):
            u = f"t{i}|{'v' if vis else 'h'}"
            obs[u] = observe({"problem_statement": "x " * int(rng.integers(20, 200)), "repo_key": f"r{i % 4}"}, vis)
            p = {"loop": 0.55 if vis else 0.45, "gate": 0.9 if vis else (0.30 if flip else 0.45)}
            for g in p:
                for s in range(2):
                    rows.append({"unit": u, "graph": g, "resolved": bool(rng.random() < p[g])})
    return Rewards(rows, obs)


def test_observe_uses_only_what_the_agent_is_given():
    o = observe({"problem_statement": "Traceback (most recent call last)\n  in foo.py", "repo_key": "r", "gold_files": 3,
                 "bug_patch": "secret"}, True, "m")
    assert o["issue_has_traceback"] and o["issue_names_file"] and o["tests_visible"]
    assert not any("gold" in k or "patch" in k for k in o)


def test_rule_router_learns_the_condition_and_beats_the_fixed_graph_when_optima_differ():
    R = world()
    half = len(R.units) // 2
    rule = RuleRouter("tests_visible").fit(R, R.units[:half])
    assert rule.table == {True: "gate", False: "loop"}
    assert evaluate(rule, R, R.units[half:])["value"] > evaluate(FixedBest().fit(R, R.units[:half]), R, R.units[half:])["value"]


def test_crossfit_reports_no_gain_when_one_graph_is_best_everywhere():
    R = world(flip=False)
    out = crossfit(R, {"rule": lambda: RuleRouter("tests_visible"), "features": FeatureRouter},
                   group_of=lambda u: u.split("|")[0], splits=60)
    assert abs(out["rule - fixed_best"]["mean"]) < 0.03
    assert out["oracle"]["mean"] > out["fixed_best"]["mean"]          # hindsight always looks better: noise


def test_crossfit_finds_the_gain_and_keeps_a_task_on_one_side_of_the_split():
    R = world()
    out = crossfit(R, {"rule": lambda: RuleRouter("tests_visible"), "features": FeatureRouter},
                   group_of=lambda u: u.split("|")[0], splits=60)
    assert out["rule - fixed_best"]["lo"] > 0 and out["features - fixed_best"]["mean"] > 0.03
    assert oracle(R, R.units) >= out["rule"]["mean"]


def test_units_missing_a_portfolio_graph_are_an_error_not_a_silent_drop():
    import pytest
    rows = [{"unit": "u1", "graph": "a", "resolved": True}, {"unit": "u1", "graph": "b", "resolved": False},
            {"unit": "u2", "graph": "a", "resolved": True}, {"unit": "u2", "graph": "c", "resolved": True}]
    with pytest.raises(ValueError, match="lack a graph"):
        Rewards(rows, {})
    R = Rewards(rows, {}, portfolio=["a"])
    assert R.units == ["u1", "u2"] and R.graphs == ["a"]
    assert Rewards(rows, {}, portfolio=["a", "b"], allow_incomplete=True).dropped == ["u2"]


def test_frozen_policies_choose_exactly_as_the_fitted_ones_after_a_json_round_trip():
    import json
    from harness_graph.router import freeze, thaw
    R = world()
    for make in (FixedBest, lambda: RuleRouter("tests_visible"), lambda: RuleRouter("repo"), FeatureRouter):
        p = make().fit(R, R.units[:120])
        q = thaw(json.loads(json.dumps(freeze(p))))
        assert [p.choose(R.obs[u]) for u in R.units] == [q.choose(R.obs[u]) for u in R.units]


def test_held_out_scores_frozen_policies_on_other_tasks_with_paired_intervals():
    from harness_graph.router import held_out
    train, test = world(seed=0), world(seed=1)
    pol = {"fixed_best": FixedBest().fit(train, train.units), "rule": RuleRouter("tests_visible").fit(train, train.units)}
    out = held_out(pol, test, group_of=lambda u: u.split("|")[0], stratum_of=lambda t: int(t[1:]) % 4, reps=500)
    assert out["n_groups"] == 120 and out["n_strata"] == 4 and out["n_units"] == 240
    assert abs(out["rule"]["mean"] - evaluate(pol["rule"], test, test.units)["value"]) < 1e-12
    assert out["rule - fixed_best"]["lo"] > 0                         # optima differ by condition in this world
    assert out["rule - fixed_best"]["lo"] <= out["rule - fixed_best"]["mean"] <= out["rule - fixed_best"]["hi"]
    same = held_out({"a": pol["rule"], "b": pol["rule"]}, test, group_of=lambda u: u.split("|")[0], reps=200)
    assert same["b - a"] == {"mean": 0.0, "lo": 0.0, "hi": 0.0}       # paired: identical policies never differ
