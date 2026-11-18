from pathlib import Path

import pytest

from harness_graph import Graph
from harness_graph.grammar import (DOMAINS, HAND_WRITTEN, Config, all_configs, canonical, compile_config, features,
                                   neighbours, signature, space_report)

ROOT = Path(__file__).resolve().parents[1]


def test_every_config_compiles_and_passes_the_static_checks():
    configs = all_configs()
    for c in configs:
        g = compile_config(c)                      # Graph.from_dict validates
        assert g.name == c.name
    assert len({c.name for c in configs}) == len(configs)


def test_space_counts():
    rep = space_report()
    raw = 1
    for dom in DOMAINS.values():
        raw *= len(dom)
    assert rep["raw_combinations"] == raw
    assert rep["after_rule_canonicalisation"] < raw
    assert rep["distinct_graph_signatures"] == rep["after_rule_canonicalisation"]      # no two canonical configs give one graph


@pytest.mark.parametrize("name", sorted(HAND_WRITTEN))
def test_hand_written_graph_is_a_point_of_the_space(name):
    hand = Graph.load(ROOT / "graphs" / f"{name}.yaml")
    assert signature(compile_config(HAND_WRITTEN[name])) == signature(hand)


def test_canonical_removes_settings_without_effect():
    a, b = Config(gate=False, retries=3, on_fail="reset"), Config(gate=False, retries=1)
    assert canonical(a) == canonical(b) and a.name == b.name
    assert Config(gate=True, retries=3).name != Config(gate=True, retries=1).name


def test_neighbours_differ_in_one_variable():
    c = Config()
    ns = neighbours(c)
    assert Config(gate=True) in ns and Config(plan=True) in ns and c not in ns
    for n in ns:
        changed = [k for k in DOMAINS if getattr(n, k) != getattr(c, k)]
        assert len(changed) == 1 or set(changed) <= {"gate", "retries", "on_fail"}


def test_features_have_one_length_and_ignore_inactive_settings():
    xs = {tuple(features(c)) for c in all_configs()}
    assert len({len(x) for x in xs}) == 1
    assert features(Config(gate=False, retries=3)) == features(Config(gate=False, retries=1))


def test_invalid_value_rejected():
    with pytest.raises(ValueError):
        Config(on_fail="pray")
