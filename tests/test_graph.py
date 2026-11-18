from pathlib import Path

import pytest

from harness_graph import END, Graph, GraphError, load_battery

ROOT = Path(__file__).resolve().parents[1]


def spec(**over):
    base = {"name": "g", "entry": "solve",
            "nodes": {"solve": {"type": "agent", "max_visits": 3}, "check": {"type": "check", "max_visits": 3}},
            "edges": [["solve", "check"], ["check", END, "pass"], ["check", "solve", "fail"]]}
    base.update(over)
    return base


def test_valid_check_retry_graph_loads():
    g = Graph.from_dict(spec())
    assert g.next_node("check", "pass") == END and g.next_node("check", "fail") == "solve"
    assert g.next_node("solve", "done") == "check"


def test_unknown_node_type_rejected():
    with pytest.raises(GraphError, match="unknown type"):
        Graph.from_dict(spec(nodes={"solve": {"type": "summon"}}, edges=[["solve", END]]))


def test_unknown_param_and_role_rejected():
    with pytest.raises(GraphError, match="unknown params"):
        Graph.from_dict(spec(nodes={"solve": {"type": "agent", "params": {"top_k": 3}}}, edges=[["solve", END]]))
    with pytest.raises(GraphError, match="unknown role"):
        Graph.from_dict(spec(nodes={"solve": {"type": "agent", "params": {"role": "pray"}}}, edges=[["solve", END]]))


def test_every_outcome_needs_an_edge():
    s = spec()
    s["edges"] = [["solve", "check"], ["check", END, "pass"]]
    with pytest.raises(GraphError, match=r"no edge for outcome\(s\) \['fail'\]"):
        Graph.from_dict(s)


def test_edge_must_name_a_real_outcome():
    with pytest.raises(GraphError, match="is not an outcome of agent"):
        Graph.from_dict(spec(nodes={"solve": {"type": "agent"}}, edges=[["solve", END, "pass"]]))


def test_params_add_outcomes():
    # compact_at makes `context_high` possible, so it needs an edge
    nodes = {"solve": {"type": "agent", "params": {"compact_at": 1000}, "max_visits": 3},
             "compact": {"type": "compact", "max_visits": 2}}
    with pytest.raises(GraphError, match="context_high"):
        Graph.from_dict(spec(nodes=nodes, edges=[["solve", END, "done"], ["compact", "solve"]]))
    Graph.from_dict(spec(nodes=nodes, edges=[["solve", END, "done"], ["solve", "compact", "context_high"],
                                             ["compact", "solve"]]))


def test_read_before_write_rejected():
    # reflect needs feedback, which only a check writes
    with pytest.raises(GraphError, match=r"reads \['feedback'\]"):
        Graph.from_dict({"name": "g", "entry": "solve",
                         "nodes": {"solve": {"type": "agent"}, "reflect": {"type": "reflect"}},
                         "edges": [["solve", "reflect"], ["reflect", END]]})


def test_unreachable_node_rejected():
    s = spec()
    s["nodes"]["plan"] = {"type": "plan"}
    s["edges"].append(["plan", "solve"])
    with pytest.raises(GraphError, match="unreachable"):
        Graph.from_dict(s)


def test_loop_without_visit_cap_rejected():
    s = spec(nodes={"solve": {"type": "agent"}, "check": {"type": "check"}})
    with pytest.raises(GraphError, match="never repeats"):
        Graph.from_dict(s)


def test_roundtrip_and_features():
    g = Graph.from_dict(spec())
    assert Graph.from_dict(g.to_dict()).to_dict() == g.to_dict()
    f = g.features()
    assert f["check"] == 1 and f["retries"] == 2 and f["plan"] == 0 and f["compact"] == 0


def test_battery_graphs_are_valid_and_distinct():
    graphs = load_battery(ROOT / "graphs")
    assert len(graphs) == 17 and graphs[0].name == "h01_react"
    assert len({tuple(sorted(g.features().items())) for g in graphs}) == 17


def test_stall_signal_adds_an_outcome_and_supplies_feedback():
    nodes = {"solve": {"type": "agent", "params": {"edit_by": 5}, "max_visits": 2}, "reflect": {"type": "reflect"}}
    with pytest.raises(GraphError, match="stalled"):
        Graph.from_dict({"name": "g", "entry": "solve", "nodes": nodes, "edges": [["solve", END, "done"]]})
    g = Graph.from_dict({"name": "g", "entry": "solve", "nodes": nodes,
                         "edges": [["solve", END, "done"], ["solve", "reflect", "stalled"], ["reflect", "solve"]]})
    assert g.features()["stall_signal"] == 1
