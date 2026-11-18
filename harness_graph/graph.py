"""Agent harnesses as typed control-flow graphs.

A harness is the program around the model: when it plans, when its work is checked, what happens after a
failed check, how the context is kept small.  Here that program is data.  Nodes are harness components, and
each node finishes with an *outcome* (``done``, ``fail``, ``context_high``, ...) that selects the next edge.

Every node type declares its possible outcomes and the state it reads and writes, so a graph is checked
before any model call:

* each outcome of each node has exactly one edge (a named one, or the catch-all ``always``);
* every key a node reads is written on every path that reaches it;
* every node is reachable, and every loop contains a node with a stated visit cap, so runs terminate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

END = "END"
ALWAYS = "always"


@dataclass(frozen=True)
class NodeType:
    name: str
    outcomes: tuple[str, ...]
    reads: frozenset[str] = frozenset()
    writes: frozenset[str] = frozenset()
    params: frozenset[str] = frozenset()
    doc: str = ""


def _nt(name, outcomes, reads=(), writes=(), params=(), doc=""):
    return NodeType(name, tuple(outcomes), frozenset(reads), frozenset(writes), frozenset(params), doc)


NODE_TYPES: dict[str, NodeType] = {t.name: t for t in [
    _nt("agent", ["done"], writes=["work"], params=["role", "max_steps", "compact_at", "edit_by", "checker"],
        doc="Model-in-the-loop segment: one shell command per step until the model says it is done."),
    _nt("plan", ["next"], writes=["plan"], doc="One model call that writes a plan before any action."),
    _nt("check", ["pass", "fail"], reads=["work"], writes=["feedback"],
        doc="Deterministic gate on the working tree: non-empty diff, files compile, reproduction script, "
            "no new failures in the visible tests."),
    _nt("reflect", ["next"], reads=["feedback"], writes=["lesson"],
        doc="One model call that diagnoses a failed check."),
    _nt("reset", ["next"], reads=["feedback"],
        doc="Restore the repository and start a fresh conversation that carries only the lesson."),
    _nt("compact", ["next"], reads=["work"],
        doc="Replace the conversation with a model-written summary and the current diff."),
]}

AGENT_ROLES = ("solve", "reproduce", "fix", "verify")


class GraphError(ValueError):
    """Raised when a graph spec is not well formed."""


@dataclass
class Node:
    name: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)
    max_visits: int = 1

    @property
    def spec(self) -> NodeType:
        return NODE_TYPES[self.type]

    @property
    def outcomes(self) -> tuple[str, ...]:
        out = list(self.spec.outcomes)
        if self.type == "agent":
            if "max_steps" in self.params:
                out.append("step_cap")           # the segment's own step cap was reached
            if "compact_at" in self.params:
                out.append("context_high")       # the prompt grew past the threshold
            if "edit_by" in self.params:
                out.append("stalled")            # this many steps and still no change to the repository
        return tuple(out)

    @property
    def writes(self) -> frozenset[str]:
        if self.type == "agent" and "edit_by" in self.params:
            return self.spec.writes | {"feedback"}   # a stall is reported the same way a failed check is
        return self.spec.writes


@dataclass
class Graph:
    name: str
    entry: str
    nodes: dict[str, Node]
    edges: dict[str, dict[str, str]]     # src -> {outcome | "always": dst}
    description: str = ""

    # ---- construction -------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "Graph":
        for key in ("name", "entry", "nodes", "edges"):
            if key not in d:
                raise GraphError(f"missing top-level key '{key}'")
        nodes = {}
        for name, nd in d["nodes"].items():
            nd = dict(nd or {})
            if name == END:
                raise GraphError("END is reserved")
            if nd.get("type") not in NODE_TYPES:
                raise GraphError(f"node '{name}': unknown type {nd.get('type')!r}")
            node = Node(name, nd["type"], dict(nd.get("params") or {}), int(nd.get("max_visits", 1)))
            unknown = set(node.params) - node.spec.params
            if unknown:
                raise GraphError(f"node '{name}': unknown params {sorted(unknown)}")
            if node.params.get("role", "solve") not in AGENT_ROLES:
                raise GraphError(f"node '{name}': unknown role {node.params['role']!r}")
            if node.max_visits < 1:
                raise GraphError(f"node '{name}': max_visits must be >= 1")
            nodes[name] = node
        edges: dict[str, dict[str, str]] = {n: {} for n in nodes}
        for e in d["edges"]:
            src, dst, cond = (list(e) + [ALWAYS])[:3]
            if src not in nodes:
                raise GraphError(f"edge from unknown node '{src}'")
            if dst != END and dst not in nodes:
                raise GraphError(f"edge to unknown node '{dst}'")
            if cond != ALWAYS and cond not in nodes[src].outcomes:
                raise GraphError(f"edge {src}->{dst}: '{cond}' is not an outcome of {nodes[src].type} "
                                 f"(has {list(nodes[src].outcomes)})")
            if cond in edges[src]:
                raise GraphError(f"node '{src}': duplicate '{cond}' edge")
            edges[src][cond] = dst
        g = cls(d["name"], d["entry"], nodes, edges, d.get("description", ""))
        g.validate()
        return g

    @classmethod
    def load(cls, path: str | Path) -> "Graph":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def to_dict(self) -> dict:
        nodes = {}
        for n in self.nodes.values():
            nd: dict[str, Any] = {"type": n.type}
            if n.params:
                nd["params"] = dict(n.params)
            if n.max_visits != 1:
                nd["max_visits"] = n.max_visits
            nodes[n.name] = nd
        edges = [[s, d] if c == ALWAYS else [s, d, c] for s, out in self.edges.items() for c, d in out.items()]
        return {"name": self.name, "description": self.description, "entry": self.entry, "nodes": nodes, "edges": edges}

    # ---- static checks ------------------------------------------------
    def validate(self) -> None:
        if self.entry not in self.nodes:
            raise GraphError(f"entry '{self.entry}' is not a node")
        for name, node in self.nodes.items():
            out = self.edges[name]
            missing = [o for o in node.outcomes if o not in out and ALWAYS not in out]
            if missing:
                raise GraphError(f"node '{name}' ({node.type}): no edge for outcome(s) {missing}")
            if ALWAYS in out and all(o in out for o in node.outcomes):
                raise GraphError(f"node '{name}': the 'always' edge can never be taken")
        seen, stack = set(), [self.entry]
        while stack:
            n = stack.pop()
            if n in seen or n == END:
                continue
            seen.add(n)
            stack.extend(self.edges[n].values())
        if seen != set(self.nodes):
            raise GraphError(f"unreachable nodes: {sorted(set(self.nodes) - seen)}")
        for cyc in self._cycles():
            if all(self.nodes[n].max_visits == 1 for n in cyc):
                raise GraphError(f"cycle {cyc} never repeats: give one of its nodes max_visits > 1")
        avail = self._must_available()
        for name, node in self.nodes.items():
            missing = node.spec.reads - avail[name]
            if missing:
                raise GraphError(f"node '{name}' ({node.type}) reads {sorted(missing)} "
                                 f"but not every path writes it first")
        if not any(n.type == "agent" for n in self.nodes.values()):
            raise GraphError("graph has no agent node, so nothing can change the repository")

    def _preds(self) -> dict[str, list[str]]:
        preds: dict[str, list[str]] = {n: [] for n in self.nodes}
        for s, out in self.edges.items():
            for d in out.values():
                if d != END:
                    preds[d].append(s)
        return preds

    def _must_available(self) -> dict[str, frozenset[str]]:
        universe = frozenset(k for t in NODE_TYPES.values() for k in t.writes)
        avail = {n: universe for n in self.nodes}
        preds = self._preds()
        changed = True
        while changed:
            changed = False
            for n in self.nodes:
                ins = [avail[p] | self.nodes[p].writes for p in preds[n]]
                new = frozenset() if n == self.entry or not ins else frozenset.intersection(*ins)
                if new != avail[n]:
                    avail[n], changed = new, True
        return avail

    def _cycles(self) -> list[list[str]]:
        out, path, on_path, done = [], [], set(), set()

        def dfs(n):
            if n == END or n in done:
                return
            if n in on_path:
                out.append(path[path.index(n):])
                return
            on_path.add(n); path.append(n)
            for d in self.edges[n].values():
                dfs(d)
            path.pop(); on_path.discard(n); done.add(n)

        dfs(self.entry)
        return out

    # ---- derived descriptors -----------------------------------------
    def next_node(self, name: str, outcome: str) -> str:
        out = self.edges[name]
        return out[outcome] if outcome in out else out[ALWAYS]

    def features(self) -> dict[str, int]:
        """Attribute vector of the harness (one row per graph in the analysis)."""
        by = lambda t: [n for n in self.nodes.values() if n.type == t]
        agents = by("agent")
        return {
            "plan": int(bool(by("plan"))),
            "reproduce_first": int(any(n.params.get("role") == "reproduce" for n in agents)),
            "verify": int(any(n.params.get("role") == "verify" for n in agents)),
            "check": int(bool(by("check"))),
            "retries": max((n.max_visits - 1 for n in by("check")), default=0),
            "reflect": int(bool(by("reflect"))),
            "reset": int(bool(by("reset"))),
            "compact": int(bool(by("compact"))),
            "stall_signal": int(any("edit_by" in n.params for n in agents)),
            "checker_tool": int(any(n.params.get("checker") for n in agents)),
            "n_nodes": len(self.nodes),
        }


def load_battery(directory: str | Path) -> list[Graph]:
    graphs = [Graph.load(p) for p in sorted(Path(directory).glob("*.yaml"))]
    names = [g.name for g in graphs]
    if len(set(names)) != len(names):
        raise GraphError("duplicate graph names in battery")
    return graphs
