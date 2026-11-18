"""A grammar of harness graphs: design variables in, a checked control-flow graph out.

A ``Config`` fixes nine design variables of a single-agent harness.  ``compile_config`` turns it into a
``Graph`` by composing fragments in a fixed order

    [plan] -> [reproduce] -> work -> [compact ->] [verify] -> [check] -> END

and wiring the side loops of the work node (context compaction, the stall rule) and of the check node
(what happens after a failed check).  Every compiled graph passes the static checks of ``Graph.validate``.

Counting.  The raw product of the variables contains settings that cannot change behaviour (a retry count
without a gate, for instance).  ``canonical`` maps those to one representative by rule, and ``signature``
detects graphs that are equal as labelled control-flow graphs whatever their node names.  Neither proves
that two different graphs behave differently or the same on a task: that is an empirical question.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, fields, replace

from .graph import END, Graph

DOMAINS: dict[str, tuple] = {
    "plan": (False, True),                              # one planning call before any action
    "first": ("none", "reproduce"),                     # a capped segment that reproduces the bug first
    "checker_tool": (False, True),                      # the agent may run the checker itself
    "verify": ("none", "same", "fresh"),                # a review segment; "fresh" compacts the conversation first
    "gate": (False, True),                              # the checker is run by the harness before the episode may end
    "on_fail": ("continue", "reflect", "reset"),        # after a failed gate
    "retries": (1, 2, 3),                               # how many times a failed gate sends the work back
    "compact_at": (None, 16000, 32000),                 # prompt size (tokens) that triggers compaction
    "stall": ("none", "nudge", "reflect", "reset", "compact"),   # no edit after STALL_STEPS steps
}
STALL_STEPS = 20
REPRODUCE_STEPS = 15
COMPACT_VISITS = 7


@dataclass(frozen=True)
class Config:
    plan: bool = False
    first: str = "none"
    checker_tool: bool = False
    verify: str = "none"
    gate: bool = False
    on_fail: str = "continue"
    retries: int = 2
    compact_at: int | None = None
    stall: str = "none"

    def __post_init__(self):
        for f in fields(self):
            if getattr(self, f.name) not in DOMAINS[f.name]:
                raise ValueError(f"{f.name}={getattr(self, f.name)!r} is not in {DOMAINS[f.name]}")

    @property
    def name(self) -> str:
        c = canonical(self)
        parts = ["g", f"p{int(c.plan)}", f"f{c.first[0]}", f"t{int(c.checker_tool)}", f"v{c.verify[0]}",
                 (f"k{c.on_fail[:3]}{c.retries}" if c.gate else "k0"),
                 f"c{(c.compact_at or 0) // 1000}", f"s{c.stall[:3]}"]
        return "-".join(parts)


def canonical(c: Config) -> Config:
    """One representative per behaviour class that the rules can see: without a gate, nothing fails it."""
    return c if c.gate else replace(c, on_fail="continue", retries=2)


def all_configs(canonical_only: bool = True) -> list[Config]:
    keys = list(DOMAINS)
    out = [Config(**dict(zip(keys, v))) for v in itertools.product(*DOMAINS.values())]
    if canonical_only:
        out = sorted({canonical(c) for c in out}, key=lambda c: c.name)
    return out


def neighbours(c: Config) -> list[Config]:
    """Configs that differ from ``c`` in one design variable (after canonicalisation)."""
    out = []
    for k, dom in DOMAINS.items():
        for v in dom:
            n = canonical(replace(c, **{k: v}))
            if n != canonical(c) and n not in out:
                out.append(n)
    return out


def compile_config(c: Config, name: str | None = None) -> Graph:
    c = canonical(c)
    nodes: dict[str, dict] = {}
    edges: list[list[str]] = []
    work = "fix" if c.first == "reproduce" else "solve"
    retries = c.retries if c.gate else 0

    params: dict = {}
    if c.first == "reproduce":
        params["role"] = "fix"
    if c.checker_tool:
        params["checker"] = True
    if c.compact_at:
        params["compact_at"] = c.compact_at
    if c.stall != "none":
        params["edit_by"] = STALL_STEPS
    visits = 1 + retries + (c.stall != "none") + (COMPACT_VISITS if c.compact_at else 0)
    nodes[work] = {"type": "agent", **({"params": params} if params else {}), "max_visits": visits}

    entry = work
    if c.first == "reproduce":
        nodes["reproduce"] = {"type": "agent", "params": {"role": "reproduce", "max_steps": REPRODUCE_STEPS}}
        edges.append(["reproduce", work])
        entry = "reproduce"
    if c.plan:
        nodes["plan"] = {"type": "plan"}
        edges.append(["plan", entry])
        entry = "plan"

    # what follows a finished work segment
    tail = END
    if c.gate:
        nodes["check"] = {"type": "check", "max_visits": retries + 1}
        edges.append(["check", END, "pass"])
        if c.on_fail == "continue":
            edges.append(["check", work, "fail"])
        else:
            nodes["reflect"] = {"type": "reflect", "max_visits": retries}
            edges.append(["check", "reflect", "fail"])
            if c.on_fail == "reflect":
                edges.append(["reflect", work])
            else:
                nodes["reset"] = {"type": "reset", "max_visits": retries}
                edges += [["reflect", "reset"], ["reset", work]]
        tail = "check"
    if c.verify != "none":
        nodes["verify"] = {"type": "agent", "params": {"role": "verify"}, "max_visits": retries + 1}
        edges.append(["verify", tail])
        tail = "verify"
        if c.verify == "fresh":
            nodes["verify_compact"] = {"type": "compact", "max_visits": retries + 1}
            edges.append(["verify_compact", "verify"])
            tail = "verify_compact"
    edges.append([work, tail, "done"])

    if c.compact_at:
        nodes["compact"] = {"type": "compact", "max_visits": COMPACT_VISITS}
        edges += [[work, "compact", "context_high"], ["compact", work]]
    if c.stall == "nudge":
        edges.append([work, work, "stalled"])
    elif c.stall == "compact":
        nodes["stall_compact"] = {"type": "compact"}
        edges += [[work, "stall_compact", "stalled"], ["stall_compact", work]]
    elif c.stall in ("reflect", "reset"):
        nodes["stall_reflect"] = {"type": "reflect"}
        edges.append([work, "stall_reflect", "stalled"])
        if c.stall == "reflect":
            edges.append(["stall_reflect", work])
        else:
            nodes["stall_reset"] = {"type": "reset"}
            edges += [["stall_reflect", "stall_reset"], ["stall_reset", work]]

    return Graph.from_dict({"name": name or c.name, "description": describe(c), "entry": entry,
                            "nodes": nodes, "edges": edges})


def describe(c: Config) -> str:
    c = canonical(c)
    bits = []
    if c.plan:
        bits.append("plan first")
    if c.first == "reproduce":
        bits.append("reproduce the bug first")
    if c.checker_tool:
        bits.append("checker available to the agent")
    if c.verify != "none":
        bits.append("review segment" + (" on a compacted conversation" if c.verify == "fresh" else ""))
    if c.gate:
        after = {"continue": "continue with the feedback", "reflect": "diagnose, then continue",
                 "reset": "diagnose, restore the repository and restart"}[c.on_fail]
        bits.append(f"enforced check, on failure {after} (up to {c.retries} times)")
    if c.compact_at:
        bits.append(f"compact the conversation above {c.compact_at} prompt tokens")
    if c.stall != "none":
        bits.append(f"no edit after {STALL_STEPS} steps: {c.stall}")
    return "; ".join(bits) or "plain loop"


def signature(g: Graph, visits: bool = False) -> tuple:
    """A node-name-free form of the graph: nodes numbered breadth-first from the entry, following the outcomes of
    each node in a fixed order, with ``always`` edges expanded to one edge per outcome."""
    order, queue = {g.entry: 0}, [g.entry]
    rows = []
    while queue:
        n = queue.pop(0)
        node = g.nodes[n]
        out = []
        for o in sorted(node.outcomes):
            d = g.next_node(n, o)
            if d != END and d not in order:
                order[d] = len(order)
                queue.append(d)
            out.append((o, -1 if d == END else order[d]))
        rows.append((order[n], node.type, tuple(sorted(node.params.items())),
                     node.max_visits if visits else 0, tuple(out)))
    return tuple(sorted(rows))


def space_report() -> dict:
    """Sizes of the space at each stage of counting."""
    raw = all_configs(canonical_only=False)
    canon = all_configs()
    sigs = {}
    for c in canon:
        sigs.setdefault(signature(compile_config(c), visits=True), []).append(c.name)
    return {"raw_combinations": len(raw), "after_rule_canonicalisation": len(canon),
            "distinct_graph_signatures": len(sigs), "all_pass_static_checks": True,
            "max_nodes": max(len(compile_config(c).nodes) for c in canon)}


FACTOR_LEVELS = [(k, v) for k, dom in DOMAINS.items() for v in dom[1:]]      # first level of each variable = reference


def features(c: Config) -> list[float]:
    """Dummy coding of the design variables plus an intercept; used by the model-based search."""
    c = canonical(c)
    d = asdict(c)
    x = [1.0]
    for k, v in FACTOR_LEVELS:
        active = d[k] == v
        if k in ("on_fail", "retries") and not c.gate:
            active = False
        x.append(1.0 if active else 0.0)
    return x


# the hand-written battery as points of the space
HAND_WRITTEN: dict[str, Config] = {
    "h01_react": Config(),
    "h02_plan": Config(plan=True),
    "h03_reproduce_first": Config(first="reproduce"),
    "h04_check_retry": Config(gate=True),
    "h05_check_reflect": Config(gate=True, on_fail="reflect"),
    "h06_check_reset": Config(gate=True, on_fail="reset"),
    "h07_compact": Config(compact_at=16000),
    "h08_reproduce_check": Config(first="reproduce", gate=True),
    "h09_full": Config(plan=True, first="reproduce", gate=True, on_fail="reflect", compact_at=16000),
    "h10_stall_nudge": Config(stall="nudge"),
    "h11_stall_reflect": Config(stall="reflect"),
    "h12_stall_reset": Config(stall="reset"),
    "h13_stall_compact": Config(stall="compact"),
    "h14_verify": Config(verify="same"),
    "h15_fresh_verify": Config(verify="fresh"),
    "h16_checker_tool": Config(checker_tool=True),
    "h17_checker_tool_gate": Config(checker_tool=True, gate=True),
}
