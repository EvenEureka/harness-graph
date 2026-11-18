"""Build docs/index.html from site/index.template.html, so that no number on the page is typed by hand.

  python scripts/build_site.py

Numbers come from results/summary/*.json.  The cost of check-and-retry over three runs is computed from the raw
episodes when they are present (results/raw is not in the repository) and otherwise taken from the last build
(results/summary/site_numbers.json, which this script writes).
"""
import datetime
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "results" / "summary"
MODELS = {"gpt-6.1-terra": "t", "gpt-6-luna": "l"}
H01, H04 = "h01_react", "h04_check_retry"
VARIANTS = [("h04_check_retry", "Check, retry (graph B)", True), ("h05_check_reflect", "Check, find cause, retry", True),
            ("h06_check_reset", "Check, find cause, start over", True), ("h14_verify", "Self-review", False),
            ("h15_fresh_verify", "Self-review after summarizing", False), ("h02_plan", "Plan first", False),
            ("h03_reproduce_first", "Reproduce the bug first", False), ("h07_compact", "Summarize long conversations", False)]

load = lambda name: json.loads((S / name).read_text())
minus = lambda s: s.replace("-", "−")
pct = lambda x: f"{100 * x:.0f}%"
pct1 = lambda x: f"{100 * x:.1f}%"
pts = lambda x: "0.0" if abs(x) < 5e-4 else minus(f"{100 * x:+.1f}")
num = lambda x: minus(f"{100 * x:.1f}")
more = lambda r: f"{100 * (r - 1):.0f}%"


def costs() -> dict:
    """h04 / h01 in model calls and tokens, tests visible, confirmation tasks, all runs."""
    raw = ROOT / "results" / "raw" / "confirm_visible"
    cache = S / "site_numbers.json"
    if not all((raw / f"{m}.jsonl").exists() for m in MODELS):
        return json.loads(cache.read_text())["cost"]
    from harness_graph import analysis as A
    tasks = {json.loads(l)["instance_id"] for l in open(ROOT / "data" / "confirm_tasks.jsonl") if l.strip()}
    out = {}
    for m in MODELS:
        df = A.load_episodes(raw / f"{m}.jsonl")
        df = df[(df.tests == "visible") & df.task_id.isin(tasks) & df.graph.isin([H01, H04])]
        out[m] = {col: A.cost_ratio(df, H04, col) for col in ("model_calls", "tokens")}
        out[m]["runs_per_task"] = int(df.seed.nunique())
    return out


def fig_main(rows) -> str:
    """Rows: (group, model, plain, check, diff text, n).  Two dots per row on a 0-100% axis."""
    x = lambda v: 250 + 360 * v
    out = ['<svg viewBox="0 0 880 266" role="img" aria-label="Repair rate of the plain loop and of check-and-retry, '
           'for two models, with tests visible and with tests hidden.">']
    for v in (0, .25, .5, .75, 1):
        out.append(f'<line class="gridl" x1="{x(v):.0f}" y1="40" x2="{x(v):.0f}" y2="226"/>'
                   f'<text class="ax" x="{x(v):.0f}" y="246" text-anchor="middle">{pct(v)}</text>')
    out.append('<text class="ax" x="650" y="28">difference [95% interval]</text>')
    out.append('<text class="ax" x="430" y="262" text-anchor="middle">share of bugs repaired</text>')
    ys = iter((62, 96, 172, 206))
    heads = iter((28, 138))
    last = None
    for group, model, a, b, diff, n in rows:
        if group != last:
            out.append(f'<text class="cap" x="20" y="{next(heads)}">{group}</text>')
            last = group
        y = next(ys)
        lo, hi = sorted((a, b))
        out.append(f'<text class="lab" x="20" y="{y + 5}">{model}</text>'
                   f'<line class="conn" x1="{x(lo):.1f}" y1="{y}" x2="{x(hi):.1f}" y2="{y}"/>')
        for v, cls, name in ((a, "d-plain", "plain loop"), (b, "d-check", "check and retry")):
            out.append(f'<circle class="{cls}" cx="{x(v):.1f}" cy="{y}" r="7"><title>{model}, {group.lower()}: {name} '
                       f'{pct1(v)} of {n} tasks</title></circle>')
        out.append(f'<text class="val" x="{x(lo) - 12:.1f}" y="{y + 5}" text-anchor="end">{pct(lo)}</text>'
                   f'<text class="val" x="{x(hi) + 12:.1f}" y="{y + 5}">{pct(hi)}</text>'
                   f'<text class="val" x="650" y="{y + 5}">{diff}</text>')
    return "".join(out) + "</svg>"


def fig_variants(cond) -> str:
    """Visible tests: each hand-designed graph minus the plain loop, one panel per model."""
    panels = {m: x0 for m, x0 in zip(MODELS, (270, 590))}
    width, lo_d, hi_d = 270, -0.10, 0.35
    out = ['<svg viewBox="0 0 880 326" role="img" aria-label="Difference in repair rate to the plain loop for eight '
           'hand-designed graphs, tests visible, two models.">']
    for m, x0 in panels.items():
        x = lambda v, x0=x0: x0 + width * (v - lo_d) / (hi_d - lo_d)
        out.append(f'<text class="lab b" x="{x0}" y="28">{m}</text>')
        for v in (-.1, 0, .1, .2, .3):
            out.append(f'<line class="{"zero" if v == 0 else "gridl"}" x1="{x(v):.1f}" y1="42" x2="{x(v):.1f}" y2="290"/>'
                       f'<text class="ax" x="{x(v):.1f}" y="308" text-anchor="middle">{minus(f"{100 * v:+.0f}") if v else "0"}</text>')
        rows = {r["graph"]: r for r in cond[m]}
        for i, (g, label, gated) in enumerate(VARIANTS):
            r, y = rows[g], 62 + 30 * i
            if m == next(iter(panels)):
                out.append(f'<text class="lab" x="20" y="{y + 5}">{label}</text>')
            k = "check" if gated else "other"
            out.append(f'<line class="w-{k}" x1="{x(r["visible_dlo"]):.1f}" y1="{y}" x2="{x(r["visible_dhi"]):.1f}" y2="{y}"/>'
                       f'<circle class="d-{k}" cx="{x(r["visible_delta"]):.1f}" cy="{y}" r="6"><title>{m}: {label} minus plain loop '
                       f'{pts(r["visible_delta"])} points [{num(r["visible_dlo"])}, {num(r["visible_dhi"])}], {r["n_tasks"]} tasks</title></circle>')
            if gated:
                out.append(f'<text class="val" x="{x(r["visible_dhi"]) + 8:.1f}" y="{y + 5}">{pts(r["visible_delta"])}</text>')
    out.append('<text class="ax" x="565" y="324" text-anchor="middle">percentage points relative to the plain loop</text>')
    return "".join(out) + "</svg>"


def main():
    N = {"updated": datetime.date.today().strftime("%-d %B %Y")}
    three, main_, enf, space = load("h04_h01_three_seeds.json"), load("main.json"), load("enforcement.json"), load("space.json")
    cost = costs()
    rows_vis, rows_hid, hid = [], [], []
    for m, k in MODELS.items():
        c = next(r for r in three["contrasts"] if r["model"] == m and r["a"] == H04 and r["b"] == H01)
        assert c["tests"] == "visible"
        N.update({f"{k}_h01": pct(c["rate_b"]), f"{k}_h04": pct(c["rate_a"]), f"{k}_delta": pts(c["delta"]),
                  f"{k}_lo": num(c["lo"]), f"{k}_hi": num(c["hi"]), f"{k}_wins": str(c["wins"]), f"{k}_losses": str(c["losses"]),
                  f"{k}_calls": more(cost[m]["model_calls"]["ratio"]), f"{k}_tokens": more(cost[m]["tokens"]["ratio"])})
        N["n_confirm"] = str(c["n_tasks"])
        rows_vis.append(("Tests visible to the agent", m, c["rate_b"], c["rate_a"],
                         f'{pts(c["delta"])} [{num(c["lo"])}, {num(c["hi"])}]', c["n_tasks"]))
        g = {r["graph"]: r for r in main_["models"][m]["confirm"]["graphs"]}
        N[f"hid_n_{k}"] = str(g[H04]["n_tasks"])
        rows_hid.append(("Tests hidden", m, g[H01]["resolved"], g[H04]["resolved"],
                         f'{pts(g[H04]["delta"])} [{num(g[H04]["dlo"])}, {num(g[H04]["dhi"])}]', g[H04]["n_tasks"]))
        others = [r for name, r in g.items() if name != H01]
        assert len(others) == 8 and all(r["dlo"] <= 0 <= r["dhi"] for r in others), "the hidden-test sentence no longer holds"
        hid += [r["delta"] for r in others]
        e = {(r["a"], r["b"]): r for r in enf["contrasts"] if r["model"] == m and r["tests"] == "visible"}
        forced, offered = e[("h17_checker_tool_gate", "h16_checker_tool")], e[("h16_checker_tool", H01)]
        N.update({f"e_{k}_plain": pct1(offered["rate_b"]), f"e_{k}_tool": pct1(offered["rate_a"]), f"e_{k}_gate": pct1(forced["rate_a"]),
                  f"e_{k}_d": pts(forced["delta"]), f"e_{k}_lo": num(forced["lo"]), f"e_{k}_hi": num(forced["hi"])})
        v = {r["graph"]: r for r in main_["conditions"][m]}
        N[f"{k}_sv"] = pts(v["h14_verify"]["visible_delta"])
        assert all(v[g_]["visible_dlo"] > 0 for g_, _, gated in VARIANTS if gated), "an enforced-check graph is not clearly above zero"
        assert v["h14_verify"]["visible_delta"] < v[H04]["visible_delta"]
    assert enf["seeds"] == [1, 2] and cost[next(iter(MODELS))]["runs_per_task"] == 3
    N.update(hid_min=pts(min(hid)), hid_max=pts(max(hid)), raw=f'{space["raw_combinations"]:,}',
             distinct=f'{space["distinct_graph_signatures"]:,}', max_nodes=str(space["max_nodes"]))
    assert space["all_pass_static_checks"] and space["distinct_graph_signatures"] == space["after_rule_canonicalisation"]
    q = load("replay_search_qwen3.8-27b_visible_panel.json")
    t, hand = q["truth_all_tasks"], space["hand_written_points"]
    rest = [v for g, v in t.items() if g not in (hand[H01], hand[H04])]
    assert len(rest) == 30
    N.update(q_h01=pct1(t[hand[H01]]), q_h04=pct1(t[hand[H04]]), q_best=pct1(max(rest)), q_tasks=str(q["n_tasks"]),
             q_graphs=str(len(q["arms"])))
    N["fig_main"] = fig_main(rows_vis + rows_hid)
    N["fig_variants"] = fig_variants(main_["conditions"])

    page = (ROOT / "site" / "index.template.html").read_text()
    used = set(re.findall(r"\{\{(\w+)\}\}", page))
    if used - set(N):
        raise SystemExit(f"template asks for unknown numbers: {sorted(used - set(N))}")
    page = re.sub(r"\{\{(\w+)\}\}", lambda mm: N[mm.group(1)], page)
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "index.html").write_text(page)
    (ROOT / "docs" / ".nojekyll").write_text("")
    (S / "site_numbers.json").write_text(json.dumps(
        {"cost": cost, "numbers": {k: v for k, v in sorted(N.items()) if not k.startswith("fig_")}}, indent=1, ensure_ascii=False))
    print(f"wrote docs/index.html ({len(page):,} characters); unused numbers: {sorted(set(N) - used)}")
    for k, v in sorted(N.items()):
        if not k.startswith("fig_"):
            print(f"  {k:12s} {v}")


if __name__ == "__main__":
    main()
