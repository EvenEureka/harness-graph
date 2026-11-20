"""Build docs/index.html from site/index.template.html, so that no number on the page is typed by hand.

  python scripts/build_site.py

Numbers come from results/summary/*.json.  The cost of check-and-retry over three runs is computed from the raw
episodes when they are present (results/raw is not in the repository) and otherwise taken from the last build
(results/summary/site_numbers.json, which this script writes).
"""
import datetime
import json
import re
from decimal import ROUND_HALF_UP, Decimal
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
r1 = lambda x: float(Decimal(repr(round(100 * x, 6))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))   # 86.25 -> 86.3
pct1 = lambda x: f"{r1(x):.1f}%"
pts = lambda x: "0.0" if abs(x) < 5e-4 else minus(f"{r1(x):+.1f}")
num = lambda x: minus(f"{r1(x):.1f}")
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


WORDS = {("checker_tool", True): "checker tool", ("verify", "same"): "self-review",
         ("verify", "fresh"): "self-review after summarizing", ("first", "reproduce"): "reproduce the bug first",
         ("plan", True): "plan first", ("on_fail", "reflect"): "find the cause after a failed check",
         ("on_fail", "reset"): "start over after a failed check", ("retries", 3): "up to 3 retries",
         ("retries", 1): "only 1 retry", ("compact_at", 16000): "summarize past 16k tokens",
         ("compact_at", 32000): "summarize past 32k tokens", ("stall", "nudge"): "reminder after a stall",
         ("stall", "reflect"): "find the cause after a stall", ("stall", "reset"): "start over after a stall",
         ("stall", "compact"): "summarize after a stall"}
PLAIN = {"plan": False, "first": "none", "checker_tool": False, "verify": "none", "gate": False, "on_fail": "continue",
         "retries": 2, "compact_at": None, "stall": "none"}
REPLAY = [("always", H01, "Always the plain loop", "plain"), ("uniform", None, "Uniform allocation", "other"),
          ("linear_thompson", None, "Thompson sampling, linear model", "other"), ("thompson", None, "Thompson sampling", "other"),
          ("successive_halving", None, "Successive halving", "other"), ("always", H04, "Always check-and-retry", "check")]
LOCAL = [("local_search_guided", "Chosen from how the last runs failed"), ("local_search_random", "Chosen at random")]


def fig_dots(rows, lo_v, hi_v, aria, axis, x0=300, x1=740, vx=800) -> str:
    """Rows: (label, class, value, tooltip, note).  One dot per row; the axis does not start at zero."""
    x = lambda v: x0 + (x1 - x0) * (v - lo_v) / (hi_v - lo_v)
    h = 62 + 30 * len(rows)
    out = [f'<svg viewBox="0 0 880 {h + 44}" role="img" aria-label="{aria}">']
    for i in range(round((hi_v - lo_v) / 0.05) + 1):
        v = lo_v + 0.05 * i
        out.append(f'<line class="gridl" x1="{x(v):.0f}" y1="24" x2="{x(v):.0f}" y2="{h}"/>'
                   f'<text class="ax" x="{x(v):.0f}" y="{h + 18}" text-anchor="middle">{pct(v)}</text>')
    out.append(f'<text class="ax" x="{(x0 + x1) / 2:.0f}" y="{h + 38}" text-anchor="middle">{axis} (axis starts at {pct(lo_v)})</text>')
    for i, (label, cls, v, tip, note) in enumerate(rows):
        y = 48 + 30 * i
        out.append(f'<text class="lab{" b" if note else ""}" x="20" y="{y + 5}">{label}</text>'
                   f'<circle class="d-{cls}" cx="{x(v):.1f}" cy="{y}" r="7"><title>{label}: {tip}</title></circle>'
                   f'<text class="val" x="{vx}" y="{y + 5}" text-anchor="end">{pct1(v)}</text>')
        if note:
            out.append(f'<text class="ax" x="{vx + 10}" y="{y + 5}">{note}</text>')
    return "".join(out) + "</svg>"


def selection_numbers(hand) -> dict:
    """Section 4, stage 4: the candidates on the selection tasks."""
    sel = load("selection.json")
    cand = json.loads((ROOT / "graphs_select" / "candidates.json").read_text())
    rows_ = {r["graph"]: r for r in sel["candidates"]}
    h01, h04, top = hand[H01], hand[H04], sel["searched_graph"]
    assert not sel["is_h04"] and len(sel["tied"]) == 1 and sel["stored_records"] == {"ok": len(rows_) * sel["n_tasks"] * len(sel["rollouts"])}
    assert all(src.startswith("panel ") for src in rows_[top]["sources"]), "the selected graph is no longer a table graph only"
    live = {run: r for run, r in cand["live"].items()}
    v04 = rows_[h04]["value"]["mean"]
    live_vals = [rows_[r["recommendation"]]["value"]["mean"] for r in live.values()]
    assert max(live_vals) == v04 and sum(v == v04 for v in live_vals) == 1, "the sentence on the ten live graphs no longer holds"
    words = {"uniform": "uniform allocation", "successive_halving": "successive halving", "thompson": "Thompson sampling",
             "linear_thompson": "Thompson sampling with a linear model"}
    how = {"local_search_guided": "from failures", "local_search_random": "at random"}

    def label(g):
        src = rows_[g]["sources"]
        panel = [words[x.split()[1]] for x in src if x.startswith("panel ")]
        if g == h01:
            return "Plain loop (graph A)"
        if g == h04:
            return "Check and retry (graph B)"
        if panel:
            return "Table: " + ", ".join(panel)
        run = live[src[0].split()[1]]
        return f"Live search, {how[run['method']]}, {run['seed'] + 1}"

    rows = []
    for r in sel["candidates"]:
        g, d = r["graph"], r["minus_h04"]
        tip = f'{pct1(r["value"]["mean"])} of {sel["n_tasks"]} tasks, two runs each' + (
            "" if g == h04 else f'; minus check-and-retry {pts(d["mean"])} points [{num(d["lo"])}, {num(d["hi"])}]')
        rows.append((label(g), "plain" if g == h01 else "check" if g == h04 else "other", r["value"]["mean"], tip,
                     "selected" if g == top else ""))
    N = {"fig_select": fig_dots(rows, 0.75, 1.0, "Repair rate of the 14 candidate graphs on the 56 selection tasks.",
                                "share of the 56 selection tasks repaired, two runs per task", x0=350, x1=730)}
    t, m, gr = rows_[top], sel["methods"], sel["method_pairs"]["local_search_guided - local_search_random"]
    d4, d1 = t["minus_h04"], rows_[h04]["minus_h01"]
    assert d4["lo"] < 0 < d4["hi"] and d1["lo"] > 0 and gr["lo"] < 0 < gr["hi"]
    assert all(r["minus_h01"]["mean"] > 0 for r in sel["candidates"] if r["graph"] != h01)
    c = cand["configs"][top]
    extras = [WORDS[(k, v)] for k, v in c.items() if k != "gate" and v != PLAIN[k]]
    js = {"checker_tool": "tool", "on_fail": "onfail", "compact_at": "compact"}
    setting = {js.get(k, k): (f"{v // 1000}k" if k == "compact_at" else v == "reproduce" if k == "first" else v)
               for k, v in c.items() if v != PLAIN[k]}
    N.update(s_graphs=str(len(rows_)), s_tasks=str(sel["n_tasks"]), s_runs=f'{sel["stored_records"]["ok"]:,}',
             s_top=pct1(t["value"]["mean"]), s_h04=pct1(v04), s_h01=pct1(rows_[h01]["value"]["mean"]),
             s_d=pts(d4["mean"]), s_lo=num(d4["lo"]), s_hi=num(d4["hi"]), s_better=str(d4["better"]), s_worse=str(d4["worse"]),
             s_extras="; ".join(extras), s_config=json.dumps(setting),
             s_guided=num(abs(m["local_search_guided"]["minus_h04"]["mean"])), s_random=num(abs(m["local_search_random"]["minus_h04"]["mean"])),
             s_gr=pts(gr["mean"]), s_gr_lo=num(gr["lo"]), s_gr_hi=num(gr["hi"]),
             s_k=pts(d1["mean"]), s_k_lo=num(d1["lo"]), s_k_hi=num(d1["hi"]))
    assert m["local_search_guided"]["minus_h04"]["mean"] < 0 and m["local_search_random"]["minus_h04"]["mean"] < 0
    return N


def confirm_numbers(hand) -> dict:
    """Section 4, stages 5 and 6: the confirmation tasks and the policies that pick a graph per task."""
    c, rt, fr, ex = load("confirmation.json"), load("routing_confirm.json"), load("router_frozen.json"), load("confirmation_exploratory.json")
    sel = load("selection.json")
    top = c["searched_graph"]
    vis, hid = c["conditions"]["visible"], c["conditions"]["hidden"]
    n_runs = sum(v["scored_records"] for v in c["cells"].values())
    assert top == sel["searched_graph"] and c["rollouts"] == {"visible": [0, 1], "hidden": [0]}
    assert all(v["engine_error_records"] == 0 and v["cells"] == v["scored_records"] for v in c["cells"].values())
    assert all(g["past_deadline"] == 0 for cond in (vis, hid) for g in cond["graphs"].values())
    p, k, kh = vis["contrasts"]["searched - h04"], vis["contrasts"]["h04 - h01"], hid["contrasts"]
    pa, ka, kn = p["all"], k["all"], k["new_repositories"]
    assert c["primary"]["mean"] == pa["mean"] and pa["lo"] < 0 < pa["hi"] < c["sesoi"] == 0.05, "the primary sentence no longer holds"
    assert ka["lo"] > 0 and kn["lo"] > 0, "check-and-retry is no longer clearly above the plain loop"
    assert all(-0.04 < d["all"]["lo"] < 0 < d["all"]["hi"] < 0.04 for d in kh.values()), "the hidden-test sentences no longer hold"
    assert 0 < pa["mean"] < next(r for r in sel["candidates"] if r["graph"] == top)["minus_h04"]["mean"], "the lead did not fall"
    ix = ex["visible_minus_hidden"]["h04 - h01"]
    assert ix["lo"] > 0
    lab = [("h01", "Plain loop (graph A)", "plain"), ("h04", "Check and retry (graph B)", "check"), ("searched", "Selected graph", "other")]
    rows = [(f"{cn} · {name}", cls, cond["graphs"][key]["resolved"],
             f'{pct1(cond["graphs"][key]["resolved"])} of {c["n_tasks"]} tasks, {runs}', "")
            for cn, cond, runs in (("Tests visible", vis, "two runs each"), ("Tests hidden", hid, "one run each")) for key, name, cls in lab]
    N = {"fig_confirm": fig_dots(rows, 0.6, 0.95, "Repair rate of the plain loop, check-and-retry and the selected graph on the 232 "
                                 "confirmation tasks, with visible and with hidden tests.",
                                 "share of the 232 confirmation tasks repaired", x0=350, x1=730)}
    g = vis["graphs"]
    N.update(c_tasks=str(c["n_tasks"]), c_runs=f"{n_runs:,}", c_old=str(c["n_tasks_by_group"]["old_repositories"]),
             c_new=str(c["n_tasks_by_group"]["new_repositories"]),
             c_top=pct1(g["searched"]["resolved"]), c_h04=pct1(g["h04"]["resolved"]), c_h01=pct1(g["h01"]["resolved"]),
             c_d=pts(pa["mean"]), c_lo=num(pa["lo"]), c_hi=num(pa["hi"]), c_better=str(pa["better"]), c_worse=str(pa["worse"]),
             c_k=pts(ka["mean"]), c_k_lo=num(ka["lo"]), c_k_hi=num(ka["hi"]),
             c_kn=pts(kn["mean"]), c_kn_lo=num(kn["lo"]), c_kn_hi=num(kn["hi"]),
             c_hid_h01=pct1(hid["graphs"]["h01"]["resolved"]), c_hid_h04=pct1(hid["graphs"]["h04"]["resolved"]),
             c_hid_top=pct1(hid["graphs"]["searched"]["resolved"]),
             c_ix=num(ix["mean"]), c_ix_lo=num(ix["lo"]), c_ix_hi=num(ix["hi"]),
             c_calls_top=f'{g["searched"]["model_calls"]:.1f}', c_calls_h04=f'{g["h04"]["model_calls"]:.1f}',
             c_tok_top=f'{round(g["searched"]["tokens"], -3):,.0f}', c_tok_h04=f'{round(g["h04"]["tokens"], -3):,.0f}',
             c_tok_ratio=f'{pa["tokens"]["ratio"]:.2f}', c_tok_lo=f'{pa["tokens"]["lo"]:.2f}', c_tok_hi=f'{pa["tokens"]["hi"]:.2f}')
    assert pa["tokens"]["lo"] < 1 < pa["tokens"]["hi"], "the token sentence no longer holds"
    hd = load("headroom_diagnostics.json")["confirmation_hidden"]["by_graph"]
    uns, wrong = [v["hidden_unsolved"] for v in hd.values()], [v["hidden_unsolved_by_exit"]["end"] for v in hd.values()]
    assert len(hd) == 3 and [v["hidden_unsolved"] for v in hd.values()] == [round((1 - hid["graphs"][k]["resolved"]) * c["n_tasks"]) for k in hd]
    N.update(d_uns_lo=str(min(uns)), d_uns_hi=str(max(uns)), d_wrong_lo=str(min(wrong)), d_wrong_hi=str(max(wrong)))
    # stage 6
    pol = fr["policies"]
    assert pol["fixed_best"]["graph"] == hand[H04] and rt["policies_file"].endswith("router_frozen.json") and not rt["units_left_out"]
    assert dict(map(tuple, pol["rule[tests_visible]"]["table"])) == {False: hand[H01], True: top}, "the rule is no longer as described"
    rf, ff = rt["rule[tests_visible] - fixed_best"], rt["feature_router - fixed_best"]
    assert rf["lo"] < 0 < rf["hi"] and ff["lo"] < 0 < ff["hi"], "the routing sentence no longer holds"
    N.update(rt_units=str(rt["n_units"]), rt_fixed=pct1(rt["fixed_best"]["mean"]), rt_rule=pct1(rt["rule[tests_visible]"]["mean"]),
             rt_feat=pct1(rt["feature_router"]["mean"]), rt_rule_d=pts(rf["mean"]), rt_rule_lo=num(rf["lo"]), rt_rule_hi=num(rf["hi"]),
             rt_feat_d=pts(ff["mean"]), rt_feat_lo=num(ff["lo"]), rt_feat_hi=num(ff["hi"]))
    return N


def search_numbers(q, hand) -> dict:
    """Section 4: the table (stage 1), replay on it (stage 2) and the live local searches (stage 3)."""
    N = {}
    tc, ls = load("table_claims.json"), load("live_search.json")
    assert tc["n_tasks"] == q["n_tasks"] and tc["n_graphs"] == len(q["arms"])
    N["q_runs"] = f'{tc["n_tasks"] * tc["n_graphs"]:,}'
    for key, k in (("all_tasks", "q"), ("used_before", "u"), ("never_used_before", "w")):
        c = tc["h04_minus_h01"][key]
        d = c["difference"]
        N.update({f"{k}_h01": pct1(c["h01"]), f"{k}_h04": pct1(c["h04"]), f"{k}_d": pts(d["mean"]), f"{k}_lo": num(d["lo"]),
                  f"{k}_hi": num(d["hi"]), f"{k}_better": str(d["better"]), f"{k}_worse": str(d["worse"]), f"{k}_n": str(d["n_tasks"])})
    a, u, w = (tc["h04_minus_h01"][k] for k in ("all_tasks", "used_before", "never_used_before"))
    assert a["difference"]["lo"] > 0 and u["difference"]["lo"] > 0 and w["difference"]["lo"] < 0 < w["difference"]["mean"] < u["difference"]["mean"]
    N.update(q_calls=more(a["model_calls"]["ratio"]), q_tokens=more(a["tokens"]["ratio"]))
    hr = tc["headroom_over_h04"]
    assert round(hr["top_minus_h04"] * tc["n_tasks"]) == 1 and hr["graphs_above_h04"] == 1 and hr["n_other_graphs"] == 31
    assert hr["upper_bound_for_every_graph"] > 0.05, "the sentence on a five-point improvement no longer holds"
    back = tc["with_replaced_episodes_put_back"]
    assert back["h04_minus_h01"] == a["difference"] and back["headroom_over_h04"]["top_minus_h04"] == 0
    N.update(q_top_d=pts(hr["top_minus_h04"]), q_bound=num(hr["upper_bound_for_every_graph"]), q_replaced=str(back["n_cells"]))

    budget = 512
    val = {(r["method"], r["budget"]): r for r in q["results"]}
    rows, reuse = [], []
    for method, graph, label, cls in REPLAY:
        key = f"always {hand[graph]}" if graph else method
        r = val[(key, budget)]
        tip = ", ".join(f'{pct1(val[(key, b)]["value"])} at {b:,} draws' for b in (256, 512, 1024)) if not graph else pct1(r["value"])
        rows.append((label, cls, r["value"], f"{tip}; {len(q['splits'])} halvings", ""))
        if not graph:
            reuse.append(r["reused_share"])
    lo_ref, hi_ref = rows[0][2], rows[-1][2]
    assert all(lo_ref < r[2] < hi_ref for r in rows[1:-1]), "the replay heading no longer holds"
    N.update(r_budget=str(budget), r_splits=str(len(q["splits"])), r_reuse_lo=pct1(min(reuse)), r_reuse_hi=pct1(max(reuse)))
    N["fig_replay"] = fig_dots(rows, 0.70, 0.85, f"Average held-out repair rate of the graph returned by each search procedure "
                               f"after {budget} table draws, next to always returning the plain loop or check-and-retry.",
                               "average repair rate of the returned graph on the held-out half")

    runs = ls["runs"]
    cfg = {r["run"]: json.loads((ROOT / "results" / "search" / r["run"] / "result.json").read_text())["recommendation_config"] for r in runs}
    assert len(runs) == 10 and all(r["spent"] == r["scored_episodes"] == 512 and r["steps"] == 16 for r in runs)
    assert all(r["engine_error_records"] == r["engine_errors_retried"] == r["past_deadline"] == 0 for r in runs)
    assert all(c["gate"] for c in cfg.values()) and len({r["recommendation"] for r in runs}) == 10
    gate_first = sum(r["first_move"] == ["gate=True"] for r in runs)
    rejected_first = 0
    for r in runs:
        tries = [s["accepted"] for s in r["trace"] if "gate=True" in s["move"]]
        assert any(tries)
        rejected_first += not tries[0]
    words = lambda n: ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"][n]
    N.update(ls_runs=words(len(runs)), ls_budget="512", ls_steps="16", ls_gate_first=words(gate_first),
             ls_gate_rejected=words(rejected_first), ls_retries3=words(sum(c["retries"] == 3 for c in cfg.values())),
             ls_compact=words(sum(c["compact_at"] is not None for c in cfg.values())),
             ls_nudge=words(sum(c["stall"] == "nudge" for c in cfg.values())))
    js = {"checker_tool": "tool", "on_fail": "onfail", "compact_at": "compact"}
    out = []
    for method, label in LOCAL:
        mine = sorted((r for r in runs if r["method"] == method), key=lambda r: r["seed"])
        assert [r["seed"] for r in mine] == [0, 1, 2, 3, 4]
        for i, r in enumerate(mine):
            c = cfg[r["run"]]
            extras = [WORDS[(k, v)] for k, v in c.items() if k != "gate" and v != PLAIN[k]]
            setting = {}
            for k, v in c.items():
                if v != PLAIN[k]:
                    setting[js.get(k, k)] = (f"{v // 1000}k" if k == "compact_at" else v == "reproduce" if k == "first" else v)
            head = f'<td rowspan="5">{label}</td>' if i == 0 else ""
            out.append(f'<tr>{head}<td class="num">{r["seed"] + 1}</td><td>{"; ".join(extras) or "nothing else"}</td>'
                       f"<td><button type=\"button\" class=\"mini\" data-config='{json.dumps(setting)}' "
                       f'aria-label="Draw the graph of search {r["seed"] + 1}, {label.lower()}, in the builder">draw</button></td></tr>')
    N["search_rows"] = "".join(out)
    return N


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
    N.update(q_tasks=str(q["n_tasks"]), q_graphs=str(len(q["arms"])))
    N.update(search_numbers(q, hand))
    N.update(selection_numbers(hand))
    N.update(confirm_numbers(hand))
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
        {"cost": cost, "numbers": {k: v for k, v in sorted(N.items()) if not k.startswith(("fig_", "search_rows"))}}, indent=1, ensure_ascii=False))
    print(f"wrote docs/index.html ({len(page):,} characters); unused numbers: {sorted(set(N) - used)}")
    for k, v in sorted(N.items()):
        if not k.startswith(("fig_", "search_rows")):
            print(f"  {k:12s} {v}")


if __name__ == "__main__":
    main()
