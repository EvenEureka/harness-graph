import json
import subprocess
import sys
from pathlib import Path

from harness_graph.grammar import Config

ROOT = Path(__file__).resolve().parents[1]
H01, H04, NEW = Config().name, Config(gate=True).name, Config(gate=True, retries=3).name
ENV = {"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"}


def episode(graph, task, seed, resolved, calls=10, status="ok"):
    return {"episode_id": f"{graph}|{task}|{seed}", "model": "m", "graph": graph, "task_id": task, "seed": seed,
            "tests": "visible", "status": status, "resolved": resolved, "model_calls": calls, "prompt_tokens": 100 * calls,
            "exit_reason": "end", "latency_s": 5.0}


def run(tmp_path, change=None):
    """12 tasks in 3 repositories.  Selected solves tasks 0-8 in both rollouts, h04 tasks 0-5 (so +0.25, all of it in
    repository r1 and half of r2... tasks 6-8), h01 tasks 0-2 on rollout 0."""
    tasks = [{"instance_id": f"t{i:02d}", "repo_key": f"r{i // 4}", "strategy": "combine_module" if i % 2 else "combine_file"} for i in range(12)]
    (tmp_path / "tasks.jsonl").write_text("\n".join(map(json.dumps, tasks)))
    (tmp_path / "sel.json").write_text(json.dumps({"searched_graph": NEW}))
    gd = tmp_path / "graphs"
    gd.mkdir(exist_ok=True)
    (gd / "a.yaml").write_text("x")
    eps = [episode(g, f"t{i:02d}", s, i < n, calls=c) for g, n, c in ((NEW, 9, 30), (H04, 6, 20)) for i in range(12) for s in (0, 1)]
    eps += [episode(H01, f"t{i:02d}", 0, i < 3) for i in range(12)]
    eps = change(eps) if change else eps
    (tmp_path / "eps.jsonl").write_text("\n".join(map(json.dumps, eps)))
    return subprocess.run([sys.executable, str(ROOT / "scripts/c1_analysis.py"), str(tmp_path / "eps.jsonl"), "--tasks", str(tmp_path / "tasks.jsonl"),
                           "--selection", str(tmp_path / "sel.json"), "--graph-dir", str(gd), "--model", "m", "--reps", "400",
                           "--out", str(tmp_path / "out.json")], capture_output=True, text=True, env=ENV)


def test_primary_secondary_and_sensitivity(tmp_path):
    r = run(tmp_path)
    assert r.returncode == 0, r.stderr
    o = json.loads((tmp_path / "out.json").read_text())
    p = o["primary"]
    assert p["mean"] == 0.25 and (p["better"], p["worse"], p["n_tasks"]) == (3, 0, 12)
    assert p["rate_selected"] == 0.75 and p["rate_h04"] == 0.5
    assert p["better_lower_bound_above_zero"] and p["gain_above_margin_lower_bound"] and not p["margin_gain_excluded_upper_bound"]
    s = o["secondary"]
    assert s["h04 - h01, rollout 0"]["mean"] == 0.25 and s["cost selected / h04"]["model_calls"]["ratio"] == 1.5
    assert s["by_repository"]["r1"]["mean"] == 0.5 and s["by_repository"]["r0"]["mean"] == 0.0 and s["by_repository"]["r2"]["mean"] == 0.25
    assert abs(s["leave_one_repository_out"]["r1"]["mean"] - 0.125) < 1e-12
    assert s["repositories_redrawn"]["lo"] <= 0.25 <= s["repositories_redrawn"]["hi"] and s["repositories_redrawn"]["lo"] < p["lo"]
    assert sum(v["n_tasks"] for v in s["by_strategy"].values()) == 12
    assert o["graphs"]["h01"]["by_rollout"] == {"0": 0.25} and o["cells"]["primary_arms"]["cells"] == 48 and o["cells"]["plain_loop"]["cells"] == 12


def test_stops_when_a_cell_is_missing_or_doubled_or_extra(tmp_path):
    assert run(tmp_path, lambda e: e[1:]).returncode != 0                                   # a cell without a record
    assert run(tmp_path, lambda e: e + [e[0]]).returncode != 0                              # a cell with two records
    assert run(tmp_path, lambda e: e + [episode(H01, "t00", 1, True)]).returncode != 0      # plain loop has rollout 0 only
    assert run(tmp_path, lambda e: e + [episode("other-graph", "t00", 0, True)]).returncode != 0
    ok = run(tmp_path, lambda e: e + [episode(NEW, "t00", 0, False, status="engine_error")])
    assert ok.returncode == 0, ok.stderr                                                     # an engine error that was rerun


def test_stops_when_the_selected_graph_is_check_and_retry(tmp_path):
    run(tmp_path)
    (tmp_path / "sel.json").write_text(json.dumps({"searched_graph": H04}))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/c1_analysis.py"), str(tmp_path / "eps.jsonl"), "--tasks", str(tmp_path / "tasks.jsonl"),
                        "--selection", str(tmp_path / "sel.json"), "--model", "m"], capture_output=True, text=True, env=ENV)
    assert r.returncode != 0
