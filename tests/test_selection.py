import json
import subprocess
import sys
from pathlib import Path

from harness_graph.grammar import Config

ROOT = Path(__file__).resolve().parents[1]
H01, H04, NEW = Config().name, Config(gate=True).name, Config(gate=True, retries=3).name


def episode(graph, task, seed, resolved, calls):
    return {"episode_id": f"{graph}|{task}|{seed}", "model": "m", "graph": graph, "task_id": task, "seed": seed,
            "tests": "visible", "status": "ok", "resolved": resolved, "model_calls": calls}


def setup(tmp_path, solved, calls, drop=None):
    tasks = [{"instance_id": f"t{i}", "repo_key": f"r{i % 2}"} for i in range(4)]
    (tmp_path / "tasks.jsonl").write_text("\n".join(map(json.dumps, tasks)))
    cand = {"configs": {g: {} for g in (H01, H04, NEW)},
            "sources": {H01: ["h01 plain loop"], H04: ["h04 check-and-retry"], NEW: ["live a", "panel uniform"]},
            "live": {"a": {"method": "local_search_guided", "recommendation": NEW},
                     "b": {"method": "local_search_random", "recommendation": H04}},
            "panel_runs": {"uniform": {"recommendation": NEW}}}
    (tmp_path / "cand.json").write_text(json.dumps(cand))
    eps = [episode(g, t["instance_id"], s, i < solved[g], calls[g])
           for g in (H01, H04, NEW) for i, t in enumerate(tasks) for s in (0, 1)]
    if drop is not None:
        eps.pop(drop)
    (tmp_path / "eps.jsonl").write_text("\n".join(map(json.dumps, eps)))
    return subprocess.run([sys.executable, str(ROOT / "scripts/select_graph.py"), str(tmp_path / "eps.jsonl"),
                           "--tasks", str(tmp_path / "tasks.jsonl"), "--candidates", str(tmp_path / "cand.json"),
                           "--reps", "200", "--model", "m", "--out", str(tmp_path / "sel.json")], capture_output=True, text=True,
                          env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})


def test_highest_mean_wins_and_differences_are_paired(tmp_path):
    r = setup(tmp_path, {H01: 1, H04: 2, NEW: 3}, {H01: 10, H04: 20, NEW: 30})
    assert r.returncode == 0, r.stderr
    sel = json.loads((tmp_path / "sel.json").read_text())
    assert sel["searched_graph"] == NEW and not sel["is_h04"] and sel["n_tasks"] == 4
    top = sel["candidates"][0]
    assert top["value"]["mean"] == 0.75 and top["minus_h04"]["mean"] == 0.25 and top["minus_h04"]["better"] == 1
    assert sel["methods"]["local_search_guided"]["value"]["mean"] == 0.75
    assert sel["method_pairs"]["local_search_guided - local_search_random"]["mean"] == 0.25


def test_a_tie_goes_to_the_graph_with_fewer_model_calls(tmp_path):
    r = setup(tmp_path, {H01: 1, H04: 3, NEW: 3}, {H01: 10, H04: 20, NEW: 30})
    assert r.returncode == 0, r.stderr
    sel = json.loads((tmp_path / "sel.json").read_text())
    assert sel["searched_graph"] == H04 and sel["is_h04"] and set(sel["tied"]) == {H04, NEW}


def test_selection_stops_when_a_cell_is_missing(tmp_path):
    r = setup(tmp_path, {H01: 1, H04: 2, NEW: 3}, {H01: 10, H04: 20, NEW: 30}, drop=5)
    assert r.returncode != 0 and "missing" in r.stderr + r.stdout


def test_selection_stops_on_records_of_another_model(tmp_path):
    setup(tmp_path, {H01: 1, H04: 2, NEW: 3}, {H01: 10, H04: 20, NEW: 30})
    with open(tmp_path / "eps.jsonl", "a") as f:
        f.write("\n" + json.dumps({**episode(NEW, "t0", 0, True, 5), "episode_id": "other", "model": "other"}))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/select_graph.py"), str(tmp_path / "eps.jsonl"),
                        "--tasks", str(tmp_path / "tasks.jsonl"), "--candidates", str(tmp_path / "cand.json"),
                        "--model", "m"], capture_output=True, text=True, env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
    assert r.returncode != 0 and "not candidate graphs" in r.stderr
