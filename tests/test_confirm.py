import json
import subprocess
import sys
from pathlib import Path

import pytest

from harness_graph import analysis as A
from harness_graph.grammar import Config

ROOT = Path(__file__).resolve().parents[1]
H01, H04, NEW = Config().name, Config(gate=True).name, Config(gate=True, retries=3).name
ENV = {"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"}


def episode(graph, task, seed, tests, resolved, calls=10, status="ok", model="m"):
    return {"episode_id": f"{graph}|{task}|{seed}", "model": model, "graph": graph, "task_id": task, "seed": seed,
            "tests": tests, "status": status, "resolved": resolved, "model_calls": calls, "prompt_tokens": 100 * calls,
            "exit_reason": "end", "latency_s": 5.0}


def write(path, rows):
    path.write_text("\n".join(map(json.dumps, rows)))
    return str(path)


def setup(tmp_path, searched=NEW, drop=None, extra=None):
    """8 tasks: 4 in repository old (also a search repository), 4 in repository new.  With visible tests the
    searched graph solves 6 tasks in both rollouts, h04 solves 4, h01 solves 2; with hidden tests all solve 2."""
    tasks = [{"instance_id": f"t{i}", "repo_key": "old" if i < 4 else "new"} for i in range(8)]
    write(tmp_path / "confirm.jsonl", tasks)
    write(tmp_path / "search.jsonl", [{"instance_id": "s0", "repo_key": "old"}])
    (tmp_path / "sel.json").write_text(json.dumps({"searched_graph": searched}))
    solved = {NEW: 6, H04: 4, H01: 2}
    graphs = list(dict.fromkeys([H01, H04, searched]))
    vis = [episode(g, f"t{i}", s, "visible", i < solved[g], calls=20 if g != H01 else 10) for g in graphs for i in range(8) for s in (0, 1)]
    hid = [episode(g, f"t{i}", 0, "hidden", i < 2) for g in graphs for i in range(8)]
    if drop is not None:
        vis.pop(drop)
    vis += extra or []
    return subprocess.run([sys.executable, str(ROOT / "scripts/confirm_analysis.py"), "--visible", write(tmp_path / "v.jsonl", vis),
                           "--hidden", write(tmp_path / "h.jsonl", hid), "--tasks", str(tmp_path / "confirm.jsonl"),
                           "--search-tasks", str(tmp_path / "search.jsonl"), "--selection", str(tmp_path / "sel.json"),
                           "--model", "m", "--reps", "300", "--out", str(tmp_path / "out.json")], capture_output=True, text=True, env=ENV)


def test_primary_contrast_groups_and_costs(tmp_path):
    r = setup(tmp_path)
    assert r.returncode == 0, r.stderr
    out = json.loads((tmp_path / "out.json").read_text())
    assert out["primary"]["mean"] == 0.25 and out["primary"]["better"] == 2 and out["primary"]["worse"] == 0
    assert out["primary"]["no_gain_of_sesoi"] is False
    vis = out["conditions"]["visible"]
    assert vis["graphs"]["searched"]["resolved"] == 0.75 and vis["graphs"]["h01"]["by_rollout"] == {"0": 0.25, "1": 0.25}
    c = vis["contrasts"]["h04 - h01"]
    assert c["all"]["mean"] == 0.25 and c["old_repositories"]["mean"] == 0.5 and c["new_repositories"]["mean"] == 0.0
    assert c["all"]["model_calls"]["ratio"] == 2.0
    assert out["repositories"] == {"old": ["old"], "new": ["new"]} and out["n_tasks_by_group"]["new_repositories"] == 4
    hid = out["conditions"]["hidden"]["contrasts"]["searched - h04"]["all"]
    assert hid["mean"] == 0.0 and hid["lo"] == hid["hi"] == 0.0 and hid["no_gain_of_sesoi"] is True


def test_searched_graph_equal_to_h04_gives_no_primary_number(tmp_path):
    r = setup(tmp_path, searched=H04)
    assert r.returncode == 0, r.stderr
    out = json.loads((tmp_path / "out.json").read_text())
    assert out["searched_is_h04"] and "mean" not in out["primary"] and list(out["conditions"]["visible"]["contrasts"]) == ["h04 - h01"]


def test_stops_on_a_missing_cell_and_on_a_second_scored_record(tmp_path):
    r = setup(tmp_path, drop=3)
    assert r.returncode != 0 and "without a scored record" in r.stderr
    r = setup(tmp_path, extra=[{**episode(H04, "t0", 0, "visible", True), "episode_id": "again"}])
    assert r.returncode != 0 and "1 with several" in r.stderr


def test_audit_counts_engine_errors_and_rejects_foreign_records(tmp_path):
    rows = [episode(H01, "t0", 0, "visible", True), episode(H01, "t0", 0, "visible", False, status="engine_error"),
            episode("other", "t0", 0, "visible", True)]
    f = write(tmp_path / "e.jsonl", rows)
    got = A.audit_cells([f], "m", [H01], ["t0"], {"visible": [0]})
    assert got == {"cells": 1, "scored_records": 1, "engine_error_records": 1, "records_of_other_graphs": 1, "cut_off_lines": 0}
    for bad in (episode(H01, "t1", 0, "visible", True), episode(H01, "t0", 1, "visible", True),
                episode(H01, "t0", 0, "visible", True, model="x")):
        with pytest.raises(ValueError):
            A.audit_cells([write(tmp_path / "b.jsonl", rows + [{**bad, "episode_id": "b"}])], "m", [H01], ["t0"], {"visible": [0]})
