"""Size of the grammar's graph space, and optionally the graphs themselves as YAML.

  python scripts/enumerate_space.py [--write DIR] [--out results/summary/space.json]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from harness_graph.grammar import HAND_WRITTEN, all_configs, compile_config, space_report

p = argparse.ArgumentParser()
p.add_argument("--write", default="")
p.add_argument("--out", default="")
a = p.parse_args()
rep = space_report()
configs = all_configs()
rep["nodes_per_graph"] = dict(sorted(Counter(len(compile_config(c).nodes) for c in configs).items()))
rep["hand_written_points"] = {k: v.name for k, v in HAND_WRITTEN.items()}
print(json.dumps(rep, indent=1))
if a.write:
    d = Path(a.write)
    d.mkdir(parents=True, exist_ok=True)
    for c in configs:
        (d / f"{c.name}.yaml").write_text(yaml.safe_dump(compile_config(c).to_dict(), sort_keys=False))
    print(f"wrote {len(configs)} graphs to {d}")
if a.out:
    Path(a.out).write_text(json.dumps(rep, indent=1))
