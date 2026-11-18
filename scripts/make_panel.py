"""The panel of candidate graphs for the search study, written as YAML so the grid runner can load it.

  python scripts/make_panel.py --n 32 --seed 0 --out graphs_panel

The plain loop and check-and-retry (the two reference graphs) are always in.  The rest is drawn by setting every
design variable independently and uniformly at random, half of the panel with the enforced check and half
without, so that the check does not dominate the panel just because it has more sub-options.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from harness_graph.grammar import DOMAINS, Config, canonical, compile_config

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=32)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="graphs_panel")
a = p.parse_args()
rng = np.random.default_rng(a.seed)
panel = [Config(), Config(gate=True)]
while len(panel) < a.n:
    gate = sum(c.gate for c in panel) < a.n // 2 and (sum(not c.gate for c in panel) >= a.n // 2 or rng.random() < 0.5)
    d = {k: dom[int(rng.integers(len(dom)))] for k, dom in DOMAINS.items()}
    d["gate"] = bool(gate)
    c = canonical(Config(**d))
    if c not in panel:
        panel.append(c)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for f in out.glob("*.yaml"):
    f.unlink()
for c in panel:
    (out / f"{c.name}.yaml").write_text(yaml.safe_dump(compile_config(c).to_dict(), sort_keys=False, allow_unicode=True))
(out / "panel.json").write_text(json.dumps({"seed": a.seed, "configs": {c.name: c.__dict__ for c in panel}}, indent=1))
print(f"{len(panel)} graphs -> {out}; with enforced check: {sum(c.gate for c in panel)}")
for c in panel:
    print(f"  {c.name:34s} {len(compile_config(c).nodes)} nodes")
