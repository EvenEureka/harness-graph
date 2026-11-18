"""Move episodes with a given exit reason out of result files, so that a resumed run does them again.

  python scripts/drop_episodes.py --exit wall_time --to results/raw/search_table/discarded_wall_time.jsonl FILE...

Used when a run was misconfigured in a way that changed how episodes end (for example too many concurrent
episodes for the model server, so that episodes ran into the time limit).  The moved records are kept.
"""
import argparse
import json
import os

p = argparse.ArgumentParser()
p.add_argument("files", nargs="+")
p.add_argument("--exit", required=True)
p.add_argument("--to", required=True)
a = p.parse_args()
with open(a.to, "a") as side:
    for path in a.files:
        keep, moved = [], 0
        for line in open(path):
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("exit_reason") == a.exit:
                side.write(line if line.endswith("\n") else line + "\n")
                moved += 1
            else:
                keep.append(line if line.endswith("\n") else line + "\n")
        with open(path + ".tmp", "w") as f:
            f.writelines(keep)
        os.replace(path + ".tmp", path)
        print(f"{path}: kept {len(keep)}, moved {moved}")
