"""Does the issue text of a task mention every defect that was planted?  One model call per task.

  python scripts/spec_audit.py --tasks data/tasks.jsonl --model gpt-6.1-terra --api-key-env OPENAI_API_KEY \
      --prices 1 0.05 5 --max-usd 3 --out data/spec_audit.jsonl

The judge sees the issue text and the patch that introduced the defects, never an agent's outcome.  A task is
"full" when every planted defect is described or would be met while fixing a described one, else "partial".
The labels are written once and not revised after looking at results.
"""
import argparse
import asyncio
import json
import os
import re
from pathlib import Path

from harness_graph.llm import OpenAIChat, Prices

PROMPT = """A bug-repair benchmark plants defects in a repository and gives the solver only an issue text.
Below are the issue text and the patch that planted the defects (lines starting with "+" are the defective
code, lines starting with "-" the original code).

Split the patch into separate defects: changes that break different behaviours count separately, and several
lines that break one behaviour count once. For each defect decide whether a careful engineer who reads only the
issue text and the repository, without any failing test, would know that this behaviour has to be repaired:
"described" if the issue reports this behaviour (even in different words), or if repairing a reported
behaviour necessarily repairs it; "not_described" otherwise.

Answer with JSON only:
{{"defects": [{{"file": "...", "what": "one line", "status": "described" | "not_described"}}]}}

<issue>
{problem}
</issue>

<patch>
{patch}
</patch>"""


async def main(a):
    tasks = [json.loads(l) for l in open(a.tasks)]
    out = Path(a.out)
    have = {json.loads(l)["instance_id"] for l in open(out)} if out.exists() else set()
    llm = OpenAIChat(a.model, "https://api.openai.com/v1", api_key=os.environ[a.api_key_env], concurrency=8,
                     temperature=None, top_p=None, reasoning_effort="medium", token_headroom=8192,
                     prices=Prices(*a.prices), max_usd=a.max_usd)

    async def one(t):
        msg = PROMPT.format(problem=t["problem_statement"], patch=t["bug_patch"][:30000])
        c = await llm.chat([{"role": "user", "content": msg}], max_tokens=2000, seed=0)
        m = re.search(r"\{.*\}", c.text, re.S)
        defects = json.loads(m.group(0))["defects"]
        n_no = sum(d["status"] == "not_described" for d in defects)
        return {"instance_id": t["instance_id"], "judge": a.model, "n_defects": len(defects), "n_not_described": n_no,
                "spec": "partial" if n_no else "full", "defects": defects}

    with open(out, "a") as f:
        for r in await asyncio.gather(*[one(t) for t in tasks if t["instance_id"] not in have], return_exceptions=True):
            if isinstance(r, Exception):
                print("FAILED", type(r).__name__, str(r)[:200])
                continue
            f.write(json.dumps(r) + "\n")
    rows = [json.loads(l) for l in open(out)]
    print(f"{len(rows)} tasks labelled, {sum(r['spec'] == 'partial' for r in rows)} partial, spent ${llm.spent:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--prices", type=float, nargs=3, required=True, metavar=("IN", "CACHED_IN", "OUT"))
    p.add_argument("--max-usd", type=float, required=True)
    p.add_argument("--out", required=True)
    asyncio.run(main(p.parse_args()))
