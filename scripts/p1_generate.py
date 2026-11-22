"""P1, step 2: run the check generator (harness_graph/gencheck.py) on tasks.  Resumable.

  python scripts/p1_generate.py --tasks data/v2/search.jsonl --seeds 0:1 --model qwen3.8-27b \
      --base-url http://127.0.0.1:8000/v1 --no-thinking --out results/raw/p1/gen_G1.jsonl
"""
import argparse
import asyncio
import concurrent.futures
import json
import time
from pathlib import Path

from harness_graph import OpenAIChat
from harness_graph import gencheck as G
from harness_graph.swe.container import ensure_sif


async def main(a):
    asyncio.get_running_loop().set_default_executor(concurrent.futures.ThreadPoolExecutor(4 * a.workers + 8))
    tasks = [json.loads(l) for l in open(a.tasks) if l.strip()]
    if a.ids:
        want = [l.strip() for l in open(a.ids) if l.strip()]
        by = {t["instance_id"]: t for t in tasks}
        tasks = [by[i] for i in want]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in open(out):
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("status") == "ok":
                done.add(e["episode_id"])
    seeds = [int(s) for s in a.seeds.split(":")]
    todo = [(t, s) for t in tasks for s in seeds if f"{a.model}/{a.generator}/{t['instance_id']}/s{s}" not in done]
    sifs = {img: ensure_sif(img) for img in sorted({t["image_name"] for t, _ in todo})}
    extra = {"top_k": 20}
    if a.no_thinking:
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    llm = OpenAIChat(a.model, a.base_url, concurrency=a.workers, extra_body=extra, token_headroom=0)
    gate = asyncio.Semaphore(a.workers)
    n = errs = valid = 0
    t0 = time.time()
    print(f"generator {a.generator}: {len(tasks)} tasks x seeds {seeds}; {len(done)} done, {len(todo)} to run", flush=True)
    with open(out, "a") as f:
        async def one(t, s):
            nonlocal n, errs, valid
            async with gate:
                rec = await G.generate(t, sifs[t["image_name"]], llm, s, a.model, a.generator)
            rec["task"] = {k: t.get(k) for k in ("repo_key", "strategy")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            errs += rec["status"] != "ok"
            valid += bool(rec["valid"])
            print(f"[{time.strftime('%H:%M:%S')}] {n}/{len(todo)} {rec['episode_id']} status={rec['status']} "
                  f"exit={rec['exit_reason']} valid={rec['valid']} first={rec['first_valid']} calls={rec['model_calls']} "
                  f"path={'>'.join(rec['path'])} t={rec.get('latency_s')}s", flush=True)
        await asyncio.gather(*[one(t, s) for t, s in todo])
    print(f"DONE {n} episodes, {valid} with a check, {errs} engine errors, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True)
    p.add_argument("--ids", default="", help="file with instance ids, one per line: run only these")
    p.add_argument("--seeds", default="0:1")
    p.add_argument("--generator", default="G1", choices=sorted(G.GENERATORS))
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", required=True)
    asyncio.run(main(p.parse_args()))
