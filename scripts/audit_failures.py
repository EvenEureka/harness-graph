"""Exploratory audit of stored episodes: why the four negative results came out as they did.  No model call.

Nothing here is a pre-specified contrast.  It reads raw episodes (results/raw, not in the repository) and writes
results/summary/failure_audit.json:
  hidden   per data set with tests hidden: how correlated independent attempts at one task are (always / never /
           sometimes solved), what the best of k attempts would reach with a perfect selector, what a label-free
           agreement selector reaches, and what the wrong patches look like (files edited against the files of the
           planted defect, share of target tests passed);
  visible  for check-and-retry with tests visible: how unsolved runs ended, how close they came, whether the failing
           count moved between the first and the last check, and repair rate by the number of files of the defect.

  python scripts/audit_failures.py --out results/summary/failure_audit.json
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H01, H04, SEL = "g-p0-fn-t0-vn-k0-c0-snon", "g-p0-fn-t0-vn-kcon2-c0-snon", "g-p0-fn-t0-vn-kref2-c0-scom"
HIDDEN = {
    "qwen3.8-27b, 232 confirmation tasks": (["results/raw/confirm/h_a/hidden/qwen3.8-27b.jsonl", "results/raw/confirm/h_b/hidden/qwen3.8-27b.jsonl"], ["data/v2/confirm.jsonl"]),
    "qwen3.8-27b, 188 search tasks": (["results/raw/search_hidden/a/hidden/qwen3.8-27b.jsonl", "results/raw/search_hidden/b/hidden/qwen3.8-27b.jsonl"], ["data/v2/search.jsonl"]),
    "gpt-6.1-terra, 81 confirmation tasks of section 2": (["results/raw/confirm_hidden/gpt-6.1-terra.jsonl"], ["data/confirm_tasks.jsonl"]),
    "gpt-6-luna, 81 confirmation tasks of section 2": (["results/raw/confirm_hidden/gpt-6-luna.jsonl"], ["data/confirm_tasks.jsonl"]),
}
VISIBLE = {
    "qwen3.8-27b, 232 confirmation tasks, two runs": (["results/raw/confirm/r0/visible/qwen3.8-27b.jsonl", "results/raw/confirm/r1/visible/qwen3.8-27b.jsonl"], ["data/v2/confirm.jsonl"]),
    "qwen3.8-27b, 900 follow-up tasks, two runs": (["results/raw/c1/r0/visible/qwen3.8-27b.jsonl", "results/raw/c1/r1/visible/qwen3.8-27b.jsonl"], ["data/c1/confirm.jsonl"]),
}


def files_of(patch: str) -> frozenset:
    return frozenset(m.group(1) for m in re.finditer(r"^\+\+\+ b/(\S+)", patch or "", re.M))


def is_test(path: str) -> bool:
    return bool(re.search(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$", path))


def norm(diff: str) -> str:
    """File names and changed lines of a diff, without hunk positions and trailing blanks."""
    return "\n".join(l.rstrip() for l in (diff or "").splitlines()
                     if (l[:1] in "+-" and not l.startswith("---")) or l.startswith("+++"))


def read(files, tests):
    seen = set()
    for f in files:
        for line in open(ROOT / f):                       # a missing file is an error, not an empty data set
            if line.strip():
                r = json.loads(line)
                if r.get("status") == "ok" and r.get("tests", tests) == tests:
                    cell = (r["graph"], r["task_id"], r["seed"], f)
                    assert cell not in seen, f"two scored records for {cell}"
                    seen.add(cell)
                    yield r


def gold_files(task_files) -> dict:
    out = {}
    for f in task_files:
        for line in open(ROOT / f):
            if line.strip():
                t = json.loads(line)
                out[t["instance_id"]] = (frozenset(p for p in files_of(t["bug_patch"]) if not is_test(p)), t.get("strategy", ""))
    return out


def complete(g: dict) -> bool:
    """The grade can be read as a count of tests: same rule as scripts/p1_patches.py."""
    return bool(g and g.get("grader_version") == "v2" and g.get("apply_ok") and not g.get("bug_apply_fail") and not g.get("timeout")
                and not g.get("ungradable") and g.get("test_rc") is not None and g.get("n_parsed") and g.get("f2p_total")
                and g.get("f2p_missing") == 0 and g.get("p2p_missing") == 0)


def relation(diff: str, gold: frozenset) -> str:
    mine = frozenset(p for p in files_of(diff) if not is_test(p))
    if not mine:
        return "no source file edited"
    if gold <= mine:
        return "edits every file of the defect"
    return "edits some files of the defect" if mine & gold else "edits none of the files of the defect"


def hidden_block(files, task_files) -> dict:
    gold = gold_files(task_files)
    by_task = defaultdict(list)
    for r in read(files, "hidden"):
        if r["task_id"] in gold:
            by_task[r["task_id"]].append(r)
    n_att = Counter(len(v) for v in by_task.values())
    k_max = min(n_att)                                      # attempts every task has
    tasks = sorted(by_task)
    solved = {t: sum(bool(r["resolved"]) for r in by_task[t]) for t in tasks}
    out = {"tasks": len(tasks), "attempts_per_task": dict(sorted(n_att.items())),
           "graphs": sorted({r["graph"] for v in by_task.values() for r in v}),
           "mean_repair_rate": sum(solved[t] / len(by_task[t]) for t in tasks) / len(tasks),
           "tasks_every_attempt_solves": sum(solved[t] == len(by_task[t]) for t in tasks),
           "tasks_no_attempt_solves": sum(solved[t] == 0 for t in tasks),
           "tasks_some_attempts_solve": sum(0 < solved[t] < len(by_task[t]) for t in tasks)}
    # the best of k attempts with a perfect selector: exact expectation over which k of the n attempts are drawn
    out["best_of_k_with_a_perfect_selector"] = {
        str(k): sum(1 - comb(len(by_task[t]) - solved[t], k) / comb(len(by_task[t]), k) for t in tasks) / len(tasks)
        for k in range(1, k_max + 1)}
    # a selector that sees no labels: take an attempt from the largest group of attempts with the same edit
    pick_same_edit = pick_same_files = pick_random = 0.0
    for t in tasks:
        rs = [r for r in by_task[t]]
        for key, acc in ((lambda r: norm(r["diff"]), "edit"), (lambda r: files_of(r["diff"]), "files")):
            groups = defaultdict(list)
            for r in rs:
                groups[key(r)].append(bool(r["resolved"]))
            top = max(len(g) for g in groups.values())
            best = [g for g in groups.values() if len(g) == top]
            val = sum(sum(g) / len(g) for g in best) / len(best)       # ties between groups broken at random
            if acc == "edit":
                pick_same_edit += val
            else:
                pick_same_files += val
        pick_random += solved[t] / len(rs)
    out["label_free_selection_over_all_attempts"] = {
        "attempts_used": "all attempts of a task", "random_attempt": pick_random / len(tasks),
        "largest_group_with_the_same_files_and_changed_lines": pick_same_edit / len(tasks),
        "largest_group_editing_the_same_files": pick_same_files / len(tasks),
        "perfect_selector": sum(solved[t] > 0 for t in tasks) / len(tasks)}
    # what the unsolved attempts look like
    wrong = [r for t in tasks for r in by_task[t] if not r["resolved"]]
    rel, by_exit, f2p_share, p2p_broken, graded = Counter(), Counter(), [], 0, 0
    rel_by_ngold = defaultdict(Counter)
    for r in wrong:
        g, _ = gold[r["task_id"]]
        by_exit[r["exit_reason"]] += 1
        k = relation(r["diff"], g)
        rel[k] += 1
        rel_by_ngold["1 file" if len(g) <= 1 else "2+ files"][k] += 1
        gr = r.get("grade") or {}
        if complete(gr):
            graded += 1
            f2p_share.append(gr["f2p_pass"] / gr["f2p_total"])
            p2p_broken += gr.get("p2p_pass", 0) < gr.get("p2p_total", 0)
    out["unsolved_attempts"] = {
        "n": len(wrong), "by_exit": dict(by_exit), "files_edited_against_files_of_the_defect": dict(rel),
        "the_same_by_number_of_files_of_the_defect": {k: dict(v) for k, v in rel_by_ngold.items()},
        "with_a_complete_grade": graded,
        "share_passing_no_target_test": sum(x == 0 for x in f2p_share) / max(1, graded),
        "share_passing_some_target_tests": sum(0 < x < 1 for x in f2p_share) / max(1, graded),
        "share_passing_all_target_tests_but_breaking_others": sum(x == 1 for x in f2p_share) / max(1, graded),
        "share_breaking_a_test_that_passed_before": p2p_broken / max(1, graded)}
    # repair rate by the number of files and by the strategy that planted the defect
    cut = defaultdict(lambda: [0, 0])
    for t in tasks:
        g, strat = gold[t]
        fam = "combine" if strat.startswith("combine") else "other"
        for key in ("defect in 1 file" if len(g) <= 1 else "defect in 2+ files", f"strategy: {fam}"):
            cut[key][0] += solved[t]
            cut[key][1] += len(by_task[t])
    out["repair_rate_by_group"] = {k: {"rate": a / b, "attempts": b} for k, (a, b) in sorted(cut.items())}
    return out


def visible_block(files, task_files) -> dict:
    gold = gold_files(task_files)
    out = {}
    for name, graph in (("plain loop", H01), ("check and retry", H04), ("selected graph", SEL)):
        rs = [r for r in read(files, "visible") if r["graph"] == graph and r["task_id"] in gold]
        if not rs:
            continue
        uns = [r for r in rs if not r["resolved"]]
        by_exit, rel, progress, close, reached = Counter(r["exit_reason"] for r in uns), Counter(), Counter(), [], Counter()
        for r in uns:
            rel[relation(r["diff"], gold[r["task_id"]][0])] += 1
            gr = r.get("grade") or {}
            if complete(gr):
                close.append(gr["f2p_pass"] / gr["f2p_total"])
            fails = [int(m.group(1)) for n in r["node_logs"] if n["type"] == "check"
                     for m in [re.search(r"new_test_failures:(\d+)", n.get("summary") or "")] if m]
            checks = [n for n in r["node_logs"] if n["type"] == "check"]
            reached["no check" if not checks else "one check" if len(checks) == 1 else "two or more checks"] += 1
            if len(fails) >= 2:
                progress["fewer failing tests at the last counted check than at the first" if fails[-1] < fails[0] else
                         "as many or more failing tests at the last counted check"] += 1
        grp = defaultdict(lambda: [0, 0, 0.0, 0])
        for r in rs:
            g = gold[r["task_id"]][0]
            key = "defect in 1 file" if len(g) <= 1 else ("defect in 2 files" if len(g) == 2 else "defect in 3+ files")
            grp[key][0] += bool(r["resolved"])
            grp[key][1] += 1
            grp[key][2] += r["model_calls"]
            grp[key][3] += r["exit_reason"] == "model_call_budget"
        mean = lambda xs: sum(xs) / len(xs) if xs else None
        out[name] = {
            "runs": len(rs), "unsolved": len(uns), "unsolved_by_exit": dict(by_exit),
            "unsolved_files_edited_against_files_of_the_defect": dict(rel),
            "unsolved_share_of_target_tests_passed": {"mean": mean(close), "none": sum(x == 0 for x in close),
                                                      "some": sum(0 < x < 1 for x in close), "all": sum(x == 1 for x in close)},
            "unsolved_with_a_complete_grade": len(close),
            "unsolved_by_checks_reached": dict(reached),
            "unsolved_with_two_counted_checks": dict(progress),
            "model_calls": {"solved": mean([r["model_calls"] for r in rs if r["resolved"]]), "unsolved": mean([r["model_calls"] for r in uns])},
            "largest_prompt_tokens": {"solved": mean([r["max_prompt_tokens"] for r in rs if r["resolved"]]),
                                      "unsolved": mean([r["max_prompt_tokens"] for r in uns])},
            "by_number_of_files_of_the_defect": {k: {"runs": n, "repair_rate": a / n, "model_calls": c / n, "ended_on_the_call_budget": b / n}
                                                 for k, (a, n, c, b) in sorted(grp.items())}}
    return out


def restart_block(files, task_files, budget=50) -> dict:
    """Parallel branches at the same total number of model calls, replayed from stored runs with tests visible.

    A task has up to four stored runs that all carry the enforced check (check-and-retry twice, the selected graph
    twice).  k branches get budget // k calls each; a branch counts as solved only if the stored run was solved and
    had used no more calls than that (a run that needed more is counted as failed, so the replay is a lower bound for
    the branches).  The task counts as solved if any of the k branches is; the visible tests are the selector.
    Averaged over every choice of k stored runs.  "k branches, full budget each" is the same without the cap on calls
    (k times the cost)."""
    from itertools import combinations
    gold = gold_files(task_files)
    by_task = defaultdict(list)
    for r in read(files, "visible"):
        if r["graph"] in (H04, SEL) and r["task_id"] in gold:
            by_task[r["task_id"]].append((r["graph"], bool(r["resolved"]), r["model_calls"]))
    tasks = [t for t in sorted(by_task) if len(by_task[t]) == 4]
    out = {"tasks": len(tasks), "runs_per_task": 4, "total_model_calls": budget}

    def rate(k, cap, pool=lambda g: True):
        tot = 0.0
        for t in tasks:
            runs = [x for x in by_task[t] if pool(x[0])]
            subs = list(combinations(runs, k))
            tot += sum(any(ok and calls <= cap for _, ok, calls in sub) for sub in subs) / len(subs)
        return tot / len(tasks)
    out["one_branch_full_budget"] = rate(1, budget)
    out["note"] = ("a stored run counts for a cutoff only if it finished solved within it; this is not the repair rate of a "
                   "branch stopped at the cutoff, and the agent was never told a budget")
    out["some_stored_run_finished_solved_within_cutoff"] = {f"{k} runs, cutoff {budget // k} calls": rate(k, budget // k) for k in (1, 2, 3, 4)}
    out["some_stored_run_finished_solved"] = {f"{k} runs, cutoff {budget} calls": rate(k, budget) for k in (1, 2, 3, 4)}
    out["check_and_retry_only"] = {"1 x 50": rate(1, budget, lambda g: g == H04), "2 x 25": rate(2, budget // 2, lambda g: g == H04),
                                   "2 x 50": rate(2, budget, lambda g: g == H04)}
    calls = [c for t in tasks for _, ok, c in by_task[t] if ok]
    n_runs = 4 * len(tasks)
    out["share_of_runs_solved_within_calls"] = {str(c): sum(x <= c for x in calls) / n_runs for c in (10, 12, 16, 20, 25, 30, 40, 50)}
    k = Counter(sum(ok for _, ok, _ in by_task[t]) for t in tasks)
    out["tasks_by_number_of_solved_runs_of_four"] = dict(sorted(k.items()))
    return out


def checkpoint_block(files, task_files, budget=50) -> dict:
    """What stored runs say about splitting the calls that remain after a checkpoint.  Tests visible, runs with the check.

    hazard: among runs not yet finished solved after c calls, the share that finish solved within the next 5.
    emulation: for a checkpoint after c0 calls, take the runs of a task that had not finished solved by c0.
      continue = one such run finishes solved within the full budget;
      split    = of two such runs, at least one finishes solved within c0 + (budget - c0) // 2 calls.
    Different stored runs of one task stand in for two continuations of one checkpoint.  Real branches share their
    history, so they are likely to be more alike than two stored runs; the emulation does not test real branching."""
    from itertools import combinations
    gold = gold_files(task_files)
    by_task = defaultdict(list)
    toks = {"solved": [], "unsolved": []}
    for r in read(files, "visible"):
        if r["graph"] in (H04, SEL) and r["task_id"] in gold:
            done = r["model_calls"] if r["resolved"] else None          # call count at which the run finished solved
            by_task[r["task_id"]].append(done)
            toks["solved" if r["resolved"] else "unsolved"].append(r["prompt_tokens"] + r["completion_tokens"])
    runs = [d for v in by_task.values() for d in v]
    hazard = {}
    for c in range(0, budget, 5):
        at_risk = [d for d in runs if d is None or d > c]
        hazard[f"{c + 1}-{c + 5}"] = {"runs_not_yet_solved": len(at_risk),
                                      "finish_solved_in_the_next_5_calls": sum(d is not None and d <= c + 5 for d in at_risk) / len(at_risk)}
    emu = {}
    for c0 in (10, 15, 20, 25, 30, 35, 40):
        half = c0 + (budget - c0) // 2
        cont = split = n = 0
        for v in by_task.values():
            open_ = [d for d in v if d is None or d > c0]
            if len(open_) < 2:
                continue
            n += 1
            cont += sum(d is not None and d <= budget for d in open_) / len(open_)
            pairs = list(combinations(open_, 2))
            split += sum(any(d is not None and d <= half for d in pr) for pr in pairs) / len(pairs)
        emu[str(c0)] = {"tasks_with_two_or_more_open_runs": n, "branch_cutoff_calls": half,
                        "continue": cont / n if n else None, "split_in_two": split / n if n else None}
    q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))] if xs else None
    return {"runs": len(runs), "hazard_by_calls": hazard, "split_emulation_by_checkpoint": emu,
            "prompt_plus_completion_tokens_at_the_end": {k: {"n": len(v), "p10": q(v, .1), "median": q(v, .5), "p90": q(v, .9), "max": q(v, 1)}
                                                         for k, v in toks.items()}}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    res = {"note": "exploratory audit of stored episodes; nothing here is a pre-specified contrast",
           "hidden": {k: hidden_block(*v) for k, v in HIDDEN.items()},
           "visible": {k: visible_block(*v) for k, v in VISIBLE.items()},
           "parallel_branches_replay_visible": {k: restart_block(*v) for k, v in VISIBLE.items()},
           "checkpoint_split_visible": {k: checkpoint_block(*v) for k, v in VISIBLE.items()}}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
