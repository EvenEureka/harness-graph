# Controlled Evaluation of Workflow Graphs and Routing for Code-Repair Agents

**Status: main experiments complete; two validity controls of the check-and-retry result are underway** (protocol in `refine-logs/EXPERIMENT_PLAN.md`, sections E1 and E2; main runs not started). Project page with an interactive graph builder: https://eveneureka.github.io/harness-graph/ · Report: [docs/harness_graph_report.pdf](docs/harness_graph_report.pdf)

A coding agent is a language model plus the program around it, the *harness*: what the model is asked to do first,
when its work is checked, and what happens after a failed check. This repository writes the harness as a graph,
compares hand-designed graphs on bug-fixing tasks, and searches a space of 3,600 graphs for better ones.

- A **node** is one unit of execution (a stretch of agent conversation, or a program such as a test run). It reads
  the shared state, writes to it, and returns an outcome label (`done`, `pass`, `fail`, ...).
- An **edge** is a control rule: node X returned outcome r, so node Y runs next. Edges carry no data; the
  conversation, the repository working copy and the latest feedback live in the shared state.

## Results

Short bug-fixing tasks from SWE-smith, 81 confirmation tasks, two models.

| Tests visible to the agent | Plain loop | Check and retry | Difference (95% interval) |
|---|---|---|---|
| gpt-6.1-terra | 75% | 98% | +23.5 points (14.8 to 32.5) |
| gpt-6-luna | 63% | 77% | +13.6 points (7.0 to 20.6) |

With the tests hidden, on the 77 and 78 tasks that every graph completed, the eight other hand-designed graphs differed
from the plain loop by −3.9 to +2.6 points, every interval including zero. Check-and-retry uses 12% and 27% more
model calls. Making the check an enforced node that returns failures and triggers retries adds 7.4 points over offering the same checker as a tool,
for both models. The page gives the figures, the comparison of all hand-designed graphs, and the limits of these
results. Every numerical result of our experiments there comes from `results/summary/` through `scripts/build_site.py`.

**Search over the 3,600 graphs** (Qwen3.8-27B, thinking disabled; 188 tasks for search, 56 for selection, 232 held out
for confirmation):

- A table of 32 graphs on 188 search tasks: check-and-retry repaired 81.9%, the plain loop 72.9% (+9.0 points,
  4.3 to 13.8). On the 80 tasks new to this project the lead is +3.8 points (−1.3 to 8.8).
- Four search procedures replayed on that table: after 512 draws each returns graphs with a higher average held-out
  repair rate than the plain loop and a lower one than check-and-retry.
- Ten live local searches over the whole space, 512 agent runs each, all starting from the plain loop: all ten ended
  on a graph with the enforced check, and the ten graphs are all different.
- Selection among 14 candidate graphs, including the two baselines, on 56 tasks: the selected graph came from the table procedures (93.8% against
  92.0% for check-and-retry, +1.8 points, −0.9 to 5.4); none of the ten graphs from live search exceeded
  check-and-retry.
- Confirmation on 232 held-out tasks (2,088 runs): with visible tests the selected graph repaired 91.4% against 90.7%
  for check-and-retry, +0.6 points (−1.3 to 2.8). Superiority was not established and a gain of five points or more
  is excluded under the rule fixed in advance. Check-and-retry led the plain loop by 6.5 points (3.4 to 9.7), and by
  7.9 points (3.3 to 12.9) on three repositories withheld from search and selection. With hidden tests the three
  graphs repaired 64.2%, 64.7% and 64.2%.
- A graph per task: neither frozen policy (a rule on test visibility, a model on test visibility and task-text features) demonstrated an
  improvement over always using check-and-retry (+0.1 points, −1.8 to 2.2; 0.0 points, −1.7 to 1.7).

- Follow-up on 900 fresh tasks (SWE-smith's `combine_file` and `combine_module` strategies, 15 unused repositories,
  4,500 runs), run because a breakdown made after the fact had shown the selected graph ahead on 40 such tasks: the
  selected graph repaired 81.5% against 81.3% for check-and-retry, +0.2 points (−1.3 to 1.7). Superiority was not
  established and a five-point gain is excluded. Using the first run of each task, check-and-retry repaired 80.4% against 56.2% for
  the plain loop, a lead of 24.2 points (20.9 to 27.3).
  As planned, no further test of these two graphs on another subset follows.

- Premise test for the mechanism planned next (a graph that writes its own executable check when the grading tests are
  hidden): can a check written by the model tell wrong patches from correct ones? 173 search tasks, two checks per
  task, two versions of the check writer, pass rule fixed in advance (reject at least 30% of known wrong patches, at
  most 2% of bug-free repositories, at most 5% of grader-accepted patches). Neither version passed. Rates average within tasks and then across tasks; detection counts a missing check
  as a miss, while false rejection is conditional on a delivered check. Version 2
  rejected 28.6% of the 169 primary wrong patches from 89 tasks (22.0 to 35.7), the bug-free repository in 18.9% of cases (14.0 to 23.8) and
  accepted patches in 15.5% (11.1 to 20.1); version 1: 27.2%, 21.3% and 19.3%. A check that fails on the buggy
  program was not necessarily a reliable judge of a fix. As planned, the generated-check approach was stopped before
  a repair experiment (`results/summary/p1_G1.json`, `p1_G2.json`; code in `harness_graph/gencheck.py`,
  `scripts/p1_*.py`).

- What the failed runs show (exploratory audit of stored runs, `results/summary/failure_audit.json`): with visible tests,
  302 of the 336 unsolved check-and-retry runs on the follow-up tasks end on the 50-call cap, and 193 of the 336 unsolved runs
  never reached the check node. With hidden tests, 215 of 248 unsolved attempts on the held-out tasks end with a submitted
  patch, and the three graphs agree on 204 of 232 tasks (135 always solved, 69 never).
- Cap extension (exploratory, `results/summary/a1_cap_extension.json`): 400 new runs of check-and-retry on 200 sampled
  follow-up tasks with a cap of 100 calls, the patch after call 50 stored and graded. The state after call 50 is a
  repair in 81.75% of cases and the final state in 90.25%, +8.5 points (5.75 to 11.5), better on 28 tasks and worse on
  none, with 49.6% more tokens in total. The 50-call cap prevented some recoverable repairs under these conditions.
  Every graph comparison above ran under the same cap; the result gives no ranking of graphs at 100 calls.

The experiments are complete. Graphs with several agent conversations were not run: one design was reviewed on stored
runs and stopped before implementation.

The protocol, with the deviations recorded, is `refine-logs/EXPERIMENT_PLAN.md` (the review records it cites are not
part of this repository). Scope and limits are listed on the project page.

## Layout

| Path | Contents |
|---|---|
| `harness_graph/graph.py`, `executor.py` | graph format, static checks, the engine that runs a graph |
| `harness_graph/grammar.py` | nine design variables, the compiler from a setting to a graph, the 3,600-graph space |
| `harness_graph/search.py`, `runner.py` | five search procedures, offline replay, live episodes |
| `harness_graph/router.py` | choosing a graph per task: fit, freeze, score on other tasks |
| `harness_graph/swe/` | SWE-smith tasks in Apptainer containers, test check, grader |
| `harness_graph/gencheck.py` | premise test: the two-node check writer, the trusted check runner, evaluation on known patches |
| `scripts/audit_failures.py`, `scripts/a1_*.py` | audit of the failed runs; the cap extension (sample, read-out) |
| `graphs/` | the 17 hand-designed graphs (YAML) |
| `graphs_panel/` | the 32-graph panel used by the search experiment |
| `data/` | task lists and splits |
| `results/summary/` | result summaries behind the page |
| `scripts/` | experiment, analysis and site scripts |
| `site/`, `docs/` | page template and the built page |
| `tests/` | unit tests and container integration tests |

## Use

```bash
pip install -e ".[dev]"
pytest -q                                   # container tests are skipped without Apptainer and task images
python scripts/enumerate_space.py           # size of the graph space and the static checks on every graph
python scripts/run_grid.py --tasks data/tasks.jsonl --model MODEL --base-url http://127.0.0.1:8000/v1 \
    --out results/raw/MODEL.jsonl           # graphs x tasks against any OpenAI-compatible endpoint
python scripts/build_site.py                # rebuild docs/index.html from results/summary
node site/check_builder.js                  # the page's graph drawing agrees with the Python grammar
```

Running episodes needs Apptainer and the SWE-smith task images. The job files of the cluster this was developed
on are not part of this repository.

## License

MIT.
