# Adaptive Graph Harness Optimization for LLM Agents

**Status: in progress.** Project page with an interactive graph builder: https://eveneureka.github.io/harness-graph/

A coding agent is a language model plus the program around it, the *harness*: what the model is asked to do first,
when its work is checked, and what happens after a failed check. This repository writes the harness as a graph,
compares hand-designed graphs on bug-fixing tasks, and searches a space of 3,600 graphs for better ones.

- A **node** is one unit of execution (a stretch of agent conversation, or a program such as a test run). It reads
  the shared state, writes to it, and returns an outcome label (`done`, `pass`, `fail`, ...).
- An **edge** is a control rule: node X returned outcome r, so node Y runs next. Edges carry no data; the
  conversation, the repository working copy and the latest feedback live in the shared state.

## Results so far

Short bug-fixing tasks from SWE-smith, 81 confirmation tasks, two models.

| Tests visible to the agent | Plain loop | Check and retry | Difference (95% interval) |
|---|---|---|---|
| gpt-6.1-terra | 75% | 98% | +23.5 points (14.8 to 32.5) |
| gpt-6-luna | 63% | 77% | +13.6 points (7.0 to 20.6) |

With the tests hidden, no graph showed an improvement over the plain loop. Check-and-retry uses 12% and 27% more
model calls. Making the check a node the patch must pass adds 7.4 points over offering the same checker as a tool,
for both models. The page gives the figures, the comparison of all hand-designed graphs, and the limits of these
results. Every number there and here comes from `results/summary/` through `scripts/build_site.py`.

The search over the graph space and the selection of a graph per task are implemented and being run; their
protocol, with the deviations recorded so far, is `refine-logs/EXPERIMENT_PLAN.md` (the review records it cites
are not part of this repository).

## Layout

| Path | Contents |
|---|---|
| `harness_graph/graph.py`, `executor.py` | graph format, static checks, the engine that runs a graph |
| `harness_graph/grammar.py` | nine design variables, the compiler from a setting to a graph, the 3,600-graph space |
| `harness_graph/search.py`, `runner.py` | five search procedures, offline replay, live episodes |
| `harness_graph/router.py` | choosing a graph per task: fit, freeze, score on other tasks |
| `harness_graph/swe/` | SWE-smith tasks in Apptainer containers, test check, grader |
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
