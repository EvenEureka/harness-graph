# Experiment plan v3 — search and pre-run selection over harness graphs (2026-11-17)

Supersedes the Part 2 (NL2Repo, multi-agent) plan, which was parked on 2026-11-16 (see REVIEW_PART2.md, REVIEW_PART2_R2.md).
Reviews: REVIEW_V3_FOUR_WORDS.md (design), REVIEW_V3_PROTOCOL.md (this protocol; Codex thread 01a1ea38, GO WITH CHANGES).

**Questions.** (1) Does budgeted search over the graph grammar return a graph that beats check-and-retry (h04) on
unseen tasks? (2) Which search method recommends better graphs at equal budget? (3) Does choosing the graph from
pre-run observables beat one fixed graph?

**Fixed.** Model Qwen3.8-27B, thinking off (premise run: thinking on gave no detectable gain and ran 6-7x slower).
SWE-smith bug-fix tasks, tests visible to the agent (primary) and hidden (secondary). Every graph: at most 50 model
calls; 3,600 s soft deadline (checked before each model call; episodes that ran past it are reported).
Splits: data/v2/search.jsonl 188, select.jsonl 56, confirm.jsonl 232 (112 old repositories + 120 from three
untouched ones). The 108 tasks used before 2026-11-16 are search data only.

| Stage | What runs | Episodes | Status |
|---|---|---|---|
| 1 Panel table | 32 graphs (graphs_panel/, seed 0: h01, h04, 30 drawn at random, 16 with the enforced check) x 188 search tasks x 1 rollout, visible tests. Second rollout if GPU time allows (preferred over more graphs). | 6,016 | done 11-18 09:08: 6,016 scored after the one engine-error cell was rerun (job 1772775) |
| 2 Panel methods, replay | uniform, successive halving, Thompson, linear Thompson on the table; budgets 64..1024; 300 task splits (search on one half, score on the other); successive halving run separately per budget; no stored episode handed out twice while unused ones remain, reuse share reported. Primary number: held-out value of the recommendation. | 0 | done on 187 tasks 11-18 (kept as ..._187tasks.json) and on 188 tasks 11-18 14:37 |
| 3 Whole-grammar search, live | local search from the plain loop, one variable per step, paired batches of 16 tasks, acceptance on the current batch only; random next change vs change proposed from failure reasons; B = 512, 5 seeds each. Engine-error retries are not charged and are reported. | 5,120 | done 11-19 11:34: ten runs, 512 scored episodes each, 0 engine errors, 0 past the deadline (results/summary/live_search.json) |
| 4 Selection | one final B = 512 replay run per panel method on all 188 search tasks (seeds fixed beforehand: 0) + every live result + h01 + h04, on the 56 select tasks x 2 rollouts. Searched graph = highest select mean; ties -> fewer model calls. If h04 wins, the primary contrast is zero by construction and is reported as such. | <= 1,800 | done 11-19 15:01: 14 graphs, 1,568 scored episodes, 0 engine errors; searched graph g-p0-fn-t0-vn-kref2-c0-scom (results/summary/selection.json) |
| 5 Confirmation | h01, h04, searched graph on 232 confirm tasks x 2 rollouts, visible; the same three, 1 rollout, hidden. | 2,088 | done 11-19 20:15: 2,088 scored episodes, 0 engine errors; primary +0.6 [-1.3, +2.8] (results/summary/confirmation.json) |
| 6 Routing | portfolio = the graphs measured in both conditions on search and confirm tasks (the three above; six if time allows). Fit FixedBest, RuleRouter[tests_visible], FeatureRouter on search tasks, freeze before stage 5, score on confirm tasks. | <= 1,500 extra | done 11-19 20:15: policies frozen in 8f0fd57, scored on 464 units (results/summary/routing_confirm.json) |

**Analysis (frozen).** Rollouts averaged within task; paired task bootstrap preserving the repository mix; 95%
intervals. Primary contrast: searched graph - h04, visible tests. Smallest effect of interest 5 points. "Better" if
the lower bound > 0; "no gain of 5+" if the upper bound < 5 (both can hold); otherwise inconclusive. Detectable
with 80% power: roughly 4.5-9 points. Secondary: h04 - h01; each contrast within old and within new repositories
(detectable 6-13 points; three new repositories support claims about those three only); cost of all three graphs;
FeatureRouter and RuleRouter each minus FixedBest, and FeatureRouter - RuleRouter.

**Wording limits.** Replay results describe a finite table: "best" is the highest observed mean and is noisy; split
intervals are not sampling intervals. Guided vs random local search tests the whole hand-written remedy table (which
proposes the enforced check first for wrong submissions) and uses grader labels during search; it does not show that
diagnosis as such causes a gain. A null result concerns the selected graph and the tested budgets, not the grammar.

**Deviations and clarifications (recorded 2026-11-18; judged in REVIEW_STAGE12_RESULTS.md).**
1. Stage 1: 1,339 episodes were completed at 28 concurrent episodes per GPU, which saturated the model server's cache.
   58 of them ended on the time limit; these were moved to results/raw/search_table/discarded_wall_time.jsonl and run
   again at 16. The other 1,281 were kept. Owed: rerun an outcome-independent stratified sample of the kept cells at 16.
2. Stage 1: one episode crashed the engine (the model put a null byte in a shell command). The record is in
   results/raw/search_table/discarded_engine_error.jsonl; the cell is rerun. From commit 765488f such a command
   returns an error message to the agent. The stage 3 jobs started before the fix count such a crash as an engine
   error (retried, not charged); jobs started after it do not crash.
3. Replay hands out unused stored episodes first within a graph-task cell, not across the table; a replay budget counts
   table draws, not fresh episodes.
4. Stage 4's panel candidates come from one standalone B = 512, seed 0 call per method on all 188 tasks, not from the
   512 checkpoint of a longer run.
5. Seven table episodes ran past 3,600 s; all ended on the model-call limit and are kept.

**Stage 4 execution (frozen 2026-11-19, after the searches ended and before any episode on the selection tasks;
reviewed in REVIEW_STAGE4_EXECUTION.md).**
6. Candidates: graphs_select/candidates.json, 14 graphs after merging (h01; h04, also the linear Thompson pick; the
   uniform pick; one graph picked by both successive halving and Thompson; ten live recommendations, all different).
   14 x 56 x 2 = 1,568 episodes. No candidate, seed or rerun is added after outcomes are seen.
7. Run: rollout 0 of every cell in one GPU job and rollout 1 in another, 16 concurrent episodes per GPU, same server
   settings as the table. A scored episode is never rerun, whatever its exit reason; an episode that ended in an
   engine error is rerun by resuming the same job, the failed record stays in the file and the count is reported.
8. Selection: scripts/select_graph.py. Exactly one scored episode per cell of the expected model, else it stops.
   Highest mean; ties -> fewer model calls per episode -> first graph name in alphabetical order.
9. Question (2) for the two live methods is answered on the selection tasks by one designated contrast: per task, the
   mean resolve rate over the five graphs a method recommended (seeds 0-4, one vote per run even if two runs name
   the same graph), guided minus random; paired bootstrap over tasks inside repositories, 4,000 replicates, seed 0,
   95% percentile interval. Wording: "for these five seeds per method"; the interval does not cover rerunning the
   searches with new seeds. The first step of a guided run has no failure reasons yet and is drawn like a random
   step, so run k of both methods starts with the same candidate; the contrast compares the two complete procedures
   and is not attributed to diagnosis. Every other comparison on the selection tasks (each candidate against h04 or
   h01, panel picks, the maximum over candidates) is descriptive; the maximum is not a simultaneous bound. Panel
   replay budgets count stored draws and are not equated with fresh live episodes.

## Stage 5-6 execution (frozen 2026-11-19 after stage 4, before any confirmation task was run)

Checked by Codex (thread 01a1f4aa; record in REVIEW_STAGE4_RESULTS.md): go with the changes below, all made first.

10. **Portfolio.** Three graphs: h01 `g-p0-fn-t0-vn-k0-c0-snon`, h04 `g-p0-fn-t0-vn-kcon2-c0-snon`, searched
    `g-p0-fn-t0-vn-kref2-c0-scom`; YAML in `graphs_confirm/` (identical to the files used in stage 4). Six graphs
    are not attempted.
11. **Stage 6, search side.** The three graphs x 188 search tasks x 1 rollout with hidden tests, two GPU jobs
    (`data/v2/search_a.jsonl`, `search_b.jsonl`), 16 concurrent episodes. The visible-test fit data are the rows of
    the three graphs in the stage-1 table (1 rollout; they include episodes run at 28 concurrent and replaced
    time-limit episodes, a qualification that stays). `scripts/route_eval.py fit` with `--visible-seeds 0
    --hidden-seeds 0` must find exactly one scored record in each of 1,128 cells (376 task-condition units).
    Policies: FixedBest, RuleRouter[tests_visible], FeatureRouter (ridge 3, the features in
    `harness_graph/router.py`, ties as coded), cost weight 0. The two conditions weigh equally after rollouts are
    averaged inside a task and condition. `results/summary/router_frozen.json` records the hashes of its inputs
    and the code version; it is committed before stage 5 starts and never refitted.
12. **Stage 5 schedule.** 232 confirmation tasks; visible rollout 0 in one GPU job and visible rollout 1 in
    another, then hidden rollout 0 as two jobs on `data/v2/confirm_a.jsonl` / `confirm_b.jsonl` (116 + 116,
    alternating inside repositories). At most two GPUs, 16 concurrent episodes, `GRAPHS=graphs_confirm`, thinking
    off, 50 model calls. Engine errors are rerun by resuming the same job; a scored episode is never rerun. A
    container image that cannot be pulled is an infrastructure failure: retry unchanged; if it stays unavailable,
    stop and write a protocol amendment (no task is dropped or replaced silently). Scored episodes that ended on
    the deadline or ran past it are kept and counted.
13. **Completeness.** Before scoring, `harness_graph.analysis.audit_cells` reads the stored records before
    duplicates are merged: 2,088 cells (464 task-condition units), exactly one scored record of the model in each,
    no scored record outside them. `--allow-incomplete` is not used.
14. **Analysis.** `scripts/confirm_analysis.py`, written and tested before any confirmation episode existed
    (`scripts/contrasts.py` is not used: its bootstrap is not stratified). Visible rollouts 0 and 1 are averaged
    inside a task; hidden uses rollout 0; a task is redrawn with both conditions together, inside its repository;
    4,000 replicates, seed 0, percentile 95% intervals. Primary: searched - h04, visible, all 232 tasks, each task
    once. Verdicts reported independently: lower bound > 0; upper bound < 0.05. Subgroups: 112 tasks of the eight
    repositories that also supplied search tasks (funcy, typeguard, sqlparse, arrow, soupsieve, pypika, glom,
    deepdiff) and 120 tasks of the three that did not (marshmallow, oauthlib, jinja).
15. **Routing score.** `scripts/route_eval.py score` with `--visible-seeds 0:1 --hidden-seeds 0`, 4,000 replicates,
    seed 0, on the same files as item 14.

## Stage 5-6: done 2026-11-19 20:15; deviations and additions

- 2,088 episodes, one scored record per cell, no engine error, no deadline overrun, nothing rerun. Results:
  `results/summary/confirmation.json`, `results/summary/routing_confirm.json`; Codex verdicts in
  `REVIEW_STAGE56_RESULTS.md`.
- Deviation from item 12: the hidden job on `confirm_a` started 5 minutes before visible rollout 1 ended (each hidden
  job was held behind one visible job). Two GPU jobs at any time; the episodes are independent of the order.
- Exploratory, added after the results were seen: `scripts/confirm_interaction.py` (a contrast with visible tests minus
  the same contrast with hidden tests) -> `results/summary/confirmation_exploratory.json`. Not a frozen contrast.


## C1: one confirmatory test of an exploratory subgroup observation (written 2026-11-20, before any C1 task was run)

Origin. After stages 5-6, an exploratory split by injection strategy (`scripts/spec_gap.py`, not pre-specified, four
overlapping groups looked at, intervals pointwise and unadjusted) showed the selected graph ahead of check-and-retry on
combine-strategy tasks with visible tests: +8.1 points on 31 confirmation tasks (five tasks better, none worse, in three
repositories), 0 on 9 selection tasks, +6.2 [2.5, 11.3] pooled. On single-function-strategy tasks (209) the difference
was 0.0 [-1.7, 1.7]. C1 tests this one observation on fresh tasks. Codex review: `REVIEW_NEXT_PHASE.md`.

16. **Hypothesis.** On combine-strategy tasks with visible tests, the selected graph `g-p0-fn-t0-vn-kref2-c0-scom`
    repairs more tasks than check-and-retry `g-p0-fn-t0-vn-kcon2-c0-snon`. The graphs are the YAML files of
    `graphs_confirm/`, unchanged (their hashes are recorded by the analysis). Nothing is searched, tuned or re-selected.
17. **Population and pool** (`scripts/make_c1_pool.py`; no rule looks at an agent outcome). SWE-smith tasks with
    strategy exactly `combine_module` or `combine_file`, an issue text, gradable, at least one failing test, every
    listed test (failing and passing) in a test file, from repositories whose name (any revision) appears in no task
    file of this project and that have at least 60 such tasks. The repositories are put in a random order (seed 0)
    and probed in that order by one job, one repository at a time. A repository is accepted if (1) its image can be
    pulled and a workspace set up, (2) the full suite of the healthy repository finishes (tests may fail) in under
    120 s (unrounded) with nothing else running in the probe job, (3) at least 40 of up to 90 randomly drawn tasks pass
    both grader controls: the reference fix applies, does not time out and resolves; on the unchanged repository the
    run does not time out, the defect applies and at least one failing test is observed to fail or error. An accepted
    repository gives min(60, passing) randomly drawn tasks; repositories are added in the random order until 900 tasks;
    the last one is cut at random. An exception inside a control is retried once for that task and then counts as a
    failed control; a repository that fails step (1) is probed once more with the same candidates; the probe log is
    append-only. The inputs of the random draws (dataset file hashes, exclusion list, repository order, candidate ids,
    NumPy version, rules) are written to `data/c1/pool_inputs.json` before the first probe and checked by every later
    step; the registry digest of every image is recorded.
    The pool is therefore a repository-capped sample of computationally tractable, grader-validated combine-strategy
    tasks from unused repositories with enough such tasks; its strategy proportions are whatever this sampling gives
    and are reported. A suite under 120 s alone does not guarantee that a check finishes inside its 300 s limit under
    16 concurrent episodes; such runs are kept and counted.
    Sample size: the target is 900 tasks; feasibility may reduce the sample to at least 600 from at least 10
    repositories (below that C1 is not run, and that is reported). Sample size is determined before agent outcomes
    are available. The fallback reduces precision and power; inconclusive results remain possible. (Planning
    approximation with variance 0.10 of the task-averaged paired difference: power for a three-point effect about 81%
    at 900 tasks and 64% at 600.)
18. **Runs.** Qwen3.8-27B, thinking off, 50 model calls, deadline 3,600 s, 16 concurrent episodes per GPU, at most two
    GPU jobs, visible tests. Job A: rollout 0 of the selected graph and check-and-retry, their episodes interleaved.
    Job B: rollout 1 of the same two graphs. Job C: rollout 0 of the plain loop, started after job A (secondary; it
    runs later than the episodes it is compared with). Engine errors are rerun by resuming the same job; a scored
    episode is never rerun; an image that cannot be pulled at run time is retried unchanged, and a task is never
    dropped or replaced silently. If required cells cannot be completed there is no confirmatory verdict: no
    complete-case decision. Episodes that end on the deadline or run past it are kept and counted. No interim analysis
    of any contrast: until every cell is complete only counts of stored records and engine errors are looked at.
    Before the first episode `data/c1/manifest.json` records the hashes of the pool file, the three graph files and
    the analysis script, the code version, the model snapshot and the image digests.
19. **Completeness.** `audit_cells`: exactly one scored record per (graph, task, rollout) cell (two rollouts for the
    selected graph and check-and-retry, one for the plain loop), none outside.
20. **Analysis** (`scripts/c1_analysis.py`, written and tested before the first C1 episode). Rollouts are averaged
    inside a task. Primary contrast: selected - check-and-retry, all C1 tasks, each task once; tasks redrawn inside
    repositories, 4,000 replicates, seed 0, 95% percentile interval. Verdicts, reported independently:
    lower bound > 0 -> the selected graph is better on the sampled repository mixture under these eligibility and
    execution rules; lower bound > 0.05 -> a gain above five points; upper bound < 0.05 -> a five-point gain is
    excluded; otherwise inconclusive about five points. If the lower bound is not above 0 this line of work stops;
    that is an operational stop, not a claim of no effect.
    Secondary, descriptive: check-and-retry rollout 0 - plain loop rollout 0, task by task (same bootstrap); model
    calls and tokens of the primary pair (equal call caps are not matched cost); the primary contrast by strategy and
    by repository; the primary contrast with repositories redrawn with replacement and all tasks of a drawn repository
    kept (task-weighted mean, 4,000 replicates, seed 0); the primary contrast with each repository left out in turn.
21. **Scope of a positive result.** It supports this fixed graph on combine-strategy tasks of the sampled repositories
    with visible tests. It is not evidence for routing, for other strategies, or for hidden tests. Whatever the
    outcome, the sequence (stages 1-6, the exploratory split, C1) is reported.

Amendments to the design first reviewed (all made before any C1 probe or episode; Codex check GO WITH CHANGES,
`REVIEW_NEXT_PHASE.md`): repositories need 60 instead of 120 population tasks and give up to 60 tasks each, because
only 18 unused repositories reach 120 and several of those are too heavy; fallback to at least 600 tasks; the plain
loop gets one rollout instead of two; stricter negative control, append-only retries, pinned inputs, image digests,
exclusion by repository name at any revision.

Run-layout change, 2026-11-20 22:42 (made while jobs A and B were running, with only record counts and engine
errors looked at): job C (plain loop, rollout 0) was split into two jobs over the odd and even lines of
`data/c1/confirm.jsonl` (`confirm_a.jsonl`, `confirm_b.jsonl`, 450 tasks each), one behind job A and one behind
job B, so that both GPUs are used. Same graph file, same settings, same cells; the analysis reads four files and
still audits every cell against the full task file. Also recorded: the task file was committed about eight minutes
after the first episode started; its hash was written to `data/c1/manifest.json` before the first episode and
matches the committed file.

C1 result, 2026-11-21: all 4,500 cells scored, no engine errors. selected - h04 = +0.2 points [-1.3, +1.7]
(80 better / 75 worse of 900): superiority not established, five-point gain excluded. h04 - h01 (rollout 0) =
+24.2 points [20.9, 27.3]. By the rule in item 20 this line stops. 15 episodes ran past the deadline (9 h04,
6 selected) and were kept. Disclosed: four per-episode outcomes were seen by accident in the first health check;
nothing was changed afterwards. File: `results/summary/c1.json`.

