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


## P1: can a generated check tell wrong patches from correct ones? (premise test, written 2026-11-21 before any generator run)

Why: stage 5, stage 6 and C1 found no gain beyond check-and-retry among graphs that reuse the supplied tests. With
hidden tests most unsolved runs end with a wrong patch submitted. The next mechanism is a check the graph writes
itself. Before any repair experiment, P1 measures the check alone. P1 uses the 188 search tasks (development
tasks); nothing held out is touched. A pass here is a premise, not a result about repair.

22. Generator G1, a frozen graph with two nodes (Qwen3.8-27B, thinking off, at most 30 model calls, hidden-test
    workspace: defect applied, files with the failing tests removed):
    - node `write_check`: an agent conversation (at most 15 commands per visit). It reads the issue and the
      repository and writes `/tmp/hg_check.py`, which must print `HG_CHECK: PROBLEM_PRESENT` and exit with status 1
      while the reported behaviour is wrong, and print `HG_CHECK: FIXED` and exit with status 0 once it is right. It
      may not change the repository. Returns `done` (finish command) or `step_cap`.
    - node `validate_check`: a program. It puts the repository back to its initial state, runs the script (120 s),
      and returns `valid` if the script exits with 1 and its last marker line is `PROBLEM_PRESENT`, else `invalid`,
      writing the reason (no script, crashed, says FIXED on the unrepaired code, timed out) into the feedback.
    - edges: `write_check` -> `validate_check` on `done` or `step_cap` (carries nothing; the script is a file);
      `validate_check` -> END on `valid`; `validate_check` -> `write_check` on `invalid` (the feedback text is added
      to the conversation), at most once.
    A (task, seed) whose last validation is not `valid` has no check. The gold fix, the tests and stored patches
    are never shown to the generator and never used to pick or filter its scripts.
23. Runs: 188 tasks x generation seeds 0 and 1 = 376 generator episodes. Pipeline debugging uses 8 selection tasks
    (first 8 of a seed-0 shuffle of `data/v2/select.jsonl`), not the 188. At most two generator versions are ever
    evaluated on the 188 tasks: G1, and one revision G2 only if G1 fails; both are reported. If G2 fails the
    generated-check line stops.
24. Trees a check is run on (program only, fresh hidden-test workspace each, 120 s, the candidate patch applied
    with test and test-runner files left out as the grader does):
    - unrepaired: the defect, no patch;
    - reference: the healthy repository;
    - correct alternatives: stored model patches the grader accepted, distinct, at most 3 per task (hidden-run
      patches first, the rest drawn with seed 0);
    - wrong, primary pool: stored model patches from hidden-test runs that the agent ended itself and the grader
      rejected with a complete grade (189 patches on 102 tasks);
    - wrong, secondary pool: other rejected hidden-run patches and up to 3 rejected visible-run patches per task
      (198 patches on 80 tasks).
    The pool is `data/p1/patches.jsonl` (`scripts/p1_patches.py`), fixed before any generator run.
    A check accepts a tree if it exits with 0 and its last marker line is `FIXED`; anything else (problem present,
    crash, time-out) is a rejection, with the reason recorded. A patch that does not apply is left out and counted.
25. Measures (tasks redrawn inside repositories, 4,000 replicates, seed 0; a task's value is the mean over its
    seeds and patches):
    - validity: share of (task, seed) with a check;
    - detection, primary: share of (task, seed, primary wrong patch) rejected; no check counts as not rejected;
    - false rejection of the reference: among (task, seed) with a check, share that reject the healthy repository;
    - false rejection of correct alternatives: among (task, seed, correct patch) with a check, share rejected.
    Secondary: detection on the secondary pool; all of the above by issue-audit label (full / partial / not
    audited), by strategy family, and by whether the wrong patch leaves target tests failing; agreement of the two
    seeds; model calls and tokens per generator episode.
26. Rule, on point estimates, intervals reported: P1 passes if detection >= 30%, false rejection of the reference
    <= 2% and false rejection of correct alternatives <= 5%. Pass -> design the repair comparison (P2: hidden tests,
    check-and-retry with and without the generated check, matched total inference budget, reviewed before any
    run). Fail -> G2 or stop, as in item 23.
27. Scope: P1 says how well this generator's checks separate stored patches of this model on these tasks. It does
    not say that enforcing the check repairs more bugs, and a pass on development tasks needs fresh tasks later.

P1 amendments after the design check (Codex: GO WITH CHANGES; all made before any generator run on the P1 tasks;
record in REVIEW_P1_DESIGN.md). Where they differ from items 22-26 above, these hold.

- Check format and verdict. The generated file defines `check()` with plain `assert` statements and begins with a
  comment block `# ISSUE -> ASSERTIONS`. A trusted runner (text in `harness_graph/gencheck.py`), not the generated
  file, decides: `fixed` if `check()` returns; `problem_present` if an `assert` written in the check file fails;
  `invalid` for anything else (import error, any other exception, an assertion raised in other code, no `check()`,
  time-out, no result). Mechanically accepted checks are not called correct specifications.
- Validation. `validate_check` copies only the check file into a fresh workspace with the defect and runs it twice;
  both runs must say `problem_present`, and the file must carry the comment block and be under 30,000 characters.
  The generator's own workspace, scratch files and repository changes are never used; the repository in the
  generator's workspace is reset before the one retry. Budgets: 15 model calls per visit, 30 in all.
- What the generator can see: the issue, the repository with the defect, and the repository's remaining tests. The
  files with the hidden grading tests are removed, in the same way on every tree a check is run on.
- Healthy-copy audit (`scripts/p1_leak_audit.py`, containers only, no outcomes) and task file
  (`scripts/p1_tasks.py`): 15 of the 188 search tasks are left out, by rules written in that script: the 12 parso
  tasks (the image holds the healthy package as an installed egg and under `build/lib` inside the repository) and 3
  funcy tasks (a backup copy `tests/test_decorators.py.orig` of their hidden test file remains). P1 uses the other
  173 tasks (`data/p1/tasks.jsonl`). Recorded, not a reason to leave out: setuptools vendors another version of
  typeguard; a few stale `.pyc` files. The same leftovers were present in every earlier hidden-test run of this
  project (12 + 3 of 188 search tasks, and 2 funcy tasks of the 232 confirmation tasks): a limit of those results.
- Patch pool (`scripts/p1_patches.py`, rebuilt on the 173 tasks: 169 primary wrong patches on 89 tasks, 164
  secondary on 68, 398 grader-accepted on 159). A patch needs a complete grade (grader v2, applied, suite finished,
  no missing designated test). Occurrences of the same diff are merged first: primary = rejected and produced by at
  least one hidden-test run the agent ended itself; a diff graded both ways is dropped and counted (none).
  Secondary = every other rejected hidden-run patch, then rejected visible-run patches up to three per task in all.
  "Correct alternatives" are grader-accepted patches, not proofs of correctness.
- Evaluation: the reference tree is run twice per check in independent fresh workspaces (primary rate = mean of the
  two; any-rejection and disagreement also reported); the unrepaired tree once more. Rejections are split into
  behavioural (an assert of the check failed) and crash or time-out. Patches are applied as the grader does (same
  file filter, `git apply`, then GNU patch with fuzz 5); a patch that does not apply is left out and counted by pool
  and reason. A run whose workspace cannot be built is tried once more, then stored as an infrastructure error.
- Estimands. For each eligible task, average over its generation seeds and patches; then average equally over
  tasks; whole tasks are redrawn inside repositories. Detection: tasks with a primary patch; no check = not
  detected. False rejection of the reference: tasks with at least one delivered check, averaged over delivered
  seeds. False rejection of accepted patches: tasks with a delivered check and an accepted patch. Also reported:
  the denominators, false rejection of the reference per generation attempt, validity overall, at the first
  validation and on the primary-pool tasks, generator calls and tokens.
- Verdict (`scripts/p1_analysis.py`): pass = detection >= 30%, reference false rejection <= 2%, accepted-patch
  false rejection <= 5% on point estimates, every expected run stored, each conditional rate resting on at least 94
  tasks. Inconclusive = thresholds met where they can be judged but runs missing or fewer than 94 tasks behind a
  conditional rate. Fail otherwise. The thresholds are engineering screens: a pass permits designing P2, it does not
  show that the true rates meet the limits, and it does not by itself authorise P2. Before P2: an explicit argument
  on cost and recovery (the generator may use up to 30 of an episode's 50 model calls; five points of repair gain
  would need roughly 40% detection at a 40% chance of repair after a rejection).
- What P1 adds over the existing reproduce-first graphs (h03, h08): a separate generation conversation, an enforced
  validation by a program, a trusted runner, and a frozen file. None of these is claimed to be an improvement
  until measured.
- Issue-audit labels are single-model judgments, used only for a secondary, exploratory split.
- Manifest: `scripts/p1_manifest.py` writes prompt, runner, script, pool and task hashes, model snapshot, images,
  budgets and the decision rule to `data/p1/manifest_G1.json` before the first episode. Pipeline debugging used the
  8 selection tasks in `data/p1/debug_ids.txt` (one seed): all 8 delivered a check at the first validation; one
  check rejected the healthy repository (it asserts a behaviour the issue does not report). The G1 prompt was not
  changed after this run.

P1, second confirmation round (Codex: three fixes required before launch; all made before any generator run on the
P1 tasks):
- the two validation runs each get their own fresh workspace;
- every P1 workspace (generation, validation, evaluation) has compiled `.pyc` files under the repository removed
  before its initial commit (the audit found bytecode of a module that reference fixes touch, in soupsieve);
- analysis: a missing or foreign record, a run made with another check than the stored one, or a patch that did not
  apply makes the verdict inconclusive (never a fail from partial data, never a pass with a silent exclusion);
- the runner counts an AssertionError raised by code in the check file, whether from an `assert` statement or an
  explicit `raise AssertionError`: a mechanical rule, not a judgment that the assertion is the right one;
- the audit records where the runner imports each package from (all nine repositories: from the repository under
  test, including the `src/` layout of typeguard); a repository imported from elsewhere would be left out;
- the manifest pins the image files by content hash, not by name.
- Debugging after these fixes: the same 8 selection tasks with seed 1, to exercise the changed code: 8 of 8 delivered
  a check, 6 at the first validation and 2 after the retry. No prompt change. In all, 16 generator episodes and 43
  evaluation runs were spent on debugging, none on the P1 tasks.

P1 result, generator G1 (2026-11-21): FAIL. 346 generator episodes, 2,425 evaluation runs, no missing or unusable
record. Validity 97.4%; detection of primary wrong patches 27.2% [19.3, 35.0] (89 tasks); false rejection of the
reference 21.3% [16.1, 26.9] (171 tasks); false rejection of grader-accepted patches 19.3% [14.2, 24.7] (159 tasks).
File: `results/summary/p1_G1.json`; verdict and wording: REVIEW_P1_G1_RESULT.md. Exploratory reading of the 72 checks
that reject the healthy repository (`results/summary/p1_G1_reference_rejections.json`, two model raters, 20 of 24
shared labels equal): in 35 the issue states an expectation the reference code does not have (22 tasks), in 31 the
expectation is the generator's own or the issue's example was altered, 5 are a defect of the check runner (the
check module was not registered in `sys.modules`; typeguard), 1 unclear.

P1, generator G2: the one permitted revision (frozen before any G2 run on the P1 tasks; Codex: GO WITH CHANGES,
applied; record in REVIEW_P1_DESIGN.md). Same 173 tasks, seeds, patch pool, budgets, validation, evaluation,
analysis and thresholds as G1. What changes, all from reading G1's reference rejections, with no correction aimed
at a particular task:
- the runner registers the check module in `sys.modules`;
- prompt: the issue's own examples unchanged (calls, values, quoting, escaping; shell commands as argument lists);
  expected values are to be looked up in the repository's remaining tests, docstrings and documentation, which win
  over the issue in details; the present output of the code is not evidence; "no longer raises" is enough only
  where the issue reports an unwanted exception; the evidence is named per assert ("issue only" is allowed, so
  corroboration is asked for, not enforced);
- the generator may decline: a file with exactly one line `# NO CHECK: <reason>` ends the episode with no check
  (no retry; counts as not detected);
- the prompt states the budget (at most 15 responses, a first version of the file by about the tenth).
Debugging, 8 selection tasks x 2 seeds, twice: round 1 (no budget sentence) 14 of 16 delivered a check, 5 at the
first validation, 9 had no file when their 15 responses ran out; 1 of 14 delivered checks rejected the healthy
repository. Then the budget sentence was added ("You have 15 commands ..."). Round 2: 16 of 16 delivered, 11 at the
first validation. After round 2 the sentence was reworded to match the implementation ("at most 15 responses, the
finish response included") and the decline rule was made strict; no run was made between that rewording and the
freeze. These are development observations, not results.
Also: evaluation records now carry the runner's hash; a file with runs made by different runners is refused. As a
separately labelled sensitivity result, all frozen G1 checks are re-evaluated with the fixed runner
(`results/summary/p1_G1_fixed_runner.json`); the G1 result above stands as the G1 result.
If G2 fails, the generated-check line stops before any repair experiment. If G2 passes, P2 still needs its own
cost-and-recovery argument and review.

P1 result, generator G2 (2026-11-21 16:12; 346 episodes, 2,431 evaluation runs, no problems in the records): FAIL on
all three thresholds. Valid check 97.7% (first validation 77.5%); detection of primary wrong patches 28.6%
[22.0, 35.7]; false rejection of the reference 18.9% [14.0, 23.8] (172 tasks); of grader-accepted patches 15.5%
[11.1, 20.1] (158 tasks); 14.2 model calls and 78k tokens per episode; the decline option was never used (0 of 346);
two seeds agree on a patch 79.3%. Sensitivity, frozen G1 checks with the corrected runner: detection 25.7%, reference
19.9%, accepted 17.7% (the runner defect does not explain G1's failure). Files: `results/summary/p1_G2.json`,
`results/summary/p1_G1_fixed_runner.json`; verdict and wording: REVIEW_P1_G2_RESULT.md.
By the rule above the generated-check line stops here: no third generator, no repair experiment (P2).

## Failure audit and the multi-agent line (2026-11-23)

Audit of the four negative results from stored episodes: `scripts/audit_failures.py` ->
`results/summary/failure_audit.json` (exploratory). Draft attributions, Codex's verdicts (all four "partly supported")
and the wording allowed: REVIEW_FAILURE_AUDIT.md.

Multi-agent graphs (several agent conversations, parallel branches): debated with Codex over three rounds, Codex proposing
and its proposal then attacked in review. Codex advanced one candidate, branching two repair conversations from a shared checkpoint against one
uninterrupted continuation, and withdrew it after the review: emulations from stored runs favoured continuing over
splitting the remaining calls at every checkpoint (follow-up tasks, checkpoint after 20 calls: 76.1% against 59.0%).
The emulation does not reproduce shared-checkpoint branches or intermediate patch correctness; it was judged
insufficient evidence to build and test branching. Status: paused before implementation; no prospective multi-agent
comparison was run. Record: REVIEW_MULTIAGENT.md.

### A1: cap-extension audit (failure-analysis follow-up; written before any A1 run)

Question. Does the 50-call cap stop repairs that more calls in the same conversation would complete? The audit found
that most unsolved check-and-retry runs end on the cap; that says where a run stopped, not whether more calls help.
- Tasks: 200 of the 900 follow-up tasks (`data/a1/tasks.jsonl`, `scripts/a1_sample.py`): 13 or 14 per repository,
  drawn with a fixed seed from sorted task ids, no outcome read. The sample is not enlarged after results are seen.
- Runs: check-and-retry (`graphs_c1/g-p0-fn-t0-vn-kcon2-c0-snon.yaml`), tests visible, Qwen3.8-27B with thinking off,
  cap 100 model calls, every other setting as in the follow-up (3,600 s limit, 65,536-token context, 16 runs in
  flight per GPU), seeds 2 and 3: 400 runs.
- Measurement: `Budget.snapshot_at = 50`. When the 51st model call is about to be made the engine stores the diff as
  it stands, without telling the agent or adding feedback; this is the state in which a 50-call run would have ended.
  It is graded with the normal rules. A run that ends before call 51 contributes its final state to both outcomes.
  The stored diff equals the final diff when nothing changed after call 50, and is then not graded twice.
- Read-out (`scripts/a1_analysis.py`): primary, descriptive: repair at the end of the 100-call run minus repair of the
  state after call 50, mean over seeds within task, tasks redrawn inside repositories (4,000 draws, seed 0), 95%
  interval; gains and losses both reported. Also: runs reaching call 51, how they end (call cap, context limit, visit
  cap, time limit), calls and tokens after call 50, runs finishing solved per block of ten calls (finishing is not the
  moment a correct patch first existed), and the stored 50-call runs of the same tasks beside the state after call 50.
- After the freeze check (REVIEW_MULTIAGENT.md, round 3; applied before the first run): the snapshot diff is read
  without leaving a file in the workspace and a failed read is an engine error (`Workspace.peek_diff`); runs that
  reach the boundary after call 50 are reported apart from runs with a completed further call, and both stay in the
  primary read-out; the time limit is read from `exec_s`, taken before the final diff and grading; the read-out also
  checks model, time limit, unique task ids and that each snapshot agrees with its run, and counts engine errors.
- No pass or fail: nothing is built on it. A clear gain supports "the 50-call cap prevented recoverable repairs in
  this setting"; it does not support branching or any multi-agent design, and it says nothing about how extra calls
  should be allocated. No gain supports "little benefit from this increase of the cap", not impossibility.
- A missing, doubled or foreign record makes the read-out incomplete; an engine error is rerun by resuming the job.
