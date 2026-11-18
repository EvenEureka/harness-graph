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
| 1 Panel table | 32 graphs (graphs_panel/, seed 0: h01, h04, 30 drawn at random, 16 with the enforced check) x 188 search tasks x 1 rollout, visible tests. Second rollout if GPU time allows (preferred over more graphs). | 6,016 | done 11-18 09:08: 6,015 scored, 1 engine error (cell set aside, rerun job 1772775) |
| 2 Panel methods, replay | uniform, successive halving, Thompson, linear Thompson on the table; budgets 64..1024; 300 task splits (search on one half, score on the other); successive halving run separately per budget; no stored episode handed out twice while unused ones remain, reuse share reported. Primary number: held-out value of the recommendation. | 0 | done on 187 tasks 11-18 (kept as ..._187tasks.json); rerun on 188 tasks queued (job 1772776) |
| 3 Whole-grammar search, live | local search from the plain loop, one variable per step, paired batches of 16 tasks, acceptance on the current batch only; random next change vs change proposed from failure reasons; B = 512, 5 seeds each. Engine-error retries are not charged and are reported. | 5,120 | running since 11-18 05:46 (jobs 1770581-1770590; two chains, guided and random alternate) |
| 4 Selection | one final B = 512 replay run per panel method on all 188 search tasks (seeds fixed beforehand: 0) + every live result + h01 + h04, on the 56 select tasks x 2 rollouts. Searched graph = highest select mean; ties -> fewer model calls. If h04 wins, the primary contrast is zero by construction and is reported as such. | <= 1,800 | after 2, 3 |
| 5 Confirmation | h01, h04, searched graph on 232 confirm tasks x 2 rollouts, visible; the same three, 1 rollout, hidden. | 2,088 | after 4 |
| 6 Routing | portfolio = the graphs measured in both conditions on search and confirm tasks (the three above; six if time allows). Fit FixedBest, RuleRouter[tests_visible], FeatureRouter on search tasks, freeze before stage 5, score on confirm tasks. | <= 1,500 extra | with 5 |

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

