# Seed 41 v3 results analysis — before SFT

Analyzed 2026-10-25 from `kg_grpo_seed41_v3_adapter_and_diagnostics.zip` in the project root. This analysis reads the exported JSON/JSONL and replays recorded tool calls against the local graph. It does not rerun model inference or modify the training implementation.

## Experiment identity

This is a **GRPO-only, pre-SFT run**. All 19 exported implementation fingerprints match commit `6b0d7a8`; all ten input fingerprints match the local source dataset. The manifest contains no supervised warm-start record. The run already includes the previous iteration's single-answer filtering and singular wording, answer-plus-retrieval reward, eight generations, gradient accumulation of eight, and 20/40/40 exposure to one-, two- and three-hop questions.

The model is `Qwen/Qwen3-1.7B` at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, trained with NF4 QLoRA, rank 8, alpha 16 and learning rate `5e-6`. The episode budget is 12 calls, 128 model tokens per turn and 1,536 completion tokens including observations. Training comprises **one smoke optimizer step followed by 30 full optimizer steps**. The selected training pool contains 933 distinct question instances; the full phase actually uses 30 distinct questions and samples eight trajectories per question.

## Evaluation results

Evaluation reward is binary correctness of the submitted entity. The retrieval bonus is reported separately and is not included in accuracy.

| Hop count | Baseline validation | Trained validation | Trained test |
| --- | --- | --- | --- |
| 1 | 2/8 (25%) | 6/8 (75%) | 8/8 (100%) |
| 2 | 0/8 (0%) | 2/8 (25%) | 4/8 (50%) |
| 3 | 1/7 (14.3%) | 5/7 (71.4%) | 1/7 (14.3%) |
| 4 | 0/7 (0%) | 0/7 (0%) | 0/7 (0%) |
| All | **3/30 (10%)** | **13/30 (43.3%)** | **13/30 (43.3%)** |

The validation evaluations use identical task IDs and questions. Ten answers change from incorrect to correct, with no correct-to-incorrect changes. Nine improved episodes initially hit the turn-token limit, and the tenth initially failed with multiple tool calls in one turn. Thus the gains involve recovery from interaction-format failures; they do not isolate an improvement in reasoning with formatting held constant.

For example, baseline `q_003530` emits a sequence of invented calls in a single 128-token turn, using `pub:author` and an invented publication ID. No call executes before truncation. After training it executes one call at a time, follows the actual `inv:authored_by` relation and submits the correct publication.

There is no baseline test evaluation, so 43.3% test accuracy cannot establish a test improvement. The three-hop validation/test gap is five versus one correct answers out of seven; these small samples do not support a reliable ranking of hop difficulty. Four-hop questions are outside the training hop range and all fail in both final evaluation splits.

## What is still failing

| Behavior | Baseline validation | Trained validation | Trained test |
| --- | --- | --- | --- |
| Submitted an answer | 8/30 | 24/30 | 24/30 |
| Wrong submitted answers | 5 | 11 | 11 |
| Wrong submissions with the wrong entity type* | 2 | 8 | 9 |
| Turn-token-limit terminations | 19 | 0 | 0 |
| Multiple-call terminations | 3 | 0 | 0 |
| Tool-budget terminations | 0 | 6 | 6 |
| Invalid calls / all calls | 4/39 | 25/192 | 7/177 |
| Average calls per episode | 1.30 | 6.40 | 5.90 |

*Compared using the dataset's typed entity-ID prefixes. A higher submission rate includes more wrong answers and is not independently evidence of better answering. Invalid-call rates omit unexecuted token-limit failures, so comparing those rates alone would misleadingly favor the baseline.

The dominant remaining errors are semantic:

- **Stopping too early:** test `q_002339` asks for the city of a researcher's university. The model follows `works_at` and submits `university_000091`, omitting `located_in_city` to `city_000158`. Similar failures submit funders for city questions and cities for country questions.
- **Skipping a required relation:** validation `q_002141` asks for the university of an author of a *cited* publication. The model follows `authored_by → works_at` directly, omitting `cites`, and submits `university_000015` rather than `university_000011`. Its calls and answer type are valid, but the route answers a different question.
- **Valid but irrelevant exploration:** several budget failures follow `authored_by → specializes_in → subfield_of` when the question requires an employment or location route. Increasing the call budget would not directly correct this choice.
- **Confusing relation IDs with entity IDs:** validation `q_003550` and `q_003512` pass the string `investigates` as an entity ID after receiving a relation list. These two episodes account for 22 undiscovered-entity errors. The remaining three validation errors are inapplicable relation operations. Test has two undiscovered-entity errors and five inapplicable-relation errors.

Successful repeated `get_relations` lookups contribute three redundant calls across three final validation episodes and eight across five test episodes. They waste budget, but the wrong routes and premature submissions would remain even if those repeats were prevented.

In both final evaluation splits, every episode that retrieves a gold entity also submits a correct answer. The failed evaluation episodes never retrieve their gold entity. This places the observed evaluation bottleneck before correct retrieval, although overshooting a retrieved answer does occur during sampled training.

## Training signal and the retrieval bonus

Full training contains 30 groups and 240 trajectories. The measured group distribution exactly matches the requested exposure: **6 one-hop, 12 two-hop and 12 three-hop questions**.

| Within-group training rewards | Groups | Reward advantages |
| --- | --- | --- |
| Eight rewards of 1.2 | 15 | Zero |
| Eight rewards of 0 | 11 | Zero |
| Mixed rewards | 4 | Nonzero |

Only steps **1, 2, 6 and 26** log positive gradient norms, approximately `0.196`, `0.071`, `0.143` and `0.117`. The other 26 steps have zero logged gradient norm. All 30 optimizer steps still execute; optimizer momentum can continue moving weights even when the current gradient is zero. A near-zero scalar GRPO loss is not itself evidence of failed optimization: the mean logged training loss is about `5.71e-9`, despite those nonzero gradients.

All 48 one-hop trajectories succeed, so none of the six one-hop groups supplies a reward difference. Two-hop success is 52/96 and three-hop success is 39/96, but many questions are either universally solved or universally failed within their group. Overall training success therefore hides the limited number of useful group comparisons.

The retrieval bonus **creates no additional mixed-reward groups** in this run. Six trajectories retrieve gold but fail to submit it correctly, all at step 6. Every trajectory in that group retrieves the answer: two receive 1.2 and six receive 0.2. The bonus adds the same constant to all eight rewards, leaving centered advantages unchanged. In every other group, retrieval coincides with correctness, so the combined reward simply scales binary reward by 1.2; group standardization largely cancels that scaling, apart from the normalization epsilon. The reward is recorded correctly, but this sample supplies no evidence that the bonus improves the GRPO learning signal.

One step-6 failure illustrates overshooting: the question asks which organization funds a publication's project. The model reaches `funder_000040`, continues through its city and country, then submits `country_000027`. This is a useful target for demonstrations teaching when to stop.

Periodic evaluation uses the same six questions: two one-hop, two two-hop, one three-hop and one four-hop. The baseline solves one. After the single smoke update it solves two; **the same two remain correct at every full-training evaluation from step 1 through 30**. This subset shows no further accuracy improvement during the full phase. Because there is no 30-question evaluation immediately after smoke, the archive cannot tell how much of the final 30-question gain occurred during smoke versus full training.

## Verification

- `smoke_verified.json` records a finite optimizer step, 196 changed LoRA tensors and a maximum parameter delta of approximately `5e-6`. This resolves the missing smoke-verification evidence in the earlier archive. The ZIP does not include the referenced optimizer checkpoint or before/after weight snapshots, so this is recorded verification rather than an independently repeated weight-delta check.
- Recomputed binary and combined rewards for all **341** exported training and final/reload evaluation trajectories agree with their submitted answers and successful neighbor observations.
- Replayed all **1,609 recorded tool events** against the matching local graph; every observation and submitted answer matches. Independently evaluated the relation programs for all **90 distinct task IDs** in those trajectories; all gold answer sets match the graph results.
- All **186 periodic per-question binary rewards** agree with their submitted-answer membership in the gold set.
- All three reloaded-adapter episodes exactly reproduce their corresponding final-validation raw assistant turns and submitted answers.
- The full phase reports 10,746 seconds, approximately **2 hours 59 minutes**, including periodic evaluation callbacks. Recorded peak allocated GPU memory is approximately **6.68 GiB** on a Tesla T4; peak reserved memory is approximately 9.27 GiB.

## Implications for the next run

The observations support trying the implemented SFT warm-start: verified demonstrations can teach waiting for tool outputs, distinguishing entity/relation IDs, following every clause of a multi-hop question, and submitting at the requested entity type. They do not establish that SFT will improve held-out performance or eliminate homogeneous GRPO groups. Easier post-SFT questions may instead become uniformly successful.

Measure **vanilla → SFT-only → SFT+GRPO** on the same validation IDs, and retain the separate final test evaluation. Track nonzero-gradient steps and mixed-reward groups alongside accuracy. The existing 200-example SFT reservation leaves a 733-question GRPO pool; a matched GRPO-only control should retain that reservation with `RUN_SFT=False`, rather than train on this archive's full 933-question pool. A larger fixed periodic validation subset would make multi-hop progress easier to detect than six questions.

The earlier archive's 35% validation and 45% test scores are not controlled comparisons with this run: only four validation IDs and five test IDs overlap, and sample sizes, wording, filtering, exposure, reward and generation count changed. The defensible measured improvement here is **10% to 43.3% on this run's paired validation set**.

Primary evidence: `runs/kg_grpo_seed41_v3/{experiment_config,fixed_task_ids,smoke_verified,run_evidence}.json`, `{training_trajectories,rollout_metrics,trainer_metrics,evaluation_metrics,evaluation_rewards}.jsonl`, `evaluation/*_trajectories.jsonl`, `selected_dataset/rl/*.jsonl`, and `grpo_adapter/training_manifest.json` inside the archive.
