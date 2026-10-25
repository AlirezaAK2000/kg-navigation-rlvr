# Seed 41 Kaggle results analysis

Analyzed 2026-10-25 from `kg_grpo_seed41_adapter_and_diagnostics.zip` in the project root. This report reads the exported JSON/JSONL and safetensors directly; it does not rerun the model or change the experiment.

## Main findings

The export records 30 full-training steps, seven with positive gradient norms. The near-zero scalar GRPO loss does not mean that every step had zero gradients. However, 23 of the 30 groups had identical rewards and therefore supplied no within-group reward advantage. Validation improved by two answers on a small fixed sample; this is preliminary evidence, not a reliable estimate of generalization improvement.

The main observed failures involve following the wrong relations and stopping at the wrong entity. Invalid tool arguments and redundant relation discovery occur too, but eliminating those errors alone would leave many failed episodes unresolved.

## Evaluation results

Each column below contains 20 episodes, five per hop category. Rewards are binary answer membership, so mean reward equals accuracy.

| Hop count | Baseline validation | Trained validation | Trained test |
| --- | --- | --- | --- |
| 1 | 5/5 (100%) | 5/5 (100%) | 4/5 (80%) |
| 2 | 0/5 (0%) | 0/5 (0%) | 2/5 (40%) |
| 3 | 0/5 (0%) | 1/5 (20%) | 3/5 (60%) |
| 4 | 0/5 (0%) | 1/5 (20%) | 0/5 (0%) |
| All | 5/20 (25%) | 7/20 (35%) | 9/20 (45%) |

The baseline and trained validation IDs match. Two questions changed from incorrect to correct (`q_002119`, three hops; `q_002215`, four hops), and none changed from correct to incorrect. Both baseline failures generated multiple invented tool calls in one turn and hit the turn-token limit; the trained model executed real tool interactions and submitted correct answers. Thus these two gains include improved interaction formatting, rather than establishing improved reasoning independently of formatting.

There is no baseline test evaluation in the archive, so 45% test accuracy cannot be presented as a test improvement. Five examples per hop are insufficient to conclude that three-hop questions are easier than two-hop questions.

| Behavior | Baseline validation | Trained validation | Trained test |
| --- | --- | --- | --- |
| Submitted an answer | 8/20 | 15/20 | 13/20 |
| Wrong submitted answers | 3 | 8 | 4 |
| Wrong submissions with wrong entity type* | 1 | 6 | 4 |
| Tool-budget terminations | 7 | 5 | 6 |
| Turn-token-limit terminations | 5 | 0 | 1 |
| Invalid calls / all calls | 3/113 (2.65%) | 2/143 (1.40%) | 7/137 (5.11%) |
| Average calls per episode | 5.65 | 7.15 | 6.85 |

*Types were compared using the dataset's typed entity-ID prefixes. The greater submission rate includes more wrong submissions; it is not independently evidence of better answering.

## Training signal

Full training sampled 120 trajectories: four per question across 30 distinct questions, from a selected pool of 80 questions.

| Full-training group outcome | Groups | Consequence with the recorded binary reward and beta=0 |
| --- | --- | --- |
| Four failures | 7 | Zero reward advantages |
| Four successes | 16 | Zero reward advantages |
| Mixed successes and failures | 7 | Nonzero reward advantages |

The seven mixed groups occur at steps 4, 5, 9, 11, 13, 15 and 25. All seven have positive recorded gradient norms (approximately 0.142–0.438); all other full-training steps record zero gradient norm. With one GRPO iteration and centered group advantages, a near-zero averaged policy loss can coexist with a nonzero gradient. The recorded final mean loss is approximately `1.74e-9`.

The actual sampled distribution was 16 one-hop groups, eight two-hop groups and six three-hop groups. Three mixed groups were one-hop, one was two-hop and three were three-hop. One-hop questions therefore supplied some usable reward comparisons, but were not sufficient to prevent homogeneous groups. There is no matched run without one-hop questions here to measure their causal effect.

Training success was 51/64 one-hop trajectories, 19/32 two-hop trajectories and 13/24 three-hop trajectories. These sampled training successes should not be compared directly with greedy validation accuracy: questions, decoding and distributions differ. The late run contains many easy one-hop groups; a rise in their training reward alone does not establish learning over time.

Periodic evaluation uses the same six validation questions throughout: two one-hop, two two-hop, one three-hop and one four-hop. During full training its reward is 1/6 at steps 1–10 and 2/6 at steps 11–30. All periodic successes are one-hop. The larger final validation evaluation finds two additional multi-hop successes outside that six-question subset. A larger balanced fixed evaluation subset would make progress easier to assess.

## Concrete failures

- **Premature submission:** test `q_002339` asks for the home city of a researcher's university. The model calls `get_relations(researcher_000607)`, follows `works_at` to `university_000091`, then submits the university. The gold answer is `city_000158`; it omitted the `located_in_city` hop. The same failure occurs for `q_002447` and `q_002319`.
- **Wrong route despite the right answer type:** validation `q_002141` asks for the university employing an author of a publication cited by the starting publication. The model follows `authored_by → works_at` directly, omitting `cites`, and submits `university_000015` instead of `university_000011`. Every executed call is valid. Neither a type check nor observed-argument constraints would correct this route by itself.
- **Repeated discovery:** the baseline calls `get_relations(publication_000311)` twelve times in `q_002071`. After training that exact loop disappears, but the answer remains wrong because the model takes an unrelated route and submits the starting publication.
- **Invalid relation direction:** trained validation attempts `publication_field` from field entities twice; the observed relation is `inv:publication_field`. Test has seven invalid calls, all involving inapplicable relations, including repeated `inv:authored_by` queries on projects, funders, cities and countries.
- **Invented multi-call turns:** the remaining test turn-token-limit failure (`q_002763`) generates multiple tool calls, including invented `wikipedia:author` and `author_000123`, without waiting for observations. The turn ends before any call executes; invalid-call counts alone therefore understate this failure mode.

## What this says about the queued changes

1. **Single-answer filtering and singular wording:** reasonable for making the objective explicit, but not established as the primary performance fix. Eighteen of twenty validation episodes already have one gold answer, and both baseline and trained models fail the two multi-answer validation episodes. The trained test model answers its single multi-answer episode correctly. This sample cannot isolate an ambiguity effect.
2. **Prevent redundant relation discovery:** directly motivated by recorded repeats. Successful duplicate `get_relations` calls total 24 across three baseline validation episodes, five across two trained validation episodes, and seven across six trained test episodes. Each count excludes the first successful lookup for an entity. A guard or decoding restriction needs a controlled comparison; simply rejecting a repeated call still consumes budget and does not guarantee recovery.
3. **Constrain arguments using observations:** directly relevant to wrong-direction and inapplicable-relation calls. It would not prevent valid but irrelevant routes, submitting an observed university for a city question, or continuing past a discovered answer. Full-format constraints would additionally target multi-call turns; that is broader than restricting argument values. Any constrained policy must be applied consistently to baseline, training and evaluation, including GRPO probabilities.

For the next experiment, preserve these traces, resolve the smoke evidence gap below, use a larger balanced fixed validation subset, and track mixed-reward groups and gradient norms alongside reward. Compare the queued changes separately with matching task IDs, budgets and seeds. The strongest behavioral targets in this run are selecting relations that match the whole question and submitting an entity of the requested type at the appropriate time. Increasing the training budget alone is not guaranteed to fix either behavior.

## Verification and limitations

- Every exported implementation fingerprint (18 files) and input fingerprint (10 files) matches the current local repository/data. The notebook itself is not included in those implementation fingerprints.
- Every reward in the baseline-validation, trained-validation and trained-test trajectories agrees with membership of the submitted ID in the recorded gold set. This checks bookkeeping consistency; it does not independently validate the gold sets.
- The exported adapter contains 392 FP32 tensors, all finite. All 196 LoRA B tensors contain nonzero values. Together with the positive gradient logs, this supports that training had an effect, but does not substitute for a saved before/after weight-delta check and optimizer/checkpoint verification.
- All three adapter-reload episodes reproduce the corresponding trained-validation raw turns and submitted answers exactly. This is a three-example reload check, not a complete equivalence test.
- **Smoke verification is incomplete:** `run_evidence.json` says `genuine_grpo_update_verified=false`, and both the evidence and adapter manifest have a null smoke result. The archive contains three smoke rollout groups and step-timing records, but only two smoke training-log/evaluation records, despite eight configured smoke steps. It contains no smoke failure report or verified smoke result. The current notebook gates full training/export on smoke verification, so the exported results and verification flags require reconciliation with the executed notebook/session. The archive cannot establish why execution continued or why smoke verification was absent.
- The archive contains the adapter and consolidated diagnostics, but no full optimizer/RNG checkpoints or precision-check files. It is sufficient for this behavioral analysis, not for confirming exact training resumption or completing all original smoke assertions.
- The recorded hardware is a Tesla T4; peak allocated memory is approximately 6.81 GiB. The full phase reports approximately 2 hours 18 minutes wall time, including its periodic evaluation callbacks.

Primary archive evidence: `runs/kg_grpo_seed41/{training_trajectories,rollout_metrics,trainer_metrics,evaluation_metrics,evaluation_rewards,step_metrics}.jsonl`, `evaluation/*_trajectories.jsonl`, `fixed_task_ids.json`, `run_evidence.json`, and `grpo_adapter/{training_manifest.json,adapter_model.safetensors}`.
