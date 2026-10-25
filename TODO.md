# Next changes

## Implemented: single-answer experiment mode

- The training notebook defaults to a subset restricted to questions with exactly one gold answer; the complete source dataset is preserved.
- Preserve train/test assignments; filter before deduplication and example limits.
- Use reviewed singular question wording that agrees with the single-entity submission tool.
- Apply the same selection rule and wording to baseline, training, periodic test monitoring, final evaluation, and adapter-reload evaluation.
- Report the actual available question and unique-instance counts by split and hop category.
- Retain binary answer accuracy; private training reward also includes the one-time verified retrieval bonus. Tests cover prompt isolation, wording and split integrity.

## Implemented: next training iteration

- Eight generations and eight gradient accumulation steps: eight trajectories for one question per update.
- Enforce 20%/40%/40% actual question-group exposure for one/two/three hops, with deterministic interleaving and within-hop shuffling/cycling.
- Use `answer_correct + 0.2 * gold_entity_retrieved`, verified from successful executed neighbor responses, with separate component logs and binary evaluation accuracy.

## Implemented: supervised warm start

- Reserve 200 unique train-split SFT instances with 40/80/80 one/two/three-hop demonstrations; exclude complete instances and all paraphrases from the GRPO pool before its cap.
- Check IDs, semantic-instance signatures and public prompts for overlap across SFT, GRPO and test; both GRPO phases use the remaining pool.
- Execute reference tool calls through the real graph environment and supervise only assistant action/EOS tokens using the native runtime codec and limits.
- Evaluate vanilla and SFT-only models before GRPO; export SFT losses, a verified adapter and checkpoint evidence separately.
- Require verified SFT before SFT + GRPO; independently initialize GRPO only and compare all four variants on identical test IDs. Graph entities and facts remain shared.

## Implemented: train/test and four-variant comparison

- Merge validation into test while preserving all question IDs, contents and train rows; retain migration provenance and regeneration checks.
- Select test IDs once for vanilla, SFT, SFT + GRPO and GRPO only; retain separate SFT and GRPO training instances.
- Retain train/test only; run independent GRPO only with the same GRPO pool and budgets as SFT + GRPO.
- Save each variant’s performance, raw and formatted trajectories, and RL returns/diagnostics before proceeding. The final cell only ZIPs the already saved files.

## Pending: restrict redundant relation discovery per episode

- Track successfully inspected entity IDs independently in every rollout; reset the history between episodes and never share it across GRPO samples.
- Reject repeated `get_relations` calls for an already-inspected entity while allowing inspection of other discovered, uninspected entities.
- Count rejected duplicates as invalid attempts consuming the existing tool-call budget; retain the binary final-answer reward.
- Tell the model to reuse previously returned relation lists; preserve access to previously discovered entities.
- Apply the same restriction during baseline, training, periodic test monitoring, final evaluation, and adapter-reload evaluation.
- Test duplicate-call rejection, valid branch switching, episode isolation, and unchanged reward semantics.
- Start with an environment execution guard. Preventing generation of invalid calls through constrained decoding is a separate possible extension, requiring GRPO-compatible probability handling.

## Pending: optional constrained sampling of tool arguments

- Limit tool argument values to choices derived from the starting input and that episode's previous tool observations.
- Track discovered entity IDs, successfully inspected entities, and the relations observed for each entity independently per rollout.
- Preserve entity/relation dependencies when constructing allowed traversal arguments; never use gold answers or reference paths to construct choices.
- Integrate constrained argument generation with Qwen's tool-call format and tokenizer boundaries, while retaining the environment's validation guards.
- Handle GRPO sampling and training probabilities consistently with the applied constraints, using compact per-token constraint state.
- Apply the same constraint mode to baseline, training, test, and adapter-reload evaluation.
- Test argument legality, episode isolation, probability calculations, and genuine optimizer updates before relying on GPU training results.
- Use the current notebook's recorded failures to prioritize this work; establish benefits with controlled constrained/unconstrained comparisons rather than assuming it fixes zero rewards.

## Pending: evaluate compact ontology context

- Test a public schema summary containing the eight entity types, sixteen forward relations, source/destination types, brief relation meanings and an explanation of inverse traversal.
- Keep entity instances, graph facts and question-specific gold answers or paths outside the schema summary; retain `get_relations` for entity-specific discovery.
- Apply the same schema context to vanilla evaluation, SFT demonstrations, SFT + GRPO and GRPO only; recheck prompt and episode token budgets.
- Compare against the current context on identical test IDs, tracking answer accuracy, wrong-type submissions, relation-direction errors and token use. Keep this optional until measured results support it.

The remaining sections marked Pending are future changes; they are not part of this iteration.
