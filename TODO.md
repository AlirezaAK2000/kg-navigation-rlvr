# Next changes

## Implemented: single-answer experiment mode

- The training notebook defaults to a subset restricted to questions with exactly one gold answer; the complete source dataset is preserved.
- Preserve existing train/validation/test assignments; filter before deduplication and example limits.
- Use reviewed singular question wording that agrees with the single-entity submission tool.
- Apply the same selection rule and wording to baseline, training, periodic validation, final evaluation, and adapter-reload evaluation.
- Report the actual available question and unique-instance counts by split and hop category.
- Retain binary answer accuracy; private training reward also includes the one-time verified retrieval bonus. Tests cover prompt isolation, wording and split integrity.

## Implemented: next training iteration

- Eight generations and eight gradient accumulation steps: eight trajectories for one question per update.
- Enforce 20%/40%/40% actual question-group exposure for one/two/three hops, with deterministic interleaving and within-hop shuffling/cycling.
- Use `answer_correct + 0.2 * gold_entity_retrieved`, verified from successful executed neighbor responses, with separate component logs and binary evaluation accuracy.

## Pending: restrict redundant relation discovery per episode

- Track successfully inspected entity IDs independently in every rollout; reset the history between episodes and never share it across GRPO samples.
- Reject repeated `get_relations` calls for an already-inspected entity while allowing inspection of other discovered, uninspected entities.
- Count rejected duplicates as invalid attempts consuming the existing tool-call budget; retain the binary final-answer reward.
- Tell the model to reuse previously returned relation lists; preserve access to previously discovered entities.
- Apply the same restriction during baseline, training, periodic validation, final evaluation, and adapter-reload evaluation.
- Test duplicate-call rejection, valid branch switching, episode isolation, and unchanged reward semantics.
- Start with an environment execution guard. Preventing generation of invalid calls through constrained decoding is a separate possible extension, requiring GRPO-compatible probability handling.

## Pending: optional constrained sampling of tool arguments

- Limit tool argument values to choices derived from the starting input and that episode's previous tool observations.
- Track discovered entity IDs, successfully inspected entities, and the relations observed for each entity independently per rollout.
- Preserve entity/relation dependencies when constructing allowed traversal arguments; never use gold answers or reference paths to construct choices.
- Integrate constrained argument generation with Qwen's tool-call format and tokenizer boundaries, while retaining the environment's validation guards.
- Handle GRPO sampling and training probabilities consistently with the applied constraints, using compact per-token constraint state.
- Apply the same constraint mode to baseline, training, validation, test, and adapter-reload evaluation.
- Test argument legality, episode isolation, probability calculations, and genuine optimizer updates before relying on GPU training results.
- Use the current notebook's recorded failures to prioritize this work; establish benefits with controlled constrained/unconstrained comparisons rather than assuming it fixes zero rewards.

The remaining sections marked Pending are future changes; they are not part of this iteration.
