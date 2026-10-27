"""Deterministic verifiers and standard TRL reward functions.

TRL passes dataset columns and trusted rollout metadata to these callables.
Gold answers are never added to the model prompt or environment observations.
"""

RETRIEVAL_BONUS = 0.2
REWARD_SPEC = {
    "version": "answer_plus_retrieval_v1",
    "answer_weight": 1.0,
    "retrieval_bonus": RETRIEVAL_BONUS,
    "retrieval_source": "successful_get_neighbors",
    "evaluation_reward": "binary_answer_equality",
}


def verify_answer(submitted_answer, answer_id):
    """Only an explicitly submitted correct entity counts as answer success."""
    return submitted_answer is not None and submitted_answer == answer_id


def verify_retrieval(tool_events, answer_id):
    """Look for gold IDs in successful, actually executed neighbor responses."""
    return any(
        event["name"] == "get_neighbors"
        and event["observation"].get("ok")
        and any(entity["id"] == answer_id for entity in event["observation"].get("entities", []))
        for event in tool_events
    )


def correctness_reward(completions, answer_id, submitted_answers, **kwargs):
    """TRL reward: one float per completion, using the environment's submission."""
    return [
        float(verify_answer(answer, gold))
        for answer, gold in zip(submitted_answers, answer_id, strict=True)
    ]


def retrieval_reward(completions, answer_id, tool_events, **kwargs):
    """TRL reward: a 0/1 retrieval signal, weighted by 0.2 in GRPOConfig."""
    return [
        float(verify_retrieval(events, gold))
        for events, gold in zip(tool_events, answer_id, strict=True)
    ]


def score_episode(submitted_answer, answer_id, tool_events):
    """The same components for evaluation diagnostics and episodic returns."""
    correct = float(verify_answer(submitted_answer, answer_id))
    retrieved = verify_retrieval(tool_events, answer_id)
    bonus = RETRIEVAL_BONUS * retrieved
    return {
        "answer_correct": correct,
        "gold_entity_retrieved": retrieved,
        "retrieval_reward": bonus,
        "training_reward": correct + bonus,
    }
