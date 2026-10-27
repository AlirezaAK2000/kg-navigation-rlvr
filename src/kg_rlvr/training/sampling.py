"""Deterministic hop exposure at the GRPO question-group level."""

from collections import Counter
import math
import random


def allocate(total, weights):
    """Largest-remainder allocation, with stable tie breaking."""
    denom = sum(weights.values())
    raw = {k: total * v / denom for k, v in weights.items()}
    counts = {k: int(v) for k, v in raw.items()}
    for key in sorted(raw, key=lambda k: (-(raw[k] - counts[k]), str(k)))[
        : total - sum(counts.values())
    ]:
        counts[key] += 1
    return counts


DEFAULT_HOP_EXPOSURE = {1: 0.2, 2: 0.4, 3: 0.4}


def task_instance_key(record):
    """A complete semantic query instance, shared by all its paraphrases."""
    return (
        record["start_entity"],
        tuple(record["relation_path"]),
        tuple(sorted(record["answer_ids"])),
    )


def check_stage_separation(store, groups):
    """Fail on ID, semantic-instance or public-prompt overlap between stages."""
    from ..data import prompt_key

    signatures = {}
    for name, ids in groups.items():
        signatures[name] = (
            set(ids),
            {task_instance_key(store.record(q)) for q in ids},
            {prompt_key(store.prompt(q)) for q in ids},
        )
    names = list(signatures)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            for label, a, b in zip(
                ("question IDs", "query instances", "public prompts"),
                signatures[left],
                signatures[right],
            ):
                if a & b:
                    raise ValueError(f"Training partition overlap ({label}): {left} and {right}")
    return {
        "disjoint": True,
        "checked": ["question_ids", "query_instances", "public_prompts"],
        "counts": {name: len(ids) for name, ids in groups.items()},
    }


def partition_training_tasks(
    store,
    hops,
    sft_count=200,
    hop_weights=None,
    max_grpo_examples=None,
    seed=41,
    sft_candidate_check=None,
):
    """Reserve unique SFT instances before applying the GRPO pool cap.

    Both stages use the source train split. Test and all paraphrases
    of reserved instances are excluded from GRPO.
    The optional checker executes reference tools to reject infeasible demos.
    """
    if not isinstance(sft_count, int) or sft_count < 0:
        raise ValueError("SFT example count must be a nonnegative integer")
    weights = {int(h): float(w) for h, w in (hop_weights or {h: 1 for h in hops}).items()}
    if set(weights) != set(hops) or any(not math.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError("SFT hop weights must be positive finite values for every training hop")
    candidates = store.select("train", hops, seed=seed, unique_instances=True)
    quotas = allocate(sft_count, weights)
    sft_ids, rejected = [], Counter()
    selected = Counter()
    for qid in candidates:
        hop = store.record(qid)["hop_count"]
        if selected[hop] == quotas[hop]:
            continue
        if sft_candidate_check is not None:
            try:
                sft_candidate_check(qid)
            except ValueError as error:
                rejected[str(error)] += 1
                continue
        sft_ids.append(qid)
        selected[hop] += 1
    if len(sft_ids) != sft_count:
        raise ValueError(
            f"Cannot reserve {sft_count} distinct SFT instances: requested {quotas}, available {dict(selected)}; no quotas relaxed"
        )
    reserved = {task_instance_key(store.record(q)) for q in sft_ids}
    grpo_ids = [q for q in candidates if task_instance_key(store.record(q)) not in reserved]
    if max_grpo_examples is not None:
        if max_grpo_examples < 1:
            raise ValueError("MAX_TRAIN_EXAMPLES must be positive")
        grpo_ids = grpo_ids[:max_grpo_examples]
    if not grpo_ids:
        raise ValueError("No GRPO training instances remain after reserving SFT data")
    report = check_stage_separation(store, {"sft": sft_ids, "grpo": grpo_ids})
    report.update(
        sft_quotas={str(h): count for h, count in quotas.items()},
        sft_candidate_rejections=dict(rejected),
        source_train_unique_instances=len(candidates),
        grpo_available_before_cap=len(candidates) - len(reserved),
        separation="query instances and their paraphrases; graph entities and relations may be shared",
    )
    return sft_ids, grpo_ids, report


def exposure_schedule(store, task_ids, group_count, hop_weights, seed=41):
    """Interleave weighted hops; shuffle questions within each hop, cycling as needed.

    This sequence has a stable prefix when a resumed run increases its step
    target. With 20/40/40 weights every five groups contain 1/2/2 questions;
    shorter runs use the nearest interleaved integer allocation. Each returned
    ID denotes ONE question group; TRL supplies its generation repetitions.
    """
    weights = {int(h): float(w) for h, w in hop_weights.items()}
    if (
        not weights
        or set(weights) - {1, 2, 3, 4}
        or any(not math.isfinite(w) or w <= 0 for w in weights.values())
    ):
        raise ValueError("Hop exposure requires positive finite weights for hops 1–4")
    if group_count < 1:
        raise ValueError("At least one training question group is required")
    buckets = {h: [] for h in sorted(weights)}
    for task_id in dict.fromkeys(task_ids):
        hop = store.record(task_id)["hop_count"]
        if hop in buckets:
            buckets[hop].append(task_id)
    missing = [h for h, ids in buckets.items() if not ids]
    if missing:
        raise ValueError(f"No selected training questions for requested exposure hops: {missing}")
    rng = random.Random(seed)
    for ids in buckets.values():
        rng.shuffle(ids)
    counts, cursors, result = Counter(), Counter(), []
    total = sum(weights.values())
    for position in range(group_count):
        hop = max(sorted(weights), key=lambda h: (position + 1) * weights[h] / total - counts[h])
        if cursors[hop] == len(buckets[hop]):
            rng.shuffle(buckets[hop])
            cursors[hop] = 0
        result.append(buckets[hop][cursors[hop]])
        cursors[hop] += 1
        counts[hop] += 1
    return result
