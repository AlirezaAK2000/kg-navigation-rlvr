"""Deterministic hop exposure at the GRPO question-group level."""

from collections import Counter
import math
import random


DEFAULT_HOP_EXPOSURE = {1: 0.2, 2: 0.4, 3: 0.4}


def exposure_schedule(store, task_ids, group_count, hop_weights, seed=41):
    """Interleave weighted hops; shuffle questions within each hop, cycling as needed.

    This sequence has a stable prefix when a resumed run increases its step
    target. With 20/40/40 weights every five groups contain 1/2/2 questions;
    shorter runs use the nearest interleaved integer allocation. Each returned
    ID denotes ONE question group; TRL supplies its generation repetitions.
    """
    weights = {int(h): float(w) for h, w in hop_weights.items()}
    if (not weights or set(weights) - {1, 2, 3, 4}
            or any(not math.isfinite(w) or w <= 0 for w in weights.values())):
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
