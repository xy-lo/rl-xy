"""Observable-history heuristic for TP-group selection; not a calibrated predictor."""
from bisect import bisect_right


def score(replica, completed_lengths):
    """Rank potential bottlenecks using only emitted tokens and completed histories.

    Never access active entries' third field or a live request's response_length.
    Historical completions are biased by censoring and the score is only a
    scheduling heuristic; its units are not seconds.
    """
    if not replica.active:
        return 0.0
    ordered = sorted(completed_lengths)
    residuals = []
    for item, context, _ in replica.active:
        elapsed = context - item['prompt_length']
        if elapsed < 0:
            raise ValueError('invalid observed context')
        i = bisect_right(ordered, elapsed)
        n = len(ordered) - i
        if n >= 8:
            # Conservative upper residual among historical finished requests.
            index = i + (9 * (n - 1)) // 10
            predicted = max(1, ordered[index] - elapsed)
        else:
            # No live terminal length is read; cold-start is intentionally fixed.
            predicted = max(64, min(2048, elapsed // 4))
        residuals.append(predicted)
    # Approximate a bottleneck in batched decode, not aggregate token cost.
    return max(residuals) * (1.0 + len(replica.active) / 16.0)
