"""Utility metrics aligned with ARX metric formulas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

ClassCounts = Mapping[tuple[str, ...], int]


@dataclass(frozen=True)
class UtilityResult:
    """Calculated utility metric values."""

    metric: str
    aggregate_function: str
    value: float
    values_by_attribute: Mapping[str, float]


def arx_precision(
    quasi_identifiers: Sequence[str],
    levels: Sequence[int],
    max_levels: Sequence[int],
    *,
    record_count: int | None = None,
    suppressed_count: int = 0,
    generalization_factor: float = 1.0,
    suppression_factor: float = 1.0,
    weights: Mapping[str, float] | None = None,
) -> UtilityResult:
    """Return ARX's arithmetic-mean Precision information loss.

    This implements ARX's Precision metric including record suppression:

    - `MetricMDNMPrecision#getInformationLossInternal`:
      per dimension `(unsuppressed * (level/height) * gFactor
      + suppressed * sFactor) / rowCount`
    - `ILMultiDimensionalArithmeticMean#getAggregate`

    `record_count` is ARX's `rowCount`: the number of released records, which
    for SafePub is the size of the beta-sampled subset
    (`Metric#getNumRecords` returns the subset size when a subset exists).
    `suppressed_count` is the number of those records that end up suppressed
    because their equivalence class is smaller than `k`. When `record_count`
    is omitted, the value reduces to the pure generalization-level formula.
    With ARX's default `gsFactor=0.5`, both factors are `1.0`.
    """

    if not quasi_identifiers:
        raise ValueError("at least one quasi-identifier is required")
    if len(quasi_identifiers) != len(levels) or len(levels) != len(max_levels):
        raise ValueError("quasi_identifiers, levels, and max_levels must align")
    if generalization_factor < 0.0:
        raise ValueError("generalization_factor must be >= 0")
    if suppression_factor < 0.0:
        raise ValueError("suppression_factor must be >= 0")
    if record_count is None:
        if suppressed_count != 0:
            raise ValueError("suppressed_count requires record_count")
    else:
        if record_count < 0:
            raise ValueError("record_count must be >= 0")
        if suppressed_count < 0 or suppressed_count > record_count:
            raise ValueError("suppressed_count must be in [0, record_count]")

    values_by_attribute: dict[str, float] = {}
    aggregate = 0.0
    dimensions = float(len(quasi_identifiers))

    for attribute, level, max_level in zip(quasi_identifiers, levels, max_levels):
        if level < 0:
            raise ValueError("generalization levels must be >= 0")
        if max_level < 0:
            raise ValueError("maximum generalization levels must be >= 0")

        value = 0.0 if max_level == 0 else float(level) / float(max_level)
        if record_count is None:
            dimension_value = value * generalization_factor
        elif record_count == 0:
            # Nothing is released; define the loss of an empty release as 0.
            dimension_value = 0.0
        else:
            unsuppressed_count = record_count - suppressed_count
            dimension_value = (
                float(unsuppressed_count) * value * generalization_factor
                + float(suppressed_count) * suppression_factor
            ) / float(record_count)
        weight = 1.0 if weights is None else float(weights.get(attribute, 1.0))
        values_by_attribute[attribute] = dimension_value
        aggregate += (dimension_value / dimensions) * weight

    return UtilityResult(
        metric="safepub_precision",
        aggregate_function="ARITHMETIC_MEAN",
        value=aggregate,
        values_by_attribute=values_by_attribute,
    )


def arx_precision_dp_score(
    quasi_identifiers: Sequence[str],
    levels: Sequence[int],
    max_levels: Sequence[int],
    *,
    record_count: int,
    k: int,
    suppressed_count: int = 0,
) -> float:
    """Return ARX's Precision score for SafePub's exponential mechanism.

    This mirrors `MetricMDNMPrecision#getScore`: per generalized dimension the
    score adds `unsuppressed * (level/height) + suppressed`, where

    - `record_count` is the total number of input records (ARX iterates all
      equivalence classes including records outside the sampled subset),
    - `suppressed_count` counts sampled records in classes smaller than `k`
      plus every record outside the sampled subset (ARX's
      `pcount - count` term), and
    - `unsuppressed_count = record_count - suppressed_count` is the number of
      sampled records in classes of size >= `k`.

    The sum is divided by the sensitivity (`dimensions`, and `k - 1` if
    `k > 1`) and negated because ARX's exponential mechanism expects larger
    scores to be better.
    """

    if not quasi_identifiers:
        raise ValueError("at least one quasi-identifier is required")
    if len(quasi_identifiers) != len(levels) or len(levels) != len(max_levels):
        raise ValueError("quasi_identifiers, levels, and max_levels must align")
    if record_count < 0:
        raise ValueError("record_count must be >= 0")
    if k < 0:
        raise ValueError("k must be >= 0")
    if suppressed_count < 0 or suppressed_count > record_count:
        raise ValueError("suppressed_count must be in [0, record_count]")

    dimensions = float(len(quasi_identifiers))
    unsuppressed_count = record_count - suppressed_count
    score = 0.0

    for level, max_level in zip(levels, max_levels):
        if level < 0:
            raise ValueError("generalization levels must be >= 0")
        if max_level < 0:
            raise ValueError("maximum generalization levels must be >= 0")

        value = 0.0 if max_level == 0 else float(level) / float(max_level)
        score += (float(unsuppressed_count) * value) + float(suppressed_count)

    score *= -1.0 / dimensions
    if k > 1:
        score /= float(k - 1)
    return score


def _validate_score_inputs(k: int, record_count: int, sample_count: int) -> None:
    if k < 1:
        raise ValueError("k must be >= 1")
    if record_count < 0:
        raise ValueError("record_count must be >= 0")
    if sample_count < 0 or sample_count > record_count:
        raise ValueError("sample_count must be in [0, record_count]")


def _class_size_sensitivity(k: int) -> float:
    """Return ARX's class-size sensitivity term `(k == 1) ? 5 : k²/(k-1) + 1`.

    Used by the Discernibility and Non-uniform entropy score functions.
    """

    return 5.0 if k == 1 else float(k) * float(k) / float(k - 1) + 1.0


def _suppressed_total(class_counts: ClassCounts, k: int, record_count: int, sample_count: int) -> int:
    """Sampled records in classes smaller than `k` plus all non-sampled records."""

    suppressed_sample = sum(count for count in class_counts.values() if count < k)
    return suppressed_sample + (record_count - sample_count)


def arx_loss_dp_score(
    class_counts: ClassCounts,
    dimension_shares: Sequence[Mapping[str, float]],
    *,
    record_count: int,
    sample_count: int,
    k: int,
) -> float:
    """Return ARX's Loss (granularity) score for SafePub's exponential mechanism.

    Mirrors `MetricMDNMLoss#getScore` (SafePub Section 5.1): per generalized
    dimension, records in classes of size >= `k` contribute the domain share
    of their generalized value, and every suppressed record (sampled records
    in classes smaller than `k`, plus records outside the sampled subset —
    ARX's `pcount - count` term) contributes 1. The sum is multiplied with
    `-1/dimensions` and divided by `k - 1` if `k > 1`.

    `class_counts` are equivalence classes of the sampled records, keyed by
    the tuple of generalized values; `dimension_shares[d]` maps a generalized
    value of dimension `d` to its domain share.
    """

    _validate_score_inputs(k, record_count, sample_count)
    if not dimension_shares:
        raise ValueError("at least one dimension is required")

    suppressed = _suppressed_total(class_counts, k, record_count, sample_count)
    dimensions = len(dimension_shares)

    score = 0.0
    for dimension, shares in enumerate(dimension_shares):
        dimension_sum = float(suppressed)
        for key, count in class_counts.items():
            if count >= k:
                value = key[dimension]
                share = shares.get(value)
                if share is None:
                    # Unknown generalized value: a fully suppressed value
                    # covers the whole domain.
                    share = 1.0
                dimension_sum += share * count
        score += dimension_sum

    score *= -1.0 / float(dimensions)
    if k > 1:
        score /= float(k - 1)
    return score


def arx_discernibility_dp_score(
    class_counts: ClassCounts,
    *,
    record_count: int,
    sample_count: int,
    k: int,
) -> float:
    """Return ARX's Discernibility score for SafePub's exponential mechanism.

    Mirrors `MetricSDNMDiscernability#getScore` (SafePub Section 5.2):

    ```text
    penalty_not_suppressed = sum over classes of size >= k of count²
    num_suppressed         = sampled records in classes < k + non-sampled records
    score = -(record_count * num_suppressed + penalty_not_suppressed)
            / (record_count * sensitivity)
    ```

    with `sensitivity = 5` if `k == 1` else `k²/(k-1) + 1`.
    """

    _validate_score_inputs(k, record_count, sample_count)
    if record_count == 0:
        raise ValueError("record_count must be >= 1")

    penalty_not_suppressed = 0.0
    num_suppressed = record_count - sample_count
    for count in class_counts.values():
        if count >= k:
            penalty_not_suppressed += float(count) * float(count)
        else:
            num_suppressed += count

    score = -(float(record_count) * float(num_suppressed) + penalty_not_suppressed)
    score /= float(record_count) * _class_size_sensitivity(k)
    return score


def arx_entropy_dp_score(
    class_counts: ClassCounts,
    root_values: Sequence[str | None],
    *,
    record_count: int,
    sample_count: int,
    k: int,
) -> float:
    """Return ARX's Non-uniform entropy score for SafePub.

    Mirrors `MetricMDNUEntropyPrecomputed#getScore` (SafePub Section 5.3).
    Per generalized dimension `d`:

    - records of classes with size >= `k` whose generalized value differs
      from the hierarchy's root value are pooled per value, contributing
      `pooled_count²`,
    - records of classes smaller than `k`, or whose value equals the root
      value (suppressed by generalization), contribute `count * record_count`,
    - non-sampled records contribute `record_count` each (ARX's
      `pcount - count` term).

    The sum is multiplied with `-1/(record_count * dimensions)` and divided
    by `5` if `k == 1` else `k²/(k-1) + 1`. `root_values[d]` is the unique
    top-level value of dimension `d`'s hierarchy, or `None` if no single
    root value exists (Java uses `-1`).
    """

    _validate_score_inputs(k, record_count, sample_count)
    if record_count == 0:
        raise ValueError("record_count must be >= 1")
    if not root_values:
        raise ValueError("at least one dimension is required")

    dimensions = len(root_values)
    score = 0.0
    for dimension, root_value in enumerate(root_values):
        pooled_counts: dict[str, int] = {}
        for key, count in class_counts.items():
            value = key[dimension]
            if count >= k and (root_value is None or value != root_value):
                pooled_counts[value] = pooled_counts.get(value, 0) + count
            else:
                score += float(count) * float(record_count)
        score += float(record_count - sample_count) * float(record_count)
        for pooled in pooled_counts.values():
            score += float(pooled) * float(pooled)

    score *= -1.0 / (float(record_count) * float(dimensions))
    score /= _class_size_sensitivity(k)
    return score


def arx_aecs_dp_score(
    class_counts: ClassCounts,
    *,
    record_count: int,
    sample_count: int,
    k: int,
) -> float:
    """Return ARX's Average equivalence class size score for SafePub.

    Mirrors `MetricSDAECS#getScore` (SafePub Section 5.4): the number of
    equivalence classes of size >= `k`, plus one if any record is suppressed
    (a sampled class smaller than `k`, or any record outside the sampled
    subset). The sensitivity is one, so no division is required. Note that
    unlike the other score functions this value is positive (more classes,
    i.e. finer granularity, is better).
    """

    _validate_score_inputs(k, record_count, sample_count)

    non_suppressed_classes = sum(1 for count in class_counts.values() if count >= k)
    has_suppressed = record_count > sample_count or any(
        count < k for count in class_counts.values()
    )
    return float(non_suppressed_classes) + (1.0 if has_suppressed else 0.0)
