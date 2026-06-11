"""Small hierarchy-based tabular utilities for running SafePub in Python."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import floor, prod
from typing import Mapping, Sequence

from ._criterion import EDDifferentialPrivacy
from ._search import DataDependentEDDPSearch, SearchResult
from ._utility import (
    UtilityResult,
    arx_aecs_dp_score,
    arx_discernibility_dp_score,
    arx_entropy_dp_score,
    arx_loss_dp_score,
    arx_precision,
    arx_precision_dp_score,
)


Row = Mapping[str, str]
Hierarchy = Mapping[str, Sequence[str]]
Hierarchies = Mapping[str, Hierarchy]
Levels = tuple[int, ...]

SUPPRESSED_VALUE = "*"

# ARX quality models whose score function (`Metric#getScore`) is supported
# for data-dependent differential privacy (`isScoreFunctionSupported()`).
SCORE_FUNCTIONS: tuple[str, ...] = (
    "safepub_precision",
    "safepub_loss",
    "safepub_discernibility",
    "safepub_entropy",
    "safepub_aecs",
    "safepub_classification",
)

_SCORE_FUNCTION_ALIASES: Mapping[str, str] = {
    "precision": "safepub_precision",
    "loss": "safepub_loss",
    "granularity": "safepub_loss",
    "discernibility": "safepub_discernibility",
    "entropy": "safepub_entropy",
    "non_uniform_entropy": "safepub_entropy",
    "aecs": "safepub_aecs",
    "average_class_size": "safepub_aecs",
    "classification": "safepub_classification",
    # Backwards-compatible former names
    "arx_precision": "safepub_precision",
    "arx_loss": "safepub_loss",
    "arx_discernibility": "safepub_discernibility",
    "arx_entropy": "safepub_entropy",
    "arx_aecs": "safepub_aecs",
    "arx_classification": "safepub_classification",
}

# ARX `DataGeneralizationScheme.GeneralizationDegree` factors.
GENERALIZATION_DEGREES: Mapping[str, float] = {
    "none": 0.0,
    "low": 0.2,
    "low_medium": 0.4,
    "medium": 0.5,
    "medium_high": 0.6,
    "high": 0.8,
    "complete": 1.0,
}


@dataclass(frozen=True)
class TabularAnonymizationResult:
    """Result from the minimal tabular SafePub anonymizer.

    `rows` mirrors ARX's output handle: one output row per input row, where a
    row is fully suppressed (every column replaced by `*`) when it was not
    part of the beta-sampled subset or when its equivalence class on the
    sampled subset is smaller than `k`.

    `score` mirrors ARX's reported information loss for data-dependent DP:
    transformations are checked with `ScoreType.DP_SCORE`, so the solution's
    information loss is the `ILScore` value of the selected transformation
    for the chosen quality model (`score_function`). Higher is better; the
    Precision, Loss, Discernibility, and Entropy scores are negative, while
    the AECS and Classification scores are positive, exactly like Java ARX.
    For data-independent (fixed scheme) runs ARX measures conventional
    information loss instead, so `score` and `score_function` are `None`.
    """

    levels: Levels
    score: float | None
    score_function: str | None
    k: int
    beta: float
    sampled_indices: tuple[int, ...]
    equivalence_class_counts: Mapping[tuple[str, ...], int]
    rows: tuple[dict[str, str], ...]
    suppressed_sample_count: int
    non_sampled_count: int
    quality_loss: float
    utility: UtilityResult
    search_strategy: str
    search_expansion_limit: int | None
    search_result: SearchResult[Levels] | None


def safe_pub_anonymize(
    data: Sequence[Row],
    quasi_identifiers: Sequence[str],
    hierarchies: Hierarchies,
    *,
    epsilon: float,
    delta: float,
    deterministic: bool = False,
    data_dependent: bool = False,
    dp_search_budget: float | None = None,
    search_expansion_limit: int | None = None,
    generalization_levels: Mapping[str, int] | None = None,
    generalization_degree: str | float | None = None,
    utility_metric: str = "safepub_precision",
    response_variables: Sequence[str] | None = None,
) -> TabularAnonymizationResult:
    """Run a minimal hierarchy-based SafePub anonymization.

    Data-dependent SafePub mirrors ARX's `DataDependentEDDPAlgorithm`: start
    at the top transformation and use the exponential mechanism to select
    predecessor transformations from a candidate set. Every transformation is
    a valid candidate; sampled records in equivalence classes smaller than
    `k` are suppressed, exactly like ARX with a 100% suppression limit.

    `utility_metric` selects the ARX quality model whose SafePub score
    function drives the data-dependent search (one of `SCORE_FUNCTIONS`).
    `safepub_classification` additionally requires `response_variables`: the
    target columns (quasi-identifying or not) whose predictability the score
    rewards, like ARX's response variables. The reference information loss
    reported as `utility` is always ARX Precision.

    Data-independent SafePub mirrors ARX's fixed `DataGeneralizationScheme`:
    the generalization levels are specified up front via
    `generalization_levels` (per attribute) and/or `generalization_degree`
    (an ARX `GeneralizationDegree` name or a factor in [0, 1]); no search is
    performed and the full epsilon is used for anonymization.
    """

    _validate_inputs(data, quasi_identifiers, hierarchies)
    score_function = _normalize_score_function(utility_metric)
    resolved_response_variables = _validate_response_variables(
        data,
        score_function,
        response_variables,
        data_dependent=data_dependent,
    )
    criterion = EDDifferentialPrivacy(
        epsilon,
        delta,
        data_dependent=data_dependent,
        dp_search_budget=dp_search_budget,
        deterministic=deterministic,
    )
    initialization = criterion.initialize(len(data))
    sampled_indices = initialization.sampled_indices

    max_levels = _max_levels(quasi_identifiers, hierarchies)
    if data_dependent:
        if generalization_levels is not None or generalization_degree is not None:
            raise ValueError(
                "generalization_levels/generalization_degree define a fixed "
                "scheme and must not be combined with data-dependent SafePub"
            )
        (
            best_levels,
            search_result,
            resolved_expansion_limit,
        ) = _find_levels_data_dependent(
            data,
            quasi_identifiers,
            hierarchies,
            max_levels,
            criterion,
            sampled_indices,
            deterministic=deterministic,
            search_expansion_limit=search_expansion_limit,
            score_function=score_function,
            response_variables=resolved_response_variables,
        )
        search_strategy = "exponential_mechanism"
    else:
        best_levels = resolve_generalization_scheme(
            quasi_identifiers,
            max_levels,
            generalization_levels,
            generalization_degree,
        )
        search_result = None
        resolved_expansion_limit = None
        search_strategy = "fixed_scheme"

    counts = equivalence_class_counts(
        data,
        quasi_identifiers,
        hierarchies,
        best_levels,
        sampled_indices,
    )
    suppressed_sample_count = _suppressed_sample_count(counts, criterion.k)

    utility = calculate_utility(
        quasi_identifiers,
        best_levels,
        max_levels,
        record_count=len(sampled_indices),
        suppressed_count=suppressed_sample_count,
    )

    return TabularAnonymizationResult(
        levels=best_levels,
        score=search_result.best_score if search_result is not None else None,
        score_function=score_function if search_result is not None else None,
        k=criterion.k,
        beta=criterion.beta,
        sampled_indices=sampled_indices,
        equivalence_class_counts=dict(counts),
        rows=_build_output_rows(
            data,
            quasi_identifiers,
            hierarchies,
            best_levels,
            sampled_indices,
            counts,
            criterion.k,
        ),
        suppressed_sample_count=suppressed_sample_count,
        non_sampled_count=len(data) - len(sampled_indices),
        quality_loss=utility.value,
        utility=utility,
        search_strategy=search_strategy,
        search_expansion_limit=resolved_expansion_limit,
        search_result=search_result,
    )


def resolve_generalization_scheme(
    quasi_identifiers: Sequence[str],
    max_levels: Levels,
    generalization_levels: Mapping[str, int] | None,
    generalization_degree: str | float | None,
) -> Levels:
    """Resolve a fixed ARX-style `DataGeneralizationScheme` to levels.

    Mirrors `DataGeneralizationScheme#getGeneralizationLevel`: an explicit
    per-attribute level wins, otherwise the degree factor is applied as
    `round(factor * max_level)` (Java `Math.round`), and the result is
    clamped to `[0, max_level]`.
    """

    if generalization_levels is None and generalization_degree is None:
        raise ValueError(
            "data-independent SafePub requires a generalization scheme: "
            "provide generalization_levels and/or generalization_degree "
            "(like ARX's DataGeneralizationScheme)"
        )

    factor: float | None = None
    if generalization_degree is not None:
        if isinstance(generalization_degree, str):
            key = generalization_degree.lower().replace("-", "_")
            if key not in GENERALIZATION_DEGREES:
                raise ValueError(
                    f"unknown generalization degree: {generalization_degree!r}; "
                    f"expected one of {sorted(GENERALIZATION_DEGREES)}"
                )
            factor = GENERALIZATION_DEGREES[key]
        else:
            factor = float(generalization_degree)
            if not 0.0 <= factor <= 1.0:
                raise ValueError("generalization degree factor must be in [0, 1]")

    explicit = dict(generalization_levels or {})
    unknown = set(explicit) - set(quasi_identifiers)
    if unknown:
        raise ValueError(
            f"generalization_levels given for non-quasi-identifiers: {sorted(unknown)}"
        )

    result: list[int] = []
    for attribute, max_level in zip(quasi_identifiers, max_levels):
        if attribute in explicit:
            level = int(explicit[attribute])
            if level < 0:
                raise ValueError("generalization levels must be >= 0")
        elif factor is not None:
            # Java Math.round(): floor(x + 0.5)
            level = int(floor(factor * float(max_level) + 0.5))
        else:
            raise ValueError(
                f"no generalization level or degree given for {attribute!r}"
            )
        result.append(min(max(level, 0), max_level))
    return tuple(result)


def generalize_row(
    row: Row,
    quasi_identifiers: Sequence[str],
    hierarchies: Hierarchies,
    levels: Levels,
) -> Row:
    """Return a row with quasi-identifiers generalized at the given levels."""

    result = dict(row)
    for attribute, level in zip(quasi_identifiers, levels):
        result[attribute] = generalize_value(row[attribute], hierarchies[attribute], level)
    return result


def generalize_value(value: str, hierarchy: Hierarchy, level: int) -> str:
    """Return one generalized value from a hierarchy."""

    if value not in hierarchy:
        if level == 0:
            return value
        return "*"
    path = hierarchy[value]
    if level < 0:
        raise ValueError("generalization level must be >= 0")
    if level >= len(path):
        return path[-1]
    return path[level]


def equivalence_class_counts(
    data: Sequence[Row],
    quasi_identifiers: Sequence[str],
    hierarchies: Hierarchies,
    levels: Levels,
    indices: Sequence[int] | None = None,
) -> Counter[tuple[str, ...]]:
    """Count equivalence classes for the selected row indices."""

    selected_indices = range(len(data)) if indices is None else indices
    counts: Counter[tuple[str, ...]] = Counter()
    for index in selected_indices:
        key = tuple(
            generalize_value(data[index][attribute], hierarchies[attribute], level)
            for attribute, level in zip(quasi_identifiers, levels)
        )
        counts[key] += 1
    return counts


def calculate_utility(
    quasi_identifiers: Sequence[str],
    levels: Levels,
    max_levels: Levels,
    metric: str = "safepub_precision",
    *,
    record_count: int | None = None,
    suppressed_count: int = 0,
) -> UtilityResult:
    """Calculate a utility metric for the selected generalization levels."""

    normalized_metric = metric.lower().replace("-", "_")
    if normalized_metric in {"safepub_precision", "arx_precision", "precision"}:
        return arx_precision(
            quasi_identifiers,
            levels,
            max_levels,
            record_count=record_count,
            suppressed_count=suppressed_count,
        )
    raise ValueError(f"unsupported utility metric: {metric}")


def _find_levels_data_dependent(
    data: Sequence[Row],
    quasi_identifiers: Sequence[str],
    hierarchies: Hierarchies,
    max_levels: Levels,
    criterion: EDDifferentialPrivacy,
    sampled_indices: Sequence[int],
    *,
    deterministic: bool,
    search_expansion_limit: int | None,
    score_function: str,
    response_variables: tuple[str, ...],
) -> tuple[Levels, SearchResult[Levels], int]:
    """Find a transformation using ARX-style data-dependent DP search.

    Like ARX, every lattice transformation is a candidate: classes smaller
    than `k` are folded into the DP score as suppressed records instead of
    being filtered out, so the candidate set never depends on the data.
    """

    resolved_expansion_limit = (
        _default_expansion_limit(max_levels)
        if search_expansion_limit is None
        else int(search_expansion_limit)
    )
    if resolved_expansion_limit < 0:
        raise ValueError("search_expansion_limit must be >= 0")

    counts_cache: dict[Levels, Counter[tuple[str, ...]]] = {}

    def counts_for(levels: Levels) -> Counter[tuple[str, ...]]:
        if levels not in counts_cache:
            counts_cache[levels] = equivalence_class_counts(
                data,
                quasi_identifiers,
                hierarchies,
                levels,
                sampled_indices,
            )
        return counts_cache[levels]

    def predecessors(levels: Levels) -> list[Levels]:
        result: list[Levels] = []
        for index, level in enumerate(levels):
            if level <= 0:
                continue
            result.append(
                tuple(
                    level_value - 1 if level_index == index else level_value
                    for level_index, level_value in enumerate(levels)
                )
            )
        return result

    score = _build_score_function(
        data,
        quasi_identifiers,
        hierarchies,
        max_levels,
        criterion,
        sampled_indices,
        counts_for,
        score_function,
        response_variables,
    )

    search = DataDependentEDDPSearch(
        top=max_levels,
        predecessors=predecessors,
        score=score,
        expansion_limit=resolved_expansion_limit,
        epsilon_search=criterion.dp_search_budget,
        deterministic=deterministic,
        level=sum,
    )
    search_result = search.traverse()
    return search_result.best, search_result, resolved_expansion_limit


def _build_score_function(
    data: Sequence[Row],
    quasi_identifiers: Sequence[str],
    hierarchies: Hierarchies,
    max_levels: Levels,
    criterion: EDDifferentialPrivacy,
    sampled_indices: Sequence[int],
    counts_for,
    score_function: str,
    response_variables: tuple[str, ...],
):
    """Build the SafePub score callable for the selected ARX quality model."""

    total_count = len(data)
    sample_count = len(sampled_indices)

    if score_function == "safepub_precision":

        def score(levels: Levels) -> float:
            suppressed_sample = _suppressed_sample_count(counts_for(levels), criterion.k)
            # ARX's `pcount - count` term: records outside the sampled subset
            # always count as suppressed.
            suppressed_total = suppressed_sample + (total_count - sample_count)
            return arx_precision_dp_score(
                quasi_identifiers,
                levels,
                max_levels,
                record_count=total_count,
                k=criterion.k,
                suppressed_count=suppressed_total,
            )

        return score

    if score_function == "safepub_loss":
        share_maps = {
            attribute: _domain_share_maps(hierarchies[attribute], max_level)
            for attribute, max_level in zip(quasi_identifiers, max_levels)
        }

        def score(levels: Levels) -> float:
            dimension_shares = [
                share_maps[attribute][level]
                for attribute, level in zip(quasi_identifiers, levels)
            ]
            return arx_loss_dp_score(
                counts_for(levels),
                dimension_shares,
                record_count=total_count,
                sample_count=sample_count,
                k=criterion.k,
            )

        return score

    if score_function == "safepub_discernibility":

        def score(levels: Levels) -> float:
            return arx_discernibility_dp_score(
                counts_for(levels),
                record_count=total_count,
                sample_count=sample_count,
                k=criterion.k,
            )

        return score

    if score_function == "safepub_entropy":
        root_values = _root_values(quasi_identifiers, hierarchies, max_levels)

        def score(levels: Levels) -> float:
            return arx_entropy_dp_score(
                counts_for(levels),
                root_values,
                record_count=total_count,
                sample_count=sample_count,
                k=criterion.k,
            )

        return score

    if score_function == "safepub_aecs":

        def score(levels: Levels) -> float:
            return arx_aecs_dp_score(
                counts_for(levels),
                record_count=total_count,
                sample_count=sample_count,
                k=criterion.k,
            )

        return score

    if score_function == "safepub_classification":

        def score(levels: Levels) -> float:
            return _classification_dp_score(
                data,
                quasi_identifiers,
                hierarchies,
                levels,
                max_levels,
                sampled_indices,
                counts_for(levels),
                criterion.k,
                response_variables,
            )

        return score

    raise ValueError(f"unsupported score function: {score_function}")


def _classification_dp_score(
    data: Sequence[Row],
    quasi_identifiers: Sequence[str],
    hierarchies: Hierarchies,
    levels: Levels,
    max_levels: Levels,
    sampled_indices: Sequence[int],
    counts: Counter[tuple[str, ...]],
    k: int,
    response_variables: tuple[str, ...],
) -> float:
    """Return ARX's Classification accuracy score for SafePub.

    Mirrors `MetricSDClassification#getScore` (extended SafePub Section 5.5):

    - For each non-quasi-identifying response variable, every equivalence
      class of size >= `k` contributes the frequency of its most frequent
      raw target value.
    - For each quasi-identifying response variable, classes are pooled with
      the target dimension projected out (`MetaHashGroupify`), each pool
      contributes the frequency of its most frequent generalized target
      value, and the sum is scaled with `1 - level/max_level` to penalize
      generalization of the target.
    - The total is divided by the sensitivity `k * len(response_variables)`.

    Like Java, the resulting score is positive (higher is better).
    """

    qi_index = {attribute: index for index, attribute in enumerate(quasi_identifiers)}

    def class_key(row: Row) -> tuple[str, ...]:
        return tuple(
            generalize_value(row[attribute], hierarchies[attribute], level)
            for attribute, level in zip(quasi_identifiers, levels)
        )

    score = 0.0
    for target in response_variables:
        if target in qi_index:
            # Pool classes with the target dimension projected out; the
            # distribution is over the target's generalized values.
            dimension = qi_index[target]
            pools: dict[tuple[str, ...], Counter[str]] = {}
            for index in sampled_indices:
                key = class_key(data[index])
                if counts[key] < k:
                    continue
                pool_key = key[:dimension] + key[dimension + 1 :]
                pools.setdefault(pool_key, Counter())[key[dimension]] += 1
            target_score = float(
                sum(max(distribution.values()) for distribution in pools.values())
            )
            # Scale between 1 (target not generalized) and 0 (fully
            # generalized) to penalize generalizing the target.
            max_level = max_levels[dimension]
            scale = 1.0 if max_level == 0 else 1.0 - levels[dimension] / max_level
            score += target_score * scale
        else:
            distributions: dict[tuple[str, ...], Counter[str]] = {}
            for index in sampled_indices:
                key = class_key(data[index])
                if counts[key] < k:
                    continue
                distributions.setdefault(key, Counter())[data[index][target]] += 1
            score += float(
                sum(max(distribution.values()) for distribution in distributions.values())
            )

    sensitivity = float(k) * float(len(response_variables))
    return score / sensitivity


def _normalize_score_function(utility_metric: str) -> str:
    normalized = utility_metric.lower().replace("-", "_")
    normalized = _SCORE_FUNCTION_ALIASES.get(normalized, normalized)
    if normalized not in SCORE_FUNCTIONS:
        raise ValueError(
            f"unsupported utility metric: {utility_metric!r}; "
            f"expected one of {list(SCORE_FUNCTIONS)}"
        )
    return normalized


def _validate_response_variables(
    data: Sequence[Row],
    score_function: str,
    response_variables: Sequence[str] | None,
    *,
    data_dependent: bool,
) -> tuple[str, ...]:
    resolved = tuple(response_variables or ())
    if score_function == "safepub_classification" and data_dependent:
        if not resolved:
            raise ValueError(
                "safepub_classification requires at least one response variable, "
                "like ARX's response variables"
            )
    elif resolved:
        raise ValueError(
            "response_variables are only used with the safepub_classification "
            "utility metric for data-dependent SafePub"
        )
    for attribute in resolved:
        if attribute not in data[0]:
            raise ValueError(f"unknown response variable: {attribute!r}")
    return resolved


def _domain_share_maps(hierarchy: Hierarchy, max_level: int) -> list[dict[str, float]]:
    """Return per-level domain shares, like ARX's `DomainShare`.

    The share of a generalized value is the number of distinct raw domain
    values it covers divided by the domain size (the number of distinct raw
    values in the hierarchy).
    """

    domain = list(hierarchy.keys())
    size = len(domain)
    maps: list[dict[str, float]] = []
    for level in range(max_level + 1):
        cover: Counter[str] = Counter(
            generalize_value(value, hierarchy, level) for value in domain
        )
        maps.append({value: count / size for value, count in cover.items()})
    return maps


def _root_values(
    quasi_identifiers: Sequence[str],
    hierarchies: Hierarchies,
    max_levels: Levels,
) -> tuple[str | None, ...]:
    """Return the unique root value per hierarchy, or None if there is none.

    Mirrors the `rootValues` initialization of
    `MetricMDNUEntropyPrecomputed`: the value every raw value generalizes to
    at the top level, or `-1` (here `None`) if the hierarchy has no single
    root value.
    """

    result: list[str | None] = []
    for attribute, max_level in zip(quasi_identifiers, max_levels):
        hierarchy = hierarchies[attribute]
        top_values = {
            generalize_value(value, hierarchy, max_level) for value in hierarchy
        }
        result.append(top_values.pop() if len(top_values) == 1 else None)
    return tuple(result)


def _build_output_rows(
    data: Sequence[Row],
    quasi_identifiers: Sequence[str],
    hierarchies: Hierarchies,
    levels: Levels,
    sampled_indices: Sequence[int],
    counts: Counter[tuple[str, ...]],
    k: int,
) -> tuple[dict[str, str], ...]:
    """Build ARX-style output rows.

    Rows outside the beta-sampled subset and sampled rows in classes smaller
    than `k` are fully suppressed (every attribute becomes `*`).
    """

    sampled = set(sampled_indices)
    rows: list[dict[str, str]] = []
    for index, row in enumerate(data):
        if index not in sampled:
            rows.append({attribute: SUPPRESSED_VALUE for attribute in row})
            continue
        key = tuple(
            generalize_value(row[attribute], hierarchies[attribute], level)
            for attribute, level in zip(quasi_identifiers, levels)
        )
        if counts[key] < k:
            rows.append({attribute: SUPPRESSED_VALUE for attribute in row})
            continue
        rows.append(dict(generalize_row(row, quasi_identifiers, hierarchies, levels)))
    return tuple(rows)


def _suppressed_sample_count(counts: Counter[tuple[str, ...]], k: int) -> int:
    return sum(count for count in counts.values() if count < k)


def _validate_inputs(
    data: Sequence[Row],
    quasi_identifiers: Sequence[str],
    hierarchies: Hierarchies,
) -> None:
    if not data:
        raise ValueError("data must contain at least one row")
    if not quasi_identifiers:
        raise ValueError("at least one quasi-identifier is required")
    for attribute in quasi_identifiers:
        if attribute not in hierarchies:
            raise ValueError(f"missing hierarchy for quasi-identifier {attribute!r}")
        for row in data:
            if attribute not in row:
                raise ValueError(f"row is missing quasi-identifier {attribute!r}")


def _max_levels(quasi_identifiers: Sequence[str], hierarchies: Hierarchies) -> Levels:
    result: list[int] = []
    for attribute in quasi_identifiers:
        max_depth = max(len(path) for path in hierarchies[attribute].values())
        result.append(max(0, max_depth - 1))
    return tuple(result)


def _default_expansion_limit(max_levels: Levels) -> int:
    """Return a finite ARX-style expansion limit for the local lattice."""

    return max(0, prod(max_level + 1 for max_level in max_levels) - 1)
