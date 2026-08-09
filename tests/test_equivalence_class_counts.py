"""Regression tests for SafePub's ``equivalence_class_counts``.

The column-wise optimization in commit ``dcb6e15`` claims the result is
unchanged from the original per-cell implementation. This module pins that
down: it keeps the original implementation as ``_naive_equivalence_class_counts``
and asserts the optimized version produces identical equivalence-class counts
over many random generalizations plus hand-picked edge cases.

Runnable either with pytest (``pytest tests/test_equivalence_class_counts.py``)
or directly (``python tests/test_equivalence_class_counts.py``); no extra
dependencies beyond the package itself.
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

try:
    from k_anonymization.algorithms.full_generalization.safepub._tabular import (
        equivalence_class_counts,
        generalize_value,
    )
except ImportError:  # allow running from a checkout without an editable install
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from k_anonymization.algorithms.full_generalization.safepub._tabular import (
        equivalence_class_counts,
        generalize_value,
    )


def _naive_equivalence_class_counts(data, quasi_identifiers, hierarchies, levels, indices=None):
    """The pre-``dcb6e15`` implementation, kept verbatim as the reference.

    Generalizes one cell at a time and counts with a running ``Counter``.
    The optimized version must match this exactly.
    """

    selected_indices = range(len(data)) if indices is None else indices
    counts: Counter = Counter()
    for index in selected_indices:
        key = tuple(
            generalize_value(data[index][attribute], hierarchies[attribute], level)
            for attribute, level in zip(quasi_identifiers, levels)
        )
        counts[key] += 1
    return counts


def _random_case(rng, n_rows=None):
    """Build a random (data, quasi_identifiers, hierarchies, levels) case."""

    n_qids = rng.randint(1, 4)
    quasi_identifiers = [f"q{i}" for i in range(n_qids)]
    hierarchies = {}
    path_lengths = []
    for q in quasi_identifiers:
        n_values = rng.randint(1, 8)
        path_length = rng.randint(1, 5)
        hierarchy = {}
        for v in range(n_values):
            raw = f"{q}_v{v}"
            # path[0] is the raw value; higher levels draw from a small token
            # set so distinct raw values can collapse into shared classes.
            path = [raw]
            for level in range(1, path_length):
                path.append(f"{q}_L{level}_{rng.randint(0, 2)}")
            hierarchy[raw] = tuple(path)
        hierarchies[q] = hierarchy
        path_lengths.append(path_length)

    # "+1" occasionally exceeds the path length, exercising the clamp branch.
    levels = tuple(rng.randint(0, pl + 1) for pl in path_lengths)

    rows = rng.randint(0, 1200) if n_rows is None else n_rows
    data = []
    for _ in range(rows):
        row = {}
        for q in quasi_identifiers:
            if rng.random() < 0.1:
                row[q] = f"{q}_unknown{rng.randint(0, 3)}"  # not in the hierarchy
            else:
                row[q] = rng.choice(list(hierarchies[q]))
        data.append(row)
    return data, quasi_identifiers, hierarchies, levels


def test_matches_naive_on_random_cases():
    rng = random.Random(20260618)
    for _ in range(60):
        data, qids, hierarchies, levels = _random_case(rng)
        assert equivalence_class_counts(
            data, qids, hierarchies, levels
        ) == _naive_equivalence_class_counts(data, qids, hierarchies, levels)

        if data:  # also check an explicit index subset
            count = rng.randint(0, len(data))
            indices = sorted(rng.sample(range(len(data)), count))
            assert equivalence_class_counts(
                data, qids, hierarchies, levels, indices
            ) == _naive_equivalence_class_counts(data, qids, hierarchies, levels, indices)


def test_matches_naive_on_large_cases():
    """Mirror the commit message's "40 random generalizations x 5000 rows"."""

    rng = random.Random(5000)
    for _ in range(40):
        data, qids, hierarchies, levels = _random_case(rng, n_rows=5000)
        assert equivalence_class_counts(
            data, qids, hierarchies, levels
        ) == _naive_equivalence_class_counts(data, qids, hierarchies, levels)


def test_edge_cases():
    hierarchies = {"x": {"a": ("a", "A", "*"), "b": ("b", "B", "*")}}

    # empty data -> empty counts
    assert equivalence_class_counts([], ["x"], hierarchies, (0,)) == Counter()
    # empty index selection -> empty counts
    assert equivalence_class_counts(
        [{"x": "a"}], ["x"], hierarchies, (0,), []
    ) == Counter()

    # single row, raw level
    assert equivalence_class_counts(
        [{"x": "a"}], ["x"], hierarchies, (0,)
    ) == Counter({("a",): 1})
    # all identical rows collapse
    assert equivalence_class_counts(
        [{"x": "a"}] * 5, ["x"], hierarchies, (1,)
    ) == Counter({("A",): 5})

    # value not in hierarchy: kept at level 0, suppressed above it
    assert equivalence_class_counts(
        [{"x": "zzz"}], ["x"], hierarchies, (0,)
    ) == Counter({("zzz",): 1})
    assert equivalence_class_counts(
        [{"x": "zzz"}], ["x"], hierarchies, (1,)
    ) == Counter({("*",): 1})
    # level beyond the path length clamps to the last element
    assert equivalence_class_counts(
        [{"x": "a"}], ["x"], hierarchies, (99,)
    ) == Counter({("*",): 1})

    # explicit indices: subset and duplicates (duplicates double-count)
    data = [{"x": "a"}, {"x": "b"}, {"x": "a"}]
    assert equivalence_class_counts(
        data, ["x"], hierarchies, (0,), [0, 2]
    ) == Counter({("a",): 2})
    assert equivalence_class_counts(
        data, ["x"], hierarchies, (0,), [0, 0]
    ) == Counter({("a",): 2})

    # a generalized value equal to the string "None" must not confuse the cache
    assert equivalence_class_counts(
        [{"y": "k"}, {"y": "k"}], ["y"], {"y": {"k": ("k", "None")}}, (1,)
    ) == Counter({("None",): 2})

    # multiple quasi-identifiers
    hierarchies2 = {"x": hierarchies["x"], "y": {"c": ("c", "C", "*")}}
    data2 = [{"x": "a", "y": "c"}, {"x": "a", "y": "c"}, {"x": "b", "y": "c"}]
    assert equivalence_class_counts(
        data2, ["x", "y"], hierarchies2, (1, 1)
    ) == Counter({("A", "C"): 2, ("B", "C"): 1})


if __name__ == "__main__":
    test_matches_naive_on_random_cases()
    test_matches_naive_on_large_cases()
    test_edge_cases()
    print("All equivalence_class_counts regression tests passed.")
