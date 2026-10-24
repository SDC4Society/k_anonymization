"""Reproducible benchmark for the SafePub ``equivalence_class_counts`` speedup.

Commit ``dcb6e15`` rewrote ``equivalence_class_counts`` (per-cell ->
column-wise with a distinct-value cache and ``Counter(zip(*cols))``) and
reported "Adult data-dependent run: ~16.9s -> ~1.8s (about 9x faster)". That
number was measured ad hoc and never committed, so it could not be reproduced.

This script re-measures the speedup using only data bundled in this repository
(``datasets.ADULT``). It times a full deterministic data-dependent
``SafePub.anonymize()`` with the current (optimized) counter and with the
original per-cell counter (monkey-patched in), and also micro-benchmarks the
counter in isolation. Both the end-to-end results and the counter output are
asserted identical, so the timing compares two implementations that agree.

Usage::

    python benchmarks/bench_equivalence_class_counts.py
    python benchmarks/bench_equivalence_class_counts.py --limit 50 --repeats 5

The absolute times and the ratio depend on the configuration (epsilon, delta,
search_expansion_limit, machine); the script prints the configuration it used.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

try:
    from k_anonymization import datasets
    from k_anonymization.algorithms.full_generalization import SafePub
    from k_anonymization.algorithms.full_generalization.safepub import _tabular
    from k_anonymization.algorithms.full_generalization.safepub._tabular import (
        generalize_value,
    )
except ImportError:  # allow running from a checkout without an editable install
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from k_anonymization import datasets
    from k_anonymization.algorithms.full_generalization import SafePub
    from k_anonymization.algorithms.full_generalization.safepub import _tabular
    from k_anonymization.algorithms.full_generalization.safepub._tabular import (
        generalize_value,
    )

# The optimized implementation currently shipped in the package.
_OPTIMIZED = _tabular.equivalence_class_counts


def _naive_equivalence_class_counts(data, quasi_identifiers, hierarchies, levels, indices=None):
    """The pre-``dcb6e15`` per-cell implementation, kept as the baseline."""

    selected_indices = range(len(data)) if indices is None else indices
    counts: Counter = Counter()
    for index in selected_indices:
        key = tuple(
            generalize_value(data[index][attribute], hierarchies[attribute], level)
            for attribute, level in zip(quasi_identifiers, levels)
        )
        counts[key] += 1
    return counts


def _run_once(impl, *, epsilon, delta, limit):
    """Run one deterministic data-dependent SafePub with ``impl`` patched in."""

    original = _tabular.equivalence_class_counts
    _tabular.equivalence_class_counts = impl
    try:
        sp = SafePub(
            dataset=datasets.ADULT,
            epsilon=epsilon,
            delta=delta,
            search_expansion_limit=limit,
            deterministic=True,
        )
        start = time.perf_counter()
        sp.anonymize()
        elapsed = time.perf_counter() - start
    finally:
        _tabular.equivalence_class_counts = original
    fingerprint = (
        tuple(sp.levels),
        sp.sampled_count,
        sp.suppressed_sample_count,
        len(sp.anon_data),
    )
    return elapsed, fingerprint


def _bench_end_to_end(epsilon, delta, limit, repeats):
    opt_times, opt_fp = [], None
    for _ in range(repeats):
        elapsed, opt_fp = _run_once(_OPTIMIZED, epsilon=epsilon, delta=delta, limit=limit)
        opt_times.append(elapsed)

    naive_times, naive_fp = [], None
    for _ in range(repeats):
        elapsed, naive_fp = _run_once(
            _naive_equivalence_class_counts, epsilon=epsilon, delta=delta, limit=limit
        )
        naive_times.append(elapsed)

    assert opt_fp == naive_fp, (
        f"end-to-end results differ!\n optimized={opt_fp}\n naive    ={naive_fp}"
    )
    return opt_times, naive_times, opt_fp


def _adult_counter_inputs():
    """Recreate the records/hierarchies the SafePub adapter feeds the counter."""

    sp = SafePub(dataset=datasets.ADULT, epsilon=2.0, delta=1e-5, deterministic=True)
    records = sp.org_data.astype(str).to_dict("records")
    qids = list(sp.dataset.qids)
    hierarchies = {}
    for qid in qids:
        hdf = sp.dataset.hierarchies[qid].hierarchy_df.astype(str)
        hierarchies[qid] = {
            row[0]: tuple(row) for row in hdf.itertuples(index=False, name=None)
        }
    levels = tuple(1 for _ in qids)
    return records, qids, hierarchies, levels


def _bench_micro(repeats):
    records, qids, hierarchies, levels = _adult_counter_inputs()
    assert _OPTIMIZED(records, qids, hierarchies, levels) == _naive_equivalence_class_counts(
        records, qids, hierarchies, levels
    )

    def timeit(fn):
        fn(records, qids, hierarchies, levels)  # warm up
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn(records, qids, hierarchies, levels)
            samples.append(time.perf_counter() - start)
        return samples

    return len(records), timeit(_OPTIMIZED), timeit(_naive_equivalence_class_counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", type=float, default=2.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--limit", type=int, default=30, help="search_expansion_limit")
    parser.add_argument("--repeats", type=int, default=3, help="end-to-end repeats")
    parser.add_argument("--micro-repeats", type=int, default=5)
    args = parser.parse_args()

    print("SafePub equivalence_class_counts benchmark (adult, data-dependent, deterministic)")
    print(
        f"config: epsilon={args.epsilon} delta={args.delta} "
        f"search_expansion_limit={args.limit} repeats={args.repeats}"
    )
    print()

    opt, naive, fp = _bench_end_to_end(args.epsilon, args.delta, args.limit, args.repeats)
    levels, sampled, suppressed, released = fp
    opt_med, naive_med = statistics.median(opt), statistics.median(naive)
    print("End-to-end SafePub.anonymize():")
    print(f"  optimized (current): median {opt_med:7.3f}s  min {min(opt):7.3f}s")
    print(f"  naive (pre-dcb6e15): median {naive_med:7.3f}s  min {min(naive):7.3f}s")
    print(f"  speedup (median):    {naive_med / opt_med:5.2f}x")
    print(
        f"  identical result:    levels={levels} sampled={sampled} "
        f"suppressed={suppressed} released={released}"
    )
    print()

    n, micro_opt, micro_naive = _bench_micro(args.micro_repeats)
    mo, mn = statistics.median(micro_opt), statistics.median(micro_naive)
    print(f"Micro-benchmark equivalence_class_counts ({n} rows, levels=all-1, "
          f"repeats={args.micro_repeats}):")
    print(f"  optimized: median {mo * 1000:8.2f} ms")
    print(f"  naive:     median {mn * 1000:8.2f} ms")
    print(f"  speedup:   {mn / mo:5.2f}x")


if __name__ == "__main__":
    main()
