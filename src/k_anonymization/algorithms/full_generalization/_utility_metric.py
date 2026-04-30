__all__ = ["UtilityMetric", "UtilityMetricBuiltIn"]

from typing import Any, Callable

from pandas import DataFrame

from k_anonymization.core.algorithm import Algorithm
from k_anonymization.evaluation.data_utility import CAVG, Discernibility, NCP

type UtilityMetric = Callable[[DataFrame, Algorithm], Any]
"""
Prototype of a utility metric for full-domain generalization algorithms.

A utility metric assigns a score to a generalized candidate, enabling
algorithms to rank or order candidates during search or solution selection.
The return value must support the ``<`` operator — typically a ``float``,
but n-dimensional vectors or any other comparable type are equally valid.
Lower scores are preferred (minimization).

Parameters
----------
generalized_df : DataFrame
    The fully generalized dataframe for one candidate.
algo : Algorithm
    The algorithm instance, providing access to ``org_data``, ``dataset``
    (hierarchies, qids_idx, is_categorical), and ``k``.

Returns
-------
Any
    A score supporting ``<`` comparison. Lower values are preferred.

See Also
--------
UtilityMetricBuiltIn :
    A set of built-in ``UtilityMetric`` implementations.
"""


class UtilityMetricBuiltIn:
    """
    A set of built-in ``UtilityMetric`` implementations.

    Each static method is a thin adapter that routes the unified
    ``(generalized_df, algo)`` call signature to the corresponding
    method in ``k_anonymization.evaluation.data_utility``, whose
    signatures differ per metric.

    See Also
    --------
    UtilityMetric :
        The type prototype for utility metrics.
    """

    @staticmethod
    def NCP(generalized_df: DataFrame, algo: Algorithm) -> float:
        """
        Normalized Certainty Penalty (lower = less information loss).

        See Also
        --------
        k_anonymization.evaluation.data_utility.NCP.calculate_for_generalization
        """
        return NCP.calculate_for_generalization(
            algo.org_data,
            generalized_df,
            algo.dataset.hierarchies,
            algo.dataset.qids_idx,
            algo.dataset.is_categorical,
        )

    @staticmethod
    def CAVG(generalized_df: DataFrame, algo: Algorithm) -> float:
        """
        Normalized average equivalence class size (lower = closer to optimal k grouping).

        See Also
        --------
        k_anonymization.evaluation.data_utility.CAVG.calculate
        """
        return CAVG.calculate(generalized_df, algo.dataset.qids_idx, algo.k)

    @staticmethod
    def DISCERNIBILITY(generalized_df: DataFrame, algo: Algorithm) -> float:
        """
        Discernibility Metric (lower = smaller equivalence classes).

        See Also
        --------
        k_anonymization.evaluation.data_utility.Discernibility.calculate
        """
        return float(Discernibility.calculate(generalized_df, algo.dataset.qids_idx))
