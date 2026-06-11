from k_anonymization.core import Algorithm, Dataset

from ._criterion import DEFAULT_SEARCH_BUDGET_RATIO
from ._parameters import ParameterCalculation
from ._tabular import (
    GENERALIZATION_DEGREES,
    SCORE_FUNCTIONS,
    generalize_value,
    safe_pub_anonymize,
)


class SafePub(Algorithm):
    """
    Implementation of the SafePub algorithm.

    SafePub provides (epsilon, delta)-differential privacy through random
    sampling, full-domain generalization, and record suppression. The
    privacy parameters (epsilon, delta) are converted into a sampling
    probability ``beta`` and a class-size threshold ``k``; the algorithm
    then releases only the records of a random ``beta``-sample whose
    equivalence classes contain at least ``k`` records.

    In the data-dependent mode (default), a part of the privacy budget is
    spent on selecting the generalization levels with the exponential
    mechanism, mirroring ARX's ``DataDependentEDDPAlgorithm``. In the
    data-independent mode the generalization levels are fixed up front,
    like ARX's ``DataGeneralizationScheme``, and the full budget is spent
    on anonymization.

    Unlike the other algorithms in this package, SafePub is randomized:
    the released subset, and in the data-dependent mode also the chosen
    generalization, differ between runs by design.

    Parameters
    ----------
    dataset : Dataset
        The Dataset object holding the original data and its metadata.
    epsilon : float
        The privacy budget of (epsilon, delta)-differential privacy.
        Must be > 0.
    delta : float
        The privacy parameter delta in (0, 1). It is recommended to use
        a value smaller than ``1 / number_of_records``.
    data_dependent : bool, default True
        Whether to select the generalization levels with the exponential
        mechanism (spending ``dp_search_budget`` of epsilon on the
        search). If False, ``generalization_levels`` and/or
        ``generalization_degree`` must be provided.
    dp_search_budget : float or None, default None
        Absolute part of epsilon reserved for the data-dependent search.
        Must satisfy ``0 < dp_search_budget < epsilon``. If neither this
        nor ``dp_search_budget_ratio`` is given, 10% of epsilon is used,
        matching the ARX GUI default.
    dp_search_budget_ratio : float or None, default None
        Fraction of epsilon reserved for the data-dependent search.
        Mutually exclusive with ``dp_search_budget``.
    search_expansion_limit : int or None, default None
        Maximum number of data-dependent search expansions. Defaults to
        the size of the generalization lattice minus one, which is only
        practical for small lattices; supply an explicit limit (for
        example 30) when many QID attributes are involved.
    generalization_levels : dict or None, default None
        Fixed generalization level per QID attribute name
        (data-independent mode).
    generalization_degree : str or None, default None
        Fixed generalization degree applied to QID attributes without an
        explicit level (data-independent mode). One of ``none``, ``low``,
        ``low_medium``, ``medium``, ``medium_high``, ``high``,
        ``complete``; the level is ``round(factor * max_level)``, like
        ARX.
    utility_metric : str, default 'safepub_precision'
        The quality model whose SafePub score function drives the
        data-dependent search. One of ``safepub_precision``,
        ``safepub_loss``, ``safepub_discernibility``, ``safepub_entropy``,
        ``safepub_aecs``, or ``safepub_classification``.
    response_variables : list or None, default None
        Target column names for ``safepub_classification`` (columns may
        be quasi-identifying or not), like ARX's response variables.
    deterministic : bool, default False
        Use a fixed random seed. **For reproducibility tests only**: a
        deterministic run provides no differential-privacy guarantee.

    Attributes
    ----------
    epsilon : float
        The privacy budget of (epsilon, delta)-differential privacy.
    delta : float
        The privacy parameter delta.
    k : int
        The class-size threshold derived from (epsilon, delta). Records
        of the sample whose equivalence class is smaller than ``k`` are
        suppressed.
    beta : float
        The sampling probability derived from (epsilon, delta).
    levels : tuple or None
        The generalization level chosen for each QID attribute by the
        last ``anonymize`` run.
    score : float or None
        The SafePub score of the chosen generalization (data-dependent
        runs only). Mirrors the information loss Java ARX reports for
        data-dependent differential privacy: higher is better, negative
        for the Precision/Loss/Discernibility/Entropy models and positive
        for AECS/Classification.
    score_function : str or None
        The quality model that produced ``score``.
    sampled_count : int or None
        Number of records in the random ``beta``-sample of the last run.
    suppressed_sample_count : int or None
        Number of sampled records suppressed because their equivalence
        class was smaller than ``k``.

    References
    ----------
    Bild R, Kuhn KA, Prasser F. "SafePub: A Truthful Data Anonymization
    Algorithm With Strong Privacy Guarantees." Proceedings on Privacy
    Enhancing Technologies. 2018(1):67-87.
    """

    def __init__(
        self,
        dataset: Dataset,
        epsilon: float,
        delta: float,
        data_dependent: bool = True,
        dp_search_budget: float = None,
        dp_search_budget_ratio: float = None,
        search_expansion_limit: int = None,
        generalization_levels: dict = None,
        generalization_degree: str = None,
        utility_metric: str = "safepub_precision",
        response_variables: list = None,
        deterministic: bool = False,
    ):
        """
        Initialize the SafePub algorithm.

        Derives the SafePub parameters ``beta`` and ``k`` from
        (epsilon, delta) — after subtracting the search budget in the
        data-dependent mode — and registers ``k`` with the base
        ``Algorithm``.

        Parameters
        ----------
        dataset : Dataset
            The Dataset object holding the original data and its metadata.
        epsilon : float
            The privacy budget. Must be > 0.
        delta : float
            The privacy parameter delta in (0, 1).
        data_dependent : bool, default True
            Whether to select the generalization levels with the
            exponential mechanism.
        dp_search_budget : float, optional
            Absolute part of epsilon reserved for the search.
        dp_search_budget_ratio : float, optional
            Fraction of epsilon reserved for the search. Mutually
            exclusive with ``dp_search_budget``.
        search_expansion_limit : int, optional
            Maximum number of data-dependent search expansions.
        generalization_levels : dict, optional
            Fixed generalization level per QID attribute name
            (data-independent mode).
        generalization_degree : str, optional
            Fixed generalization degree (data-independent mode).
        utility_metric : str, optional
            The quality model driving the data-dependent search.
        response_variables : list, optional
            Target columns for ``safepub_classification``.
        deterministic : bool, optional
            Use a fixed random seed (reproducibility tests only).
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")
        if dp_search_budget is not None and dp_search_budget_ratio is not None:
            raise ValueError(
                "dp_search_budget and dp_search_budget_ratio are mutually exclusive"
            )

        resolved_budget = None
        epsilon_anonymization = epsilon
        if data_dependent:
            if dp_search_budget is not None:
                resolved_budget = float(dp_search_budget)
            elif dp_search_budget_ratio is not None:
                if not 0 < dp_search_budget_ratio < 1:
                    raise ValueError("dp_search_budget_ratio must be in (0, 1)")
                resolved_budget = epsilon * float(dp_search_budget_ratio)
            else:
                resolved_budget = epsilon * DEFAULT_SEARCH_BUDGET_RATIO
            if not 0 < resolved_budget < epsilon:
                raise ValueError(
                    "the search budget must satisfy 0 < budget < epsilon"
                )
            epsilon_anonymization = epsilon - resolved_budget

        parameter_calculation = ParameterCalculation(epsilon_anonymization, delta)
        super().__init__(dataset, parameter_calculation.get_k())

        self.epsilon = epsilon
        self.delta = delta
        self.beta = parameter_calculation.get_beta()
        self.levels = None
        self.score = None
        self.score_function = None
        self.sampled_count = None
        self.suppressed_sample_count = None
        self.__data_dependent = data_dependent
        self.__dp_search_budget = resolved_budget
        self.__search_expansion_limit = search_expansion_limit
        self.__generalization_levels = generalization_levels
        self.__generalization_degree = generalization_degree
        self.__utility_metric = utility_metric
        self.__response_variables = response_variables
        self.__deterministic = deterministic

    def anonymize(self):
        """
        Run the SafePub algorithm.

        Draws a random ``beta``-sample, selects the generalization levels
        (with the exponential mechanism in the data-dependent mode, or
        from the fixed scheme otherwise), and releases the generalized
        sampled records whose equivalence classes contain at least ``k``
        records. Records outside the sample and records of smaller
        classes are suppressed: they are removed from ``anon_data`` and
        the affected equivalence classes are listed in
        ``suppressed_qids`` as ``{"qid": ..., "count": ...}``.
        """
        qids = list(self.dataset.qids)
        records = self.org_data.astype(str).to_dict("records")
        hierarchies = {qid: self.__hierarchy_mapping(qid) for qid in qids}

        result = safe_pub_anonymize(
            records,
            tuple(qids),
            hierarchies,
            epsilon=self.epsilon,
            delta=self.delta,
            deterministic=self.__deterministic,
            data_dependent=self.__data_dependent,
            dp_search_budget=self.__dp_search_budget,
            search_expansion_limit=self.__search_expansion_limit,
            generalization_levels=self.__generalization_levels,
            generalization_degree=self.__generalization_degree,
            utility_metric=self.__utility_metric,
            response_variables=self.__response_variables,
        )

        self.levels = result.levels
        self.score = result.score
        self.score_function = result.score_function
        self.sampled_count = len(result.sampled_indices)
        self.suppressed_sample_count = result.suppressed_sample_count

        # Records of the beta-sample whose class reaches k are released;
        # everything else (non-sampled records, smaller classes) is
        # suppressed, i.e. removed from the output.
        counts = result.equivalence_class_counts
        released_rows = []
        for index in result.sampled_indices:
            key = tuple(
                generalize_value(records[index][qid], hierarchies[qid], level)
                for qid, level in zip(qids, result.levels)
            )
            if counts[key] >= result.k:
                released_rows.append(result.rows[index])

        self.suppressed_qids = [
            {"qid": key, "count": count}
            for key, count in counts.items()
            if count < result.k
        ]

        columns = list(self.org_data)
        data = [[row[column] for column in columns] for row in released_rows]
        self._construct_anon_data(
            data, columns=columns, table_name="SafePub Anonymized Data"
        )

    def __hierarchy_mapping(self, qid: str) -> dict:
        """
        Convert a ``Hierarchy`` into the mapping used by the SafePub core.

        The core represents a hierarchy as a mapping from each raw value
        to the tuple of its generalizations per level (level 0 being the
        raw value itself). All values are normalized to strings, matching
        the string conversion applied to the data.

        Parameters
        ----------
        qid : str
            The QID attribute name.

        Returns
        -------
        dict
            Mapping of raw value to its per-level generalizations.
        """
        hierarchy_df = self.dataset.hierarchies[qid].hierarchy_df.astype(str)
        return {
            row[0]: tuple(row)
            for row in hierarchy_df.itertuples(index=False, name=None)
        }


SafePub.SCORE_FUNCTIONS = SCORE_FUNCTIONS
"""Quality models available for ``utility_metric``."""

SafePub.GENERALIZATION_DEGREES = tuple(sorted(GENERALIZATION_DEGREES))
"""Degree names available for ``generalization_degree``."""
