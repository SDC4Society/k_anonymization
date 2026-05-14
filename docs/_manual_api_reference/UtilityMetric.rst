.. type:: UtilityMetric
    :no-contents-entry:
    :annotation: : (DataFrame, Algorithm) -> Any

    Prototype of a utility metric for full-domain generalization algorithms.

    A utility metric assigns a score to a generalized candidate, enabling
    algorithms to rank or order candidates during search or solution selection.
    The return value must support the ``<`` operator — typically a ``float``,
    but n-dimensional vectors or any other comparable type are equally valid.
    Lower scores are preferred (minimization).

    :param generalized_df: The fully generalized dataframe for one candidate.
    :type generalized_df: DataFrame
    :param algo: The algorithm instance, providing access to ``org_data``, ``dataset``
        (hierarchies, qids_idx, is_categorical), and ``k``.
    :type algo: Algorithm

    :returns: *Any* -- A score supporting ``<`` comparison. Lower values are preferred.

    .. seealso::

        :class:`UtilityMetricBuiltIn`
            A set of built-in ``UtilityMetric`` implementations.
