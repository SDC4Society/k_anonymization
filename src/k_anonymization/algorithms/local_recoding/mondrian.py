from __future__ import annotations

from numpy import median, ndarray
from pandas import factorize, unique

from k_anonymization.core.dataset import Dataset

from .local_recoding_algorithm import (
    GroupAnonymization,
    GroupAnonymizationBuiltIn,
    LocalRecodingAlgorithm,
)


class MondrianClassic(LocalRecodingAlgorithm):
    """
    Implementation of Classic Mondrian algorithm.

    Classic Mondrian uses a top-down, greedy domain partitioning
    approach. It divides data into smaller groups by recursively
    splitting the widest-range QID attribute (of the local region at
    each step) on its median value, until there is no longer an
    allowable split satisfying `k`-anonymity (i.e., no possible split
    that provides 2 groups of size ≥ `k`).

    Parameters
    ----------
    dataset : Dataset
        Dataset object holding the original data and its metadata.
    k : int
        Privacy parameter `k`.
    group_anonymization : GroupAnonymization
        The method to anonymize the resulting groups after applying
        local recoding.
        It is possible to use an example method in
        ``GroupAnonymizationBuiltIn``, or create a custom method
        ``custom_group_anonymization(group: list, props: Any) -> list``.
        Default: ``GroupAnonymizationBuiltIn.SUMMARIZATION``
    """

    def __init__(
        self,
        dataset: Dataset,
        k: int,
        group_anonymization: GroupAnonymization = GroupAnonymizationBuiltIn.SUMMARIZATION,
    ):
        """
        Initialize Classic Mondrian.

        Parameters
        ----------
        dataset : Dataset
            The Dataset object holding the original data and its metadata.
        k : int
            The privacy parameter `k`.
        group_anonymization
            The method to anonymize the resulting groups after applying
            local recoding.
            It is possible to use an example method in
            ``GroupAnonymizationBuiltIn``, or create a custom method
            ``custom_group_anonymization(group: list, props: Any) -> list``.
            Default: ``GroupAnonymizationBuiltIn.SUMMARIZATION``
        """
        super().__init__(dataset, k, group_anonymization)
        self.qids_idx_categorical = self.dataset.qids_idx_categorical

    def sort_qids_idx(self, slice_data: ndarray, max_ranges: dict[int]):
        """
        Determine the order of QID attributes to attempt splitting.

        Heuristically choose the QID attributes with the widest normalized
        range first to minimize information loss. If the normalized ranges
        are tied, pick the attribute with more distinct values.

        Parameters
        ----------
        slice_data : numpy.ndarray
            The records within the current partition.
        max_ranges : dict[int]
            Dictionary storing the ranges of each attribute (for normalization)

        Returns
        -------
        list
            Sorted list of QID indices for splitting attempts.
        """

        def get_normalized_range(idx):
            if idx in self.qids_idx_categorical:
                return unique(slice_data[:, idx]).size / max_ranges[idx]
            return (slice_data[:, idx].max() - slice_data[:, idx].min()) / (
                max_ranges[idx]
            )

        return sorted(
            self.qids_idx,
            key=lambda x: (-get_normalized_range(x), -unique(slice_data[:, x]).size),
        )

    def do_partitioning(self, slice_data: ndarray, sorted_qids_idx: list[int]):
        """
        Iterate through each QID attribute in ``sorted_qids_idx`` and try
        partitioning the data (slice).

        Partitioning method: Median partitioning.

        Parameters
        ----------
        slice_data : numpy.ndarray
            Data slice.
        sorted_qids_idx : list[int]
            Sorted list of QID attribute indices.

        Returns
        -------
        list[numpy.ndarray] | None
            List of resulting partitions, or `None` if allowable split is done.
        """
        for idx in sorted_qids_idx:
            median_val = median(slice_data[:, idx])
            left = slice_data[slice_data[:, idx] <= median_val]
            if left.shape[0] < self.k:
                continue
            right = slice_data[slice_data[:, idx] > median_val]
            if right.shape[0] < self.k:
                continue
            return [left, right]
        return None

    def do_mondrian(self, np_data: ndarray):
        """
        The core of Classic Mondrian's partitioning process. At every step:

        1. Create a sorted list of QID attributes indices: by the
           widest normalized ranges, followed by cardinalities.

        2. Iterate through each QID attribute in the sorted list and try
           partitioning the data (slice). The loop stops immediately when
           an allowable split is done (a split resulting in sub-groups all
           of sizes ≥ `k`). If no splits are allowed, the data (slice) is
           kept as-is.

           Partitioning method: Median partitioning.

        Parameters
        ----------
        np_data : numpy.ndarray
            Original data in ``numpy.ndarray`` type.

        Returns
        -------
        list
            List of resulting partitions.
        """
        max_ranges = {}
        cat_value_map = {}

        for idx in self.qids_idx:
            if idx in self.qids_idx_categorical:
                # Store categorical attribute ranges as their cardinalities
                # Factorize categorical attributes to allow for median-based partitioning
                # Store factorized-original value maps to revert them later.
                encode, value_map = factorize(np_data[:, idx])
                max_ranges[idx] = value_map.size
                cat_value_map[idx] = value_map
                np_data[:, idx] = encode
                np_data[:, idx].astype(int)
            else:
                # Store numerical attribute ranges as max - min
                max_ranges[idx] = np_data[:, idx].max() - np_data[:, idx].min()

        # Function to revert categorical attributes from factorized values
        # back to their original values
        def ___restore_cat_qids_idx(slice_data: ndarray):
            for idx in self.qids_idx_categorical:
                slice_data[:, idx] = cat_value_map[idx][slice_data[:, idx].astype(int)]
            return slice_data.tolist()

        groups = []

        partition_ables = self.do_partitioning(
            np_data, self.sort_qids_idx(np_data, max_ranges)
        )
        # If no split provide 2 partitions of size >= k
        if partition_ables is None:
            return [___restore_cat_qids_idx(np_data)]

        while len(partition_ables) > 0:
            slice_data = partition_ables.pop()
            # If size < k*2 => impossible split into 2 partitions of size >= k
            if slice_data.shape[0] < self.k * 2:
                groups.append(___restore_cat_qids_idx(slice_data))
                continue
            _partition_ables = self.do_partitioning(
                slice_data, self.sort_qids_idx(slice_data, max_ranges)
            )
            # If no split provide 2 partitions of size >= k
            if _partition_ables is None:
                groups.append(___restore_cat_qids_idx(slice_data))
            else:
                partition_ables[:0] = _partition_ables

        return groups

    def do_local_recoding(self):
        """
        Prepare data and initiate the Mondrian partitioning process.

        Returns
        -------
        list
            List of resulting partitions
        """
        # Using numpy.ndarray is significantly faster than pandas.DataFrame
        np_data = self.anon_data.values

        # If data size < k*2 => impossible split into 2 partitions of size >= k
        if np_data.shape[0] < self.k * 2:
            return [np_data.tolist()]

        return self.do_mondrian(np_data)


class MondrianClassicUnOptimized(LocalRecodingAlgorithm):
    """
    Implementation of Classic Mondrian algorithm. This class uses recursion
    for partitioning, which is less memory efficient and potentially causes
    stack overflow.

    Classic Mondrian uses a top-down, greedy domain partitioning
    approach. It divides data into smaller groups by recursively
    splitting the widest-range QID attribute (of the local region at
    each step) on its median value, until there is no longer an
    allowable split satisfying `k`-anonymity (i.e., no possible split
    that provides 2 groups of size ≥ `k`).

    Parameters
    ----------
    dataset : Dataset
        The Dataset object holding the original data and its metadata.
    k : int
        The privacy parameter `k`.
    group_anonymization : GroupAnonymization
        The method to anonymize the resulting groups after applying
        local recoding.
        It is possible to use an example method in
        ``GroupAnonymizationBuiltIn``, or create a custom method
        ``custom_group_anonymization(group: list, props: Any) -> list``.
        Default: ``GroupAnonymizationBuiltIn.SUMMARIZATION``
    """

    def __init__(
        self,
        dataset: Dataset,
        k: int,
        group_anonymization: GroupAnonymization = GroupAnonymizationBuiltIn.SUMMARIZATION,
    ):
        """
        Initialize Classic Mondrian.

        Parameters
        ----------
        dataset : Dataset
            The Dataset object holding the original data and its metadata.
        k : int
            The privacy parameter `k`.
        group_anonymization
            The method to anonymize the resulting groups after applying
            local recoding.
            It is possible to use an example method in
            ``GroupAnonymizationBuiltIn``, or create a custom method
            ``custom_group_anonymization(group: list, props: Any) -> list``.
            Default: ``GroupAnonymizationBuiltIn.SUMMARIZATION``
        """
        super().__init__(dataset, k, group_anonymization)
        self.__ranges = {}
        self.__cat_uniques = {}
        self.__cat_encodes = {}
        for pos, qid in enumerate(dataset.qids):
            if self.is_categorical[pos]:
                # Store categorical attribute range as the size of unique values
                # Store the original and encoded categorical values
                encode, unique = factorize(self.org_data[qid])
                self.__ranges[self.qids_idx[pos]] = unique.size
                self.__cat_uniques[self.qids_idx[pos]] = unique
                self.__cat_encodes[self.qids_idx[pos]] = encode
            else:
                # Store numerical attribute range as [min, max]
                self.__ranges[self.qids_idx[pos]] = [
                    self.org_data[qid].min(),
                    self.org_data[qid].max(),
                ]

    def sort_qids_idx(self, slice_data: ndarray):
        """
        Determine the order of QID attributes to attempt splitting.

        Heuristically choose the QID attributes with the widest normalized
        range first to minimize information loss. If the normalized ranges
        are tied, pick the attribute with more distinct values.

        Parameters
        ----------
        slice_data : numpy.ndarray
            The records within the current partition.

        Returns
        -------
        list
            Sorted list of QID indices for splitting attempts.
        """

        def get_normalized_range(idx):
            slice_col = slice_data[:, idx]
            # If idx is a categorical attribute
            # (self.__ranges[idx] stores a single integer):
            if isinstance(self.__ranges[idx], int):
                return unique(slice_col).size / self.__ranges[idx]
            else:
                return (slice_col.max() - slice_col.min()) / (
                    self.__ranges[idx][1] - self.__ranges[idx][0]
                )

        return sorted(
            self.qids_idx,
            key=lambda x: (
                -get_normalized_range(x),
                -unique(slice_data[:, x]).size,
            ),
        )

    def __restore_cat_qids_idx(self, slice_data):
        """
        Retore original values of categorical attribute.

        Classic Mondrian processes categorical data as integers;
        this method reverses that mapping before final output.

        Parameters
        ----------
        slice_data : numpy.ndarray
            The records within the current partition.
        """
        for pos, idx in enumerate(self.qids_idx):
            if self.is_categorical[pos]:
                slice_data[:, idx] = self.__cat_uniques[idx][
                    slice_data[:, idx].astype(int)
                ]

    def do_classic_mondrian(self, slice_data: ndarray):
        """
        The recursive core of the Classic Mondrian algorithm.

        Splits a group into two smaller sub-groups at the median
        of the chosen dimension. A split is only accepted if both
        resulting sub-groups contain at least `k` records.

        Parameters
        ----------
        slice_data : numpy.ndarray
            The records within the current partition.

        Returns
        -------
        list
            A list of final partitions.
        """
        if slice_data.shape[0] < self.k * 2:
            self.__restore_cat_qids_idx(slice_data)
            return [slice_data.tolist()]

        sorted_qids_idx = self.sort_qids_idx(slice_data)
        for idx in sorted_qids_idx:
            median_value = median(slice_data[:, idx])
            lhs = slice_data[slice_data[:, idx] <= median_value]
            rhs = slice_data[slice_data[:, idx] > median_value]

            if lhs.shape[0] < self.k or rhs.shape[0] < self.k:
                continue
            else:
                return self.do_classic_mondrian(lhs) + self.do_classic_mondrian(rhs)
        self.__restore_cat_qids_idx(slice_data)
        return [slice_data.tolist()]

    def do_local_recoding(self):
        """
        Prepare data and initiate the Mondrian partitioning process.

        Factors categorical attributes into integers to allow for
        median-based splitting, then calls the recursive partitioner.

        Returns
        -------
        list
            The collection of partitions (equivalence classes) produced
            by the algorithm.
        """
        np_data = self.anon_data.values
        for pos, idx in enumerate(self.qids_idx):
            if self.is_categorical[pos]:
                # TODO: preserve ordinal order
                np_data[:, idx] = self.__cat_encodes[idx]
        partitions = self.do_classic_mondrian(np_data)
        return partitions
