from __future__ import annotations

from numpy import isin, median, ndarray

from k_anonymization.core.dataset import Dataset

from ..local_recoding_algorithm import (
    GroupAnonymization,
    GroupAnonymizationBuiltIn,
    LocalRecodingAlgorithm,
)


class Mondrian(LocalRecodingAlgorithm):
    """
    Implementation of Mondrian algorithm.

    Mondrian uses a top-down, greedy domain partitioning approach.
    It divides data into smaller groups by recursively splitting the
    widest-range QID attribute (of the local region at each step) until
    there is no longer an allowable split satisfying `k`-anonymity
    (i.e., no possible split that provides 2 groups of size ≥ `k`).
     on its median value,

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
        Default: ``GroupAnonymizationBuiltIn.GENERALIZATION``
    """

    def __init__(
        self,
        dataset: Dataset,
        k: int,
        group_anonymization: GroupAnonymization = GroupAnonymizationBuiltIn.GENERALIZATION,
    ):
        """
        Initialize Mondrian.

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
            Default: ``GroupAnonymizationBuiltIn.GENERALIZATION``
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
                return len(set(slice_data[:, idx])) / max_ranges[idx]
            return (slice_data[:, idx].max() - slice_data[:, idx].min()) / (
                max_ranges[idx]
            )

        return sorted(
            self.qids_idx,
            key=lambda x: (-get_normalized_range(x), -len(set(slice_data[:, x]))),
        )

    def __do_partitioning_numerical(self, slice_data: ndarray, idx: int):
        """
        Try partitioning a numerical QID attribute on its median value.

        Parameters
        ----------
        slice_data : numpy.ndarray
            Data slice.
        idx : int
            Column index of the QID attribute in ``slice_data``.

        Returns
        -------
        list[numpy.ndarray] | None
            List of resulting partitions, or `None` if no allowable split is done.
        """
        median_val = median(slice_data[:, idx])
        left = slice_data[slice_data[:, idx] <= median_val]
        if left.shape[0] < self.k:
            return None
        right = slice_data[slice_data[:, idx] > median_val]
        if right.shape[0] < self.k:
            return None
        return [left, right]

    def __do_partitioning_categorical(self, slice_data: ndarray, idx: int):
        """
        Try partitioning a categorical QID attribute.

        Find the lowest common ancestor (LCA), then for every direct child
        of the LCA, group its leaf nodes as a partition.

        Parameters
        ----------
        slice_data : numpy.ndarray
            Data slice.
        idx : int
            Column index of the QID attribute in ``slice_data``.

        Returns
        -------
        list[numpy.ndarray] | None
            List of resulting partitions, or `None` if no allowable split is done.
        """
        unique_vals = set(slice_data[:, idx])
        if len(unique_vals) == 1:
            return None
        lca_height = self.hierarchies[idx].get_lowest_common_ancestor(
            unique_vals, get_type="height"
        )
        partitions = []
        allowable = True
        if lca_height == 1:
            for value in unique_vals:
                partition = slice_data[slice_data[:, idx] == value]
                if partition.shape[0] < self.k:
                    allowable = False
                    break
                partitions.append(partition)
        else:
            hierarchy = self.hierarchies[idx]
            for value in hierarchy.hierarchy_df[lca_height - 1].unique():
                leaves = hierarchy.get_leaves_under_node(value)
                partition = slice_data[isin(slice_data[:, idx], leaves)]
                if partition.shape[0] == 0:
                    continue
                if partition.shape[0] < self.k:
                    allowable = False
                    break
                partitions.append(partition)
        if allowable is True:
            return partitions
        return None

    def do_partitioning(self, slice_data: ndarray, sorted_qids_idx: list[int]):
        """
        Iterate through each QID attribute in ``sorted_qids_idx`` and try
        partitioning the data (slice).

        Partitioning method:

        - Numerical Attributes: Median partitioning.

        - Categorical Attributes: Find the lowest common ancestor (LCA),
          then for every direct child of the LCA, group its leaf nodes as
          a partition.

        Parameters
        ----------
        slice_data : numpy.ndarray
            Data slice.
        sorted_qids_idx : list[int]
            Sorted list of QID attribute indices.

        Returns
        -------
        list[numpy.ndarray] | None
            List of resulting partitions, or `None` if no allowable split is done.
        """
        for idx in sorted_qids_idx:
            if idx in self.qids_idx_categorical:
                partitions = self.__do_partitioning_categorical(slice_data, idx)
            else:
                partitions = self.__do_partitioning_numerical(slice_data, idx)
            if partitions is None:
                continue
            return partitions
        return None

    def do_mondrian(self, np_data: ndarray):
        """
        The core of Mondrian's partitioning process.

        Original data is recursively partitioned into smaller data slices
        until no further partitioning is possible. For each data slice:

        1. Create a sorted list of QID attributes indices: by the
           widest normalized ranges, followed by cardinalities.

        2. Iterate through each QID attribute in the sorted list and try
           partitioning the data (slice). The loop stops immediately when
           an allowable split is done (a split resulting in sub-groups all
           of sizes ≥ `k`). If no splits are allowed, the data (slice) is
           kept as-is.

           Partitioning method:

            - Numerical Attributes: Median partitioning.

            - Categorical Attributes: Find the lowest common ancestor (LCA),
                then for every direct child of the LCA, group its leaf nodes as
                a partition.

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

        for idx in self.qids_idx:
            if idx in self.qids_idx_categorical:
                # Store categorical attribute ranges as their cardinalities.
                max_ranges[idx] = len(set(np_data[:, idx]))
            else:
                # Store numerical attribute ranges as max - min
                max_ranges[idx] = np_data[:, idx].max() - np_data[:, idx].min()

        groups = []

        partition_ables = self.do_partitioning(
            np_data, self.sort_qids_idx(np_data, max_ranges)
        )
        # If no split provide 2 partitions of size >= k
        if partition_ables is None:
            return [np_data.tolist()]

        while len(partition_ables) > 0:
            slice_data = partition_ables.pop()
            # If size < k*2 => impossible split into 2 partitions of size >= k
            if slice_data.shape[0] < self.k * 2:
                groups.append(slice_data.tolist())
                continue
            _partition_ables = self.do_partitioning(
                slice_data, self.sort_qids_idx(slice_data, max_ranges)
            )
            # If no split provide 2 partitions of size >= k
            if _partition_ables is None:
                groups.append(slice_data.tolist())
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
