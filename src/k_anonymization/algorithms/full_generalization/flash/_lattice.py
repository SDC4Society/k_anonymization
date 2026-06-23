import itertools
from typing import List

import numpy as np

from k_anonymization.core import Dataset

from ._node import Node


class Lattice:
    """
    Generalization lattice for the Flash algorithm.

    Stores all possible generalization nodes in a multi-dimensional numpy
    array, where each axis corresponds to a QID attribute and its size
    equals the attribute's maximum generalization level + 1.

    Parameters
    ----------
    dataset : Dataset
        The Dataset object providing QID definitions and hierarchies.

    Attributes
    ----------
    qids : list[str]
        Names of the QID attributes, in positional order.
    max_levels : list[int]
        Maximum generalization level for each QID attribute.
    distinct_counts : list[list[int]]
        ``distinct_counts[j][level]`` is the number of distinct values
        at the given generalization level for the j-th QID.
    _row_codes : list[numpy.ndarray]
        ``_row_codes[j]`` holds the integer code of each row's level-0
        value for the j-th QID. Used for DataFrame-free k-anonymity checks.
    _level_lut : list[list[numpy.ndarray]]
        ``_level_lut[j][level][c]`` is the integer code of the generalized
        value at the given level for the level-0 code ``c`` of the j-th QID.
    shape : tuple
        Shape of the lattice array.
    max_height : int
        Maximum height (sum of all max generalization levels).
    top : Node
        The top node where all QIDs are at their maximum generalization level.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.qids: list[str] = dataset.qids
        hierarchies = [dataset.hierarchies[qid] for qid in self.qids]
        self.max_levels: list[int] = [h.height for h in hierarchies]
        self.distinct_counts: list[list[int]] = []
        self._row_codes: list[np.ndarray] = []
        self._level_lut: list[list[np.ndarray]] = []
        for qid, hierarchy, max_level in zip(self.qids, hierarchies, self.max_levels):
            h_df = hierarchy.hierarchy_df
            self.distinct_counts.append(
                [h_df[level].nunique() for level in range(max_level + 1)]
            )

            column = dataset.df[qid]
            if column.isna().any():
                raise ValueError(
                    f"QID '{qid}' contains missing values (NaN/NA); Flash's "
                    "integer-encoded k-anonymity check requires complete QID columns."
                )
            categories, row_codes = np.unique(column.to_numpy(), return_inverse=True)
            self._row_codes.append(row_codes.astype(np.int64))

            luts: list[np.ndarray] = []
            for level in range(max_level + 1):
                level_map = dict(zip(h_df[0], h_df[level]))
                generalized = np.array(
                    [level_map[value] for value in categories], dtype=object
                )
                _, generalized_codes = np.unique(generalized, return_inverse=True)
                luts.append(generalized_codes.astype(np.int64))
            self._level_lut.append(luts)

        self.shape: tuple = tuple(ml + 1 for ml in self.max_levels)
        self.max_height: int = sum(self.max_levels)

        self.lattice = np.full(self.shape, fill_value=None, dtype=object)
        for g_tuple in np.ndindex(self.shape):
            node = Node(g_tuple)
            node.calculate_criterion(self.max_levels, self.distinct_counts)
            self.lattice[g_tuple] = node

        self.top: Node = self.lattice[tuple(self.max_levels)]

    def __getitem__(self, key: tuple) -> Node:
        """
        Retrieve the node at the given generalization tuple.

        Parameters
        ----------
        key : tuple
            A generalization tuple indexing into the lattice.

        Returns
        -------
        Node
            The node at the specified position.
        """
        return self.lattice[key]

    def k_anonymity(self, generalization_tuple: tuple) -> int:
        """
        Compute the k-anonymity of a generalization via integer encoding.

        Returns the size of the smallest equivalence class produced by the
        full-domain generalization defined by ``generalization_tuple``,
        without materializing a (string) DataFrame.

        Each QID column is reduced to its precomputed generalized integer
        codes (``_level_lut[j][level][_row_codes[j]]``); the per-QID codes
        are combined into a single key (mixed-radix when it fits in int64,
        otherwise a structured row view) and equivalence-class sizes are
        counted with ``numpy``. Because the per-level encoding is bijective,
        the resulting size multiset is identical to
        ``df.groupby(qids).size()`` on the generalized string data.

        Parameters
        ----------
        generalization_tuple : tuple
            The generalization level for each QID attribute.

        Returns
        -------
        int
            The minimum equivalence-class size (the level of k-anonymity).
        """
        columns = []
        radices = []
        for j, level in enumerate(generalization_tuple):
            generalized = self._level_lut[j][level][self._row_codes[j]]
            columns.append(generalized)
            radices.append(self.distinct_counts[j][level])

        product = 1
        for radix in radices:
            product *= radix

        if product < (1 << 62):
            key = columns[0].copy()
            for column, radix in zip(columns[1:], radices[1:]):
                key = key * radix + column
            if product < (1 << 22):
                counts = np.bincount(key)
                return int(counts[counts > 0].min())
            counts = np.unique(key, return_counts=True)[1]
            return int(counts.min())

        stacked = np.stack(columns, axis=1)
        view = np.ascontiguousarray(stacked).view(
            [("", stacked.dtype)] * stacked.shape[1]
        )
        return int(np.unique(view, return_counts=True)[1].min())

    def get_nodes_at_height(self, height: int) -> List[Node]:
        """
        Get all nodes whose generalization levels sum to ``height``.

        Uses a stars-and-bars combinatorial enumeration to find all
        integer partitions of ``height`` across the lattice dimensions,
        filtering out those that exceed the per-dimension bounds.

        Parameters
        ----------
        height : int
            The target height.

        Returns
        -------
        list[Node]
            All nodes at the specified height.
        """
        dim = len(self.shape)
        node_indices = []
        for bars in itertools.combinations(range(height + dim - 1), dim - 1):
            seps = [-1] + list(bars) + [height + dim - 1]
            coordinate = tuple(seps[i + 1] - seps[i] - 1 for i in range(dim))
            if all(coordinate[i] < self.shape[i] for i in range(dim)):
                node_indices.append(coordinate)
        return [self[idx] for idx in node_indices]
