import itertools
from typing import List

from k_anonymization.core import Dataset

from ._node import LightningNode


class Lattice:
    """
    Generalization lattice for the Lightning algorithm.

    Unlike the Flash lattice (which pre-allocates all nodes in a numpy
    array), this lattice lazily creates nodes on first access using a
    dictionary. This avoids the upfront cost of materializing every
    node in high-dimensional lattices.

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
    shape : tuple
        Shape of the lattice (each dimension is max_level + 1).
    max_height : int
        Maximum height (sum of all max generalization levels).
    top : LightningNode
        The top node where all QIDs are at their maximum generalization level.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.qids: list[str] = dataset.qids
        hierarchies = [dataset.hierarchies[qid] for qid in self.qids]
        self.max_levels: list[int] = [h.height for h in hierarchies]
        self.distinct_counts: list[list[int]] = []
        for hierarchy, max_level in zip(hierarchies, self.max_levels):
            h_df = hierarchy.hierarchy_df
            self.distinct_counts.append(
                [h_df[level].nunique() for level in range(max_level + 1)]
            )

        self.shape: tuple = tuple(ml + 1 for ml in self.max_levels)
        self.max_height: int = sum(self.max_levels)

        self.__nodes: dict[tuple, LightningNode] = {}
        self.top: LightningNode = self[tuple(self.max_levels)]

    def __getitem__(self, key: tuple) -> LightningNode:
        """
        Retrieve or create the node at the given generalization tuple.

        Parameters
        ----------
        key : tuple
            A generalization tuple indexing into the lattice.

        Returns
        -------
        LightningNode
            The node at the specified position.
        """
        if key not in self.__nodes:
            node = LightningNode(key)
            node.calculate_criterion(self.max_levels, self.distinct_counts)
            self.__nodes[key] = node
        return self.__nodes[key]

    def get_nodes_at_height(self, height: int) -> List[LightningNode]:
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
        list[LightningNode]
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
