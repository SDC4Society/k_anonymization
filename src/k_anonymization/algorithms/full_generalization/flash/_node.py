from typing import List

import numpy as np


class Node:
    """
    A node in the generalization lattice for the Flash algorithm.

    Each node represents a specific combination of generalization levels
    for all QID attributes, identified by a tuple of integers. Nodes are
    ordered by their criterion vector ``(c1, c2, c3)`` for use in
    priority-queue-based search.

    Parameters
    ----------
    generalization_tuple : tuple
        A tuple of integers specifying the generalization level for each
        QID attribute. For example, ``(0, 1, 2)`` means the first QID is
        at level 0, the second at level 1, and the third at level 2.

    Attributes
    ----------
    generalization_tuple : tuple
        The generalization level for each QID attribute.
    height : int
        Sum of all generalization levels.
    tagged : bool
        Whether this node has been visited during search.
    k_ano : bool
        Whether this node satisfies k-anonymity.
    criterion : tuple
        Priority vector ``(c1, c2, c3)`` computed by ``calculate_criterion``.
    """

    def __init__(self, generalization_tuple: tuple):
        self.generalization_tuple: tuple = generalization_tuple
        self.height: int = sum(generalization_tuple)
        self.tagged: bool = False
        self.k_ano: bool = False
        self.criterion: tuple = (None, None, None)

    def calculate_criterion(
        self,
        max_levels: list[int],
        distinct_counts: list[list[int]],
    ) -> None:
        """
        Compute the priority criterion vector ``(c1, c2, c3)``.

        Defined in Section 5-B of the Flash paper. Lower values are
        preferred for all three components.

        Parameters
        ----------
        max_levels : list[int]
            Maximum generalization level for each QID attribute.
        distinct_counts : list[list[int]]
            ``distinct_counts[j][level]`` is the number of distinct values
            at the given generalization level for the j-th QID.
        """
        c1 = self.height
        c2 = np.average(
            [
                h / max_levels[j]
                for j, h in enumerate(self.generalization_tuple)
            ]
        )
        c3 = 1 - np.average(
            [
                distinct_counts[j][h] / distinct_counts[j][0]
                for j, h in enumerate(self.generalization_tuple)
            ]
        )
        self.criterion = (c1, float(c2), float(c3))

    def get_upper_neighbor_index(self, max_levels: list[int]) -> List[tuple]:
        """
        Return generalization tuples of all direct upper (more generalized) neighbors.

        Parameters
        ----------
        max_levels : list[int]
            Maximum generalization level for each QID attribute.

        Returns
        -------
        list[tuple]
            Generalization tuples of reachable upper neighbors.
        """
        neighbors = []
        for i in range(len(self.generalization_tuple)):
            candidate = list(self.generalization_tuple)
            candidate[i] += 1
            if candidate[i] <= max_levels[i]:
                neighbors.append(tuple(candidate))
        return neighbors

    def get_lower_neighbor_index(self, max_levels: list[int]) -> List[tuple]:
        """
        Return generalization tuples of all direct lower (less generalized) neighbors.

        Parameters
        ----------
        max_levels : list[int]
            Maximum generalization level for each QID attribute.

        Returns
        -------
        list[tuple]
            Generalization tuples of reachable lower neighbors.
        """
        neighbors = []
        for i in range(len(self.generalization_tuple)):
            candidate = list(self.generalization_tuple)
            candidate[i] -= 1
            if candidate[i] >= 0:
                neighbors.append(tuple(candidate))
        return neighbors

    def tag(self) -> None:
        """Mark this node as visited."""
        self.tagged = True

    def check_as_k_ano(self) -> None:
        """Mark this node as satisfying k-anonymity."""
        self.k_ano = True

    def __le__(self, other):
        return self.criterion <= other.criterion

    def __lt__(self, other):
        return self.criterion < other.criterion

    def __repr__(self):
        return f"Node{self.generalization_tuple}"
