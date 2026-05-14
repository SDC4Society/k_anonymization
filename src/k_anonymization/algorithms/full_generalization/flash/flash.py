from queue import PriorityQueue
from typing import List

from k_anonymization.algorithms.utils import generalize_column
from k_anonymization.core import Algorithm, Dataset
from k_anonymization.evaluation.anonymity import is_k_anonymous

from ._lattice import Lattice
from ._node import Node


class Flash(Algorithm):
    """
    Implementation of the Flash algorithm.

    Flash explores the generalization lattice using a combination of
    bottom-up breadth-first traversal and binary search along greedy
    paths to efficiently find the optimal k-anonymous generalization
    with the lowest criterion score.

    The search leverages the monotonicity property of generalization:
    if a node satisfies k-anonymity, all of its ancestors do as well;
    if a node does not, none of its descendants do either.

    Parameters
    ----------
    dataset : Dataset
        The Dataset object holding the original data and its metadata.
    k : int
        The privacy parameter `k`.
    """

    def __init__(self, dataset: Dataset, k: int):
        """
        Initialize the Flash algorithm.

        Parameters
        ----------
        dataset : Dataset
            The Dataset object holding the original data and its metadata.
        k : int
            The privacy parameter `k`.
        """
        super().__init__(dataset, k)
        self.__lattice: Lattice = Lattice(dataset)
        self.__qids: list[str] = dataset.qids
        self.__qids_idx: list[int] = dataset.qids_idx

    def anonymize(self):
        """
        Run the Flash algorithm.

        Explores the generalization lattice bottom-up, constructing greedy
        paths and applying binary search on each path to locate the
        k-anonymity boundary. The k-anonymous node with the lowest
        criterion score is selected as the optimal solution.

        Raises
        ------
        RuntimeError
            If no generalization satisfying k-anonymity is found.
        """
        heap = PriorityQueue()
        self.__best_score = None
        self.__best_df = None

        for level in range(self.__lattice.max_height + 1):
            for node in self.__lattice.get_nodes_at_height(level):
                if not node.tagged:
                    path = self.__find_path(node)
                    self.__check_path(path, heap)

                    while not heap.empty():
                        failed_node = heap.get()
                        upper_indices = failed_node.get_upper_neighbor_index(
                            self.__lattice.max_levels
                        )
                        for idx in upper_indices:
                            child = self.__lattice[idx]
                            if not child.tagged:
                                path = self.__find_path(child)
                                self.__check_path(path, heap)

        if self.__best_df is None:
            raise RuntimeError("No generalization satisfies k-anonymity.")

        self._construct_anon_data(
            self.__best_df.values, columns=list(self.__best_df)
        )

    def __apply_generalization(self, generalization_tuple: tuple):
        """
        Apply full-domain generalization defined by the given tuple.

        For each QID attribute, generalizes the original data from level 0
        to the level specified in the tuple using ``generalize_column``.

        Parameters
        ----------
        generalization_tuple : tuple
            The generalization level for each QID attribute.

        Returns
        -------
        DataFrame
            A copy of the original data with generalization applied.
        """
        generalized_df = self.org_data.copy()
        for i, level in enumerate(generalization_tuple):
            if level == 0:
                continue
            qid = self.__qids[i]
            generalized_values, _ = generalize_column(
                generalized_df[qid],
                self.dataset.hierarchies[qid],
                level_from=0,
                level_to=level,
            )
            generalized_df[qid] = generalized_values
        return generalized_df

    def __find_path(self, start_node: Node) -> List[Node]:
        """
        Build a greedy path of untagged nodes from start_node toward the top.

        Extends the path by selecting the first untagged upper neighbor at
        each step. Stops when the top node is reached or all upper neighbors
        are already tagged.

        Parameters
        ----------
        start_node : Node
            The starting node of the path.

        Returns
        -------
        list[Node]
            Ordered list of nodes forming the path (lower to higher).
        """
        path = [start_node]
        node = start_node

        while node != self.__lattice.top:
            upper_indices = node.get_upper_neighbor_index(self.__lattice.max_levels)
            found_next = False
            for idx in upper_indices:
                child = self.__lattice[idx]
                if not child.tagged:
                    path.append(child)
                    node = child
                    found_next = True
                    break
            if not found_next:
                break

        return path

    def __check_path(
        self,
        path: List[Node],
        heap: PriorityQueue,
    ) -> None:
        """
        Binary search on a path to locate the k-anonymity boundary.

        Nodes that fail k-anonymity are added to ``heap`` as candidates
        for further exploration. When a k-anonymous node is found, its
        criterion score is compared against the current best and the
        global optimum is updated if it is lower.

        Parameters
        ----------
        path : list[Node]
            The path to search (lower index = less generalized).
        heap : PriorityQueue
            Queue collecting non-k-anonymous nodes for further exploration.
        """
        low, high = 0, len(path) - 1

        while low <= high:
            mid = (low + high) // 2
            node = path[mid]
            node.tag()

            generalized_df = self.__apply_generalization(node.generalization_tuple)
            k_ano = is_k_anonymous(generalized_df, self.k, self.__qids_idx)

            if k_ano:
                node.check_as_k_ano()
                self.__tagging_upper_nodes(node)
                if self.__best_score is None or node.criterion < self.__best_score:
                    self.__best_score = node.criterion
                    self.__best_df = generalized_df
                high = mid - 1
            else:
                heap.put(node)
                self.__tagging_lower_nodes(node)
                low = mid + 1

    def __tagging_upper_nodes(self, start_node: Node) -> set[Node]:
        """
        Recursively tag all reachable untagged upper nodes as k-anonymous.

        Leverages the monotonicity property: if a node satisfies
        k-anonymity, all of its more generalized ancestors do as well.

        Parameters
        ----------
        start_node : Node
            The node from which to propagate upward.

        Returns
        -------
        set[Node]
            The set of newly tagged nodes.
        """
        found = set()
        upper_indices = start_node.get_upper_neighbor_index(self.__lattice.max_levels)
        for idx in upper_indices:
            upper = self.__lattice[idx]
            if not upper.tagged:
                found.add(upper)
                upper.tag()
                upper.check_as_k_ano()
                found.update(self.__tagging_upper_nodes(upper))
        return found

    def __tagging_lower_nodes(self, start_node: Node) -> set[Node]:
        """
        Recursively tag all reachable untagged lower nodes as not k-anonymous.

        Leverages the monotonicity property: if a node does not satisfy
        k-anonymity, none of its less generalized descendants do either.

        Parameters
        ----------
        start_node : Node
            The node from which to propagate downward.

        Returns
        -------
        set[Node]
            The set of newly tagged nodes.
        """
        found = set()
        lower_indices = start_node.get_lower_neighbor_index(self.__lattice.max_levels)
        for idx in lower_indices:
            lower = self.__lattice[idx]
            if not lower.tagged:
                found.add(lower)
                lower.tag()
                found.update(self.__tagging_lower_nodes(lower))
        return found
