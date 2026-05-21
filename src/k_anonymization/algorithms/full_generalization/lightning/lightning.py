import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import PriorityQueue

from k_anonymization.algorithms.full_generalization._generalization_scoring import (
    GeneralizationScoring,
)
from k_anonymization.algorithms.utils import generalize_column
from k_anonymization.core import Algorithm, Dataset
from k_anonymization.evaluation.anonymity import is_k_anonymous

from ._lattice import Lattice
from ._node import LightningNode


class Lightning(Algorithm):
    """
    Implementation of the Lightning algorithm.

    Lightning explores the generalization lattice using a combination of
    best-first (priority queue) and greedy (depth-first) search to find
    a k-anonymous generalization with high data utility.

    The search alternates between expansion steps (best-first) and greedy
    steps (depth-first), switching every ``greedy_interval`` steps. Nodes
    are ordered by a criterion vector ``(c1, c2, c3)`` that favors less
    generalized states.

    Parameters
    ----------
    dataset : Dataset
        The Dataset object holding the original data and its metadata.
    k : int
        The privacy parameter ``k``.
    generalization_scoring : GeneralizationScoring or None
        The scoring function used to select the best solution among all
        k-anonymous candidates found during search.
        If ``None`` (default), the internal criterion vector is used for
        solution selection, reproducing the original Lightning behavior.
    greedy_interval : int
        Frequency of greedy (depth-first) steps. A greedy step is
        performed every ``greedy_interval`` steps; all other steps use
        best-first expansion. Smaller values increase depth-first
        exploration, pushing the search toward higher lattice heights
        more aggressively. Larger values favor breadth-first expansion,
        exploring more nodes at each height before moving up.
        Default: 3.
    max_workers : int
        Number of parallel workers for node checking during expansion.
        Set to 1 for sequential execution. Default: 1.

    Attributes
    ----------
    generalization_scoring : GeneralizationScoring or None
        The scoring function used to select the best solution.
    """

    def __init__(
        self,
        dataset: Dataset,
        k: int,
        generalization_scoring: GeneralizationScoring | None = None,
        greedy_interval: int = 3,
        max_workers: int = 1,
    ):
        """
        Initialize the Lightning algorithm.

        Parameters
        ----------
        dataset : Dataset
            The Dataset object holding the original data and its metadata.
        k : int
            The privacy parameter ``k``.
        generalization_scoring : GeneralizationScoring or None
            The scoring function used to select the best solution among
            all k-anonymous candidates found during search.
            If ``None`` (default), the internal criterion vector is
            used, reproducing the original Lightning behavior.
        greedy_interval : int
            Frequency of greedy (depth-first) steps. Default: 3.
        max_workers : int
            Number of parallel workers for node checking. Default: 1.
        """
        super().__init__(dataset, k)
        self.generalization_scoring = generalization_scoring
        self.__greedy_interval: int = greedy_interval
        self.__lattice: Lattice = Lattice(dataset)
        self.__qids: list[str] = dataset.qids
        self.__qids_idx: list[int] = dataset.qids_idx
        self.__max_workers: int = max_workers
        self.__best_score = None
        self.__best_criterion: tuple | None = None
        self.__best_generalization: tuple | None = None
        self.__state_lock = threading.Lock()

    def anonymize(self):
        """
        Run the Lightning algorithm.

        Explores the generalization lattice using a priority queue.
        Every ``greedy_interval`` steps, switches from best-first
        expansion to greedy depth-first search. Criterion-based pruning
        is applied to skip nodes that cannot improve upon the current
        best solution.

        Raises
        ------
        RuntimeError
            If no generalization satisfying k-anonymity is found.
        """
        search_queue = PriorityQueue()

        bottom_node = self.__lattice.get_nodes_at_height(0)[0]
        self.__check(bottom_node)
        search_queue.put(bottom_node)
        step = 0

        while not search_queue.empty():
            next_node = search_queue.get()

            if self.__should_prune(next_node):
                continue

            step += 1
            if step % self.__greedy_interval == 0:
                self.__greedy(next_node, search_queue)
            else:
                self.__expand(next_node, search_queue)

        if self.__best_generalization is None:
            raise RuntimeError("No generalization satisfies k-anonymity.")

        best_df = self.__apply_generalization(self.__best_generalization)
        self._construct_anon_data(best_df.values, columns=list(best_df))

    def __should_prune(self, node: LightningNode) -> bool:
        """
        Check whether a node can be pruned from the search.

        A node is pruned if a k-anonymous solution has already been
        found and the node's criterion is worse (higher) than the
        current best. Pruning is always based on criterion, regardless
        of whether ``generalization_scoring`` is set.

        Parameters
        ----------
        node : LightningNode
            The node to evaluate.

        Returns
        -------
        bool
            Whether the node should be skipped.
        """
        return (
            self.__best_criterion is not None and node.criterion > self.__best_criterion
        )

    def __expand(
        self,
        node: LightningNode,
        search_queue: PriorityQueue,
    ) -> None:
        """
        Expand a node by checking all unvisited upper neighbors.

        Neighbors are checked (possibly in parallel) and added to the
        search queue if not already tagged.

        Parameters
        ----------
        node : LightningNode
            The node to expand.
        search_queue : PriorityQueue
            The global search queue to add discovered nodes to.
        """
        node.mark_as_expanded()
        successors = [
            self.__lattice[idx]
            for idx in node.get_upper_neighbor_index(self.__lattice.max_levels)
        ]
        to_check = [s for s in successors if not s.tagged and not s.expanded]

        with ThreadPoolExecutor(max_workers=self.__max_workers) as executor:
            future_to_node = {
                executor.submit(self.__check, successor): successor
                for successor in to_check
            }
            for future in as_completed(future_to_node):
                successor = future_to_node[future]
                future.result()
                with self.__state_lock:
                    if not successor.tagged:
                        successor.tag()
                        search_queue.put(successor)

    def __greedy(
        self,
        node: LightningNode,
        search_queue: PriorityQueue,
    ) -> None:
        """
        Perform a greedy depth-first search from the given node.

        Expands the node into a local queue, then recursively follows
        the most promising (lowest criterion) unvisited successor.
        Remaining successors are returned to the global search queue.

        Parameters
        ----------
        node : LightningNode
            The starting node for greedy search.
        search_queue : PriorityQueue
            The global search queue to return remaining nodes to.
        """
        local_queue = PriorityQueue()
        self.__expand(node, local_queue)

        if not local_queue.empty():
            next_node = local_queue.get()
            self.__greedy(next_node, search_queue)

        while not local_queue.empty():
            search_queue.put(local_queue.get())

    def __check(self, node: LightningNode) -> None:
        """
        Check a node for k-anonymity and update the best solution.

        Applies the generalization defined by the node's tuple, checks
        k-anonymity, and updates the criterion bound for pruning.
        The best solution is selected by criterion (when
        ``generalization_scoring`` is ``None``) or by the provided
        scoring function.

        Parameters
        ----------
        node : LightningNode
            The node to check.
        """
        generalized_df = self.__apply_generalization(node.generalization_tuple)
        k_ano = is_k_anonymous(generalized_df, self.k, self.__qids_idx)

        if not k_ano:
            return

        node.check_as_k_ano()

        with self.__state_lock:
            # Update pruning bound (always criterion-based; see __should_prune)
            is_new_best = (
                self.__best_criterion is None or node.criterion < self.__best_criterion
            )
            if is_new_best:
                self.__best_criterion = node.criterion

            # Select best solution among k-anonymous candidates
            if self.generalization_scoring is None:
                if is_new_best:
                    self.__best_generalization = node.generalization_tuple
            else:
                score = self.generalization_scoring(generalized_df, self)
                if self.__best_score is None or score < self.__best_score:
                    self.__best_score = score
                    self.__best_generalization = node.generalization_tuple

    def __apply_generalization(self, generalization_tuple: tuple):
        """
        Apply full-domain generalization defined by the given tuple.

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
