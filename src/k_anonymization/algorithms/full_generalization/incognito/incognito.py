from queue import PriorityQueue

from k_anonymization.algorithms.utils import generalize_column
from k_anonymization.core import Algorithm, Dataset
from k_anonymization.core.frame import ITableDF
from k_anonymization.evaluation.anonymity import is_k_anonymous

from ._lattice import Lattice


class Incognito(Algorithm):
    """
    Implementation of Incognito algorithm.

    Incognito is a bottom-up algorithm that explores the generalization lattice to
    identify all transformations satisfying k-anonymity.

    It efficiently prunes the search space by leveraging the monotonicity property
    of generalization, where if a node in the lattice satisfies `k`-anonymity, all
    of its direct ancestors (more generalized nodes) are guaranteed to satisfy it
    as well.

    Parameters
    ----------
    dataset : Dataset
        The Dataset object holding the original data and its metadata.
    k : int
        The privacy parameter `k`.

    Attributes
    ----------
    solutions : list[ITableDF]
        All anonymized tables that satisfy k-anonymity. Best solution is stored in
        ``anon_data``. Configurable utility metric support will be added in a
        follow-up update.
    """

    def __init__(self, dataset: Dataset, k: int):
        """
        Initialize the Incognito algorithm.

        Parameters
        ----------
        dataset : Dataset
            The Dataset object holding the original data and its metadata.
        k : int
            The privacy parameter `k`.
        """
        super().__init__(dataset, k)
        self.solutions: list[ITableDF] = []
        self.__lattice: Lattice = Lattice(dataset)
        self.__num_qids: int = len(dataset.qids)
        self.__pqueue = PriorityQueue()
        self.__qids_idx_map = {
            qid: dataset.qids_idx[pos] for pos, qid in enumerate(dataset.qids)
        }

    def __apply_node_generalization(self, generalization: list[tuple[str, int]]):
        """Apply one lattice node's full-domain generalization to original data."""
        generalized_df = self.org_data.copy()

        for qid, level in generalization:
            if level == 0:
                continue
            generalized_values, _ = generalize_column(
                generalized_df[qid],
                self.dataset.hierarchies[qid],
                level_from=0,
                level_to=level,
            )
            generalized_df[qid] = generalized_values

        return generalized_df

    def anonymize(self):
        """Run Incognito and store all valid solutions plus one selected result."""
        self.solutions = []

        # Bottom-up by combination size of quasi-identifiers.
        for _ in range(self.__num_qids):
            self.__lattice.increment_attributes()

            # Initialize queue with current roots.
            self.__pqueue = PriorityQueue()
            for node in self.__lattice.nodes:
                if node.is_root() and not node.deleted:
                    self.__pqueue.put(node)

            while not self.__pqueue.empty():
                node = self.__pqueue.get()

                if node.is_marked() or node.deleted:
                    continue

                generalized_df = self.__apply_node_generalization(node.generalization)
                node_qids_idx = [self.__qids_idx_map[qid] for qid, _ in node.generalization]
                k_anonymous = is_k_anonymous(generalized_df, self.k, node_qids_idx)

                if k_anonymous:
                    node.mark()
                    for dst_node in node.to_nodes:
                        dst_node.mark()
                else:
                    for dst_node in node.to_nodes:
                        self.__pqueue.put(dst_node)
                    node.delete()

        # Gather all valid solutions.
        best_df = None
        best_height = None
        for node in self.__lattice.nodes:
            if node.deleted:
                continue

            generalized_df = self.__apply_node_generalization(node.generalization)
            solution_df = ITableDF(
                generalized_df,
                table_name=f"Incognito solution: {sorted(node.generalization, key=lambda x: x[0])}",
            )
            self.solutions.append(solution_df)

            # Hardcoded utility for now: lower total generalization height is better.
            if best_height is None or node.height < best_height:
                best_height = node.height
                best_df = generalized_df

        if best_df is not None:
            self._construct_anon_data(best_df.values, columns=list(best_df))
