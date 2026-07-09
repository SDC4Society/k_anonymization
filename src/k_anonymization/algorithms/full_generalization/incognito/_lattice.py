import itertools
from typing import List

import numpy as np

from ._node import Node
from k_anonymization.core import Dataset, HierarchiesDict


class Lattice:
    """
    Generalization lattice for the Incognito algorithm.

    Builds a lattice of generalization nodes by incrementally increasing
    the number of QID attributes considered. At each step, nodes are
    generated combinatorially from the previous dimension's active nodes
    and connected via directed edges that encode the partial order of
    generalization (lower -> higher).

    Attributes
    ----------
    _qid_pos : dict[str, int]
        Maps a QID name to its position in ``self.qids``. Used to look up
        ``distinct_counts``/``_row_codes``/``_level_lut`` by name, since
        Incognito's nodes reference QIDs by name over a variable-length
        subset rather than by a fixed position (unlike Flash/Lightning).
    distinct_counts : list[list[int]]
        ``distinct_counts[j][level]`` is the number of distinct values
        at the given generalization level for the j-th QID.
    _row_codes : list[numpy.ndarray]
        ``_row_codes[j]`` holds the integer code of each row's level-0
        value for the j-th QID. Used for DataFrame-free k-anonymity checks.
    _level_lut : list[list[numpy.ndarray]]
        ``_level_lut[j][level][c]`` is the integer code of the generalized
        value at the given level for the level-0 code ``c`` of the j-th QID.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.nodes: List[Node] = []
        self.qids: List[str] = dataset.qids
        self.hierarchy: HierarchiesDict = dataset.hierarchies
        self.attributes: int = 0

        # --- integer encoding for DataFrame-free k-anonymity checks ---
        self._qid_pos: dict = {qid: j for j, qid in enumerate(self.qids)}
        hierarchies = [self.hierarchy[qid] for qid in self.qids]
        self.max_levels: list = [h.height for h in hierarchies]
        self.distinct_counts: list = []
        self._row_codes: list = []
        self._level_lut: list = []
        for qid, hierarchy, max_level in zip(self.qids, hierarchies, self.max_levels):
            h_df = hierarchy.hierarchy_df
            self.distinct_counts.append(
                [h_df[level].nunique() for level in range(max_level + 1)]
            )

            column = dataset.df[qid]
            if column.isna().any():
                raise ValueError(
                    f"QID '{qid}' contains missing values (NaN/NA); Incognito's "
                    "integer-encoded k-anonymity check requires complete QID columns."
                )
            categories, row_codes = np.unique(column.to_numpy(), return_inverse=True)
            self._row_codes.append(row_codes.astype(np.int64))

            luts: list = []
            for level in range(max_level + 1):
                level_map = dict(zip(h_df[0], h_df[level]))
                generalized = np.array(
                    [level_map[value] for value in categories], dtype=object
                )
                _, generalized_codes = np.unique(generalized, return_inverse=True)
                luts.append(generalized_codes.astype(np.int64))
            self._level_lut.append(luts)
        # --- end integer encoding ---

    def __single_attribute_initialization(self) -> None:
        """Initialize the lattice with single-attribute generalization chains.

        For each QID, create a chain of nodes from level 0 (original) to
        the maximum generalization level, linked by predecessor/successor edges.
        """
        for qid in self.qids:
            tmp_nodes = []
            # Create one node per generalization level (0 = original value)
            for generalization_level in range(self.hierarchy[qid].height + 1):
                node = Node([(qid, generalization_level)])
                tmp_nodes.append(node)
            # Link consecutive levels: level_i -> level_(i+1)
            for i in range(1, len(tmp_nodes)):
                tmp_nodes[i].add_src_node(tmp_nodes[i - 1])
                tmp_nodes[i - 1].add_dst_node(tmp_nodes[i])

            self.nodes.extend(tmp_nodes)

    def __node_generation(self) -> None:
        """Generate nodes for the next attribute dimension combinatorially.

        For each pair (p, q) of active nodes from the current dimension,
        if they share the same first (n-1) attributes at the same levels
        and p's last attribute name < q's last attribute name, merge them
        into a new (n+1)-attribute node.
        """
        active_nodes = []
        new_nodes_tmp = []
        for node in self.nodes:
            if not node.deleted:
                active_nodes.append(node)

        for p, q in itertools.permutations(active_nodes, 2):
            p.generalization = sorted(p.generalization, key=lambda x: x[0])
            q.generalization = sorted(q.generalization, key=lambda x: x[0])
            if (p.generalization[:-1] == q.generalization[:-1]) and (
                p.generalization[-1][0] < q.generalization[-1][0]
            ):
                new_generalization = set(p.generalization) | set(q.generalization)
                new_generalization = sorted(new_generalization, key=lambda x: x[0])
                append_node = Node(new_generalization)
                append_node.add_inclement_parent([p, q])
                new_nodes_tmp.append(append_node)

        self.nodes = new_nodes_tmp

    def __edge_generation(self) -> None:
        """Connect nodes with directed edges based on generalization order.

        Each new-dimension node has two generating parents (graph_gen_parents).
        A directed edge p -> q (less generalized -> more generalized) exists
        when the generating parents of p and q satisfy one of three conditions:

        - Cond 1: Same first parent; second parents are adjacent in the
          previous lattice (i.e., one is the direct predecessor of the other).
        - Cond 2: First parents are adjacent; same second parent.
        - Cond 3: Both first and second parent pairs are adjacent.

        The edge direction is determined by node height (sum of levels):
        the lower-height node becomes the predecessor (p -> q when
        p.height < q.height).
        """
        for p, q in itertools.combinations(self.nodes, 2):
            # Cond 1: same first parent, second parents are adjacent
            cond_1 = (p.graph_gen_parents[0] == q.graph_gen_parents[0]) and (
                (  # p -> q direction
                    q.graph_gen_parents[1] in p.graph_gen_parents[1].to_nodes
                    and p.graph_gen_parents[1] in q.graph_gen_parents[1].from_nodes
                )
                and (  # q -> p direction
                    p.graph_gen_parents[1] in q.graph_gen_parents[1].to_nodes
                    and q.graph_gen_parents[1] in p.graph_gen_parents[1].from_nodes
                )
            )

            # Cond 2: first parents are adjacent, same second parent
            cond_2 = (p.graph_gen_parents[1] == q.graph_gen_parents[1]) and (
                (  # p -> q direction
                    q.graph_gen_parents[0] in p.graph_gen_parents[0].to_nodes
                    and p.graph_gen_parents[0] in q.graph_gen_parents[0].from_nodes
                )
                and (  # q -> p direction
                    p.graph_gen_parents[0] in q.graph_gen_parents[0].to_nodes
                    and q.graph_gen_parents[0] in p.graph_gen_parents[0].from_nodes
                )
            )

            # Cond 3: both parent pairs are adjacent
            cond_3 = (
                q.graph_gen_parents[0] in p.graph_gen_parents[0].to_nodes
                and p.graph_gen_parents[0] in q.graph_gen_parents[0].from_nodes
            ) and (
                q.graph_gen_parents[1] in p.graph_gen_parents[1].to_nodes
                and p.graph_gen_parents[1] in q.graph_gen_parents[1].from_nodes
            )

            if cond_1 or cond_2 or cond_3:
                if p.height > q.height:
                    # q -> p (q is less generalized)
                    q.add_dst_node(p)
                    p.add_src_node(q)
                else:
                    # p -> q (p is less generalized)
                    p.add_dst_node(q)
                    q.add_src_node(p)

    def __graph_generation(self) -> None:
        """Build the next-dimension lattice by generating nodes and edges."""
        self.__node_generation()
        self.__edge_generation()

    def increment_attributes(self) -> None:
        """Expand the lattice by one QID attribute dimension."""
        if self.attributes == 0:
            self.__single_attribute_initialization()
        else:
            self.__graph_generation()
        self.attributes += 1

    def k_anonymity(self, generalization: List[tuple]) -> int:
        """
        Compute the k-anonymity of a generalization via integer encoding.

        Returns the size of the smallest equivalence class produced by
        generalizing the QID subset named in ``generalization`` (a
        variable-length, named subset -- Incognito's nodes don't always
        cover all dataset QIDs, unlike Flash/Lightning's fixed-length
        generalization tuples), without materializing a (string) DataFrame.

        Each named QID is reduced to its precomputed generalized integer
        codes (``_level_lut[j][level][_row_codes[j]]``); the per-QID codes
        are combined into a single key (mixed-radix when it fits in int64,
        otherwise a structured row view) and equivalence-class sizes are
        counted with ``numpy``. Because the per-level encoding is bijective,
        the resulting size multiset is identical to
        ``df.groupby(qids).size()`` on the generalized string data.

        Parameters
        ----------
        generalization : list[tuple[str, int]]
            The (QID name, generalization level) pairs to check.

        Returns
        -------
        int
            The minimum equivalence-class size (the level of k-anonymity).
        """
        columns = []
        radices = []
        for qid, level in generalization:
            j = self._qid_pos[qid]
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
