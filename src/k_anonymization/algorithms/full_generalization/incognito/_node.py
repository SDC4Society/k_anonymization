from typing import List


class Node:
    """
    A node in the generalization lattice.

    Each node represents a specific combination of generalization levels
    for one or more QID attributes, connected by directed edges reflecting
    the partial order of generalization (lower -> higher).
    """

    def __init__(self, generalization: List[tuple], **kwargs):
        self.height: int
        self.generalization: List[tuple] = generalization
        self.from_nodes: list["Node"] = []
        self.to_nodes: list["Node"] = []
        self.marked: bool = False
        self.deleted: bool = False
        self.graph_gen_parents: list["Node"] = []

        self.height = sum(level for _, level in self.generalization)

    def is_root(self) -> bool:
        """Check whether this node has no predecessors."""
        return len(self.from_nodes) == 0

    def is_marked(self) -> bool:
        """Check whether this node has been marked as k-anonymous."""
        return self.marked

    def mark(self) -> None:
        """Mark this node as k-anonymous."""
        self.marked = True

    def add_dst_node(self, dst: "Node") -> None:
        """Add a successor (more generalized) node."""
        self.to_nodes.append(dst)

    def add_src_node(self, src: "Node") -> None:
        """Add a predecessor (less generalized) node."""
        self.from_nodes.append(src)

    def add_inclement_parent(self, parent: List["Node"]) -> None:
        """Register the node pair that generated this node during dimension increment."""
        self.graph_gen_parents.extend(parent)

    def __lt__(self, other):
        return self.height < other.height

    def delete(self) -> None:
        """Prune this node from the lattice and disconnect from successors."""
        self.deleted = True
        for dst_node in self.to_nodes:
            dst_node.from_nodes.remove(self)
