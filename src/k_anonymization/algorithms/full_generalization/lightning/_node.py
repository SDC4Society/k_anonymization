from ..flash._node import Node


class LightningNode(Node):
    """
    A node in the generalization lattice for the Lightning algorithm.

    Extends :class:`Node` with an ``expanded`` flag to track whether
    the node's neighbors have already been explored during best-first
    search.

    Parameters
    ----------
    generalization_tuple : tuple
        A tuple of integers specifying the generalization level for each
        QID attribute.

    Attributes
    ----------
    expanded : bool
        Whether this node has been expanded (its neighbors explored).
    """

    def __init__(self, generalization_tuple: tuple):
        super().__init__(generalization_tuple)
        self.expanded: bool = False

    def mark_as_expanded(self) -> None:
        """Mark this node as expanded."""
        self.expanded = True
