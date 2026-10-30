import json
from functools import cached_property
from os.path import isfile
from typing import Literal

from ipywidgets import Dropdown, Label, Stack, VBox, jslink
from pandas import DataFrame, read_csv

from k_anonymization.utils.data_table import get_ITable_widget

from .frame import ITableDF


class Hierarchy:
    """
    Attribute's generalization hierarchy.

    This class stores the generalization mapping of a QID attribute's
    generalization hierarchy and provides utility functions for acquiring
    necessary properties of the hierarchy and its nodes.

    Parameters
    ----------
    name : str
        Name of the attribute.
    hierarchy_df : DataFrame
        DataFrame storing generalization hierarchy. Column names must be
        progressive integers, where Column `0` stores the raw data and
        subsequent columns store progressively higher generalized values.

    Attributes
    ----------
    name : str
        Name of the attribute.
    hierarchy_df : ITableDF
        DataFrame storing generalization hierarchy.
    """

    def __init__(self, name: str, hierarchy_df: DataFrame):
        """
        Initialize hierarchy from a pre-computed mapping DataFrame.

        Parameters
        ----------
        name : str
            Hierarchy name.
        hierarchy_df : DataFrame
            Pre-computed hierarchy mapping DataFrame.
        """
        hierarchy_df.sort_values(
            hierarchy_df.columns[::-1].to_list(),
            ignore_index=True,
            inplace=True,
        )
        self.hierarchy_df = ITableDF(hierarchy_df)
        self.name = name

    @classmethod
    def make_default(cls, name: str, org_column: DataFrame):
        """
        Initialize default a hierarchy where all values are suppressed.

        Parameters
        ----------
        name : str
            Hierarchy name.
        org_column : DataFrame
            Data column from the original dataset to use as Level 0.
        """
        unique_values = org_column.unique()
        hierarchy_df = DataFrame({0: unique_values, 1: ["*"] * unique_values.size})
        return cls(name, hierarchy_df)

    @classmethod
    def from_csv(cls, name: str, path: str, sep: str = ","):
        """
        Initialize hierarchy from a pre-computed mapping CSV file.

        The CSV should have no header, where the first column is the raw
        data and each following column is a higher level of generalization.

        Parameters
        ----------
        name : str
            Hierarchy name.
        path : str
            Path to the pre-computed mapping CSV file.
        sep : str, default ','
            Separator string (delimiter) of the CSV file.
        """
        return cls(name, read_csv(path, sep=sep, header=None))

    @classmethod
    def from_json(cls, name: str, org_column: DataFrame, path: str):
        """
        Initialize hierarchy from a JSON configuration file.

        Supports two types of definition for generalization:

        1. ``lambda``: Apply lambda functions to derive the next
           generalization level based on the current value.

        2. ``tree``: Explicitly map a list of ``original`` values to a
           ``generalized`` value.

        Parameters
        ----------
        name : str
            Hierarchy name.
        org_column : DataFrame
            Data column from the original dataset to use as Level 0.
        path : str
            Path to the JSON configuration file.
        """
        try:
            with open(path) as f:
                props = json.load(f)
            return cls.from_dict(name, org_column, props)
        except FileNotFoundError:
            raise FileNotFoundError(f'Cannot find hierarchy file at "{path}".')

    @classmethod
    def from_dict(cls, name: str, org_column: DataFrame, props: dict):
        """
        Initialize hierarchy from a dictionary storing configurations.

        Supports two types of definition for generalization:

        1. ``lambda``: Apply lambda functions to derive the next
            generalization level based on the current value.

        2. ``tree``: Explicitly map a list of ``original`` values to a
            ``generalized`` value.

        Parameters
        ----------
        name : str
            Hierarchy name.
        org_column : DataFrame
            Data column from the original dataset to use as Level 0.
        props : dict
            Dictionary storing the ``lambda`` or ``tree`` configurations.
        """
        hierarchy_df = DataFrame({0: org_column.unique()})
        if "lambda" in props:
            for level in range(len(props["lambda"])):
                f = eval(props["lambda"][level])
                hierarchy_df[level + 1] = hierarchy_df[level].apply(f)
        else:
            for level in range(len(props["tree"])):
                _hi = hierarchy_df[[level]].copy()
                if props["tree"][level]["is_suppressed"]:
                    hierarchy_df[level + 1] = "*"
                else:
                    for v in props["tree"][level]["values"]:
                        _hi[_hi.isin(v["original"]).any(axis=1)] = v["generalized"]
                    hierarchy_df[level + 1] = _hi[level]

        return cls(name, hierarchy_df)

    def _repr_html_(self):
        """
        Get the HTML representation of the hierarchy.
        """
        return self.hierarchy_df._repr_html_()

    @cached_property
    def height(self):
        """
        Height of this hierarchy.

        Returns
        -------
        int
        """
        return self.hierarchy_df.shape[1] - 1

    @cached_property
    def leaves(self):
        """
        List of leaves (values at level 0).

        Returns
        -------
        list
        """
        return self.hierarchy_df[0].to_list()

    def contains(self, node_value: any):
        """
        Check whether a node exists anywhere in the hierarchy.

        Parameters
        ----------
        node_value : any
            The value of the node to inspect.

        Returns
        -------
        bool
        """
        if node_value in self.leaves or node_value == "*":
            return True

        return True if self.hierarchy_df.eq(node_value).any().sum() != 0 else False

    def get_leaves_under_node(self, node_value: any):
        """
        Get all leaves (values at level 0) under the given node.

        Parameters
        ----------
        node_value : any
            The value of the node to inspect.

        Returns
        -------
        list
            A list of leaves.
        """
        if node_value in self.leaves:
            return []

        if node_value == "*":
            return self.leaves

        search_df = self.hierarchy_df[self.hierarchy_df.eq(node_value).any(axis=1)]
        if search_df.shape[0] == 0:
            raise ValueError(
                f'Node "{node_value}" (type: {type(node_value)}) not found in the hierarchy.'
            )
        return search_df[0].to_list()

    def get_height_of_node(self, node_value: any):
        """
        Get the generalization level for a specific node.

        Parameters
        ----------
        node_value : any
            The value of the node to inspect.

        Returns
        -------
        int
            Generalization level of the input node.
        """
        if node_value in self.leaves:
            return 0

        if node_value == "*":
            return self.height

        search_df = self.hierarchy_df.columns[self.hierarchy_df.eq(node_value).any()]
        if search_df.size == 0:
            raise ValueError(
                f'Node "{node_value}" (type: {type(node_value)}) not found in the hierarchy.'
            )
        return search_df.item()

    def get_lowest_common_ancestor(
        self,
        node_values: list,
        get_type: Literal["value", "height"] = "value",
    ):
        """
        Find the lowest common ancestor (LCA) of the given nodes.

        This is used to find the lowest-level generalized value that can
        hide a group of different values.

        Parameters
        ----------
        node_values : list
            The list of values to inspect.
        get_type : {'value', 'height'}, default 'value'
            Whether to return the LCA value or its generalization level.

        Returns
        -------
        str or int
            The LCA value or its height (generalization level).
        """
        assert len(node_values) > 1, "node_values must contains at least 2 values."

        for node_value in set(node_values):
            if not self.contains(node_value):
                raise ValueError(
                    f'Node "{node_value}" (type: {type(node_value)}) not found in the hierarchy.'
                )

        search_df = self.hierarchy_df[self.hierarchy_df.isin(node_values).any(axis=1)]

        for level in range(self.height):
            _unique = search_df[level].unique()
            if _unique.size == 1:
                if get_type == "value":
                    return _unique.item()
                return level

        if get_type == "value":
            return "*"
        return self.height


class HierarchiesDict(dict):
    """
    Dictionary-like manager for attributes' generalization hierarchies.

    This class maps attribute names or their column indices to their
    respective hierarchy objects.

    Parameters
    ----------
    hierarchies_dict : dict[str, Hierarchy]
        Dictionary storing the hierarchy of every column.
    df : DataFrame, default None
        Dataset to ensure column index of every QID attribute
        matches its index in `df.columns`.
        If `None`, set column indices to their indices in
        `hierarchies_dict.keys()`.
    """

    def __init__(
        self,
        hierarchies_dict: dict[str, Hierarchy] | dict[str, DataFrame],
        df: DataFrame = None,
    ):
        """
        Initialize the HierarchiesDict.

        Also construct an internal mapping of QID attribute names to their
        specific column indices to allow for flexible lookup.

        Parameters
        ----------
        hierarchies_dict : dict[str, Hierarchy] or dict[str, DataFrame]
            Dictionary storing the hierarchy of every QID attribute.
        df : DataFrame, default None
            Dataset to ensure column index of every QID attribute
            matches its index in `df.columns`.
            If `None`, set column indices to their indices in
            `hierarchies_dict.keys()`.
        """
        _hierarchies_dict = {
            name: (
                hierarchy
                if isinstance(hierarchy, Hierarchy)
                else Hierarchy(name, hierarchy)
            )
            for name, hierarchy in hierarchies_dict.items()
        }
        self.update(_hierarchies_dict)

        if df is None:
            self._qids_idx_mapping = {
                pos: qid for pos, qid in enumerate(hierarchies_dict.keys())
            }
        else:
            self._qids_idx_mapping = {
                pos: att
                for pos, att in enumerate(df.columns)
                if att in _hierarchies_dict
            }

    @classmethod
    def make_default(
        cls,
        df: DataFrame,
        qids: list[str] = None,
    ):
        """
        Initialize a default HierarchiesDict where all columns are suppressed.

        Parameters
        ----------
        df : DataFrame
            Original data.
        qids : dict[int, str], default None
            Specify QID attributes. If `None`, select all columns in `df`.
        """
        if qids is None:
            return cls(
                {att: Hierarchy.make_default(att, df[att]) for att in df.columns}
            )
        return cls(
            {att: Hierarchy.make_default(att, df[att]) for att in qids},
            df,
        )

    @classmethod
    def from_dir(
        cls,
        hierarchies_dir: str,
        df: DataFrame,
        qids: list[str] = None,
        csv_sep: str = ",",
    ):
        """
        Initialize the HierarchiesDict from CSV or JSON files.
        If an attribute has both CSV and JSON files, CSV file is selected.
        If both CSV and JSON files are not found, fall back to default
        hierarchy (i.e., suppress all values).

        Parameters
        ----------
        hierarchies_dir : str
            Path to the directory containing all hierarchy files.
        df : DataFrame
            Original data.
        qids : list[str], default None
            Specify QID attributes. If `None`, select all columns in `df`.
        csv_sep : str, default ','
            Separator string (delimiter) of CSV files.
        """
        hierarchies_dict = {}
        columns = df.columns
        if qids is None:
            qids = columns
        for att in columns:
            if att in qids:
                if isfile(f"{hierarchies_dir}/{att}.csv"):
                    hierarchies_dict[att] = Hierarchy.from_csv(
                        att, f"{hierarchies_dir}/{att}.csv", csv_sep
                    )
                elif isfile(f"{hierarchies_dir}/{att}.json"):
                    hierarchies_dict[att] = Hierarchy.from_json(
                        att, df[att], f"{hierarchies_dir}/{att}.json"
                    )
                else:
                    hierarchies_dict[att] = Hierarchy.make_default(att, df[att])

        return cls(
            hierarchies_dict,
            df,
        )

    def __getitem__(self, key) -> Hierarchy:
        """
        Retrieve a Hierarchy by index or attribute name.

        If the hierarchy is not already loaded in memory, this method
        attempts to find and load the corresponding JSON file from the
        `hierarchies_dir`.

        Parameters
        ----------
        key : str or int
            The attribute name (str) or its column index (int) to look up.

        Returns
        -------
        Hierarchy
            The hierarchy object associated with the specified attribute.
        """
        if isinstance(key, int):
            _key = self._qids_idx_mapping.get(key, None)
            if _key is None:
                raise AttributeError(
                    f"Cannot find hierarchy for attribute at index '{key}'"
                )
        else:
            _key = key
            if _key is None or _key not in self.keys():
                raise AttributeError(f"Cannot find attribute '{_key}'")
        return super().__getitem__(_key)

    @cached_property
    def all_hierarchies_df(self):
        """
        Interactive widget to display all loaded hierarchies on IPython.

        Generates a UI component for IPython consisting of a dropdown menu
        and a display area showing the generalization hierarchy of every
        QID attribute.

        Returns
        -------
        ipywidgets.VBox
        """
        qids = self.keys()
        children = [
            get_ITable_widget(self.__getitem__(qid).hierarchy_df) for qid in qids
        ]
        stack = Stack(children, selected_index=0)
        dropdown = Dropdown(options=list(qids))
        jslink((dropdown, "index"), (stack, "selected_index"))
        return VBox([Label("Generalization Trees:"), dropdown, stack])

    def describe(self):
        """
        Interactive widget to display all loaded hierarchies on IPython.

        Generates a UI component for IPython consisting of a dropdown menu
        and a display area showing the generalization hierarchy of every
        QID attribute.

        Returns
        -------
        ipywidgets.VBox
        """
        return self.all_hierarchies_df

    def copy(self):
        return HierarchiesDict(super().copy(), qids_idx=self._qids_idx_mapping.keys())
