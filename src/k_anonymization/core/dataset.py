import json
from functools import cached_property
from typing import Literal

from itables import show as itables_show
from pandas import DataFrame, read_csv

from .frame import ITableDF
from .hierarchy import HierarchiesDict


class Dataset:
    """
    Management class for handling dataset and its properties.

    This class serves as a central hub for a dataset, orchestrating
    the loading of raw data, attribute properties, and generalization
    hierarchies.

    Parameters
    ----------
    name : str
        The name of the dataset. This should correspond to a directory
        in `k_anonymization/datasets/`.

    Attributes
    ----------
    name : str
        The name assigned to the dataset instance.
    """

    # TODO: implement classes for QIDs

    def __init__(
        self,
        name: str,
        df: DataFrame,
        qids_idx: list[int],
        is_categorical: list[bool] = None,
        hierarchies: HierarchiesDict = None,
        target: str = None,
    ):
        """
        Initializes the Dataset instance, and registers it in the global
        `all_datasets` list.

        Parameters
        ----------
        name : str
            The name of the dataset, which corresponds to a directory in
            `k_anonymization/datasets/`.
        """
        self.name = name
        self.df = ITableDF(df, table_name=name)
        self.hierarchies = (
            hierarchies
            if hierarchies is not None
            else HierarchiesDict.make_default(df, qids_idx)
        )
        self.qids_idx = qids_idx
        self.is_categorical = (
            is_categorical if is_categorical is not None else [False] * len(qids_idx)
        )
        self.target = target

        self.qids = []
        self.qids_categorical = []
        self.qids_idx_categorical = []
        self.qids_numerical = []
        self.qids_idx_numerical = []
        for pos, qid in enumerate(df.columns):
            if pos in qids_idx:
                self.qids.append(qid)
                if is_categorical[qids_idx.index(pos)]:
                    self.qids_categorical.append(qid)
                    self.qids_idx_categorical.append(pos)
                else:
                    self.qids_numerical.append(qid)
                    self.qids_idx_numerical.append(pos)

    @classmethod
    def from_dir(
        cls,
        name: str,
        path: str,
        hierarchies_file_type: Literal["csv", "json"],
        hierarchies_file_sep: str = ",",
    ):

        df = read_csv(f"{path}/data.csv")
        with open(f"{path}/props.json") as f:
            props = json.load(f)
        hierarchies = HierarchiesDict.from_dir(
            f"{path}/hierarchies",
            df,
            props["qids_idx"],
            hierarchies_file_type,
            hierarchies_file_sep,
        )
        return cls(
            name,
            df,
            props["qids_idx"],
            props["is_categorical"],
            hierarchies,
            props["target"],
        )

    def __str__(self):
        """
        Return the uppercase name of the dataset.

        Returns
        -------
        str
            The dataset name in all caps.
        """
        return self.name

    # TODO: Update other modules to use
    # - qids_categorical
    # - qids_numerical
    # - qids_idx_categorical
    # - qids_idx_numerical

    @cached_property
    def info(self):
        """
        Generate a summary DataFrame of the dataset's attributes.

        Provides a breakdown of each column's status as a QID, numerical,
        or categorical attribute, along with the count of unique values.

        Returns
        -------
        pandas.DataFrame
            A summary table with metadata labels as the index.
        """
        _info_all = {}
        for col_idx, col in enumerate(list(self.df)):
            if col_idx not in self.qids_idx:
                _info = ["", "", ""]
            else:
                _info = [
                    "o" if col_idx in self.qids_idx else "",
                    (
                        "o"
                        if self.is_categorical[self.qids_idx.index(col_idx)] is False
                        else ""
                    ),
                    (
                        "o"
                        if self.is_categorical[self.qids_idx.index(col_idx)] is True
                        else ""
                    ),
                ]
            _info.append(self.df[col].unique().size)
            _info_all[col] = _info
        return DataFrame(
            _info_all,
            index=[
                "Quasi-Identifier (QID)",
                "Numerical Attribute",
                "Categorical Attribute",
                "No. of Unique Values",
            ],
        )

    def describe(self):
        """
        Render interactive dataframe showing dataset info.
        """
        itables_show(
            DataFrame(
                [
                    str(self),
                    self.df.shape[0],
                    self.df.shape[1],
                ],
                index=[
                    "Dataset Name",
                    "No. of Records",
                    "No. of Attributes",
                ],
            ),
            scrollX=True,
            ordering={"indicators": False},
            columnDefs=[
                {
                    "className": "no-header",
                    "target": "_all",
                }
            ],
        )

        itables_show(
            self.info,
            fixedColumns={"start": 1},
            scrollX=True,
            ordering={"indicators": False},
            columnDefs=[
                {
                    "className": "dt-center",
                    "target": "_all",
                }
            ],
        )

    def sample(self, n=None, frac=None, seed=None, ignore_index=True):
        """
        Create a stratified sample of the dataset.

        Samples rows based on the distribution of the target attribute.

        Parameters
        ----------
        n : int, optional
            Number of rows to sample.
        frac : float, optional
            Fraction of data to sample.
        seed : int, optional
            Random state for reproducibility.
        ignore_index : bool, default True
            If True, the resulting index will be labeled 0, 1, …, n - 1.

        Returns
        -------
        SampleDataset
            A new dataset object containing the sampled records.
        """
        if n is None:
            _frac = frac
        else:
            _frac = float(n / self.df.shape[0])
        return SampleDataset(
            self,
            self.df.groupby(
                self.target,
                group_keys=False,
            )[self.df.columns]
            .apply(
                lambda x: x.sample(
                    frac=_frac,
                    random_state=seed,
                    ignore_index=ignore_index,
                )
            )
            .reset_index(drop=True),
        )

    def _repr_html_(self):
        """
        Get the HTML representation of the dataset.

        Returns
        -------
        str
            HTML string for Jupyter display.
        """
        return self.df._repr_html_()


class SampleDataset(Dataset):
    """
    A specialized Dataset class for representing a sample of records.

    This class is typically instantiated via the `sample` method of the
    base `Dataset`. It maintains a reference to the original dataset's
    metadata while holding only a sample of the data.

    Parameters
    ----------
    dataset : Dataset
        The parent Dataset instance from which this sample was derived.
    df : DataFrame
        The sampled data records.

    Attributes
    ----------
    """

    def __init__(self, base_dataset: Dataset, sampled_df: DataFrame):
        """
        Initialize the SampleDataset.

        Parameters
        ----------
        dataset : Dataset
            The source dataset.
        df : DataFrame
            The filtered/sampled records to be stored.
        """
        super().__init__(
            base_dataset.name,
            sampled_df,
            base_dataset.hierarchies,
            base_dataset.qids_idx,
            base_dataset.is_categorical,
            base_dataset.target,
        )

    def __str__(self):
        """
        Return the string representation of the sample.

        Returns
        -------
        str
            The word 'SAMPLE' followed by the uppercase dataset name.
        """
        return self.name + " sample"
