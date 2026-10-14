import json
from functools import cached_property

from itables import show as itables_show
from pandas import DataFrame, read_csv

from .frame import ITableDF
from .hierarchy import HierarchiesDict


class Dataset:
    """
    Management class for handling dataset and its properties.

    This class serves as a central hub for a dataset, storing data,
    attribute properties, and generalization hierarchies.

    Parameters
    ----------
    name : str
        Name of the dataset.
    df : DataFrame
        Dataset itself.
    qids : list[str], default None
        Specify QID attributes. If `None`, select all columns in `df`
        except `target` as QID attributes.
    is_categorical : list[bool], default None
        Specify whether a QID attribute is categorical (`True`) or
        numerical (`False`). If `None`, determine from `df.dtypes`.
    hierarchies : HierarchiesDict, default None
        Specify generalization hierarchies for all QID attributes.
        If `None`, use default hierarchies, i.e., suppress all values.
    target : str, default None
        Sensitive attribute, a.k.a., target for data analysis tasks.

    Attributes
    ----------
    name : str
        Name of the dataset.
    df : ITableDF
        Dataset itself.
    qids : list[str]
        List of names of the QID attributes.
    qids_categorial : list[str]
        List of names of the categorical QID attributes.
    qids_numerical : list[str]
        List of names of the numerical QID attributes.
    qids_idx : list[int]
        List of column indices of the QID attributes in `df`.
    qids_idx_categorical : list[int]
        List of column indices of the categorical QID attributes in `df`.
    qids_idx_numerical : list[int]
        List of column indices of the numerical QID attributes in `df`.
    is_categorical : list[bool]
        List of booleans indicating if a QID attribute is categorical.
        Its order matches the orders of `qids` and `qids_idx`.
    hierarchies : HierarchiesDict
        Generalization hierarchies manager of all QID attributes.
    target : str
        Sensitive attribute, a.k.a., target for data analysis tasks.
    """

    # TODO: implement classes for attributes and/or QIDs

    def __init__(
        self,
        name: str,
        df: DataFrame,
        qids: list[str] = None,
        is_categorical: list[bool] = None,
        hierarchies: HierarchiesDict = None,
        target: str = None,
    ):
        """
        Initialize the Dataset.

        Parameters
        ----------
        name : str
            Name of the dataset.
        df : DataFrame
            Dataset itself.
        qids : list[str], default None
            Specify QID attributes. If `None`, select all columns in `df`
            except `target` as QID attributes.
        is_categorical : list[bool], default None
            Specify whether a QID attribute is categorical (`True`) or
            numerical (`False`). If `None`, determine from `df.dtypes`.
        hierarchies : HierarchiesDict, default None
            Specify generalization hierarchies for all QID attributes.
            If `None`, use default hierarchies, i.e., suppress all values.
        target : str, default None
            Sensitive attribute, a.k.a., target for data analysis tasks.
        """
        self.name = name
        self.df = ITableDF(df, table_name=name)

        columns = df.columns

        if qids is not None:
            self.qids = qids
            self.qids_idx = [idx for idx, qid in enumerate(columns) if qid in qids]
        else:
            self.qids = [col for col in columns if col != target]
            self.qids_idx = [idx for idx, col in enumerate(columns) if col in self.qids]

        self.qids = (
            qids if qids is not None else [col for col in columns if col != target]
        )

        self.hierarchies = (
            hierarchies
            if hierarchies is not None
            else HierarchiesDict.make_default(df, self.qids)
        )

        self.target = target

        self.is_categorical = is_categorical
        if self.is_categorical is None:
            _qids_categorical = df.select_dtypes(exclude="number")
            self.is_categorical = [
                True if qid in _qids_categorical else False for qid in self.qids
            ]

        self.qids_idx = []
        self.qids_categorical = []
        self.qids_idx_categorical = []
        self.qids_numerical = []
        self.qids_idx_numerical = []

        for idx, qid in enumerate(columns):
            if qid in self.qids:
                self.qids_idx.append(idx)
                if self.is_categorical[self.qids.index(qid)]:
                    self.qids_categorical.append(qid)
                    self.qids_idx_categorical.append(idx)
                else:
                    self.qids_numerical.append(qid)
                    self.qids_idx_numerical.append(idx)

    @classmethod
    def from_dir(
        cls,
        name: str,
        dataset_dir: str,
        hierarchies_csv_sep: str = ",",
    ):
        """
        Initialize the Dataset from stored data in the given directory.

        The directory contains:

        - `data.csv` (Required): a CSV file storing the data, where the top row stores
        the header (attribute) names.

        - `props.json` (Optional): a JSON file storing data properties,
        including `qids`, `is_categorical`, and `target`. If any of the
        above properties is not provided, use its default value.
        If `props.json` does not exist, use default values for all
        properties.

        - `hierarchies/` (Optional): a directory containing hierarchy files
        of all QID attributes. If the hierarchy file of an attribute is not
        provided, use its default hierarchy, i.e., suppress all values.
        If `hierarchies/` does not exist, use default hierarchies for all
        attributes.


        Parameters
        ----------
        name : str
            Name of the dataset.
        dataset_dir : str
            Path to directory storing the data, its properties, and
            generalization hierarchies.
        hierarchies_csv_sep :  str, default ','
            Separator string (delimiter) of CSV hierarchy files.
        """

        df = read_csv(f"{dataset_dir}/data.csv")
        try:
            with open(f"{dataset_dir}/props.json") as f:
                props = json.load(f)
        except:
            props = {}
        hierarchies = HierarchiesDict.from_dir(
            f"{dataset_dir}/hierarchies",
            df,
            props.get("qids", None),
            hierarchies_csv_sep,
        )
        return cls(
            name,
            df,
            qids=props.get("qids", None),
            is_categorical=props.get("is_categorical", None),
            hierarchies=hierarchies,
            target=props.get("target", None),
        )

    def __str__(self):
        """
        Return the string representation of the sample dataset.

        Returns
        -------
        str
            The dataset name.
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

        Provide a breakdown of each column's status as a QID, numerical,
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

        Sample rows based on the distribution of the target attribute.

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
        The sampled data.

    Attributes
    ----------
    """

    def __init__(self, base_dataset: Dataset, sampled_df: DataFrame):
        """
        Initialize the SampleDataset.

        Parameters
        ----------
        dataset : Dataset
            The source dataset instance.
        df : DataFrame
            The sampled data.
        """
        super().__init__(
            base_dataset.name,
            sampled_df,
            base_dataset.qids,
            base_dataset.is_categorical,
            base_dataset.hierarchies,
            base_dataset.target,
        )

    def __str__(self):
        """
        Return the string representation of the sample dataset.

        Returns
        -------
        str
            The dataset name followed by the word 'sample'.
        """
        return self.name + " sample"
