# +
from .type import Dataset, DataFrameTable, SampleDataset
# -



ADULT = Dataset("adult")
MINI_CRIME = Dataset("mini_crime")
MINI_PATIENT = Dataset("mini_patient")

__all__ = ["DataFrameTable", "Dataset", "SampleDataset"] + [x.name.upper() for x in Dataset.all_datasets]
