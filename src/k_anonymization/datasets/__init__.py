"""
Load example data sets
"""

from os import path as os_path

from k_anonymization import __file__ as top_dir
from k_anonymization.core import Dataset


def load_adult():
    """
    Load Adult dataset.
    """
    return Dataset.from_dir("adult", f"{os_path.dirname(top_dir)}/datasets/adult")


def load_london_housing():
    """
    Load London Housing dataset.
    """
    return Dataset.from_dir(
        "london_housing", f"{os_path.dirname(top_dir)}/datasets/london_housing"
    )


def load_mini_crime():
    """
    Load Mini Crime - a 6-record toy dataset.
    """
    return Dataset.from_dir(
        "mini_crime", f"{os_path.dirname(top_dir)}/datasets/mini_crime"
    )


def load_mini_patient():
    """
    Load Mini Patient - a 6-record toy dataset.
    """
    return Dataset.from_dir(
        "mini_patient", f"{os_path.dirname(top_dir)}/datasets/mini_patient"
    )
