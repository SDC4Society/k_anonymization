from os import path as os_path

from k_anonymization import __file__ as top_dir
from k_anonymization.core import Dataset

# ADULT = Dataset.from_dir("adult")
# LONDON_HOUSE_PRICE = Dataset.from_dir("london_house_price")
# MINI_CRIME = Dataset.from_dir("mini_crime")
# MINI_PATIENT = Dataset.from_dir("mini_patient")


def load_adult():
    return Dataset.from_dir(
        "adult",
        f"{os_path.dirname(top_dir)}/datasets/adult",
        "json",
    )


def load_london_housing():
    return Dataset.from_dir(
        "london_house_price",
        f"{os_path.dirname(top_dir)}/datasets/london_house_price",
        "json",
    )


def load_mini_crime():
    return Dataset.from_dir(
        "mini_crime",
        f"{os_path.dirname(top_dir)}/datasets/mini_crime",
        "json",
    )


def load_mini_patient():
    return Dataset.from_dir(
        "mini_patient",
        f"{os_path.dirname(top_dir)}/datasets/mini_patient",
        "json",
    )
