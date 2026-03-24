from k_anonymization.core import Dataset

ADULT = Dataset("adult")
LONDON_HOUSE_PRICE = Dataset("london_house_price")
MINI_CRIME = Dataset("mini_crime")
MINI_PATIENT = Dataset("mini_patient")

__all__ = [x.name.upper() for x in Dataset.all_datasets]
