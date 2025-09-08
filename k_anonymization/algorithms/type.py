# +
from abc import ABC, abstractmethod

from ..datasets import Dataset


# -

class Algorithm(ABC):
    def __init__(
        self,
        dataset: Dataset,
        k: int,
    ):
        self.k = k
        self.dataset = dataset
        self.anon_data = None

    @property
    def org_data(self):
        return self.dataset.df

    def __reset_anon_data(self, anonymize_func):
        def wrapper(*args, **kwargs):
            self.anon_data = self.org_data[:]
            return anonymize_func(self, *args, **kwargs)

        return wrapper

    def __getattribute__(self, name):
        if name == "anonymize":
            func = getattr(type(self), "anonymize")
            return self.__reset_anon_data(func)
        return object.__getattribute__(self, name)

    @abstractmethod
    def anonymize(self):
        pass


