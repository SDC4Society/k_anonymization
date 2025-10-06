# +
from numpy import ndarray
from pandas.core.frame import DataFrame

from ..algorithms.type import Algorithm
from .utils import count_equivalent_qids


# -

def discernibility(
    equivalent_qids: list,
    suppression_counts: int = 0,
    org_data_size: int = 0,
):
    return sum([x**2 for x in equivalent_qids]) + suppression_counts * org_data_size


def discernibility_from_data(
    anon_data: DataFrame | ndarray,
    qids_idx: list = [],
    suppression_counts: int = 0,
    org_data_size: int = 0,
):
    equivalent_qids = count_equivalent_qids(anon_data, qids_idx=qids_idx)
    return discernibility(equivalent_qids, suppression_counts, org_data_size)


def discernibility_from_algo(algo: Algorithm):
    return discernibility_from_data(
        algo.anon_data,
        algo.dataset.qids_idx,
        (
            sum([x["count"] for x in algo.suppressed_qids])
            if algo.suppressed_qids
            else 0
        ),
        algo.org_data.shape[0],
    )
