from numpy import ndarray
from pandas import DataFrame


def find_not_k_anonymized_qids(
    data: DataFrame | ndarray, k: int = 2, qids_idx: list = []
):
    """
    Find QIDs that violate `k`-anonymity.

    Parameters
    ----------
    data : DataFrame or ndarray
        The dataset to inspect.
    k : int, default 2
        The minimum required size for an equivalence class.
    qids_idx : list, optional
        The column indices of the Quasi-Identifiers.

    Returns
    -------
    list of dict
        A list of dictionaries, each containing the 'qid' values and
        the 'count' of records in that non-compliant group.
    """

    return get_more_than_k_equivalence_qids(data, k, qids_idx)


def get_equivalence_qids(data: DataFrame | ndarray, qids_idx: list = []):
    return get_more_than_k_equivalence_qids(data, float("inf"), qids_idx)


def get_more_than_k_equivalence_qids(data: DataFrame | ndarray, k, qids_idx: list = []):
    if isinstance(data, ndarray):
        _df = DataFrame(data)
        _qids = qids_idx
    else:
        _df = data
        _attr_names = _df.keys()
        _qids = [_attr_names[i] for i in qids_idx]
    return [
        {"qid": key, "count": value}
        for key, value in _df.groupby(_qids).size().items()
        if value < k
    ]


def is_k_anonymized(data: DataFrame | ndarray, k: int = 2, qids_idx: list = []):

    if isinstance(data, ndarray):
        _df = DataFrame(data)
        _qids = qids_idx
    else:
        _df = data
        _attr_names = _df.keys()
        _qids = [_attr_names[i] for i in qids_idx]

    return _df.groupby(_qids).size().min() >= k
