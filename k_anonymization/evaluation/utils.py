from pandas.core.frame import DataFrame


def find_not_k_anonymized_qids(
    df: DataFrame, k: int = 2, qids: list = [], get_qids: bool = True
):
    return count_equivalent_qids(df, k, qids, get_qids)


def count_equivalent_qids(
    df: DataFrame, k: int = -1, qids: list = [], get_qids: bool = False
):
    if len(qids) == 0:
        qids = list(df.keys())
    unique = [df[x].unique().tolist() for x in qids]
    for i in range(0, len(unique)):
        if len(unique[i]) > 1:
            results = []
            for u in unique[i]:
                results.extend(
                    count_equivalent_qids(df[df[qids[i]] == u], k, qids, get_qids)
                )
            return results
    num_of_vals = df.shape[0]
    if num_of_vals >= k and k != -1:
        return []
    else:
        return (
            [{"qid": list(zip(*unique))[0], "count": num_of_vals}]
            if get_qids is True
            else [num_of_vals]
        )


def is_k_anonymized(df: DataFrame, k: int = 2, qids: list = []):
    if len(qids) == 0:
        qids = list(df.keys())
    unique = [df[x].unique().tolist() for x in qids]
    for i in range(0, len(unique)):
        if len(unique[i]) > 1:
            for u in unique[i]:
                result = is_k_anonymized(df[df[qids[i]] == u], k, qids)
                if result is False:
                    return False
            return True
    return df.shape[0] >= k
