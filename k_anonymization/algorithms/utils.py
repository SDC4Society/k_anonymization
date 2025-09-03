from pandas.core.frame import DataFrame


def generalize(df: DataFrame, hierarchy: dict, level: int):
    is_suppressed = False

    if hierarchy["type"] == "lambda":
        df[hierarchy["name"]] = df[hierarchy["name"]].map(
            eval(hierarchy["lambda"][level])
        )
        if level == len(hierarchy["lambda"]) - 1:
            is_suppressed = True
    elif hierarchy["type"] == "list":
        is_suppressed = hierarchy["hierarchy"][level]["is_suppressed"]
        if is_suppressed:
            df[hierarchy["name"]] = df[hierarchy["name"]].map(lambda x: "*")
        else:
            generalized_values = hierarchy["hierarchy"][level]["values"]

            def find_generalized_value(x):
                for generalized_value in generalized_values:
                    if x in generalized_value["original"]:
                        return generalized_value["generalized"]

            df[hierarchy["name"]] = df[hierarchy["name"]].map(
                lambda x: find_generalized_value(x)
            )

    return df, is_suppressed


def find_not_k_anonymized_qids(df: DataFrame, k: int = 2, qids: list = []):
    if len(qids) == 0:
        qids = list(df.keys())
    unique = [df[x].unique().tolist() for x in qids]
    for i in range(0, len(unique)):
        if len(unique[i]) > 1:
            results = []
            for u in unique[i]:
                results.extend(
                    find_not_k_anonymized_qids(df[df[qids[i]] == u], k, qids)
                )
            return results
    num_of_vals = df.shape[0]
    if num_of_vals >= k:
        return []
    else:
        return [{"qid": list(zip(*unique))[0], "count": num_of_vals}]


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