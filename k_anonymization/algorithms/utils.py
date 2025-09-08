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