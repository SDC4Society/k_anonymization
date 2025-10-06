def get_num_ranges(data, qids_idx, is_cat):
    num_ranges = {}

    columns = list(zip(*data))
    for pos, idx in enumerate(qids_idx):
        if is_cat[pos] == True:
            continue
        num_ranges[str(idx)] = max(columns[idx]) - min(columns[idx])

    return num_ranges


def get_distance(r, record, qids_idx, is_cat, num_ranges, hierarchies):
    distances = []

    for pos, idx in enumerate(qids_idx):
        if is_cat[pos] == True:
            distances.append(
                get_categorical_distance([r[idx], record[idx]], hierarchies[idx])
            )
        else:
            distances.append(abs(r[idx] - record[idx]) / num_ranges[str(idx)])

    return sum(distances)


def get_information_loss(record, cluster, qids_idx, is_cat, num_ranges, hierarchies):
    information_losses = []
    if record == None:
        size = len(cluster)
        columns = list(zip(*cluster))
    else:
        size = len(cluster) + 1
        columns = list(zip(*(cluster + [record])))

    for pos, idx in enumerate(qids_idx):
        if is_cat[pos] == True:
            information_losses.append(
                get_categorical_distance(columns[idx], hierarchies[idx])
            )
        else:
            information_losses.append(
                (max(columns[idx]) - min(columns[idx])) / num_ranges[str(idx)]
            )

    return size * sum(information_losses)


def get_categorical_distance(values, hierarchy):
    current_level = 0
    generalized_values = values[:]
    height = (
        len(hierarchy["lambda"])
        if hierarchy["type"] == "lambda"
        else len(hierarchy["hierarchy"])
    )

    def generalize(_values, level):
        if hierarchy["type"] == "lambda":
            f = eval(hierarchy["lambda"][level])
            if level == len(hierarchy["lambda"]) - 1:
                is_suppressed = True
        elif hierarchy["type"] == "list":
            is_suppressed = hierarchy["hierarchy"][level]["is_suppressed"]
            if is_suppressed:
                f = lambda x: "*"
            else:
                generalized_values = hierarchy["hierarchy"][level]["values"]

                def find_generalized_value(x):
                    for generalized_value in generalized_values:
                        if x in generalized_value["original"]:
                            return generalized_value["generalized"]

                f = lambda x: find_generalized_value(x)
        return list(map(f, _values))

    while len(set(generalized_values)) > 1:
        generalized_values = generalize(generalized_values, current_level)
        current_level += 1

    return current_level / height
