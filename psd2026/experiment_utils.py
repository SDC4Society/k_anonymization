import time
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.base import BaseEstimator
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import TargetEncoder

from k_anonymization.algorithms.full_generalization import Incognito
from k_anonymization.algorithms.local_recoding import (
    GroupAnonymizationBuiltIn,
    LocalRecodingAlgorithm,
)
from k_anonymization.algorithms.utils import generalize_column
from k_anonymization.core import Algorithm, Dataset, SampleDataset
from k_anonymization.evaluation import data_utility as UT

BASE_RESULTS = {
    "METHOD": None,
    "K": 1,
    "RUN_TIME": None,
    "UT_NCP": None,
    "UT_CAVG": None,
    "UT_DM": None,
    "UT_RM": None,
    "UT_CM_BIN": None,
    "UT_CM_MUL": None,
    "ML_REG_RMSE": None,
    "ML_REG_R2": None,
    "ML_CLS_BIN_F1": None,
    "ML_CLS_MUL_F1": None,
    "VUL_DBRL": None,
}


def create_unique_df(dataset: Dataset, features: list, target: str):
    unique_df = (
        dataset.df.groupby(features, as_index=False, observed=False)
        .count()
        .query(f"{target} == 1")
        .reset_index(drop=True)
    )
    return pd.merge(
        dataset.df,
        unique_df[features],
        on=features,
        how="right",
    ).reset_index(drop=True)


def make_default_ML_pipe(
    target_type: Literal["binary", "multiclass", "continous"],
    seed: int = None,
):
    model = (
        HistGradientBoostingRegressor(random_state=seed)
        if target_type == "continuous"
        else HistGradientBoostingClassifier(random_state=seed)
    )
    return make_ML_pipe(model, target_type, seed)


def make_ML_pipe(
    model: BaseEstimator,
    target_type: Literal["binary", "multiclass", "continuous"],
    seed: int = None,
):
    return make_pipeline(
        make_column_transformer(
            (
                TargetEncoder(target_type=target_type, random_state=seed),
                make_column_selector(dtype_include="category"),
            ),
            remainder="passthrough",
            n_jobs=-1,
        ),
        model,
    )


def get_ML_reg_result(
    pipe, features, target, train_df: pd.DataFrame, validation_df: pd.DataFrame
):
    y_pred = pipe.fit(train_df[features], train_df[target]).predict(
        validation_df[features]
    )
    max_range = train_df[target].max() - train_df[target].min()
    return np.sqrt(
        mean_squared_error(validation_df[target], y_pred)
    ).item() / max_range, r2_score(validation_df[target], y_pred)


def get_ML_cls_result(
    pipe, features, target, train_df: pd.DataFrame, validation_df: pd.DataFrame
):
    y_pred = pipe.fit(train_df[features], train_df[target]).predict(
        validation_df[features]
    )
    return f1_score(validation_df[target], y_pred, average="weighted")


def get_utility_all(
    algo: LocalRecodingAlgorithm | Incognito,
    targets: dict,
    features: list,
):
    ut_scores = {}
    ut_scores["UT_NCP"] = (
        UT.NCP.calculate_for_local_recoding_mean_mode(
            algo.org_data,
            algo.groups,
            algo.qids_idx,
            algo.is_categorical,
        )
        if isinstance(algo, LocalRecodingAlgorithm)
        else algo.best_score
    )
    ut_scores["UT_CAVG"] = UT.CAVG.calculate(
        algo.anon_data, algo.dataset.qids_idx, algo.k
    )
    ut_scores["UT_DM"] = (
        UT.Discernibility.calculate(algo.anon_data, algo.dataset.qids_idx)
        / algo.dataset.df.shape[0]
    )
    ut_scores.update(
        get_utility_ML(
            algo.anon_data,
            features,
            targets["reg"],
            targets["bin"],
            targets["mul"],
        )
    )
    return ut_scores


def get_utility_ML(
    data: pd.DataFrame,
    qids: list[str],
    reg_target: str = None,
    bin_target: str = None,
    mul_target: str = None,
):
    ml_ut_scores = {}
    ml_ut_scores["UT_RM"] = (
        None if reg_target is None else UT.RM.calculate(data, qids, reg_target)
    )
    ml_ut_scores["UT_CM_BIN"] = (
        None if bin_target is None else UT.CM.calculate(data, qids, bin_target)
    )
    ml_ut_scores["UT_CM_MUL"] = (
        None if mul_target is None else UT.CM.calculate(data, qids, mul_target)
    )
    return ml_ut_scores


def generalize_df(df, generalization, hierarchies, qids_numerical):
    _df = df.copy()
    for attr, level in generalization:
        if level == 0:
            continue
        if level == -1:
            if attr in qids_numerical:
                _df[attr] = 0
            else:
                _df[attr] = "*"
        else:
            _df[attr], _ = generalize_column(_df[attr], hierarchies[attr], 0, level)
    return _df


def get_DBRL_attack_result(
    dataset: Dataset,
    anon_df: pd.DataFrame,
    columns: list,
    numerical_columns: list,
    seed: int = None,
):
    shuffle_df = anon_df.sample(frac=1, random_state=seed)
    cost = np.zeros((anon_df.shape[0], anon_df.shape[0]))
    for col in columns:
        anon_col = shuffle_df[col].values
        org_col = dataset.df[col].values
        if col not in dataset.qids:
            cost += (anon_col != org_col[:, None]).astype(float)
        else:
            if col in numerical_columns:
                dist = np.abs(anon_col - org_col[:, None]).astype(float)
                cost += dist / (org_col.max() - org_col.min())
            else:
                cost += (anon_col != org_col[:, None]).astype(float)

    positions, guesses = linear_sum_assignment(cost.T)
    attack_result = pd.DataFrame(
        {
            "true_idx": shuffle_df.index,
            "guessed_idx": guesses,
            "distance": cost.T[positions, guesses],
        }
    )
    attack_result["correct"] = attack_result["true_idx"] == attack_result["guessed_idx"]
    return attack_result


def get_data(base_dataset: Dataset, split: dict):
    SAMPLE = SampleDataset(base_dataset, base_dataset.df.loc[split["train_idx"]].copy())
    VALIDATION = base_dataset.df.loc[split["test_idx"]].copy()
    SAMPLE.df[SAMPLE.qids_categorical] = SAMPLE.df[SAMPLE.qids_categorical].astype(
        "category"
    )
    VALIDATION[SAMPLE.qids_categorical] = VALIDATION[SAMPLE.qids_categorical].astype(
        "category"
    )

    return (SAMPLE, VALIDATION)


def anonymize_data(algo_class: Algorithm, dataset: Dataset, k: int, seed: int = None):
    time_start = time.perf_counter()
    if issubclass(algo_class, LocalRecodingAlgorithm):
        ALGO = algo_class(
            dataset,
            k,
            group_anonymization=GroupAnonymizationBuiltIn.MEAN_MODE,
            seed=seed,
        )
    else:
        ALGO = algo_class(dataset, k)
    ALGO.anonymize()
    run_time = time.perf_counter() - time_start
    return (ALGO, run_time)


def get_ut_and_ml_metrics(
    algo: Algorithm,
    validation_df: pd.DataFrame,
    pipes: dict,
    features: list,
    targets: dict,
    generalization: list[tuple] = None,
):
    results = {}
    cat_qids = algo.dataset.qids_categorical.copy()

    if generalization is not None:
        VALIDATION_DF = generalize_df(
            validation_df,
            generalization,
            algo.dataset.hierarchies,
            algo.dataset.qids_numerical,
        )
        for attr, level in generalization:
            if (attr in algo.dataset.qids_numerical) and (level != 0):
                cat_qids.append(attr)
    else:
        VALIDATION_DF = validation_df

    anon_df = algo.anon_data.copy()
    anon_df[cat_qids] = anon_df[cat_qids].astype("category")
    VALIDATION_DF[cat_qids] = VALIDATION_DF[cat_qids].astype("category")

    results["ML_REG_RMSE"], results["ML_REG_R2"] = get_ML_reg_result(
        pipes["reg"], features, targets["reg"], anon_df, VALIDATION_DF
    )
    results["ML_CLS_BIN_F1"] = get_ML_cls_result(
        pipes["bin"], features, targets["bin"], anon_df, VALIDATION_DF
    )
    results["ML_CLS_MUL_F1"] = get_ML_cls_result(
        pipes["mul"], features, targets["mul"], anon_df, VALIDATION_DF
    )

    results.update(get_utility_all(algo, targets, features))

    return results


def get_attack_results(
    algo: Algorithm,
    dataset: Dataset,
    features: list,
    generalization: list[tuple] = None,
    seed: int = None,
):
    NUM_COLS = dataset.qids_numerical
    if generalization is not None:
        SAMPLE = SampleDataset(
            dataset,
            generalize_df(
                dataset.df,
                generalization,
                algo.dataset.hierarchies,
                algo.dataset.qids_numerical,
            ),
        )
        for attr, level in generalization:
            if (attr in dataset.qids_numerical) and (level != 0):
                NUM_COLS.remove(attr)
    else:
        SAMPLE = dataset

    ATTACK_RESULTS_DF = get_DBRL_attack_result(
        SAMPLE, algo.anon_data, features, NUM_COLS, seed
    )
    VUL_DBRL = ATTACK_RESULTS_DF.correct.sum() / ATTACK_RESULTS_DF.shape[0]

    return (ATTACK_RESULTS_DF, VUL_DBRL)
