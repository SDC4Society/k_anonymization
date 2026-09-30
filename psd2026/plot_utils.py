# ruff: noqa: F401

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import linregress

# Set specific label sizes
mpl.rcParams["axes.labelsize"] = 20  # x and y labels
mpl.rcParams["xtick.labelsize"] = 16  # x-axis ticks
mpl.rcParams["ytick.labelsize"] = 16  # y-axis ticks
mpl.rcParams["legend.fontsize"] = 16  # legend font
mpl.rcParams["axes.titlesize"] = 25  # title font
mpl.rcParams["lines.markersize"] = 12  # marker size
mpl.rcParams["grid.linewidth"] = 1.5
mpl.rcParams["lines.linewidth"] = 2.5
mpl.rcParams["axes.linewidth"] = 1.5

K = [2, 5, 10] + list(range(20, 101, 10))
K_TICKS = [2, 10] + list(range(20, 101, 10))

ALPHA = 0.7
WIDTH = 5
HEIGHT = 4
NORM = colors.Normalize(vmin=2, vmax=100)
CMAP_BLUE = colors.LinearSegmentedColormap.from_list("blue_seq", ["lightblue", "blue"])
CMAP_ORANGE = colors.LinearSegmentedColormap.from_list(
    "orange_seq", ["navajowhite", "chocolate"]
)
CMAP_PURPLE = colors.LinearSegmentedColormap.from_list("purple_seq", ["pink", "purple"])
NOTES = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p"]

METRICS_SPECS = {
    "UT_CAVG": {"name": r"$C_{AVG}$"},
    "UT_DM": {"name": r"$DM$"},
    "UT_NCP": {"name": r"$NCP$"},
    "UT_RM": {"name": r"$\mathcal{L}_{REG}$", "marker": "o", "color": "tab:blue"},
    "UT_CM_BIN": {"name": r"$\mathcal{L}_{BIN}$", "marker": "X", "color": "tab:orange"},
    "UT_CM_MUL": {"name": r"$\mathcal{L}_{MUL}$", "marker": "^", "color": "tab:purple"},
    "UT_RM+NCP": {
        "name": r"m$(\mathcal{L}_{REG},NCP)$",
        "marker": "o",
        "color": "tab:blue",
    },
    "UT_CM_BIN+NCP": {
        "name": r"m$(\mathcal{L}_{BIN},NCP)$",
        "marker": "X",
        "color": "tab:orange",
    },
    "UT_CM_MUL+NCP": {
        "name": r"m$(\mathcal{L}_{MUL,NCP)}$",
        "marker": "^",
        "color": "tab:purple",
    },
    "ML_REG_RMSE": {"name": r"$\mathcal{E}_{REG}$", "marker": "o", "color": "tab:blue"},
    "ML_CLS_BIN_F1_LOSS": {
        "name": r"$\mathcal{E}_{BIN}$",
        "marker": "X",
        "color": "tab:orange",
    },
    "ML_CLS_MUL_F1_LOSS": {
        "name": r"$\mathcal{E}_{MUL}$",
        "marker": "^",
        "color": "tab:purple",
    },
    "VUL_DBRL": {"name": r"$\mathcal{V}$"},
}

CORR_SPECS = [
    {
        "marker": "o",
        "note": "●",
        "color": "tab:blue",
        "cmap": CMAP_BLUE,
    },
    {
        "marker": "X",
        "note": "✖",
        "color": "tab:orange",
        "cmap": CMAP_ORANGE,
    },
    {
        "marker": "^",
        "note": "▲",
        "color": "tab:purple",
        "cmap": CMAP_PURPLE,
    },
    {
        "marker": "s",
        "note": "◼",
        "color": "tab:red",
        "cmap": colors.LinearSegmentedColormap.from_list(
            "red_seq", ["lightcoral", "crimson"]
        ),
    },
]


def draw_trend_line(AX, X, Y, fit: str = "linear", xlim=None, ylim=None):
    x_flatten = [item for sublist in X for item in sublist]
    x_trend = (
        np.linspace(np.min(x_flatten), np.max(x_flatten), 100)
        if xlim is None
        else np.linspace(xlim[0], xlim[1], 100)
    )

    y_flatten = [item for sublist in Y for item in sublist]

    if fit == "linear":
        res = linregress(x_flatten, y_flatten)
        y_trend = res.slope * x_trend + res.intercept
    elif fit == "logarithmic":
        _x = np.log(x_flatten)
        y_trend = (
            np.linspace(np.min(y_flatten), np.max(y_flatten), 100)
            if ylim is None
            else np.linspace(ylim[0], ylim[1], 100)
        )
        res = linregress(y_flatten, _x)
        x_trend = np.exp(res.intercept) * np.exp(y_trend * res.slope)
    elif fit == "exponential":
        _y = np.log(y_flatten)
        res = linregress(x_flatten, _y)
        y_trend = np.exp(res.intercept) * np.exp(x_trend * res.slope)
    elif fit == "power":
        _x = np.log(x_flatten)
        _y = np.log(y_flatten)
        res = linregress(_x, _y)
        y_trend = np.exp(res.intercept) * x_trend**res.slope

    AX.plot(
        x_trend,
        y_trend,
        color="tab:red",
        alpha=ALPHA,
        zorder=0.8,
        label=f"{r'$r^2$'}={res.rvalue**2:.2f}, "
        + f"p={f'{res.pvalue:.2e}' if res.pvalue < 0.001 else f'{res.pvalue:.3f}'}",
    )


def load_results(dataset_name: str):
    results_df = pd.read_csv(f"./results/{dataset_name}/results_summary.csv")
    results_df["ML_CLS_BIN_F1_LOSS"] = 1 - results_df["ML_CLS_BIN_F1"]
    results_df["ML_CLS_MUL_F1_LOSS"] = 1 - results_df["ML_CLS_MUL_F1"]

    results = {}
    for algo in results_df.METHOD.unique():
        result = (
            results_df.query(f'METHOD == "{algo}" & K.isin({[1] + K})')
            .sort_values("K")
            .reset_index(drop=False)
        )
        results[algo] = result.copy()
    return results


def plot_single_vs_k(
    anon_results: pd.DataFrame,
    draw_specs: list[list[dict]],
    original_results: pd.DataFrame = None,
    alpha=0.7,
    ax_title_offset=-0.4,
    wspace=0.25,
    hspace=0.5,
):
    def draw(_axis, _X, _Y, _label, _color="tab:blue", _marker="o", _linestyle=":"):
        _axis.plot(
            _X,
            _Y,
            label=_label,
            color=_color,
            marker=_marker,
            markeredgecolor="k",
            markeredgewidth=0.2,
            linestyle=_linestyle,
            alpha=alpha,
        )

    nrows = len(draw_specs)
    ncols = max([len(r) for r in draw_specs])

    fig, ax = plt.subplots(
        nrows, ncols, figsize=[ncols * WIDTH, nrows * HEIGHT], dpi=150
    )

    i_subplot = -1
    for i_row, row in enumerate(draw_specs):
        for i_col, col in enumerate(row):
            i_subplot += 1
            axis = ax[i_row, i_col]
            for metric in col["metrics"]:
                draw(
                    axis,
                    anon_results["K"],
                    anon_results[metric],
                    _label=METRICS_SPECS[metric]["name"],
                    _color=(
                        "tab:blue"
                        if "color" not in METRICS_SPECS[metric]
                        else METRICS_SPECS[metric]["color"]
                    ),
                    _marker=(
                        "o"
                        if "marker" not in METRICS_SPECS[metric]
                        else METRICS_SPECS[metric]["marker"]
                    ),
                )
                if original_results is not None:
                    if metric in original_results and not np.isnan(
                        original_results.loc[0, metric]
                    ):
                        draw(
                            axis,
                            [K[0], K[-1]],
                            [original_results.loc[0, metric]] * 2,
                            _label="original",
                            _color="tab:green",
                            _marker=None,
                            _linestyle="-",
                        )
            if len(col["metrics"]) > 1:
                axis.legend(ncols=2 if len(col["metrics"]) > 2 else 1)

            caption = (
                col["caption"]
                if "caption" in col
                else METRICS_SPECS[col["metrics"][0]]["name"]
            )
            axis.set_title(
                f"({NOTES[i_subplot]}) {caption}",
                y=ax_title_offset,
            )
            axis.set_xlabel("k")
            axis.set_xlim([0, 102])
            axis.set_xticks(K_TICKS)
            axis.set_ylabel(caption)

            if "ylim" in col:
                axis.set_ylim(col["ylim"])
            if "yticks" in col:
                axis.set_yticks(col["yticks"])
            axis.grid(linestyle=":")

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    return fig


def plot_corr(
    results: list[pd.DataFrame],
    draw_pairs: list[list[dict]],
    draw_specs: dict,
    alpha=0.7,
    ax_title_offset=-0.4,
    wspace=0.25,
    hspace=0.5,
):
    def draw(_axis, _X, _Y, _cmap=CMAP_BLUE, _marker="o", c=K, dotted=True):
        if dotted:
            _axis.plot(_X, _Y, zorder=0.8, linestyle=":", color="k", alpha=alpha)
        _axis.scatter(
            _X,
            _Y,
            marker=_marker,
            edgecolors="k",
            linewidths=1,
            alpha=alpha,
            s=150,
            c=c,
            cmap=_cmap,
            vmin=2,
            vmax=100,
        )

    nrows = len(draw_pairs)
    ncols = max([len(r) for r in draw_pairs])

    fig, ax = plt.subplots(
        nrows, ncols, figsize=[ncols * WIDTH, nrows * HEIGHT], dpi=150
    )
    i_subplot = -1
    for i_row, row in enumerate(draw_pairs):
        for i_col, col in enumerate(row):
            i_subplot += 1
            axis = ax[i_row, i_col]
            if col["X"] != "LEGEND":
                results_X = (
                    [result[col["X"]].tolist() for result in results]
                    if "filter" not in col
                    else [
                        result[col["X"]][col["filter"][0] : col["filter"][1]].tolist()
                        for result in results
                    ]
                )
                results_Y = (
                    [result[col["Y"]].tolist() for result in results]
                    if "filter" not in col
                    else [
                        result[col["Y"]][col["filter"][0] : col["filter"][1]].tolist()
                        for result in results
                    ]
                )
                _c = (
                    K if "filter" not in col else K[col["filter"][0] : col["filter"][1]]
                )
                if "fit" in col:
                    draw_trend_line(
                        axis,
                        results_X,
                        results_Y,
                        fit=None if "fit" not in col else col["fit"],
                        xlim=(
                            col["xlim"]
                            if "xlim" in col
                            else draw_specs[col["X"]]["lim"]
                        ),
                        ylim=(
                            col["ylim"]
                            if "ylim" in col
                            else draw_specs[col["Y"]]["lim"]
                        ),
                    )
                    axis.legend()
                for i_result, result in enumerate(results):
                    draw(
                        axis,
                        results_X[i_result],
                        results_Y[i_result],
                        _cmap=CORR_SPECS[i_result]["cmap"],
                        _marker=CORR_SPECS[i_result]["marker"],
                        c=_c,
                        dotted=col["dotted"] if "dotted" in col else True,
                    )
                X_name = METRICS_SPECS[col["X"]]["name"]
                Y_name = METRICS_SPECS[col["Y"]]["name"]
                axis.set_title(
                    f"({NOTES[i_subplot]}) {X_name} vs {Y_name}"
                    + f"{' (' + col['note'] + ')' if 'note' in col else ''}",
                    y=ax_title_offset,
                )
                axis.set_xlabel(X_name)
                axis.set_xlim(
                    col["xlim"] if "xlim" in col else draw_specs[col["X"]]["lim"]
                )
                axis.set_xticks(
                    col["xticks"] if "xticks" in col else draw_specs[col["X"]]["ticks"]
                )

                axis.set_ylabel(Y_name)
                axis.set_ylim(
                    col["ylim"] if "ylim" in col else draw_specs[col["Y"]]["lim"]
                )
                axis.set_yticks(
                    col["yticks"] if "yticks" in col else draw_specs[col["Y"]]["ticks"]
                )
                axis.grid(linestyle=":")
            else:
                ax_color_top = ax[i_row, i_col]
                divider = make_axes_locatable(ax_color_top)
                ax_color_mid = divider.append_axes("bottom", size="100%", pad=0.9)
                fig.add_axes(ax_color_mid)
                ax_color_bot = divider.append_axes("bottom", size="100%", pad=0.9)
                fig.add_axes(ax_color_bot)
                if len(col["Y"]) == 1:
                    ax_color_top.set_visible(False)
                    ax_color_bot.set_visible(False)
                    ax_colors = [ax_color_mid]
                elif len(col["Y"]) == 2:
                    ax_color_mid.set_visible(False)
                    ax_colors = [ax_color_top, ax_color_bot]
                else:
                    ax_colors = [ax_color_top, ax_color_mid, ax_color_bot]

                for i_ax_color, ax_color in enumerate(ax_colors):
                    fig.colorbar(
                        mpl.cm.ScalarMappable(
                            norm=NORM, cmap=CORR_SPECS[i_ax_color]["cmap"]
                        ),
                        cax=ax_color,
                        orientation="horizontal",
                        label=f'{CORR_SPECS[i_ax_color]["note"]} {col["Y"][i_ax_color]}',
                        ticks=[2, 50, 100],
                        shrink=1,
                    )
                    ax_color.set_xticklabels(["k=2", "k=50", "k=100"])

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    return fig
