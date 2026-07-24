#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build paper figures for main-text experiment summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from unified_figure_style import METHOD_COLORS_BY_KEY, USER_PALETTE, apply_paper_style


DEFAULT_RUN_ROOT = Path(
    "main_text_experiments/runs/latest_online_krr_ckks_parts_20260605_105428"
)
DEFAULT_FIG_DIR = Path("latex/fig")
DEFAULT_DATA_DIR = Path(
    "main_text_experiments/paper_ready/latest_online_krr_ckks_parts_20260605_105428"
)

DATASETS = ["hateful_memes", "hatespeech", "mscoco"]
DATASET_LABELS: Dict[str, str] = {
    "hateful_memes": "Hateful Memes",
    "hatespeech": "HateSpeech",
    "mscoco": "MSCOCO",
}

MARKETS = [
    "all_average",
    "noise",
    "coreset_far_wrong",
    "cosine",
    "uncertainty_badge",
    "kmeans_center",
    "typiclust_dense",
]
MARKET_LABELS: Dict[str, str] = {
    "all_average": "Mixed",
    "noise": "Noise/CoreSet",
    "coreset_far_wrong": "CoreSet",
    "cosine": "Cosine",
    "uncertainty_badge": "BADGE",
    "kmeans_center": "KMeans",
    "typiclust_dense": "TypiClust",
}

METHODS = [
    "market_random_select",
    "market_coreset_select",
    "market_cosine_select",
    "market_badge_select",
    "market_kmeans_center_select",
    "market_typiclust_select",
    "market_uncertainty_select",
    "ours_kernel_ridge_student",
    "ours_sample_package_krr",
]
METHOD_LABELS: Dict[str, str] = {
    "market_random_select": "Random",
    "market_coreset_select": "CoreSet",
    "market_cosine_select": "Cosine",
    "market_badge_select": "BADGE",
    "market_kmeans_center_select": "KMeans",
    "market_typiclust_select": "TypiClust",
    "market_uncertainty_select": "Uncertainty",
    "ours_kernel_ridge_student": "Raw KRR",
    "ours_sample_package_krr": "Pkg KRR",
}

PALETTE: Dict[str, str] = METHOD_COLORS_BY_KEY

ABLATION_VARIANTS = [
    ("Task\noperator", 20.81, 15.37, USER_PALETTE["light_gray"]),
    ("+ Loss-red.\nKRR", 33.92, 15.65, USER_PALETTE["blue"]),
    ("+ Val.\nalignment", 39.63, 11.04, USER_PALETTE["magenta"]),
]
TEACHER_REFERENCE_MEAN = 49.46


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper experiment figures.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--score-dir", default="main_score_5k")
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--package-override-summary-dir",
        type=Path,
        default=None,
        help="Optional summary directory whose package-KRR column replaces the base run.",
    )
    return parser.parse_args()


def load_pivot_summary(path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    long = df.melt(
        id_vars=["dataset", "profile"],
        value_vars=[m for m in METHODS if m in df.columns],
        var_name="method",
        value_name=value_name,
    )
    long["Dataset"] = long["dataset"].map(DATASET_LABELS)
    long["Market"] = long["profile"].map(MARKET_LABELS)
    long["Method"] = long["method"].map(METHOD_LABELS)
    long["dataset_order"] = long["dataset"].map({v: i for i, v in enumerate(DATASETS)})
    long["market_order"] = long["profile"].map({v: i for i, v in enumerate(MARKETS)})
    long["method_order"] = long["method"].map({v: i for i, v in enumerate(METHODS)})
    return long.sort_values(["dataset_order", "market_order", "method_order"])


def apply_package_override(base_path: Path, override_path: Path) -> pd.DataFrame:
    base = pd.read_csv(base_path)
    override = pd.read_csv(override_path)
    key = ["dataset", "profile"]
    column = "ours_sample_package_krr"
    if column not in override.columns:
        raise ValueError(f"Missing {column} in {override_path}")
    merged = base.merge(
        override[key + [column]].rename(columns={column: f"{column}_override"}),
        on=key,
        how="left",
        validate="one_to_one",
    )
    replacement = f"{column}_override"
    if merged[replacement].isna().any():
        missing = merged.loc[merged[replacement].isna(), key]
        raise ValueError(f"Package override is incomplete:\n{missing}")
    merged[column] = merged.pop(replacement)
    return merged


def pivot_frame_to_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    long = df.melt(
        id_vars=["dataset", "profile"],
        value_vars=[m for m in METHODS if m in df.columns],
        var_name="method",
        value_name=value_name,
    )
    long["Dataset"] = long["dataset"].map(DATASET_LABELS)
    long["Market"] = long["profile"].map(MARKET_LABELS)
    long["Method"] = long["method"].map(METHOD_LABELS)
    long["dataset_order"] = long["dataset"].map({v: i for i, v in enumerate(DATASETS)})
    long["market_order"] = long["profile"].map({v: i for i, v in enumerate(MARKETS)})
    long["method_order"] = long["method"].map({v: i for i, v in enumerate(METHODS)})
    return long.sort_values(["dataset_order", "market_order", "method_order"])


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.65, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(axis="both", colors="#333333")


def save_figure(fig: plt.Figure, fig_dir: Path, stem: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_std_grouped_bar(std_long: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 7.8), sharex=True, sharey=True)
    market_x = np.arange(len(MARKETS))
    width = 0.082
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2.0) * width

    y_max = max(1.0, std_long["std"].max() * 1.18)
    for ax, dataset in zip(axes, DATASETS):
        sub = std_long[std_long["dataset"] == dataset]
        for i, method in enumerate(METHODS):
            values: List[float] = []
            for market in MARKETS:
                row = sub[(sub["profile"] == market) & (sub["method"] == method)]
                values.append(float(row["std"].iloc[0]) if not row.empty else 0.0)
            edge = "#222222" if method.startswith("ours_") else "white"
            line_width = 0.75 if method.startswith("ours_") else 0.35
            ax.bar(
                market_x + offsets[i],
                values,
                width=width,
                color=PALETTE[method],
                edgecolor=edge,
                linewidth=line_width,
                label=METHOD_LABELS[method] if dataset == DATASETS[0] else None,
            )
        ax.set_title(DATASET_LABELS[dataset], loc="left", fontweight="bold")
        ax.set_ylim(0, y_max)
        ax.set_ylabel("Std. of useful rows")
        style_axes(ax)

    axes[-1].set_xticks(market_x)
    axes[-1].set_xticklabels([MARKET_LABELS[m] for m in MARKETS], rotation=22, ha="right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.015),
    )
    fig.suptitle(
        "Cross-seed standard deviation by dataset, market, and method",
        y=1.055,
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    save_figure(fig, fig_dir, "fig_main5k_std_by_scenario_grouped_bars")


def build_mean_std_grouped_bar(
    mean_long: pd.DataFrame, std_long: pd.DataFrame, fig_dir: Path
) -> None:
    merged = mean_long.merge(
        std_long[["dataset", "profile", "method", "std"]],
        on=["dataset", "profile", "method"],
        how="left",
    )
    merged["std"] = merged["std"].fillna(0.0)

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.85), sharex=True, sharey=True)
    market_x = np.arange(len(MARKETS))
    width = 0.058
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2.0) * width

    for ax, dataset in zip(axes, DATASETS):
        sub = merged[merged["dataset"] == dataset]
        for i, method in enumerate(METHODS):
            means: List[float] = []
            stds: List[float] = []
            for market in MARKETS:
                row = sub[(sub["profile"] == market) & (sub["method"] == method)]
                means.append(float(row["mean"].iloc[0]) if not row.empty else 0.0)
                stds.append(float(row["std"].iloc[0]) if not row.empty else 0.0)
            edge = "#222222" if method.startswith("ours_") else "white"
            line_width = 0.75 if method.startswith("ours_") else 0.35
            ax.bar(
                market_x + offsets[i],
                means,
                width=width,
                yerr=stds,
                capsize=1.6,
                error_kw={"elinewidth": 0.65, "capthick": 0.65, "ecolor": "#333333"},
                color=PALETTE[method],
                edgecolor=edge,
                linewidth=line_width,
                label=METHOD_LABELS[method] if dataset == DATASETS[0] else None,
            )
        ax.axhline(5, color="#555555", linestyle="--", linewidth=0.8, alpha=0.65)
        ax.set_title(DATASET_LABELS[dataset], loc="left", fontweight="bold")
        ax.set_ylim(0, 55)
        if dataset == DATASETS[0]:
            ax.set_ylabel("Useful rows\n(mean $\pm$ std.)")
        style_axes(ax)

    for ax in axes:
        ax.set_xticks(market_x)
        ax.set_xticklabels([MARKET_LABELS[m] for m in MARKETS], rotation=48, ha="right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(METHODS),
        frameon=False,
        fontsize=6.1,
        handlelength=0.75,
        columnspacing=0.38,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.81, bottom=0.29, wspace=0.13)
    save_figure(fig, fig_dir, "fig_main5k_mean_std_by_scenario_grouped_bars")


def build_mean_market_line(mean_long: pd.DataFrame, fig_dir: Path) -> None:
    agg = (
        mean_long.groupby(["profile", "method", "Market", "Method"], dropna=False)["mean"]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg["market_order"] = agg["profile"].map({v: i for i, v in enumerate(MARKETS)})
    agg["method_order"] = agg["method"].map({v: i for i, v in enumerate(METHODS)})
    agg = agg.sort_values(["method_order", "market_order"])

    fig, ax = plt.subplots(figsize=(4.55, 4.55))
    x = np.arange(len(MARKETS))
    for method in METHODS:
        sub = agg[agg["method"] == method].set_index("profile").reindex(MARKETS)
        y = sub["mean"].astype(float).values
        yerr = sub["std"].fillna(0.0).astype(float).values
        line_width = 2.4 if method.startswith("ours_") else 1.55
        marker_size = 6 if method.startswith("ours_") else 4.5
        ax.plot(
            x,
            y,
            marker="o",
            markersize=marker_size,
            linewidth=line_width,
            color=PALETTE[method],
            label=METHOD_LABELS[method],
        )
        if method == "ours_sample_package_krr":
            ax.fill_between(
                x,
                np.maximum(0, y - yerr),
                np.minimum(50, y + yerr),
                color=PALETTE[method],
                alpha=0.075,
                linewidth=0,
            )

    ax.axhline(5, color="#555555", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([MARKET_LABELS[m] for m in MARKETS], rotation=38, ha="right")
    ax.set_ylim(-1.2, 52)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_ylabel("Useful rows selected (mean)")
    style_axes(ax)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=True,
        facecolor="white",
        edgecolor="#D4D4D4",
        framealpha=0.86,
        ncol=2,
        fontsize=7.0,
        handlelength=1.1,
        columnspacing=0.5,
        borderpad=0.25,
        labelspacing=0.25,
    )
    fig.subplots_adjust(left=0.17, right=0.99, top=0.99, bottom=0.25)
    save_figure(fig, fig_dir, "fig_main5k_mean_by_market_lines")


def build_std_heatmap(std_long: pd.DataFrame, fig_dir: Path) -> None:
    agg = (
        std_long.groupby(["profile", "method"], dropna=False)["std"]
        .mean()
        .reset_index()
    )
    matrix = (
        agg.pivot(index="method", columns="profile", values="std")
        .reindex(index=METHODS, columns=MARKETS)
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    im = ax.imshow(matrix.values, cmap="viridis", aspect="auto", vmin=0)
    ax.set_xticks(np.arange(len(MARKETS)))
    ax.set_xticklabels([MARKET_LABELS[m] for m in MARKETS], rotation=28, ha="right")
    ax.set_yticks(np.arange(len(METHODS)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS])
    ax.set_title("Average standard deviation across datasets", fontsize=12, fontweight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.values[i, j]
            color = "white" if val > matrix.values.max() * 0.55 else "#222222"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Std. of useful rows")
    fig.tight_layout()
    save_figure(fig, fig_dir, "fig_main5k_std_heatmap")


def build_component_ablation_bar(fig_dir: Path) -> None:
    labels = [row[0] for row in ABLATION_VARIANTS]
    means = np.asarray([row[1] for row in ABLATION_VARIANTS], dtype=float)
    stds = np.asarray([row[2] for row in ABLATION_VARIANTS], dtype=float)
    colors = [row[3] for row in ABLATION_VARIANTS]

    fig, ax = plt.subplots(figsize=(3.4, 2.05))
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=2.5,
        width=0.62,
        color=colors,
        edgecolor="#222222",
        linewidth=0.6,
        error_kw={"elinewidth": 0.7, "capthick": 0.7, "ecolor": "#333333"},
    )
    for bar, hatch in zip(bars, ["", "//", "\\\\"]):
        bar.set_hatch(hatch)

    ax.axhline(
        TEACHER_REFERENCE_MEAN,
        color="#222222",
        linewidth=0.7,
        linestyle="--",
        alpha=0.85,
    )
    ax.text(
        x[0] - 0.28,
        TEACHER_REFERENCE_MEAN + 1.15,
        "Teacher/ref. (non-deployable)",
        ha="left",
        va="bottom",
        fontsize=9.0,
        color="#222222",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.6},
    )

    ax.set_ylim(0, 58)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("Useful-good rows", fontsize=9.2, labelpad=2.0)
    ax.grid(axis="y", color="#D4D4D4", linewidth=0.65, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(axis="both", colors="#333333")
    ax.tick_params(axis="x", labelsize=9.0)
    ax.tick_params(axis="y", labelsize=9.0)
    fig.subplots_adjust(left=0.18, right=0.99, top=0.94, bottom=0.29)
    save_figure(fig, fig_dir, "ablation_downstream_signals")
    save_figure(fig, fig_dir, "fig_component_ablation_online")


def main() -> None:
    args = parse_args()
    summary_dir = args.run_root / args.score_dir / "summary"
    mean_path = summary_dir / "good_count_pivot_mean.csv"
    std_path = summary_dir / "good_count_pivot_std.csv"
    if not mean_path.exists() or not std_path.exists():
        raise FileNotFoundError(f"Missing mean/std summary under {summary_dir}")

    if args.package_override_summary_dir is None:
        mean_long = load_pivot_summary(mean_path, "mean")
        std_long = load_pivot_summary(std_path, "std")
    else:
        override_dir = args.package_override_summary_dir
        mean_frame = apply_package_override(
            mean_path,
            override_dir / "good_count_pivot_mean.csv",
        )
        std_frame = apply_package_override(
            std_path,
            override_dir / "good_count_pivot_std.csv",
        )
        mean_long = pivot_frame_to_long(mean_frame, "mean")
        std_long = pivot_frame_to_long(std_frame, "std")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    mean_long.to_csv(args.data_dir / "figure_main5k_mean_long.csv", index=False)
    std_long.to_csv(args.data_dir / "figure_main5k_std_long.csv", index=False)

    apply_paper_style(base_size=10.5)
    build_std_grouped_bar(std_long, args.fig_dir)
    build_mean_std_grouped_bar(mean_long, std_long, args.fig_dir)
    build_mean_market_line(mean_long, args.fig_dir)
    build_std_heatmap(std_long, args.fig_dir)
    build_component_ablation_bar(args.fig_dir)
    print(f"Wrote figures to {args.fig_dir.resolve()}")


if __name__ == "__main__":
    main()
