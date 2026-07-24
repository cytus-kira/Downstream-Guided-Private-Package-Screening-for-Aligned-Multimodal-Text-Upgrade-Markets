#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the buyer-synchronized package-size sensitivity figure."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from unified_figure_style import DATASET_COLORS, apply_paper_style


DATASETS = ["hateful_memes", "hatespeech", "mscoco"]
DATASET_LABELS: Dict[str, str] = {
    "hateful_memes": "Hateful Memes",
    "hatespeech": "HateSpeech",
    "mscoco": "MSCOCO",
}
COLORS: Dict[str, str] = DATASET_COLORS


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--package-sizes", default="2,4,6,8")
    parser.add_argument("--purchase-total", type=int, default=48)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    return parser.parse_args()


def load_results(run_root: Path, package_sizes: List[int]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for package_size in package_sizes:
        size_root = run_root / f"package_size_{package_size}"
        paths = sorted(size_root.glob("seed_*/results.csv"))
        if not paths:
            raise FileNotFoundError(f"No results.csv files under {size_root}")
        for path in paths:
            frame = pd.read_csv(path)
            frame["package_size_sweep"] = int(package_size)
            frame["run_seed"] = int(path.parent.name.removeprefix("seed_"))
            frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result = result[result["method"] == "ours_sample_package_krr"].copy()
    if result.empty:
        raise ValueError("No ours_sample_package_krr rows found.")
    return result


def summarize(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed = (
        results.groupby(["package_size_sweep", "dataset", "run_seed"], as_index=False)[
            "good_count"
        ]
        .mean()
        .rename(columns={"good_count": "market_mean_good_count"})
    )
    by_dataset = (
        per_seed.groupby(["package_size_sweep", "dataset"])[
            "market_mean_good_count"
        ]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    overall_per_seed = (
        results.groupby(["package_size_sweep", "run_seed"], as_index=False)["good_count"]
        .mean()
        .rename(columns={"good_count": "dataset_market_mean_good_count"})
    )
    overall = (
        overall_per_seed.groupby("package_size_sweep")[
            "dataset_market_mean_good_count"
        ]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    return by_dataset, overall


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(colors="#333333")


def build_figure(
    by_dataset: pd.DataFrame,
    overall: pd.DataFrame,
    package_sizes: List[int],
    purchase_total: int,
    seed_count: int,
    figure_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(3.55, 2.9))
    for dataset in DATASETS:
        sub = by_dataset[by_dataset["dataset"] == dataset].sort_values(
            "package_size_sweep"
        )
        ax.errorbar(
            sub["package_size_sweep"],
            sub["mean"],
            yerr=sub["std"].fillna(0.0),
            color=COLORS[dataset],
            marker="o",
            markersize=5,
            linewidth=1.8,
            capsize=3,
            label=DATASET_LABELS[dataset],
        )

    overall = overall.sort_values("package_size_sweep")
    ax.errorbar(
        overall["package_size_sweep"],
        overall["mean"],
        yerr=overall["std"].fillna(0.0),
        color=COLORS["overall"],
        marker="s",
        markersize=5.5,
        linewidth=2.2,
        capsize=3,
        linestyle="--",
        label="Overall",
    )
    ax.set_xticks(package_sizes)
    ax.set_xlabel("Package size")
    ax.set_ylabel(f"Selected useful rows (budget = {purchase_total})")
    ax.set_ylim(0, purchase_total + 2)
    style_axes(ax)
    ax.legend(
        frameon=True,
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        facecolor="white",
        edgecolor="#D4D4D4",
        framealpha=0.86,
        fontsize=8.0,
        handlelength=1.3,
        columnspacing=0.7,
        borderpad=0.25,
        labelspacing=0.25,
    )
    fig.subplots_adjust(left=0.22, right=0.99, top=0.98, bottom=0.22)

    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = "fig_package_size_sensitivity_buyer_sync"
    fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cli = parse_args()
    apply_paper_style(base_size=10.8)
    package_sizes = [int(x) for x in parse_csv(cli.package_sizes)]
    results = load_results(cli.run_root, package_sizes)
    by_dataset, overall = summarize(results)

    cli.data_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(cli.data_dir / "package_size_sweep_all_rows.csv", index=False)
    by_dataset.to_csv(
        cli.data_dir / "package_size_sweep_by_dataset.csv",
        index=False,
    )
    overall.to_csv(cli.data_dir / "package_size_sweep_overall.csv", index=False)
    build_figure(
        by_dataset,
        overall,
        package_sizes,
        int(cli.purchase_total),
        int(results["run_seed"].nunique()),
        cli.figure_dir,
    )
    print(f"[figure] {cli.figure_dir / 'fig_package_size_sensitivity_buyer_sync.pdf'}")
    print(f"[data] {cli.data_dir}")


if __name__ == "__main__":
    main()
