#!/usr/bin/env python
"""Replay exact and quartic KRR acquisition on the saved 63 main cases."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from unified_figure_style import METHOD_COLORS_BY_LABEL, apply_paper_style


VARIANTS = {
    "raw_krr": {
        "label": "Raw KRR",
        "method": "ours_kernel_ridge_student",
        "poly_key": "krr_score",
        "exact_key": "krr_score_exact",
        "color": METHOD_COLORS_BY_LABEL["Raw KRR"],
    },
    "package_krr": {
        "label": "Pkg KRR",
        "method": "ours_sample_package_krr",
        "poly_key": "sample_package_krr_full_score",
        "exact_key": "sample_package_krr_full_score_exact",
        "color": METHOD_COLORS_BY_LABEL["Pkg KRR"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay selected-good fidelity for exact and quartic KRR."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("main_text_experiments/runs/poly4_krr_main5k"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "main_text_experiments/runs/poly4_krr_main5k/fidelity_replay"
        ),
    )
    parser.add_argument("--figure-dir", type=Path, default=Path("latex/fig"))
    parser.add_argument("--purchase-total", type=int, default=50)
    parser.add_argument(
        "--main-cases",
        action="store_true",
        help="Require the canonical 3 datasets x 7 markets x 3 seeds.",
    )
    parser.add_argument(
        "--with-selected-good-delta",
        action="store_true",
        help="Retained as an explicit reproducibility flag for this replay.",
    )
    return parser.parse_args()


def package_members(market: np.lib.npyio.NpzFile) -> list[np.ndarray]:
    offsets = np.asarray(market["sample_package_offsets"], dtype=np.int64)
    members = np.asarray(market["sample_package_members"], dtype=np.int64)
    return [
        members[offsets[i] : offsets[i + 1]].astype(np.int64)
        for i in range(len(offsets) - 1)
    ]


def select_rows(
    market: np.lib.npyio.NpzFile,
    score: np.ndarray,
    variant: str,
    purchase_total: int,
) -> np.ndarray:
    score = np.asarray(score, dtype=np.float32)
    if variant == "raw_krr":
        return np.argsort(-score, kind="mergesort")[:purchase_total].astype(np.int64)

    packages = package_members(market)
    order = np.argsort(-score, kind="mergesort")
    chosen: list[int] = []
    for package_id in order.tolist():
        rows = packages[int(package_id)].tolist()
        if len(chosen) + len(rows) > purchase_total:
            continue
        chosen.extend(int(row_id) for row_id in rows)
        if len(chosen) == purchase_total:
            break
    return np.asarray(chosen, dtype=np.int64)


def replay_case(
    path: Path,
    seed: int,
    dataset: str,
    market_name: str,
    variant: str,
    purchase_total: int,
) -> dict[str, object]:
    spec = VARIANTS[variant]
    with np.load(path, allow_pickle=True) as market:
        is_good = np.asarray(market["is_good"], dtype=np.int64)
        exact_score = np.asarray(market[spec["exact_key"]], dtype=np.float32)
        quartic_score = np.asarray(market[spec["poly_key"]], dtype=np.float32)
        exact_rows = select_rows(
            market, exact_score, variant, purchase_total
        )
        quartic_rows = select_rows(
            market, quartic_score, variant, purchase_total
        )
        package_sizes = np.asarray(
            market.get("sample_package_sizes", np.ones(len(is_good))),
            dtype=np.int64,
        )

    if len(exact_rows) != purchase_total or len(quartic_rows) != purchase_total:
        raise ValueError(
            f"{path}: expected {purchase_total} selected rows, got "
            f"{len(exact_rows)} exact and {len(quartic_rows)} quartic."
        )
    exact_good = int(np.sum(is_good[exact_rows]))
    quartic_good = int(np.sum(is_good[quartic_rows]))
    overlap = len(set(exact_rows.tolist()) & set(quartic_rows.tolist())) / purchase_total
    return {
        "variant": variant,
        "variant_label": spec["label"],
        "case_id": f"{dataset}/{market_name}/seed{seed}",
        "dataset": dataset,
        "market": market_name,
        "seed": seed,
        "purchase_total": purchase_total,
        "package_size_min": int(np.min(package_sizes)) if variant == "package_krr" else 1,
        "package_size_max": int(np.max(package_sizes)) if variant == "package_krr" else 1,
        "exact_selected_good": exact_good,
        "quartic_selected_good": quartic_good,
        "abs_delta": abs(exact_good - quartic_good),
        "signed_delta": quartic_good - exact_good,
        "top50_overlap": overlap,
        "max_score_error": float(np.max(np.abs(quartic_score - exact_score))),
    }


def load_deployed_results(run_root: Path) -> pd.DataFrame:
    frames = []
    for seed_dir in sorted(run_root.glob("seed_*")):
        path = seed_dir / "results.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["run_seed"] = int(seed_dir.name.split("_", 1)[1])
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No seed_*/results.csv files under {run_root}")
    return pd.concat(frames, ignore_index=True)


def validate_replay(detail: pd.DataFrame, run_root: Path, require_main: bool) -> None:
    expected_datasets = {"hateful_memes", "hatespeech", "mscoco"}
    expected_markets = {
        "noise",
        "coreset_far_wrong",
        "typiclust_dense",
        "kmeans_center",
        "uncertainty_badge",
        "cosine",
        "all_average",
    }
    expected_seeds = {42, 43, 44}
    if require_main:
        for variant, group in detail.groupby("variant"):
            if len(group) != 63:
                raise ValueError(f"{variant}: expected 63 cases, found {len(group)}")
            if set(group["dataset"]) != expected_datasets:
                raise ValueError(f"{variant}: dataset set mismatch")
            if set(group["market"]) != expected_markets:
                raise ValueError(f"{variant}: market set mismatch")
            if set(group["seed"]) != expected_seeds:
                raise ValueError(f"{variant}: seed set mismatch")

    deployed = load_deployed_results(run_root)
    for variant, spec in VARIANTS.items():
        replay_rows = detail[detail["variant"] == variant]
        deployed_rows = deployed[deployed["method"] == spec["method"]].copy()
        merged = replay_rows.merge(
            deployed_rows[["dataset", "profile", "run_seed", "good_count"]],
            left_on=["dataset", "market", "seed"],
            right_on=["dataset", "profile", "run_seed"],
            how="left",
            validate="one_to_one",
        )
        if merged["good_count"].isna().any():
            raise ValueError(f"{variant}: missing deployed rows during validation")
        mismatch = merged[
            merged["quartic_selected_good"].astype(int)
            != merged["good_count"].astype(int)
        ]
        if not mismatch.empty:
            columns = [
                "case_id",
                "quartic_selected_good",
                "good_count",
            ]
            raise ValueError(
                f"{variant}: quartic replay mismatch:\n"
                + mismatch[columns].to_string(index=False)
            )


def aggregate(detail: pd.DataFrame) -> pd.DataFrame:
    summary = (
        detail.groupby(["variant", "variant_label"], as_index=False)
        .agg(
            cases=("case_id", "size"),
            mean_max_score_error=("max_score_error", "mean"),
            worst_score_error=("max_score_error", "max"),
            mean_top50_overlap=("top50_overlap", "mean"),
            minimum_top50_overlap=("top50_overlap", "min"),
            exact_selected_good_mean=("exact_selected_good", "mean"),
            quartic_selected_good_mean=("quartic_selected_good", "mean"),
            mean_abs_selected_good_delta=("abs_delta", "mean"),
            worst_selected_good_delta=("abs_delta", "max"),
        )
    )
    order = {"raw_krr": 0, "package_krr": 1}
    summary["_order"] = summary["variant"].map(order)
    return summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def write_latex(summary: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for row in summary.itertuples(index=False):
        rows.append(
            f"{row.variant_label} & ${row.mean_max_score_error:.4f}$ & "
            f"${row.worst_score_error:.4f}$ & "
            f"${row.mean_top50_overlap:.4f}$ & "
            f"${row.minimum_top50_overlap:.2f}$ & "
            f"${row.exact_selected_good_mean:.2f}$ & "
            f"${row.quartic_selected_good_mean:.2f}$ & "
            f"${row.mean_abs_selected_good_delta:.2f}$ & "
            f"${int(row.worst_selected_good_delta)}$ \\\\"
        )
    (output_dir / "table6_quartic_fidelity.tex").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def plot_fidelity(detail: pd.DataFrame, summary: pd.DataFrame, figure_dir: Path) -> None:
    apply_paper_style(base_size=10)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), sharex=True, sharey=True)
    for axis, variant in zip(axes, ["raw_krr", "package_krr"]):
        spec = VARIANTS[variant]
        group = detail[detail["variant"] == variant]
        counts = Counter(
            zip(
                group["exact_selected_good"].astype(int),
                group["quartic_selected_good"].astype(int),
            )
        )
        x = np.asarray([pair[0] for pair in counts], dtype=float)
        y = np.asarray([pair[1] for pair in counts], dtype=float)
        frequency = np.asarray(list(counts.values()), dtype=float)
        axis.plot([0, 50], [0, 50], color="#666666", linewidth=1.0, linestyle="--")
        axis.scatter(
            x,
            y,
            s=25 + 18 * frequency,
            color=spec["color"],
            alpha=0.72,
            edgecolor="white",
            linewidth=0.6,
        )
        row = summary[summary["variant"] == variant].iloc[0]
        axis.text(
            0.04,
            0.96,
            f"mean |delta| = {row['mean_abs_selected_good_delta']:.2f}\n"
            f"worst |delta| = {int(row['worst_selected_good_delta'])}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "pad": 3},
        )
        axis.set_title(spec["label"], fontweight="bold")
        axis.set_xlim(-1, 51)
        axis.set_ylim(-1, 51)
        axis.set_xticks([0, 10, 20, 30, 40, 50])
        axis.set_yticks([0, 10, 20, 30, 40, 50])
        axis.grid(True, color="#DDDDDD", linewidth=0.6, alpha=0.8)
        axis.set_xlabel("Exact-exponential selected-good")
    axes[0].set_ylabel("Quartic selected-good")
    fig.suptitle(
        "Acquisition-level fidelity over 63 dataset-market-seed cases",
        y=1.01,
        fontsize=10.5,
        fontweight="bold",
    )
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = figure_dir / f"fig_quartic_selected_good_fidelity.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for seed_dir in sorted(args.run_root.glob("seed_*")):
        seed = int(seed_dir.name.split("_", 1)[1])
        for path in sorted((seed_dir / "markets").rglob("*.npz")):
            dataset = path.parents[2].name
            market_name = path.stem
            for variant in VARIANTS:
                rows.append(
                    replay_case(
                        path,
                        seed,
                        dataset,
                        market_name,
                        variant,
                        args.purchase_total,
                    )
                )
    detail = pd.DataFrame(rows)
    if detail.empty:
        raise FileNotFoundError(f"No market NPZ files found under {args.run_root}")
    validate_replay(detail, args.run_root, args.main_cases)
    summary = aggregate(detail)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(
        args.output_dir / "quartic_fidelity_selected_good.csv", index=False
    )
    summary.to_csv(
        args.output_dir / "table6_quartic_fidelity_summary.csv", index=False
    )
    write_latex(summary, args.output_dir)
    plot_fidelity(detail, summary, args.figure_dir)

    print(summary.to_string(index=False))
    print(f"[done] cases={len(detail) // len(VARIANTS)} output={args.output_dir}")


if __name__ == "__main__":
    main()
