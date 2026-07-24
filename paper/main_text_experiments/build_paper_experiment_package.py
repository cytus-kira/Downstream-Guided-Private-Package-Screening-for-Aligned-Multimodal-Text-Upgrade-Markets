#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build paper-ready experiment tables from completed main-text runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


DEFAULT_RUN_ROOT = Path(
    "main_text_experiments/runs/latest_online_krr_ckks_parts_20260605_105428"
)
DEFAULT_CKKS_FULLFLOW = Path(
    "encrypted_benchmarks/results/ckks_seal_summary.csv"
)

DATASET_LABELS: Dict[str, str] = {
    "hateful_memes": "Hateful Memes",
    "hatespeech": "HateSpeech",
    "mscoco": "MSCOCO",
}

PROFILE_LABELS: Dict[str, str] = {
    "noise": "noise",
    "coreset_far_wrong": "coreset_far_wrong",
    "typiclust_dense": "typiclust_dense",
    "kmeans_center": "kmeans_center",
    "uncertainty_badge": "uncertainty_badge",
    "cosine": "cosine",
    "all_average": "all_average",
}

METHOD_LABELS: Dict[str, str] = {
    "market_random_select": "Random",
    "market_cosine_select": "Cosine",
    "market_uncertainty_select": "Uncertainty",
    "market_coreset_select": "CoreSet",
    "market_badge_select": "BADGE",
    "market_kmeans_center_select": "KMeans",
    "market_typiclust_select": "TypiClust",
    "ours_downstream_direct": "Teacher-direct",
    "ours_influence_only": "Teacher-no-loss",
    "ours_loss_reduction_only": "Teacher-no-influence",
    "ours_task_operator": "Task operator",
    "ours_krr_influence_only": "Raw KRR-no-loss",
    "ours_krr_loss_reduction_only": "Raw KRR-no-influence",
    "ours_kernel_ridge_student": "Raw KRR",
    "ours_sample_package_direct": "Teacher-pkg-direct",
    "ours_sample_package_influence_only": "Teacher-pkg-no-loss",
    "ours_sample_package_loss_reduction_only": "Teacher-pkg-no-influence",
    "ours_sample_package_task_operator": "Pkg task operator",
    "ours_sample_package_krr_influence_only": "Pkg KRR-no-loss",
    "ours_sample_package_krr_loss_reduction_only": "Pkg KRR-no-influence",
    "ours_sample_package_krr": "Pkg KRR",
    "min_targeted_baselines": "Target baseline",
}

BASELINE_GOOD_COLUMNS: Dict[str, str] = {
    "coreset_good": "CoreSet",
    "cosine_good": "Cosine",
    "badge_good": "BADGE",
    "kmeans_good": "KMeans",
    "typiclust_good": "TypiClust",
    "uncertainty_good": "Uncertainty",
}

ADVANCED_METHOD_ORDER = [
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

ABLATION_ORDER = [
    "ours_downstream_direct",
    "ours_sample_package_direct",
    "ours_loss_reduction_only",
    "ours_sample_package_loss_reduction_only",
    "ours_influence_only",
    "ours_sample_package_influence_only",
    "ours_task_operator",
    "ours_sample_package_task_operator",
    "ours_kernel_ridge_student",
    "ours_sample_package_krr",
    "ours_krr_loss_reduction_only",
    "ours_sample_package_krr_loss_reduction_only",
    "ours_krr_influence_only",
    "ours_sample_package_krr_influence_only",
]

COMPACT_ABLATION_ORDER = [
    "ours_downstream_direct",
    "ours_loss_reduction_only",
    "ours_influence_only",
    "ours_kernel_ridge_student",
    "ours_krr_loss_reduction_only",
    "ours_krr_influence_only",
    "ours_task_operator",
    "ours_sample_package_krr",
]


class TableWriter:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.manifest: List[Dict[str, str]] = []

    def write(self, name: str, df: pd.DataFrame, source: str) -> None:
        csv_path = self.out_dir / f"{name}.csv"
        tex_path = self.out_dir / f"{name}.tex"
        df.to_csv(csv_path, index=False)
        tex_path.write_text(
            df.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.2f}"),
            encoding="utf-8",
        )
        self.manifest.append(
            {"name": name, "csv": str(csv_path), "tex": str(tex_path), "source": source}
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ordered paper experiment tables from completed summaries."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--ckks-fullflow", type=Path, default=DEFAULT_CKKS_FULLFLOW)
    parser.add_argument("--main-score-dir", default="main_score_5k")
    parser.add_argument("--scale-score-dir", default="main_score_20k")
    parser.add_argument("--ablation-dir", default="main_ablation_5k")
    parser.add_argument("--ratio-dir", default="ratio_downstream_5k")
    return parser.parse_args()


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def summary_path(run_root: Path, section: str, filename: str) -> Path:
    return run_root / section / "summary" / filename


def order_by(values: Iterable[str]) -> Dict[str, int]:
    return {value: i for i, value in enumerate(values)}


def add_common_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "dataset" in out.columns:
        out["Dataset"] = out["dataset"].map(DATASET_LABELS).fillna(out["dataset"])
    if "profile" in out.columns:
        out["Market"] = out["profile"].map(PROFILE_LABELS).fillna(out["profile"])
    return out


def round_numeric(df: pd.DataFrame, digits: int = 2) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(digits)
    return out


def best_label(row: pd.Series) -> str:
    available = {col: label for col, label in BASELINE_GOOD_COLUMNS.items() if col in row}
    if not available:
        return ""
    best_col = max(available, key=lambda col: row[col])
    return available[best_col]


def build_advanced_tables(
    writer: TableWriter,
    run_root: Path,
    score_dir: str,
    prefix: str,
) -> Dict[str, float]:
    targeted_path = summary_path(run_root, score_dir, "targeted_summary_mean.csv")
    aggregate_path = summary_path(run_root, score_dir, "method_good_count_aggregate.csv")
    targeted = add_common_labels(read_csv(targeted_path))
    aggregate = read_csv(aggregate_path)

    baseline_cols = [col for col in BASELINE_GOOD_COLUMNS if col in targeted.columns]
    targeted["Best advanced"] = targeted[baseline_cols].max(axis=1)
    targeted["Best advanced method"] = targeted.apply(best_label, axis=1)
    targeted["Ours > target"] = targeted["ours_krr_good"] > targeted["target_good_count"]
    targeted["Ours >= best advanced"] = targeted["ours_krr_good"] >= targeted["Best advanced"]
    targeted["Target < 10"] = targeted["target_good_count"] < 10

    table = targeted[
        [
            "Dataset",
            "Market",
            "target_label",
            "target_good_count",
            "coreset_good",
            "cosine_good",
            "badge_good",
            "kmeans_good",
            "typiclust_good",
            "uncertainty_good",
            "Best advanced",
            "Best advanced method",
            "ours_krr_good",
            "ours_pkg_krr_good",
            "Ours > target",
            "Ours >= best advanced",
            "Target < 10",
        ]
    ].rename(
        columns={
            "target_label": "Targeted method",
            "target_good_count": "Target good",
            "coreset_good": "CoreSet",
            "cosine_good": "Cosine",
            "badge_good": "BADGE",
            "kmeans_good": "KMeans",
            "typiclust_good": "TypiClust",
            "uncertainty_good": "Uncertainty",
            "ours_krr_good": "Ours online KRR",
            "ours_pkg_krr_good": "Ours package KRR",
        }
    )
    writer.write(f"{prefix}_advanced_comparison", round_numeric(table), str(targeted_path))

    aggregate = aggregate.copy()
    aggregate["method_label"] = aggregate["method_label"].fillna(
        aggregate["method"].map(METHOD_LABELS)
    )
    aggregate["_order"] = aggregate["method"].map(order_by(ADVANCED_METHOD_ORDER)).fillna(999)
    aggregate = aggregate.sort_values(["_order", "method"]).drop(columns=["_order"])
    agg_table = aggregate[
        [
            "method_label",
            "mean",
            "std",
            "min",
            "max",
            "count",
            "scoring_semantics",
            "online_scoring_uses_downstream_model",
        ]
    ].rename(
        columns={
            "method_label": "Method",
            "mean": "Good mean",
            "std": "Good std",
            "min": "Good min",
            "max": "Good max",
            "count": "Rows",
            "scoring_semantics": "Scoring semantics",
            "online_scoring_uses_downstream_model": "Uses downstream online",
        }
    )
    writer.write(f"{prefix}_advanced_method_aggregate", round_numeric(agg_table), str(aggregate_path))

    collapse = pd.DataFrame(
        [
            {
                "Score run": score_dir,
                "Markets": len(targeted),
                "Target good mean": targeted["target_good_count"].mean(),
                "Target good max": targeted["target_good_count"].max(),
                "Target < 10 count": int(targeted["Target < 10"].sum()),
                "Ours > target count": int(targeted["Ours > target"].sum()),
                "Ours >= best advanced count": int(targeted["Ours >= best advanced"].sum()),
                "Ours mean": targeted["ours_krr_good"].mean(),
                "Ours package mean": targeted["ours_pkg_krr_good"].mean(),
                "Best advanced mean": targeted["Best advanced"].mean(),
            }
        ]
    )
    writer.write(f"{prefix}_targeted_collapse_summary", round_numeric(collapse), str(targeted_path))

    return {
        "target_mean": float(targeted["target_good_count"].mean()),
        "target_max": float(targeted["target_good_count"].max()),
        "target_lt_10": int(targeted["Target < 10"].sum()),
        "ours_gt_target": int(targeted["Ours > target"].sum()),
        "ours_ge_best": int(targeted["Ours >= best advanced"].sum()),
        "markets": int(len(targeted)),
        "ours_mean": float(targeted["ours_krr_good"].mean()),
        "best_advanced_mean": float(targeted["Best advanced"].mean()),
    }


def variant_descriptor(method: str) -> Dict[str, str]:
    table: Dict[str, Dict[str, str]] = {
        "ours_downstream_direct": {
            "Variant": "Teacher full operator",
            "Scoring stage": "teacher/reference",
            "Influence term": "yes",
            "Loss-reduction term": "yes",
            "Student distillation": "no",
            "Sample package": "no",
        },
        "ours_loss_reduction_only": {
            "Variant": "Teacher without influence",
            "Scoring stage": "teacher/reference",
            "Influence term": "removed",
            "Loss-reduction term": "yes",
            "Student distillation": "no",
            "Sample package": "no",
        },
        "ours_influence_only": {
            "Variant": "Teacher without loss reduction",
            "Scoring stage": "teacher/reference",
            "Influence term": "yes",
            "Loss-reduction term": "removed",
            "Student distillation": "no",
            "Sample package": "no",
        },
        "ours_kernel_ridge_student": {
            "Variant": "Online KRR student",
            "Scoring stage": "online",
            "Influence term": "distilled",
            "Loss-reduction term": "distilled",
            "Student distillation": "KRR",
            "Sample package": "no",
        },
        "ours_krr_loss_reduction_only": {
            "Variant": "Online KRR without influence",
            "Scoring stage": "online",
            "Influence term": "removed",
            "Loss-reduction term": "distilled",
            "Student distillation": "KRR",
            "Sample package": "no",
        },
        "ours_krr_influence_only": {
            "Variant": "Online KRR without loss reduction",
            "Scoring stage": "online",
            "Influence term": "distilled",
            "Loss-reduction term": "removed",
            "Student distillation": "KRR",
            "Sample package": "no",
        },
        "ours_task_operator": {
            "Variant": "Online task operator",
            "Scoring stage": "online",
            "Influence term": "closed form",
            "Loss-reduction term": "closed form",
            "Student distillation": "no",
            "Sample package": "no",
        },
        "ours_sample_package_krr": {
            "Variant": "Online package KRR student",
            "Scoring stage": "online",
            "Influence term": "distilled",
            "Loss-reduction term": "distilled",
            "Student distillation": "KRR",
            "Sample package": "yes",
        },
    }
    return table.get(
        method,
        {
            "Variant": METHOD_LABELS.get(method, method),
            "Scoring stage": "unknown",
            "Influence term": "unknown",
            "Loss-reduction term": "unknown",
            "Student distillation": "unknown",
            "Sample package": "unknown",
        },
    )


def build_ablation_tables(writer: TableWriter, run_root: Path, ablation_dir: str) -> Dict[str, float]:
    overall_path = summary_path(run_root, ablation_dir, "ablation_summary_overall.csv")
    by_market_path = summary_path(run_root, ablation_dir, "ablation_summary_by_market.csv")
    overall = read_csv(overall_path)
    by_market = read_csv(by_market_path)

    overall = overall.copy()
    overall["_order"] = overall["method"].map(order_by(ABLATION_ORDER)).fillna(999)
    overall = overall.sort_values(["_order", "method"]).drop(columns=["_order"])
    overall_table = overall[
        ["method_label", "mean", "std", "min", "max", "count", "scoring_semantics"]
    ].rename(
        columns={
            "method_label": "Method",
            "mean": "Good mean",
            "std": "Good std",
            "min": "Good min",
            "max": "Good max",
            "count": "Rows",
            "scoring_semantics": "Scoring semantics",
        }
    )
    writer.write("02_ablation_overall_all_variants", round_numeric(overall_table), str(overall_path))

    compact = overall[overall["method"].isin(COMPACT_ABLATION_ORDER)].copy()
    compact["_order"] = compact["method"].map(order_by(COMPACT_ABLATION_ORDER)).fillna(999)
    compact = compact.sort_values(["_order", "method"]).drop(columns=["_order"])
    desc = pd.DataFrame([variant_descriptor(method) for method in compact["method"]])
    compact_table = pd.concat([desc.reset_index(drop=True), compact.reset_index(drop=True)], axis=1)
    compact_table = compact_table[
        [
            "Variant",
            "Scoring stage",
            "Influence term",
            "Loss-reduction term",
            "Student distillation",
            "Sample package",
            "mean",
            "std",
            "min",
            "max",
        ]
    ].rename(
        columns={
            "mean": "Good mean",
            "std": "Good std",
            "min": "Good min",
            "max": "Good max",
        }
    )
    writer.write("02_ablation_math_components", round_numeric(compact_table), str(overall_path))

    if not by_market.empty:
        by_market = add_common_labels(by_market)
        by_market["_order"] = by_market["method"].map(order_by(COMPACT_ABLATION_ORDER)).fillna(999)
        by_market_table = by_market[by_market["method"].isin(COMPACT_ABLATION_ORDER)].sort_values(
            ["Dataset", "Market", "_order", "method"]
        )
        by_market_table = by_market_table[
            ["Dataset", "Market", "method_label", "mean", "std", "min", "max", "count"]
        ].rename(
            columns={
                "method_label": "Method",
                "mean": "Good mean",
                "std": "Good std",
                "min": "Good min",
                "max": "Good max",
                "count": "Rows",
            }
        )
        writer.write("02_ablation_by_market", round_numeric(by_market_table), str(by_market_path))

    metric = overall.set_index("method")["mean"]
    return {
        "teacher_direct": float(metric.get("ours_downstream_direct", np.nan)),
        "teacher_no_influence": float(metric.get("ours_loss_reduction_only", np.nan)),
        "teacher_no_loss": float(metric.get("ours_influence_only", np.nan)),
        "online_krr": float(metric.get("ours_kernel_ridge_student", np.nan)),
        "online_no_influence": float(metric.get("ours_krr_loss_reduction_only", np.nan)),
        "online_no_loss": float(metric.get("ours_krr_influence_only", np.nan)),
        "online_task": float(metric.get("ours_task_operator", np.nan)),
        "online_pkg_krr": float(metric.get("ours_sample_package_krr", np.nan)),
    }


def build_raw_package_selection_table(
    writer: TableWriter, run_root: Path, score_dirs: List[str]
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for score_dir in score_dirs:
        per_seed_path = summary_path(run_root, score_dir, "targeted_summary_per_seed.csv")
        if not per_seed_path.exists():
            continue
        per_seed = read_csv(per_seed_path)
        if "ours_krr_good" not in per_seed.columns or "ours_pkg_krr_good" not in per_seed.columns:
            continue
        delta = per_seed["ours_pkg_krr_good"] - per_seed["ours_krr_good"]
        rows.append(
            {
                "Score run": score_dir,
                "Rows": int(len(per_seed)),
                "Raw online KRR mean": per_seed["ours_krr_good"].mean(),
                "Package online KRR mean": per_seed["ours_pkg_krr_good"].mean(),
                "Mean package-minus-raw": delta.mean(),
                "Max abs delta": delta.abs().max(),
                "Identical rows": int((delta.abs() < 1e-12).sum()),
            }
        )
    table = pd.DataFrame(rows)
    writer.write(
        "03_internal_raw_vs_package_selection",
        round_numeric(table),
        ";".join(score_dirs),
    )
    return table


def scheme_label(scheme: str) -> str:
    if scheme == "ours_student_row_ctpt":
        return "Raw row-wise KRR"
    if scheme == "ours_student_pkg_ctpt":
        return "Sample-package KRR"
    return scheme


def build_ckks_tables(writer: TableWriter, ckks_fullflow: Path) -> Dict[str, float]:
    ckks = read_csv(ckks_fullflow, required=False)
    if ckks.empty:
        return {}

    keep_cols = [
        "scheme",
        "logical_rows",
        "chunks",
        "scored_objects",
        "raw_rows_per_scored_object",
        "input_prepare_encrypt_ms_mean",
        "encrypted_compute_ms_mean",
        "decrypt_decode_ms_mean",
        "total_full_flow_ms_mean",
        "rows_per_second_mean",
        "input_ciphertexts_mean",
        "output_ciphertexts_mean",
        "total_communication_bytes_mean",
        "ct_pt_mults_mean",
        "additions_mean",
        "rescales_mean",
    ]
    keep_cols = [col for col in keep_cols if col in ckks.columns]
    table = ckks[keep_cols].copy()
    table["Scheme"] = table["scheme"].map(scheme_label)
    table["Total communication MB"] = table["total_communication_bytes_mean"] / (1024 * 1024)
    table = table.sort_values(["logical_rows", "scheme"])
    table = table[
        [
            "Scheme",
            "logical_rows",
            "chunks",
            "scored_objects",
            "raw_rows_per_scored_object",
            "input_prepare_encrypt_ms_mean",
            "encrypted_compute_ms_mean",
            "decrypt_decode_ms_mean",
            "total_full_flow_ms_mean",
            "rows_per_second_mean",
            "input_ciphertexts_mean",
            "output_ciphertexts_mean",
            "Total communication MB",
            "ct_pt_mults_mean",
            "additions_mean",
            "rescales_mean",
        ]
    ].rename(
        columns={
            "logical_rows": "Rows",
            "chunks": "Chunks",
            "scored_objects": "Scored objects",
            "raw_rows_per_scored_object": "Rows per object",
            "input_prepare_encrypt_ms_mean": "Input encrypt ms",
            "encrypted_compute_ms_mean": "Encrypted compute ms",
            "decrypt_decode_ms_mean": "Decrypt/decode ms",
            "total_full_flow_ms_mean": "Full-flow ms",
            "rows_per_second_mean": "Rows/sec",
            "input_ciphertexts_mean": "Input ciphertexts",
            "output_ciphertexts_mean": "Output ciphertexts",
            "ct_pt_mults_mean": "Ct-pt mults",
            "additions_mean": "Additions",
            "rescales_mean": "Rescales",
        }
    )
    writer.write("03_ckks_fullflow_raw_package", round_numeric(table), str(ckks_fullflow))

    rows: List[Dict[str, float]] = []
    for logical_rows, group in ckks.groupby("logical_rows"):
        raw = group[group["scheme"] == "ours_student_row_ctpt"]
        pkg = group[group["scheme"] == "ours_student_pkg_ctpt"]
        if raw.empty or pkg.empty:
            continue
        raw_row = raw.iloc[0]
        pkg_row = pkg.iloc[0]
        rows.append(
            {
                "Rows": int(logical_rows),
                "Raw full-flow ms": raw_row["total_full_flow_ms_mean"],
                "Package full-flow ms": pkg_row["total_full_flow_ms_mean"],
                "Full-flow speedup": raw_row["total_full_flow_ms_mean"]
                / pkg_row["total_full_flow_ms_mean"],
                "Raw encrypted compute ms": raw_row["encrypted_compute_ms_mean"],
                "Package encrypted compute ms": pkg_row["encrypted_compute_ms_mean"],
                "Encrypted compute speedup": raw_row["encrypted_compute_ms_mean"]
                / pkg_row["encrypted_compute_ms_mean"],
                "Raw communication MB": raw_row["total_communication_bytes_mean"] / (1024 * 1024),
                "Package communication MB": pkg_row["total_communication_bytes_mean"] / (1024 * 1024),
                "Communication reduction": raw_row["total_communication_bytes_mean"]
                / pkg_row["total_communication_bytes_mean"],
                "Raw input ciphertexts": raw_row["input_ciphertexts_mean"],
                "Package input ciphertexts": pkg_row["input_ciphertexts_mean"],
            }
        )
    speedup = pd.DataFrame(rows).sort_values("Rows")
    writer.write("03_ckks_packaging_speedup", round_numeric(speedup), str(ckks_fullflow))

    if speedup.empty:
        return {}
    return {
        "min_fullflow_speedup": float(speedup["Full-flow speedup"].min()),
        "max_fullflow_speedup": float(speedup["Full-flow speedup"].max()),
        "min_comm_reduction": float(speedup["Communication reduction"].min()),
        "max_comm_reduction": float(speedup["Communication reduction"].max()),
    }


def build_ratio_tables(writer: TableWriter, run_root: Path, ratio_dir: str) -> Dict[str, float]:
    aggregate_path = summary_path(run_root, ratio_dir, "method_good_count_aggregate.csv")
    metrics_path = summary_path(run_root, ratio_dir, "downstream_metrics_mean_std.csv")
    aggregate = read_csv(aggregate_path, required=False)
    metrics = read_csv(metrics_path, required=False)

    stats: Dict[str, float] = {}
    if not aggregate.empty:
        aggregate = aggregate.copy()
        aggregate["_order"] = aggregate["method"].map(order_by(ADVANCED_METHOD_ORDER)).fillna(999)
        aggregate = aggregate.sort_values(["_order", "method"]).drop(columns=["_order"])
        table = aggregate[
            [
                "method_label",
                "mean",
                "std",
                "min",
                "max",
                "count",
                "scoring_semantics",
                "online_scoring_uses_downstream_model",
            ]
        ].rename(
            columns={
                "method_label": "Method",
                "mean": "Purchased-good mean",
                "std": "Purchased-good std",
                "min": "Purchased-good min",
                "max": "Purchased-good max",
                "count": "Rows",
                "scoring_semantics": "Scoring semantics",
                "online_scoring_uses_downstream_model": "Uses downstream online",
            }
        )
        writer.write("03_downstream_purchase_good_count_aggregate", round_numeric(table), str(aggregate_path))
        ours = aggregate[aggregate["method"] == "ours_kernel_ridge_student"]
        if not ours.empty:
            stats["ratio_ours_good_mean"] = float(ours.iloc[0]["mean"])

    if not metrics.empty:
        grouped = (
            metrics.groupby(["method", "method_label", "purchase_ratio"], dropna=False)
            .agg(
                test_auroc_mean=("test_auroc_mean", "mean"),
                test_macro_f1_mean=("test_macro_f1_mean", "mean"),
                test_acc_mean=("test_acc_mean", "mean"),
                good_count_mean=("good_count_mean", "mean"),
                purchase_total_mean=("purchase_total", "mean"),
            )
            .reset_index()
        )
        grouped["_order"] = grouped["method"].map(order_by(ADVANCED_METHOD_ORDER)).fillna(999)
        grouped = grouped.sort_values(["_order", "purchase_ratio", "method"]).drop(columns=["_order"])
        table = grouped[
            [
                "method_label",
                "purchase_ratio",
                "purchase_total_mean",
                "good_count_mean",
                "test_auroc_mean",
                "test_macro_f1_mean",
                "test_acc_mean",
            ]
        ].rename(
            columns={
                "method_label": "Method",
                "purchase_ratio": "Purchase ratio",
                "purchase_total_mean": "Purchase total",
                "good_count_mean": "Good count",
                "test_auroc_mean": "AUROC",
                "test_macro_f1_mean": "Macro-F1",
                "test_acc_mean": "Accuracy",
            }
        )
        writer.write("03_downstream_purchase_ratio_metrics", round_numeric(table), str(metrics_path))
    return stats


def build_summary_markdown(
    out_dir: Path,
    run_root: Path,
    main_stats: Dict[str, float],
    scale_stats: Dict[str, float],
    ablation_stats: Dict[str, float],
    ckks_stats: Dict[str, float],
    ratio_stats: Dict[str, float],
    raw_package_table: pd.DataFrame,
    manifest: List[Dict[str, str]],
) -> None:
    if raw_package_table.empty:
        raw_package_summary = "- Raw/package selection comparison was not available."
    else:
        row = raw_package_table.iloc[0]
        raw_package_summary = (
            f"- {row['Score run']}: raw online KRR averages "
            f"{float(row['Raw online KRR mean']):.2f}/50 and true pre-score package KRR averages "
            f"{float(row['Package online KRR mean']):.2f}/50; package-minus-raw is "
            f"{float(row['Mean package-minus-raw']):.2f}, with "
            f"{int(row['Identical rows'])}/{int(row['Rows'])} identical cases."
        )
    lines = [
        "# Paper Experiment Package",
        "",
        f"- Source run root: `{run_root}`",
        "- Section order: advanced-method comparison -> ablation -> internal comparison.",
        "",
        "## 01 Advanced-Method Comparison",
        "",
        (
            f"- Main 5k: target baselines average {main_stats.get('target_mean', np.nan):.2f}/50, "
            f"max {main_stats.get('target_max', np.nan):.2f}/50; "
            f"target < 10 in {main_stats.get('target_lt_10', 0)}/{main_stats.get('markets', 0)} markets."
        ),
        (
            f"- Main 5k: ours online KRR average {main_stats.get('ours_mean', np.nan):.2f}/50; "
            f"ours > targeted baseline in {main_stats.get('ours_gt_target', 0)}/{main_stats.get('markets', 0)} markets; "
            f"ours >= best advanced method in {main_stats.get('ours_ge_best', 0)}/{main_stats.get('markets', 0)} markets."
        ),
        (
            f"- Scale 20k: target baselines average {scale_stats.get('target_mean', np.nan):.2f}/50; "
            f"ours average {scale_stats.get('ours_mean', np.nan):.2f}/50; "
            f"ours > targeted baseline in {scale_stats.get('ours_gt_target', 0)}/{scale_stats.get('markets', 0)} markets."
        ),
        "",
        "## 02 Ablation",
        "",
        (
            f"- Teacher/reference full operator: {ablation_stats.get('teacher_direct', np.nan):.2f}/50."
        ),
        (
            f"- Online KRR student: {ablation_stats.get('online_krr', np.nan):.2f}/50; "
            f"without influence: {ablation_stats.get('online_no_influence', np.nan):.2f}/50; "
            f"without loss reduction: {ablation_stats.get('online_no_loss', np.nan):.2f}/50."
        ),
        (
            f"- Closed-form online task operator diagnostic: {ablation_stats.get('online_task', np.nan):.2f}/50."
        ),
        "",
        "## 03 Internal Comparison",
        "",
        raw_package_summary,
        (
            f"- CKKS package full-flow speedup range: "
            f"{ckks_stats.get('min_fullflow_speedup', np.nan):.2f}x to {ckks_stats.get('max_fullflow_speedup', np.nan):.2f}x."
        ),
        (
            f"- CKKS package communication reduction range: "
            f"{ckks_stats.get('min_comm_reduction', np.nan):.2f}x to {ckks_stats.get('max_comm_reduction', np.nan):.2f}x."
        ),
        (
            f"- Downstream purchase auxiliary good-count mean for ours online KRR: "
            f"{ratio_stats.get('ratio_ours_good_mean', np.nan):.2f}."
        ),
        "",
        "## Files",
        "",
    ]
    for item in manifest:
        lines.append(f"- `{Path(item['csv']).name}` from `{item['source']}`")
    (out_dir / "00_paper_experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_root = args.run_root
    if not run_root.exists():
        raise FileNotFoundError(run_root)

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = Path("main_text_experiments/paper_ready") / run_root.name
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = TableWriter(out_dir)
    main_stats = build_advanced_tables(writer, run_root, args.main_score_dir, "01_main5k")
    scale_stats = build_advanced_tables(writer, run_root, args.scale_score_dir, "01_scale20k")
    ablation_stats = build_ablation_tables(writer, run_root, args.ablation_dir)
    raw_package_table = build_raw_package_selection_table(
        writer, run_root, [args.main_score_dir, args.scale_score_dir]
    )
    ckks_stats = build_ckks_tables(writer, args.ckks_fullflow)
    ratio_stats = build_ratio_tables(writer, run_root, args.ratio_dir)

    manifest_path = out_dir / "paper_experiment_tables_manifest.json"
    manifest_path.write_text(
        json.dumps(writer.manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    build_summary_markdown(
        out_dir,
        run_root,
        main_stats,
        scale_stats,
        ablation_stats,
        ckks_stats,
        ratio_stats,
        raw_package_table,
        writer.manifest,
    )
    print(f"Wrote paper experiment package to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
