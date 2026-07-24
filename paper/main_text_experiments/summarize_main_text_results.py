#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize main-text targeted-market score-only experiment outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


TARGET_METHOD: Dict[str, str] = {
    "noise": "market_coreset_select",
    "coreset_far_wrong": "market_coreset_select",
    "typiclust_dense": "market_typiclust_select",
    "kmeans_center": "market_kmeans_center_select",
    "uncertainty_badge": "market_badge_select",
    "cosine": "market_cosine_select",
    "all_average": "min_targeted_baselines",
}

BASELINE_METHODS = [
    "market_coreset_select",
    "market_cosine_select",
    "market_badge_select",
    "market_kmeans_center_select",
    "market_typiclust_select",
    "market_uncertainty_select",
]

PIVOT_METHODS = [
    "market_random_select",
    "market_cosine_select",
    "market_uncertainty_select",
    "market_coreset_select",
    "market_badge_select",
    "market_kmeans_center_select",
    "market_typiclust_select",
    "ours_downstream_direct",
    "ours_influence_only",
    "ours_loss_reduction_only",
    "ours_task_operator",
    "ours_krr_influence_only",
    "ours_krr_loss_reduction_only",
    "ours_kernel_ridge_student",
    "ours_sample_package_direct",
    "ours_sample_package_influence_only",
    "ours_sample_package_loss_reduction_only",
    "ours_sample_package_task_operator",
    "ours_sample_package_krr_influence_only",
    "ours_sample_package_krr_loss_reduction_only",
    "ours_sample_package_krr",
]

METHOD_LABELS = {
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
    "oracle_downstream_gain": "Teacher-oracle",
    "min_targeted_baselines": "Target baseline",
}


def discover_run_dirs(run_root: Path) -> List[Path]:
    run_root = Path(run_root)
    if (run_root / "results.csv").exists():
        return [run_root]
    out = [p for p in sorted(run_root.iterdir()) if p.is_dir() and (p / "results.csv").exists()]
    if not out:
        raise FileNotFoundError(f"No results.csv found under {run_root}")
    return out


def seed_from_run_dir(run_dir: Path) -> int:
    cfg = run_dir / "config.json"
    if cfg.exists():
        data = json.loads(cfg.read_text(encoding="utf-8"))
        if "seed" in data:
            return int(data["seed"])
    stem = run_dir.name
    if stem.startswith("seed_"):
        return int(stem.split("_", 1)[1])
    return 0


def load_tables(run_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    result_parts: List[pd.DataFrame] = []
    market_parts: List[pd.DataFrame] = []
    for run_dir in discover_run_dirs(run_root):
        run_seed = seed_from_run_dir(run_dir)
        meta_path = run_dir / "experiment_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        res = pd.read_csv(run_dir / "results.csv")
        res["run_seed"] = int(run_seed)
        res["run_dir"] = str(run_dir)
        for key, value in meta.items():
            if key not in res.columns:
                res[key] = value
        result_parts.append(res)
        market_path = run_dir / "market_summary.csv"
        if market_path.exists():
            market = pd.read_csv(market_path)
            market["run_seed"] = int(run_seed)
            market["run_dir"] = str(run_dir)
            for key, value in meta.items():
                if key not in market.columns:
                    market[key] = value
            market_parts.append(market)
    results = pd.concat(result_parts, ignore_index=True)
    markets = pd.concat(market_parts, ignore_index=True) if market_parts else pd.DataFrame()
    return results, markets


def fmt_mean_std(mean: float, std: float | None, digits: int = 1) -> str:
    if mean is None or (isinstance(mean, float) and not math.isfinite(mean)):
        return ""
    if std is None or not math.isfinite(float(std)) or float(std) == 0.0:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def summarize_pivot(results: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    per_seed = results.pivot_table(
        index=["run_seed", "dataset", "profile"],
        columns="method",
        values="good_count",
        aggfunc="first",
    ).reset_index()
    keep = ["run_seed", "dataset", "profile"] + [m for m in PIVOT_METHODS if m in per_seed.columns]
    per_seed = per_seed[keep]
    per_seed.to_csv(out_dir / "good_count_pivot_per_seed.csv", index=False)

    value_cols = [c for c in per_seed.columns if c not in {"run_seed", "dataset", "profile"}]
    mean = per_seed.groupby(["dataset", "profile"], dropna=False)[value_cols].mean().reset_index()
    std = per_seed.groupby(["dataset", "profile"], dropna=False)[value_cols].std().reset_index()
    mean.to_csv(out_dir / "good_count_pivot_mean.csv", index=False)
    std.to_csv(out_dir / "good_count_pivot_std.csv", index=False)
    return mean


def targeted_rows(results: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (run_seed, dataset, profile), group in results.groupby(["run_seed", "dataset", "profile"], dropna=False):
        vals = {str(r.method): int(r.good_count) for r in group.itertuples()}
        target = TARGET_METHOD.get(str(profile), "")
        if target == "min_targeted_baselines":
            present = [vals[m] for m in BASELINE_METHODS if m in vals]
            target_good = min(present) if present else np.nan
        else:
            target_good = vals.get(target, np.nan)
        rows.append(
            {
                "run_seed": int(run_seed),
                "dataset": dataset,
                "profile": profile,
                "target": target,
                "target_label": METHOD_LABELS.get(target, target),
                "target_good_count": target_good,
                "coreset_good": vals.get("market_coreset_select", np.nan),
                "cosine_good": vals.get("market_cosine_select", np.nan),
                "badge_good": vals.get("market_badge_select", np.nan),
                "kmeans_good": vals.get("market_kmeans_center_select", np.nan),
                "typiclust_good": vals.get("market_typiclust_select", np.nan),
                "uncertainty_good": vals.get("market_uncertainty_select", np.nan),
                "ours_direct_good": vals.get("ours_downstream_direct", np.nan),
                "ours_task_good": vals.get("ours_task_operator", np.nan),
                "ours_krr_good": vals.get("ours_kernel_ridge_student", np.nan),
                "ours_pkg_direct_good": vals.get("ours_sample_package_direct", np.nan),
                "ours_pkg_task_good": vals.get("ours_sample_package_task_operator", np.nan),
                "ours_pkg_krr_good": vals.get("ours_sample_package_krr", np.nan),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "profile", "run_seed"]).reset_index(drop=True)


def summarize_targeted(results: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    long = targeted_rows(results)
    long.to_csv(out_dir / "targeted_summary_per_seed.csv", index=False)
    value_cols = [
        "target_good_count",
        "coreset_good",
        "cosine_good",
        "badge_good",
        "kmeans_good",
        "typiclust_good",
        "uncertainty_good",
        "ours_direct_good",
        "ours_task_good",
        "ours_krr_good",
        "ours_pkg_direct_good",
        "ours_pkg_task_good",
        "ours_pkg_krr_good",
    ]
    mean = long.groupby(["dataset", "profile", "target", "target_label"], dropna=False)[value_cols].mean().reset_index()
    std = long.groupby(["dataset", "profile", "target", "target_label"], dropna=False)[value_cols].std().reset_index()
    mean.to_csv(out_dir / "targeted_summary_mean.csv", index=False)
    std.to_csv(out_dir / "targeted_summary_std.csv", index=False)
    return mean


def summarize_methods(results: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    agg = (
        results.groupby("method", dropna=False)["good_count"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    agg["method_label"] = agg["method"].map(METHOD_LABELS).fillna(agg["method"])
    if "scoring_semantics" in results.columns:
        semantics = results.groupby("method", dropna=False)["scoring_semantics"].agg(lambda x: str(x.dropna().iloc[0]) if len(x.dropna()) else "").to_dict()
        agg["scoring_semantics"] = agg["method"].map(semantics).fillna("")
    if "online_scoring_uses_downstream_model" in results.columns:
        uses = results.groupby("method", dropna=False)["online_scoring_uses_downstream_model"].max().to_dict()
        agg["online_scoring_uses_downstream_model"] = agg["method"].map(uses).fillna("")
    if "student_supervision" in results.columns:
        supervision = results.groupby("method", dropna=False)["student_supervision"].agg(lambda x: str(x.dropna().iloc[0]) if len(x.dropna()) else "").to_dict()
        agg["student_supervision"] = agg["method"].map(supervision).fillna("")
    agg.to_csv(out_dir / "method_good_count_aggregate.csv", index=False)
    return agg


def summarize_ablation(results: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    ablation_methods = [
        "ours_downstream_direct",
        "ours_influence_only",
        "ours_loss_reduction_only",
        "ours_task_operator",
        "ours_krr_influence_only",
        "ours_krr_loss_reduction_only",
        "ours_kernel_ridge_student",
        "ours_sample_package_direct",
        "ours_sample_package_influence_only",
        "ours_sample_package_loss_reduction_only",
        "ours_sample_package_task_operator",
        "ours_sample_package_krr_influence_only",
        "ours_sample_package_krr_loss_reduction_only",
        "ours_sample_package_krr",
    ]
    subset = results[results["method"].isin(ablation_methods)].copy()
    if len(subset) == 0:
        return pd.DataFrame()
    subset["method_label"] = subset["method"].map(METHOD_LABELS).fillna(subset["method"])
    per_profile = (
        subset.groupby(["dataset", "profile", "method", "method_label"], dropna=False)["good_count"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    per_profile.to_csv(out_dir / "ablation_summary_by_market.csv", index=False)
    overall = (
        subset.groupby(["method", "method_label"], dropna=False)["good_count"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    if "scoring_semantics" in subset.columns:
        semantics = subset.groupby("method", dropna=False)["scoring_semantics"].agg(lambda x: str(x.dropna().iloc[0]) if len(x.dropna()) else "").to_dict()
        overall["scoring_semantics"] = overall["method"].map(semantics).fillna("")
    overall.to_csv(out_dir / "ablation_summary_overall.csv", index=False)
    return overall


def summarize_markets(markets: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    if len(markets) == 0:
        return pd.DataFrame()
    cols = [
        "run_seed",
        "dataset",
        "profile",
        "market_size",
        "good_count",
        "good_ratio",
        "decoy_type_good_ratio",
        "static_target_top_good_count",
        "static_target_top_good_ratio",
        "static_target_top_downstream_gain_mean",
        "downstream_gain_mean_good",
        "downstream_gain_mean_bad",
        "anchor_test_auroc",
        "anchor_test_macro_f1",
        "anchor_test_acc",
    ]
    cols = [c for c in cols if c in markets.columns]
    markets[cols].to_csv(out_dir / "market_diagnostic_per_seed.csv", index=False)
    value_cols = [c for c in cols if c not in {"run_seed", "dataset", "profile"}]
    mean = markets.groupby(["dataset", "profile"], dropna=False)[value_cols].mean().reset_index()
    mean.to_csv(out_dir / "market_diagnostic_mean.csv", index=False)
    return mean


def summarize_ckks(results: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    ckks_cols = [
        "ckks_simd_enabled",
        "ckks_scheme",
        "sample_packaging_enabled",
        "package_size",
        "scored_objects",
        "ckks_est_input_prepare_encrypt_ms",
        "ckks_est_encrypted_compute_ms",
        "ckks_est_decrypt_decode_ms",
        "ckks_est_total_full_flow_ms",
        "ckks_scaled_from_reference_ms",
        "ckks_rows_per_second_mean",
        "ckks_ref_input_prepare_encrypt_ms_mean",
        "ckks_ref_encrypted_compute_ms_mean",
        "ckks_ref_decrypt_decode_ms_mean",
        "ckks_ref_total_full_flow_ms_mean",
        "ckks_ref_output_ciphertexts_mean",
        "ckks_ref_decoded_values_mean",
        "ckks_ref_input_ciphertexts_mean",
        "ckks_ref_input_ciphertext_bytes_mean",
        "ckks_ref_output_ciphertext_bytes_mean",
        "ckks_ref_total_communication_bytes_mean",
        "ckks_ref_ct_ct_mults_mean",
        "ckks_ref_ct_pt_mults_mean",
        "ckks_ref_rotations_mean",
        "ckks_ref_additions_mean",
        "ckks_ref_relinearizations_mean",
        "ckks_ref_rescales_mean",
        "ckks_ref_reference_count_mean",
        "ckks_ref_poly_degree_mean",
        "ckks_ref_ctct_nonlinear_depth_mean",
    ]
    present = [c for c in ckks_cols if c in results.columns]
    if not present:
        return pd.DataFrame()
    meta_cols = ["method", "dataset", "profile", "run_seed"]
    for col in ["scoring_semantics", "online_scoring_uses_downstream_model", "student_calibration_size", "student_landmark_count"]:
        if col in results.columns:
            meta_cols.append(col)
    subset = results[meta_cols + present].copy()
    subset = subset[subset.get("ckks_simd_enabled", 0).fillna(0).astype(float) > 0]
    if len(subset) == 0:
        return pd.DataFrame()
    subset.to_csv(out_dir / "ckks_summary_per_seed.csv", index=False)
    num_cols = [c for c in present if c != "ckks_scheme" and pd.api.types.is_numeric_dtype(subset[c])]
    grouped = subset.groupby(["method", "ckks_scheme"], dropna=False)[num_cols].mean().reset_index()
    grouped["method_label"] = grouped["method"].map(METHOD_LABELS).fillna(grouped["method"])
    if "scoring_semantics" in subset.columns:
        semantics = subset.groupby(["method", "ckks_scheme"], dropna=False)["scoring_semantics"].agg(lambda x: str(x.dropna().iloc[0]) if len(x.dropna()) else "").reset_index()
        grouped = grouped.merge(semantics, on=["method", "ckks_scheme"], how="left")
    grouped.to_csv(out_dir / "ckks_summary_mean.csv", index=False)
    return grouped


def summarize_downstream_metrics(results: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    metric_cols = [c for c in ["test_auroc", "test_macro_f1", "test_acc", "eval_time"] if c in results.columns]
    if not metric_cols:
        return pd.DataFrame()
    subset = results.copy()
    valid = subset[metric_cols].notna().any(axis=1)
    subset = subset[valid]
    if len(subset) == 0:
        return pd.DataFrame()
    subset.to_csv(out_dir / "downstream_metrics_per_seed.csv", index=False)
    group_cols = ["dataset", "profile", "method"]
    for col in ["purchase_ratio", "purchase_total"]:
        if col in subset.columns:
            group_cols.append(col)
    grouped = (
        subset.groupby(group_cols, dropna=False)[metric_cols + ["good_count"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        c if isinstance(c, str) else (c[0] if c[1] == "" else f"{c[0]}_{c[1]}")
        for c in grouped.columns.to_flat_index()
    ]
    grouped["method_label"] = grouped["method"].map(METHOD_LABELS).fillna(grouped["method"])
    grouped.to_csv(out_dir / "downstream_metrics_mean_std.csv", index=False)
    return grouped


def write_latex_tables(
    targeted_mean: pd.DataFrame,
    method_agg: pd.DataFrame,
    ckks_mean: pd.DataFrame,
    ablation_mean: pd.DataFrame,
    out_dir: Path,
) -> None:
    base_cols = [
        ("dataset", "Dataset"),
        ("profile", "Market"),
        ("target_label", "Target"),
        ("target_good_count", "Target good"),
    ]
    optional_cols = [
        ("ours_task_good", "Online task"),
        ("ours_pkg_task_good", "Online pkg task"),
        ("ours_krr_good", "Online KRR"),
        ("ours_pkg_krr_good", "Online pkg KRR"),
        ("ours_direct_good", "Teacher direct"),
        ("ours_pkg_direct_good", "Teacher pkg direct"),
    ]
    chosen_cols = base_cols + [
        (col, label)
        for col, label in optional_cols
        if col in targeted_mean.columns and targeted_mean[col].notna().any()
    ]
    table = targeted_mean[[col for col, _ in chosen_cols]].copy()
    table.columns = [label for _, label in chosen_cols]
    (out_dir / "table_targeted_main.tex").write_text(
        table.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.1f}"),
        encoding="utf-8",
    )

    agg_cols = ["method_label", "mean", "std", "min", "max"]
    if "scoring_semantics" in method_agg.columns:
        agg_cols.insert(1, "scoring_semantics")
    agg = method_agg[agg_cols].copy()
    agg.columns = ["Method", "Scoring semantics", "Mean good", "Std", "Min", "Max"] if "scoring_semantics" in agg_cols else ["Method", "Mean good", "Std", "Min", "Max"]
    (out_dir / "table_method_aggregate.tex").write_text(
        agg.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.1f}"),
        encoding="utf-8",
    )
    if len(ckks_mean):
        cols = [
            "method_label",
            "ckks_scheme",
            "package_size",
            "scored_objects",
            "ckks_est_input_prepare_encrypt_ms",
            "ckks_est_encrypted_compute_ms",
            "ckks_est_decrypt_decode_ms",
            "ckks_est_total_full_flow_ms",
            "ckks_rows_per_second_mean",
            "ckks_ref_input_ciphertexts_mean",
            "ckks_ref_input_ciphertext_bytes_mean",
            "ckks_ref_output_ciphertext_bytes_mean",
            "ckks_ref_total_communication_bytes_mean",
            "ckks_ref_ct_ct_mults_mean",
            "ckks_ref_ct_pt_mults_mean",
            "ckks_ref_rotations_mean",
            "ckks_ref_additions_mean",
            "ckks_ref_relinearizations_mean",
            "ckks_ref_rescales_mean",
            "ckks_ref_reference_count_mean",
            "ckks_ref_poly_degree_mean",
            "ckks_ref_ctct_nonlinear_depth_mean",
        ]
        cols = [c for c in cols if c in ckks_mean.columns]
        ckks_table = ckks_mean[cols].copy()
        ckks_table.columns = [
            "Method",
            "CKKS scheme",
            "Pkg size",
            "Scored objects",
            "Input+encrypt ms",
            "Enc. compute ms",
            "Decrypt ms",
            "Total flow ms",
            "Rows/s",
            "Input cts",
            "Input bytes",
            "Output bytes",
            "Total comm bytes",
            "Ct-ct mults",
            "Ct-pt mults",
            "Rotations",
            "Additions",
            "Relinearizations",
            "Rescales",
            "References",
            "Poly degree",
            "CT-CT depth",
        ][: len(ckks_table.columns)]
        (out_dir / "table_ckks.tex").write_text(
            ckks_table.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.2f}"),
            encoding="utf-8",
        )
    if len(ablation_mean):
        abl_cols = ["method_label", "mean", "std", "min", "max"]
        if "scoring_semantics" in ablation_mean.columns:
            abl_cols.insert(1, "scoring_semantics")
        abl = ablation_mean[abl_cols].copy()
        abl.columns = ["Variant", "Scoring semantics", "Mean good", "Std", "Min", "Max"] if "scoring_semantics" in abl_cols else ["Variant", "Mean good", "Std", "Min", "Max"]
        (out_dir / "table_ablation.tex").write_text(
            abl.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.1f}"),
            encoding="utf-8",
        )


def write_markdown_summary(targeted_mean: pd.DataFrame, method_agg: pd.DataFrame, out_dir: Path) -> None:
    direct = method_agg[method_agg["method"] == "ours_downstream_direct"]
    task = method_agg[method_agg["method"] == "ours_task_operator"]
    krr = method_agg[method_agg["method"] == "ours_kernel_ridge_student"]
    lines = [
        "# Main Text Score-Only Summary",
        "",
        "All values are selected good rows out of the purchase budget of 50.",
        "Teacher/reference variants use downstream-model signals and are not online scoring methods; online variants score candidates with the fitted student/KRR only.",
        "",
    ]
    if len(direct):
        r = direct.iloc[0]
        lines.append(f"- Teacher direct reference: mean {r['mean']:.2f}, min {int(r['min'])}, max {int(r['max'])}.")
    if len(task):
        r = task.iloc[0]
        lines.append(f"- Ours online task operator: mean {r['mean']:.2f}, min {int(r['min'])}, max {int(r['max'])}.")
    if len(krr):
        r = krr.iloc[0]
        lines.append(f"- Ours online KRR: mean {r['mean']:.2f}, min {int(r['min'])}, max {int(r['max'])}.")
    target_vals = targeted_mean["target_good_count"].dropna() if len(targeted_mean) else pd.Series(dtype=float)
    if len(target_vals):
        target_mean = float(target_vals.mean())
        target_max = float(target_vals.max())
        lines.append(f"- Target baselines: mean {target_mean:.2f}, max {target_max:.1f}.")
    lines.append("")
    lines.append("## Files")
    for name in [
        "targeted_summary_mean.csv",
        "good_count_pivot_mean.csv",
        "method_good_count_aggregate.csv",
        "market_diagnostic_mean.csv",
        "ckks_summary_mean.csv",
        "ablation_summary_overall.csv",
        "downstream_metrics_mean_std.csv",
        "table_targeted_main.tex",
        "table_method_aggregate.tex",
        "table_ckks.tex",
        "table_ablation.tex",
    ]:
        lines.append(f"- `{name}`")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--summary-dir", type=Path, default=None)
    return ap.parse_args()


def main() -> None:
    cli = build_cli()
    run_root = Path(cli.run_root)
    out_dir = Path(cli.summary_dir) if cli.summary_dir else run_root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    results, markets = load_tables(run_root)
    results.to_csv(out_dir / "combined_results.csv", index=False)
    if len(markets):
        markets.to_csv(out_dir / "combined_market_summary.csv", index=False)
    summarize_pivot(results, out_dir)
    targeted_mean = summarize_targeted(results, out_dir)
    method_agg = summarize_methods(results, out_dir)
    summarize_markets(markets, out_dir)
    ckks_mean = summarize_ckks(results, out_dir)
    ablation_mean = summarize_ablation(results, out_dir)
    summarize_downstream_metrics(results, out_dir)
    write_latex_tables(targeted_mean, method_agg, ckks_mean, ablation_mean, out_dir)
    write_markdown_summary(targeted_mean, method_agg, out_dir)
    print(f"[summary] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
