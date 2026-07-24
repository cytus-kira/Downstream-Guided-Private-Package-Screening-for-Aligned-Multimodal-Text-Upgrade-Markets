#!/usr/bin/env python
"""Build paper-ready approximation and CKKS comparison tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial import Chebyshev, Polynomial


SCHEME_LABELS = {
    "baseline_random_noop": "Random",
    "baseline_cosine_ctpt": "Cosine",
    "baseline_uncertainty_poly4_ctpt": "Uncertainty",
    "baseline_coreset_all_distances_ctpt": "CoreSet (10 rounds)",
    "baseline_badge_components_poly4_ctpt": "BADGE",
    "baseline_kmeans_all_distances_ctpt": "KMeans-center",
    "baseline_typiclust_sqrt_poly4_ctpt": "TypiClust",
    "ours_krr_row_exp_poly4_ctpt": "Ours-KRR-raw",
    "ours_krr_pkg_exp_poly4_ctpt": "Ours-KRR-package",
}


def approximation_row(name: str, fn, lo: float, hi: float, degree: int = 4) -> dict[str, object]:
    cheb = Chebyshev.interpolate(fn, deg=degree, domain=[lo, hi])
    poly = cheb.convert(kind=Polynomial)
    x = np.linspace(lo, hi, 200_001, dtype=np.float64)
    truth = fn(x)
    approx = poly(x)
    error = np.abs(approx - truth)
    return {
        "function": name,
        "interval_min": lo,
        "interval_max": hi,
        "degree": degree,
        "coefficients_ascending": ";".join(f"{v:.12g}" for v in poly.coef),
        "max_abs_error": float(np.max(error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "max_relative_error": float(np.max(error / np.maximum(np.abs(truth), 1e-12))),
    }


def build_approximation_table(out_dir: Path) -> pd.DataFrame:
    sigmoid_uncertainty = lambda z: (1.0 / (1.0 + np.exp(-z))) * (
        1.0 - 1.0 / (1.0 + np.exp(-z))
    )
    rows = [
        approximation_row("exp", np.exp, -4.0, 0.0),
        approximation_row("sigmoid_uncertainty", sigmoid_uncertainty, -3.0, 3.0),
        approximation_row("sqrt", np.sqrt, 0.05, 4.0),
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "poly4_approximation_summary.csv", index=False)

    latex = frame[
        ["function", "interval_min", "interval_max", "degree", "max_abs_error", "rmse"]
    ].copy()
    latex.columns = ["Function", "Interval min", "Interval max", "Degree", "Max abs. error", "RMSE"]
    (out_dir / "table_poly4_approximation.tex").write_text(
        latex.to_latex(index=False, escape=True, float_format=lambda value: f"{value:.3e}"),
        encoding="utf-8",
    )
    return frame


def build_ckks_table(
    summary_path: Path,
    out_dir: Path,
    krr_summary_path: Path | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(summary_path)
    if krr_summary_path is not None and krr_summary_path.exists():
        krr = pd.read_csv(krr_summary_path)
        krr_schemes = {
            "ours_krr_row_exp_poly4_ctpt",
            "ours_krr_pkg_exp_poly4_ctpt",
        }
        replacement = krr[krr["scheme"].isin(krr_schemes)].copy()
        replacement_schemes = set(replacement["scheme"].tolist())
        frame = frame[~frame["scheme"].isin(replacement_schemes)]
        frame = pd.concat([frame, replacement], ignore_index=True)
    frame = frame[frame["logical_rows"].astype(int) == 5000].copy()
    frame["method"] = frame["scheme"].map(SCHEME_LABELS).fillna(frame["scheme"])
    frame["full_flow_s"] = frame["total_full_flow_ms_mean"] / 1000.0
    frame["communication_mb"] = frame["total_communication_bytes_mean"] / (1024.0 * 1024.0)
    frame["complete_protocol_input_encrypt_ms"] = frame["input_prepare_encrypt_ms_mean"]
    frame["complete_protocol_encrypted_compute_ms"] = frame["encrypted_compute_ms_mean"]
    frame["complete_protocol_decrypt_decode_ms"] = frame["decrypt_decode_ms_mean"]
    frame["complete_protocol_full_flow_s"] = (
        frame["complete_protocol_input_encrypt_ms"]
        + frame["complete_protocol_encrypted_compute_ms"]
        + frame["complete_protocol_decrypt_decode_ms"]
    ) / 1000.0
    frame["complete_protocol_communication_mb"] = frame["communication_mb"]

    # The implemented CoreSet selector buys 5 rows per round until 50 rows.
    # Each round refreshes nearest-reference distances, so the complete online
    # cost is ten measured scans. Candidate ciphertexts are encrypted once and
    # reused; encrypted evaluation, output release, and output communication
    # repeat each round. Later rounds add only 0--45 references to the
    # registered 800, so the fixed-reference scan is slightly conservative.
    core_mask = frame["scheme"] == "baseline_coreset_all_distances_ctpt"
    frame.loc[core_mask, "complete_protocol_encrypted_compute_ms"] *= 10.0
    frame.loc[core_mask, "complete_protocol_decrypt_decode_ms"] *= 10.0
    frame.loc[core_mask, "complete_protocol_full_flow_s"] = (
        frame.loc[core_mask, "complete_protocol_input_encrypt_ms"]
        + frame.loc[core_mask, "complete_protocol_encrypted_compute_ms"]
        + frame.loc[core_mask, "complete_protocol_decrypt_decode_ms"]
    ) / 1000.0
    frame.loc[core_mask, "complete_protocol_communication_mb"] = (
        frame.loc[core_mask, "input_ciphertext_bytes_mean"]
        + 10.0 * frame.loc[core_mask, "output_ciphertext_bytes_mean"]
    ) / (1024.0 * 1024.0)
    frame["complete_protocol_input_encrypt_s"] = (
        frame["complete_protocol_input_encrypt_ms"] / 1000.0
    )
    frame["complete_protocol_encrypted_compute_s"] = (
        frame["complete_protocol_encrypted_compute_ms"] / 1000.0
    )
    frame["complete_protocol_decrypt_decode_s"] = (
        frame["complete_protocol_decrypt_decode_ms"] / 1000.0
    )

    cols = [
        "method",
        "scheme",
        "logical_rows",
        "scored_objects",
        "reference_count_mean",
        "poly_degree_mean",
        "ctct_nonlinear_depth_mean",
        "input_prepare_encrypt_ms_mean",
        "encrypted_compute_ms_mean",
        "decrypt_decode_ms_mean",
        "complete_protocol_input_encrypt_ms",
        "complete_protocol_encrypted_compute_ms",
        "complete_protocol_decrypt_decode_ms",
        "complete_protocol_input_encrypt_s",
        "complete_protocol_encrypted_compute_s",
        "complete_protocol_decrypt_decode_s",
        "full_flow_s",
        "complete_protocol_full_flow_s",
        "output_ciphertexts_mean",
        "communication_mb",
        "complete_protocol_communication_mb",
    ]
    result = frame[cols].sort_values("complete_protocol_full_flow_s").reset_index(drop=True)
    result.to_csv(out_dir / "ckks_poly4_all_methods_summary.csv", index=False)

    latex = result[
        [
            "method",
            "reference_count_mean",
            "poly_degree_mean",
            "ctct_nonlinear_depth_mean",
            "complete_protocol_input_encrypt_s",
            "complete_protocol_encrypted_compute_s",
            "complete_protocol_decrypt_decode_s",
            "complete_protocol_full_flow_s",
            "complete_protocol_communication_mb",
        ]
    ].copy()
    latex.columns = [
        "Method",
        "References",
        "Poly. degree",
        "CT-CT depth",
        "Input and encrypt (s)",
        "Encrypted compute (s)",
        "Decrypt and decode (s)",
        "Complete flow (s)",
        "Communication (MB)",
    ]
    amortized = build_amortized_throughput_table(result, out_dir)
    amortized_latex = amortized[
        [
            "method",
            "covered_raw_rows",
            "encrypted_object_label",
            "full_flow_s",
            "ms_per_covered_raw_row",
            "covered_raw_rows_per_s",
            "ms_per_encrypted_scoring_object",
            "encrypted_scoring_objects_per_s",
            "communication_kb_per_covered_raw_row",
        ]
    ].copy()
    amortized_latex.columns = [
        "Method",
        "Covered raw rows",
        "Encrypted objects",
        "Full flow (s)",
        "ms/raw row",
        "Raw rows/s",
        "ms/object",
        "Objects/s",
        "KB/raw row",
    ]
    combined_latex = (
        "% Panel (a): complete full-flow benchmark\n"
        + latex.to_latex(index=False, escape=True, float_format=lambda value: f"{value:.2f}")
        + "\n% Panel (b): amortized batch-screening throughput\n"
        + amortized_latex.to_latex(
            index=False, escape=True, float_format=lambda value: f"{value:.2f}"
        )
    )
    (out_dir / "table_ckks_poly4_all_methods.tex").write_text(
        combined_latex,
        encoding="utf-8",
    )
    return result


def build_amortized_throughput_table(frame: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Derive batch-throughput context from the measured full-flow rows."""
    selected = frame[
        frame["scheme"].isin(
            {
                "ours_krr_pkg_exp_poly4_ctpt",
                "ours_krr_row_exp_poly4_ctpt",
                "baseline_coreset_all_distances_ctpt",
            }
        )
    ].copy()

    rows: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        covered_rows = int(row.logical_rows)
        rounds = 10 if row.scheme == "baseline_coreset_all_distances_ctpt" else 1
        encrypted_objects = int(row.scored_objects) * rounds
        if row.scheme == "ours_krr_pkg_exp_poly4_ctpt":
            object_label = f"{encrypted_objects:,} packages"
            object_unit = "package"
        elif row.scheme == "baseline_coreset_all_distances_ctpt":
            object_label = f"{encrypted_objects:,} row-rounds"
            object_unit = "row-round"
        else:
            object_label = f"{encrypted_objects:,} rows"
            object_unit = "row"

        full_flow_s = float(row.complete_protocol_full_flow_s)
        communication_mb = float(row.complete_protocol_communication_mb)
        rows.append(
            {
                "method": row.method,
                "covered_raw_rows": covered_rows,
                "encrypted_scoring_objects": encrypted_objects,
                "encrypted_object_label": object_label,
                "encrypted_object_unit": object_unit,
                "acquisition_rounds": rounds,
                "full_flow_s": full_flow_s,
                "ms_per_covered_raw_row": 1000.0 * full_flow_s / covered_rows,
                "covered_raw_rows_per_s": covered_rows / full_flow_s,
                "ms_per_encrypted_scoring_object": 1000.0
                * full_flow_s
                / encrypted_objects,
                "encrypted_scoring_objects_per_s": encrypted_objects / full_flow_s,
                # Table 7 reports communication in MB, so use decimal KB/MB
                # for this deterministic amortization of its displayed unit.
                "communication_kb_per_covered_raw_row": 1000.0
                * communication_mb
                / covered_rows,
            }
        )

    result = pd.DataFrame(rows)
    order = {
        "Ours-KRR-package": 0,
        "Ours-KRR-raw": 1,
        "CoreSet (10 rounds)": 2,
    }
    result["_order"] = result["method"].map(order)
    result = result.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    result.to_csv(out_dir / "ckks_amortized_batch_throughput.csv", index=False)

    return result


def build_fidelity_table(results_root: Path | None, out_dir: Path) -> pd.DataFrame:
    if results_root is None or not results_root.exists():
        return pd.DataFrame()
    files = sorted(results_root.glob("seed_*/results.csv"))
    if not files:
        return pd.DataFrame()
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    frame = frame[
        frame["method"].isin(["ours_kernel_ridge_student", "ours_sample_package_krr"])
    ].copy()
    grouped = (
        frame.groupby("method", as_index=False)
        .agg(
            max_score_error=("krr_poly4_score_max_abs_error", "max"),
            mean_score_error=("krr_poly4_score_max_abs_error", "mean"),
            mean_top50_overlap=("krr_poly4_top_budget_overlap", "mean"),
            minimum_top50_overlap=("krr_poly4_top_budget_overlap", "min"),
            cases=("krr_poly4_top_budget_overlap", "size"),
        )
    )
    grouped.to_csv(out_dir / "krr_poly4_selection_fidelity.csv", index=False)
    (out_dir / "table_krr_poly4_selection_fidelity.tex").write_text(
        grouped.to_latex(index=False, escape=True, float_format=lambda value: f"{value:.4f}"),
        encoding="utf-8",
    )
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckks-summary",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / "ckks_poly4_all_methods"
        / "ckks_seal_summary.csv",
    )
    parser.add_argument("--selection-results-root", type=Path, default=None)
    parser.add_argument("--krr-summary", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "ckks_poly4_all_methods",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_approximation_table(args.output_dir)
    build_ckks_table(args.ckks_summary, args.output_dir, args.krr_summary)
    build_fidelity_table(args.selection_results_root, args.output_dir)
    print(f"[done] {args.output_dir}")


if __name__ == "__main__":
    main()
