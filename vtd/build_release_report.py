#!/usr/bin/env python
"""Aggregate real 2-of-2 VTD runs and generate paper-ready artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STYLE_DIR = ROOT / "paper" / "main_text_experiments"
sys.path.insert(0, str(STYLE_DIR))
from unified_figure_style import USER_PALETTE, apply_paper_style  # noqa: E402


EXPECTED_ATTACKS = {
    "wrong-target",
    "wrong-key",
    "malformed-share",
    "large-contribution-tampering",
    "wrong-level",
    "proof-tampering",
    "replay",
    "unauthorized-output",
}


def load_run(path: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = path / "results"
    summary = json.loads((results / "vtd_2of2_summary.json").read_text(encoding="utf-8"))
    release = pd.read_csv(results / "vtd_2of2_release_runs.csv")
    attacks = pd.read_csv(results / "vtd_2of2_attacks.csv")
    attacks["attack_type"] = attacks["attack_type"].replace(
        {"excessive-noise": "large-contribution-tampering"}
    )
    attacks.insert(0, "run_id", path.name)
    attacks["protocol"] = "2-of-2"
    attacks["proof_backend"] = release["proof_backend"].iloc[0]
    attacks["ckks_parameter_digest"] = release["ckks_parameter_digest"].iloc[0]
    audit = pd.read_csv(results / "post_purchase_audit.csv")
    return summary, release, attacks, audit


def validate_attack_matrix(attacks: pd.DataFrame) -> None:
    required = {
        "run_id",
        "dataset",
        "seed",
        "batch_size",
        "attack_type",
        "attempted_cases",
        "accepted_cases",
        "rejected_cases",
        "expected",
        "observed",
        "protocol",
        "proof_backend",
        "ckks_parameter_digest",
    }
    missing = required.difference(attacks.columns)
    if missing:
        raise ValueError(f"Attack CSV is missing required columns: {sorted(missing)}")

    duplicate_key = ["run_id", "seed", "batch_size", "attack_type"]
    if attacks.duplicated(duplicate_key).any():
        duplicates = attacks.loc[attacks.duplicated(duplicate_key, keep=False), duplicate_key]
        raise ValueError(f"Duplicate attack records found:\n{duplicates.to_string(index=False)}")

    for (run_id, batch_size), group in attacks.groupby(["run_id", "batch_size"], sort=True):
        observed_attacks = set(group["attack_type"])
        if observed_attacks != EXPECTED_ATTACKS:
            missing_attacks = sorted(EXPECTED_ATTACKS.difference(observed_attacks))
            extra_attacks = sorted(observed_attacks.difference(EXPECTED_ATTACKS))
            raise ValueError(
                f"Incomplete attack matrix for {run_id} batch {batch_size}: "
                f"missing={missing_attacks}, extra={extra_attacks}"
            )

    if not (attacks["attempted_cases"] > 0).all():
        raise ValueError("Every attack record must contain at least one attempted case")
    if not (attacks["accepted_cases"] == 0).all():
        raise ValueError("At least one injected invalid case was accepted")
    if not (attacks["rejected_cases"] == attacks["attempted_cases"]).all():
        raise ValueError("Rejected-case counts do not match attempted-case counts")
    if not (attacks["observed"] == attacks["expected"]).all():
        raise ValueError("At least one observed attack outcome differs from expectation")


def tex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def build(args: argparse.Namespace) -> None:
    summaries: list[dict] = []
    release_frames: list[pd.DataFrame] = []
    attack_frames: list[pd.DataFrame] = []
    audit_frames: list[pd.DataFrame] = []
    for run in args.runs:
        summary, release, attacks, audit = load_run(run.resolve())
        summaries.append(summary)
        release_frames.append(release)
        attack_frames.append(attacks)
        audit_frames.append(audit)

    release = pd.concat(release_frames, ignore_index=True).sort_values("batch_size")
    attacks = pd.concat(attack_frames, ignore_index=True)
    validate_attack_matrix(attacks)
    audit = pd.concat(audit_frames, ignore_index=True)
    rows: list[dict] = []
    for summary, (_, run) in zip(summaries, release.groupby("batch_size", sort=True)):
        batch = int(run["batch_size"].iloc[0])
        rows.append(
            {
                "batch_size": batch,
                "seed": int(run["seed"].iloc[0]),
                "encrypted_score_ms": float(summary["scorer"]["encrypted_compute_ms"]),
                "partial_decryption_ms": float(run["partial_decryption_ms"].mean()),
                "proof_generation_ms": float(run["proof_generation_ms"].mean()),
                "verification_ms": float(run["verification_ms"].mean()),
                "reconstruction_ms": float(run["reconstruction_ms"].mean()),
                "decoding_ms": float(run["ckks_decoding_ms"].mean()),
                "end_to_end_process_ms": float(run["end_to_end_localhost_ms"].mean()),
                "proof_bytes": int(run["proof_bytes"].iloc[0]),
                "response_bytes": int(run["communication_bytes"].iloc[0]),
                "max_abs_error": float(run["decoded_max_abs_error"].max()),
                "mean_abs_error": float(run["decoded_mean_abs_error"].mean()),
                "constraints": int(run["constraints"].iloc[0]),
                "public_variables": int(run["public_variables"].iloc[0]),
                "honest_accepted": int(run["accepted"].sum()),
                "honest_attempted": int(len(run)),
                "proof_generation_ms_per_score": float(run["proof_generation_ms"].mean()) / batch,
                "verification_ms_per_score": float(run["verification_ms"].mean()) / batch,
                "end_to_end_process_ms_per_score": float(run["end_to_end_localhost_ms"].mean()) / batch,
                "proof_bytes_per_score": float(run["proof_bytes"].iloc[0]) / batch,
                "response_bytes_per_score": float(run["communication_bytes"].iloc[0]) / batch,
            }
        )
    scale = pd.DataFrame(rows).sort_values("batch_size")

    attack_summary = (
        attacks.groupby("attack_type", as_index=False)
        .agg(attempted_cases=("attempted_cases", "sum"), accepted_cases=("accepted_cases", "sum"), rejected_cases=("rejected_cases", "sum"))
        .sort_values("attack_type")
    )
    attack_summary["rejection_rate"] = attack_summary["rejected_cases"] / attack_summary["attempted_cases"]
    audit_summary = (
        audit.groupby("case", as_index=False)
        .agg(attempted_cases=("case", "size"), accepted_cases=("accepted", "sum"), detected_cases=("detected", "sum"))
        .sort_values("case")
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scale.to_csv(args.output_dir / "vtd_2of2_scaling_summary.csv", index=False)
    attack_summary.to_csv(args.output_dir / "vtd_2of2_attack_summary.csv", index=False)
    audit_summary.to_csv(args.output_dir / "vtd_post_purchase_audit_summary.csv", index=False)
    release.to_csv(args.output_dir / "vtd_2of2_release_runs_combined.csv", index=False)
    attacks.to_csv(args.output_dir / "vtd_2of2_attacks.csv", index=False)
    attacks.to_csv(args.output_dir / "vtd_2of2_attacks_combined.csv", index=False)

    table_lines = [
        r"\begin{tabular}{r r r r r r}",
        r"\toprule",
        r"Scores & Prove (s) & Verify (ms) & Proof (B) & Response (KiB) & Max error \\",
        r"\midrule",
    ]
    for row in scale.itertuples(index=False):
        table_lines.append(
            f"{row.batch_size} & {row.proof_generation_ms / 1000:.2f} & {row.verification_ms:.1f} & "
            f"{row.proof_bytes} & {row.response_bytes / 1024:.1f} & ${row.max_abs_error:.2e}$ \\\\"
        )
    table_lines += [r"\bottomrule", r"\end{tabular}"]
    (args.output_dir / "table_vtd_2of2_overhead_rows.tex").write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    attack_lines = [r"\begin{tabular}{lrr}", r"\toprule", r"Case & Rejected & Attempted \\", r"\midrule"]
    for row in attack_summary.itertuples(index=False):
        attack_lines.append(
            f"{tex_escape(row.attack_type)} & {int(row.rejected_cases)} & "
            f"{int(row.attempted_cases)} " + r"\\"
        )
    attack_lines += [r"\bottomrule", r"\end{tabular}"]
    (args.output_dir / "table_vtd_2of2_attacks_rows.tex").write_text("\n".join(attack_lines) + "\n", encoding="utf-8")

    apply_paper_style(9.2)
    batches = scale["batch_size"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.58))
    ax = axes[0]
    ax.plot(
        batches,
        scale["proof_generation_ms"] / 1000,
        marker="o",
        lw=1.8,
        color=USER_PALETTE["navy"],
        label="Proof generation",
    )
    ax.plot(
        batches,
        scale["verification_ms"] / 1000,
        marker="s",
        lw=1.6,
        color=USER_PALETTE["magenta"],
        label="Proof verification",
    )
    ax.plot(
        batches,
        scale["end_to_end_process_ms"] / 1000,
        marker="^",
        lw=1.6,
        color=USER_PALETTE["salmon"],
        label="Local two-process path",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(batches, [str(x) for x in batches])
    ax.set_xlabel("Scores / ciphertext")
    ax.set_ylabel("Measured time (s)")
    ax.set_title("(a) Stage and local-path latency")
    ax.grid(True, which="both", axis="y", color="#D4D4D4", linewidth=0.55)
    ax.legend(frameon=False, loc="center left", fontsize=7.0, handlelength=1.5)

    ax = axes[1]
    ax.plot(
        batches,
        scale["proof_generation_ms_per_score"],
        marker="o",
        lw=1.8,
        color=USER_PALETTE["navy"],
        label="Proof generation",
    )
    ax.plot(
        batches,
        scale["verification_ms_per_score"],
        marker="s",
        lw=1.6,
        color=USER_PALETTE["magenta"],
        label="Proof verification",
    )
    ax.plot(
        batches,
        scale["end_to_end_process_ms_per_score"],
        marker="^",
        lw=1.6,
        color=USER_PALETTE["salmon"],
        label="Local two-process path",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(batches, [str(x) for x in batches])
    ax.set_xlabel("Scores / ciphertext")
    ax.set_ylabel("Amortized time (ms / score)")
    ax.set_title("(b) Amortized computation")
    ax.grid(True, which="both", axis="y", color="#D4D4D4", linewidth=0.55)
    ax.legend(frameon=False, loc="lower left", fontsize=7.0, handlelength=1.5)

    ax = axes[2]
    ax.plot(
        batches,
        scale["proof_bytes_per_score"] / 1024,
        marker="o",
        lw=1.8,
        color=USER_PALETTE["navy"],
        label="Proof",
    )
    ax.plot(
        batches,
        scale["response_bytes_per_score"] / 1024,
        marker="s",
        lw=1.6,
        color=USER_PALETTE["sky"],
        label="Full market response",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(batches, [str(x) for x in batches])
    ax.set_xlabel("Scores / ciphertext")
    ax.set_ylabel("Communication (KiB / score)")
    ax.set_title("(c) Amortized communication")
    ax.grid(True, which="both", axis="y", color="#D4D4D4", linewidth=0.55)
    ax.legend(frameon=False, loc="lower left", fontsize=7.0, handlelength=1.5)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(w_pad=0.75)
    for suffix in ("pdf", "png"):
        fig.savefig(args.figure_dir / f"fig_vtd_2of2_release_scaling.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "runs": [str(p.resolve()) for p in args.runs],
        "proof_backend": "gnark-v0.15.0 Groth16 over BN254",
        "he_backend": "Lattigo-v6.2.0 multiparty CKKS",
        "proof_relation": "exact coefficient-wise NTT-domain modular release and registered CKG-share relations",
        "production_parameters": {"logN": 13, "logQ": [45, 32, 32, 32, 32], "scale_bits": 32, "output_level": 0, "landmarks": 1000},
        "attack_artifact": "vtd_2of2_attacks.csv",
        "attack_batches": sorted(int(value) for value in attacks["batch_size"].unique()),
        "attack_types": sorted(attacks["attack_type"].unique()),
        "attack_attempted_cases": int(attacks["attempted_cases"].sum()),
        "attack_rejected_cases": int(attacks["rejected_cases"].sum()),
        "attack_matrix_complete": True,
        "timing_repetitions_per_point": 1,
        "scope_note": "Production-parameter application-path measurements on three cached seeds; single timing observation per batch point because isolated proof-artifact loading dominates the prototype E2E path.",
    }
    (args.output_dir / "vtd_2of2_report_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper" / "derived_experiment_tables" / "vtd_2of2")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "paper" / "latex" / "fig")
    args = parser.parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    build(args)


if __name__ == "__main__":
    main()
