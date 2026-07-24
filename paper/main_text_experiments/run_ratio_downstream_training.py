#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run ratio-purchase downstream-training experiments.

Unlike the main score-only runner, this script intentionally does not pass
``--score-only`` to the base experiment.  Each selected purchase set is added
to the buyer's local training data, then the downstream pair model is trained
and evaluated.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "run_quick_downstream_collapse_experiment.py"
SUMMARY_SCRIPT = Path(__file__).resolve().parent / "summarize_main_text_results.py"

DATASETS = "hateful_memes,hatespeech,mscoco"
PROFILES = "noise,coreset_far_wrong,typiclust_dense,kmeans_center,uncertainty_badge,cosine,all_average"
METHODS = (
    "market_random_select,"
    "market_cosine_select,"
    "market_coreset_select,"
    "market_badge_select,"
    "market_kmeans_center_select,"
    "market_typiclust_select,"
    "ours_kernel_ridge_student,"
    "ours_sample_package_krr"
)

PRESETS: Dict[str, Dict[str, object]] = {
    "smoke": {
        "candidate_pool": 500,
        "search_pool": 1000,
        "student_calibration_pool": 500,
        "market_size": 1000,
        "initial_noisy_size": 300,
        "validation_size": 300,
        "operator_feature_dim": 64,
        "krr_train_size": 200,
        "anchor_epochs": 1,
        "anchor_patience": 1,
        "batch_size": 512,
    },
    "ratio_5k": {
        "candidate_pool": 4000,
        "search_pool": 6000,
        "student_calibration_pool": 4000,
        "market_size": 5000,
        "initial_noisy_size": 800,
        "validation_size": 800,
        "operator_feature_dim": 64,
        "krr_train_size": 1000,
        "anchor_epochs": 5,
        "anchor_patience": 2,
        "batch_size": 512,
    },
}


def parse_csv(text: str | Iterable[object]) -> List[str]:
    if isinstance(text, str):
        return [x.strip() for x in text.split(",") if x.strip()]
    return [str(x).strip() for x in text if str(x).strip()]


def ratio_slug(x: float) -> str:
    return f"{float(x):.5g}".replace("-", "m").replace(".", "p")


def build_command(cli: argparse.Namespace, seed: int, ratio: float, purchase_total: int, out_dir: Path, preset: Dict[str, object]) -> List[object]:
    cmd: List[object] = [
        cli.python,
        BASE_SCRIPT,
        "--datasets",
        cli.datasets,
        "--profiles",
        cli.profiles,
        "--methods",
        cli.methods,
        "--output-dir",
        out_dir,
        "--seed",
        int(seed),
        "--device",
        cli.device,
        "--good-count",
        0,
        "--good-ratio",
        cli.good_ratio,
        "--good-source",
        cli.good_source,
        "--good-target-avoid-weight",
        cli.good_target_avoid_weight,
        "--purchase-total",
        int(purchase_total),
        "--round-budget",
        min(int(cli.round_budget), int(purchase_total)),
        "--package-size",
        cli.package_size,
        "--downstream-epochs",
        cli.downstream_epochs,
        "--downstream-patience",
        cli.downstream_patience,
    ]
    for key, value in preset.items():
        cmd.extend(["--" + key.replace("_", "-"), value])
    if cli.class_balanced:
        cmd.append("--class-balanced")
    return cmd


def run_cmd(cmd: List[object], dry_run: bool) -> None:
    printable = " ".join(str(x) for x in cmd)
    print("\n[cmd]\n" + printable + "\n", flush=True)
    if dry_run:
        return
    subprocess.run([str(x) for x in cmd], cwd=str(ROOT), check=True)


def build_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), default="ratio_5k")
    ap.add_argument("--ratios", default="0.0025,0.005,0.01,0.02")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "runs" / "ratio_downstream_5k")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--datasets", default=DATASETS)
    ap.add_argument("--profiles", default=PROFILES)
    ap.add_argument("--methods", default=METHODS)
    ap.add_argument("--good-ratio", type=float, default=0.10)
    ap.add_argument("--good-source", choices=["downstream_any", "useful_clean"], default="downstream_any")
    ap.add_argument("--good-target-avoid-weight", type=float, default=0.05)
    ap.add_argument("--round-budget", type=int, default=5)
    ap.add_argument("--package-size", type=int, default=2)
    ap.add_argument("--downstream-epochs", type=int, default=6)
    ap.add_argument("--downstream-patience", type=int, default=2)
    ap.add_argument("--class-balanced", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-summarize", action="store_true")
    return ap.parse_args()


def main() -> None:
    cli = build_cli()
    preset = dict(PRESETS[cli.preset])
    output_root = Path(cli.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in parse_csv(cli.seeds)]
    ratios = [float(x) for x in parse_csv(cli.ratios)]
    market_size = int(preset["market_size"])
    manifest = {
        "preset": cli.preset,
        "seeds": seeds,
        "ratios": ratios,
        "datasets": parse_csv(cli.datasets),
        "profiles": parse_csv(cli.profiles),
        "methods": parse_csv(cli.methods),
        "score_only": False,
        "good_source": cli.good_source,
        "good_target_avoid_weight": float(cli.good_target_avoid_weight),
        "package_size": int(cli.package_size),
        "sample_packaging_stage": "buyer_pca_synchronized_indices",
        "sample_package_summary": "mean_seller_phi_by_buyer_membership",
        "seller_packaging_rule": "reuse_buyer_package_membership",
        "preset_args": preset,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    for ratio in ratios:
        purchase_total = max(1, int(round(market_size * float(ratio))))
        for seed in seeds:
            out_dir = output_root / f"ratio_{ratio_slug(ratio)}_seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "purchase_ratio": float(ratio),
                "purchase_total": int(purchase_total),
                "ratio_market_size": int(market_size),
            }
            (out_dir / "experiment_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            cmd = build_command(cli, seed, ratio, purchase_total, out_dir, preset)
            run_cmd(cmd, cli.dry_run)
    if not cli.no_summarize:
        run_cmd([cli.python, SUMMARY_SCRIPT, "--run-root", output_root], cli.dry_run)


if __name__ == "__main__":
    main()
