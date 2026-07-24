#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run buyer-synchronized package-size sensitivity experiments and plot them."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).resolve().parent / "run_main_text_experiments.py"
PLOTTER = Path(__file__).resolve().parent / "build_package_size_sweep_figure.py"


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-sizes", default="2,4,6,8")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--preset", default="main_score_5k")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--purchase-total", type=int, default=48)
    parser.add_argument(
        "--market-size",
        type=int,
        default=4992,
        help="Use a size divisible by 2, 4, 6, and 8 for whole-package fairness.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / "buyer_sync_package_size_sweep_4992",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=ROOT / "latex" / "fig",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "paper_ready"
        / "buyer_sync_package_size_sweep_4992",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(cmd: List[str], dry_run: bool) -> None:
    print("[cmd] " + subprocess.list2cmdline(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    cli = parse_args()
    package_sizes = [int(x) for x in parse_csv(cli.package_sizes)]
    if not package_sizes or any(x <= 0 for x in package_sizes):
        raise ValueError("Package sizes must be positive integers.")
    incompatible = [x for x in package_sizes if cli.purchase_total % x != 0]
    if incompatible:
        raise ValueError(
            f"purchase-total={cli.purchase_total} must be divisible by every "
            f"package size; incompatible={incompatible}"
        )
    market_incompatible = [x for x in package_sizes if cli.market_size % x != 0]
    if market_incompatible:
        raise ValueError(
            f"market-size={cli.market_size} must be divisible by every package "
            f"size; incompatible={market_incompatible}"
        )

    cli.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "buyer_pca_synchronized_indices",
        "package_sizes": package_sizes,
        "seeds": [int(x) for x in parse_csv(cli.seeds)],
        "preset": cli.preset,
        "market_size": int(cli.market_size),
        "purchase_total": int(cli.purchase_total),
        "method": "ours_sample_package_krr",
    }
    (cli.output_root / "sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    for package_size in package_sizes:
        run_root = cli.output_root / f"package_size_{package_size}"
        cmd = [
            cli.python,
            str(RUNNER),
            "--preset",
            cli.preset,
            "--seeds",
            cli.seeds,
            "--device",
            cli.device,
            "--methods",
            "ours_sample_package_krr",
            "--package-size",
            str(package_size),
            "--purchase-total",
            str(cli.purchase_total),
            "--market-size",
            str(cli.market_size),
            "--output-root",
            str(run_root),
            "--score-only",
        ]
        run(cmd, cli.dry_run)

    plot_cmd = [
        cli.python,
        str(PLOTTER),
        "--run-root",
        str(cli.output_root),
        "--package-sizes",
        ",".join(str(x) for x in package_sizes),
        "--purchase-total",
        str(cli.purchase_total),
        "--figure-dir",
        str(cli.figure_dir),
        "--data-dir",
        str(cli.data_dir),
    ]
    run(plot_cmd, cli.dry_run)


if __name__ == "__main__":
    main()
