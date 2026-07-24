#!/usr/bin/env python
"""Perform data-free integrity checks for the clean experiment release."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "README.md",
    "EXPERIMENT_SUMMARY.md",
    "CODE_INVENTORY.md",
    "requirements.txt",
    "hm_tdsc_nonpackage_ablations.py",
    "paper/run_quick_downstream_collapse_experiment.py",
    "paper/main_text_experiments/run_main_text_experiments.py",
    "encrypted_benchmarks/krr_ckks_seal/ckks_seal_bench.cpp",
    "vtd/go.mod",
    "vtd/internal/vtd/he_test.go",
    "evidence/rq1_main/method_good_count_aggregate.csv",
    "evidence/rq5_vtd/vtd_2of2_attack_summary.csv",
]
FORBIDDEN_PARTS = {"__pycache__", "feature_cache", "paper_ready", "runs", "private", "messages", "build"}
PRIVATE_SUFFIXES = {".key", ".pem", ".bin", ".pyc", ".exe", ".dll", ".pdb"}
HOST_MARKERS = (
    "F:\\codex_workspase",
    "F:\\github",
    "F:\\python_env",
    "C:\\Users\\Admin",
)
TEXT_SUFFIXES = {".py", ".ps1", ".go", ".cpp", ".h", ".md", ".txt", ".json", ".csv", ".mod"}


def check_tree() -> None:
    missing = [rel for rel in REQUIRED if not (ROOT / rel).is_file()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")

    violations: list[str] = []
    host_leaks: list[str] = []
    python_files: list[Path] = []
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if path.is_dir() and path.name in FORBIDDEN_PARTS:
            violations.append(str(rel))
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() in PRIVATE_SUFFIXES:
            violations.append(str(rel))
        if path.suffix.lower() == ".py":
            python_files.append(path)
        if path.suffix.lower() in TEXT_SUFFIXES and path != Path(__file__).resolve():
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            if any(marker in text for marker in HOST_MARKERS):
                host_leaks.append(str(rel))

    if violations:
        raise SystemExit(f"Generated/private artifacts present: {violations}")
    if host_leaks:
        raise SystemExit(f"Host-specific paths present: {host_leaks}")

    for path in python_files:
        source = path.read_text(encoding="utf-8-sig")
        compile(source, str(path), "exec")
    print(f"[ok] required files, clean tree, and Python syntax ({len(python_files)} files)")


def check_go() -> None:
    go = shutil.which("go")
    if not go:
        raise SystemExit("Go was requested but is not on PATH")
    subprocess.run([go, "test", "./..."], cwd=ROOT / "vtd", check=True)
    print("[ok] Go tests")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-go", action="store_true", help="also run go test ./... in vtd")
    args = parser.parse_args()
    check_tree()
    if args.with_go:
        check_go()


if __name__ == "__main__":
    main()
