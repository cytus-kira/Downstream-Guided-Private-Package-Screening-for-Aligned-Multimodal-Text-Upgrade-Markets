#!/usr/bin/env python3

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


NUMERIC = {
    "input_prepare_encrypt_ms",
    "encrypted_compute_ms",
    "decrypt_decode_ms",
    "total_full_flow_ms",
    "threshold_parties",
    "rows_per_second",
    "scored_objects_per_second",
    "output_ciphertexts",
    "decoded_values",
    "input_ciphertexts",
    "input_ciphertext_bytes",
    "output_ciphertext_bytes",
    "total_communication_bytes",
    "ct_ct_mults",
    "ct_pt_mults",
    "rotations",
    "additions",
    "relinearizations",
    "rescales",
    "reference_count",
    "poly_degree",
    "ctct_nonlinear_depth",
}


def mean(xs):
    return sum(xs) / max(1, len(xs))


def std(xs):
    if len(xs) <= 1:
        return 0.0
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def read_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            for key in NUMERIC:
                if key in row and row[key] != "":
                    row[key] = float(row[key])
            rows.append(row)
        return rows


def write_summary(rows, out_path):
    group_cols = [
        "scheme",
        "poly_modulus_degree",
        "coeff_modulus_bits",
        "scale_bits",
        "slot_count",
        "logical_rows",
        "chunks",
        "scored_objects",
        "raw_rows_per_scored_object",
        "feature_dim",
        "student_summary_dim",
        "package_size",
        "threshold_parties",
    ]
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row.get(c, "") for c in group_cols)
        groups[key].append(row)

    fields = list(group_cols)
    for key in [
        "input_prepare_encrypt_ms",
        "encrypted_compute_ms",
        "decrypt_decode_ms",
        "total_full_flow_ms",
        "rows_per_second",
        "scored_objects_per_second",
        "output_ciphertexts",
        "decoded_values",
        "input_ciphertexts",
        "input_ciphertext_bytes",
        "output_ciphertext_bytes",
        "total_communication_bytes",
        "ct_ct_mults",
        "ct_pt_mults",
        "rotations",
        "additions",
        "relinearizations",
        "rescales",
        "reference_count",
        "poly_degree",
        "ctct_nonlinear_depth",
    ]:
        fields.append(key + "_mean")
        fields.append(key + "_std")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, items in sorted(groups.items()):
            out = {col: val for col, val in zip(group_cols, key)}
            for metric in [
                "input_prepare_encrypt_ms",
                "encrypted_compute_ms",
                "decrypt_decode_ms",
                "total_full_flow_ms",
                "rows_per_second",
                "scored_objects_per_second",
                "output_ciphertexts",
                "decoded_values",
                "input_ciphertexts",
                "input_ciphertext_bytes",
                "output_ciphertext_bytes",
                "total_communication_bytes",
                "ct_ct_mults",
                "ct_pt_mults",
                "rotations",
                "additions",
                "relinearizations",
                "rescales",
                "reference_count",
                "poly_degree",
                "ctct_nonlinear_depth",
            ]:
                vals = [float(r[metric]) for r in items if metric in r]
                out[metric + "_mean"] = f"{mean(vals):.6f}"
                out[metric + "_std"] = f"{std(vals):.6f}"
            writer.writerow(out)


def write_markdown(rows, out_path):
    key_cols = ["scheme", "logical_rows", "feature_dim"]
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(c, "") for c in key_cols)].append(row)

    lines = [
        "| Scheme | Rows | Dim | Enc ms | Eval ms | Dec ms | Full ms | Total comm MB | Raw rows/s | CT-CT mult | CT-PT mult | Rotations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, items in sorted(groups.items()):
        scheme, rows_n, dim = key
        enc = mean([float(r.get("input_prepare_encrypt_ms", 0.0)) for r in items])
        t = mean([float(r["encrypted_compute_ms"]) for r in items])
        dec = mean([float(r.get("decrypt_decode_ms", 0.0)) for r in items])
        full = mean([float(r.get("total_full_flow_ms", t)) for r in items])
        comm_mb = mean([float(r.get("total_communication_bytes", 0.0)) for r in items]) / (1024.0 * 1024.0)
        rps = mean([float(r["rows_per_second"]) for r in items])
        ctct = mean([float(r["ct_ct_mults"]) for r in items])
        ctpt = mean([float(r["ct_pt_mults"]) for r in items])
        rot = mean([float(r["rotations"]) for r in items])
        lines.append(
            f"| {scheme} | {rows_n} | {dim} | {enc:.3f} | {t:.3f} | {dec:.3f} | {full:.3f} | {comm_mb:.2f} | {rps:.2f} | {ctct:.0f} | {ctpt:.0f} | {rot:.0f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(input_path)
    write_summary(rows, out_dir / "ckks_seal_summary.csv")
    write_markdown(rows, out_dir / "ckks_seal_summary.md")
    print("[done]", out_dir / "ckks_seal_summary.csv")


if __name__ == "__main__":
    main()
