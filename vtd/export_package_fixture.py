#!/usr/bin/env python
"""Export real saved package summaries for the two-party CKKS release path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


EXP_POLY4 = np.asarray(
    [
        0.9963358096138180,
        0.9534382874063090,
        0.3987763197687612,
        0.0819512647685308,
        0.0066488054692884,
    ],
    dtype=np.float64,
)


def deterministic_reference(index: int, dim: int) -> np.ndarray:
    d = np.arange(1, dim + 1, dtype=np.float64)
    ref = np.sin(0.017 * (index + 1) * d) + 0.5 * np.cos(0.031 * (index + 3) * d)
    return ref / max(float(np.linalg.norm(ref)), 1e-18)


def plaintext_scores(features: np.ndarray, landmarks: int) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    norm2 = np.sum(features * features, axis=1)
    out = np.zeros(len(features), dtype=np.float64)
    for index in range(landmarks):
        ref = deterministic_reference(index, features.shape[1])
        u = 2.0 * (features @ ref) - norm2 - 1.0
        u2 = u * u
        p = EXP_POLY4[0] + EXP_POLY4[1] * u + EXP_POLY4[2] * u2 + EXP_POLY4[3] * u * u2 + EXP_POLY4[4] * u2 * u2
        alpha = math.sin(0.013 * (index + 1)) / math.sqrt(float(landmarks))
        out += alpha * p
    return out


def row_digest(phi_row: np.ndarray, source_idx: int) -> str:
    h = hashlib.sha256()
    h.update(b"BALL-MARKET-VTD-ROW-V1\x00")
    h.update(np.asarray(phi_row, dtype="<f4").tobytes(order="C"))
    h.update(int(source_idx).to_bytes(8, "big", signed=True))
    return h.hexdigest()


def resolve_npz(repo: Path, requested_seed: int, dataset: str, profile: str) -> tuple[Path, int]:
    seeds = [requested_seed]
    if requested_seed in (0, 1, 2):
        seeds.append(42 + requested_seed)
    for seed in seeds:
        root = repo / "paper" / "main_text_experiments" / "runs" / "poly4_krr_main5k" / f"seed_{seed}" / "markets" / dataset
        matches = sorted(root.rglob(f"{profile}.npz")) if root.exists() else []
        if matches:
            return matches[0], seed
    raise FileNotFoundError(f"no cached real market for seed={requested_seed}, dataset={dataset}, profile={profile}")


def export_fixture(repo: Path, output: Path, scores: int, seed: int, dataset: str, profile: str, landmarks: int) -> dict:
    npz_path, resolved_seed = resolve_npz(repo, seed, dataset, profile)
    with np.load(npz_path, allow_pickle=True) as market:
        package_phi = np.asarray(market["sample_package_phi"], dtype=np.float64)
        offsets = np.asarray(market["sample_package_offsets"], dtype=np.int64)
        flat = np.asarray(market["sample_package_members"], dtype=np.int64)
        phi = np.asarray(market["phi"], dtype=np.float32)
        source_idx = np.asarray(market["base_source_idx"], dtype=np.int64)
        count = min(int(scores), len(package_phi))
        package_ids = list(range(count))
        members = [flat[offsets[i] : offsets[i + 1]].astype(int).tolist() for i in package_ids]
        features = package_phi[:count]
        expected = plaintext_scores(features, landmarks)
        digests = [
            hashlib.sha256(
                ("|".join(row_digest(phi[row], int(source_idx[row])) for row in group)).encode("ascii")
            ).hexdigest()
            for group in members
        ]
        payload = {
            "dataset": dataset,
            "seed": int(resolved_seed),
            "requested_seed": int(seed),
            "profile": profile,
            "source_npz": str(npz_path.resolve()),
            "package_ids": package_ids,
            "features": features.tolist(),
            "members": members,
            "rows_digest": digests,
            "plaintext_scores": expected.tolist(),
            "landmarks": int(landmarks),
            "feature_dim": int(features.shape[1]),
            "source_package_score_key": "sample_package_krr_full_score",
            "scorer": "RQ4 deterministic quartic CKKS cost-path scorer applied to saved real package summaries",
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="hateful_memes")
    parser.add_argument("--profile", default="noise")
    parser.add_argument("--landmarks", type=int, default=1000)
    args = parser.parse_args()
    payload = export_fixture(args.repo, args.output, args.scores, args.seed, args.dataset, args.profile, args.landmarks)
    print(json.dumps({k: payload[k] for k in ["dataset", "seed", "profile", "source_npz", "feature_dim", "landmarks"]}, indent=2))


if __name__ == "__main__":
    main()
