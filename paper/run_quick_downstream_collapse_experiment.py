#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick single-seed downstream-operator collapse experiments.

The script is intentionally self-contained and writes outputs under the current
paper workspace. It reads the existing feature cache and model helpers from
the repository root without modifying cached inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import hm_tdsc_nonpackage_ablations as hm  # noqa: E402


DATASETS = {
    "hateful_memes": {
        "base_dir": REPO_DIR / "feature_cache" / "hateful_memes" / "clip_vit_base_patch32" / "base",
        "primary_metric": "good_count",
    },
    "hatespeech": {
        "base_dir": REPO_DIR / "feature_cache" / "hatespeech" / "clip_vit_base_patch32" / "base",
        "primary_metric": "good_count",
    },
    "mscoco": {
        "base_dir": REPO_DIR / "feature_cache" / "mscoco" / "clip_vit_base_patch32" / "base",
        "primary_metric": "good_count",
    },
}

PROFILES = [
    "noise",
    "coreset_far_wrong",
    "typiclust_dense",
    "kmeans_center",
    "uncertainty_badge",
    "cosine",
    "all_average",
]
PROFILE_DECOY_MIXES: Dict[str, Dict[str, float]] = {
    "noise": {
        "noise_decoy": 0.72,
        "coreset_far_wrong_decoy": 0.10,
        "uncertainty_badge_decoy": 0.08,
    },
    "coreset_far_wrong": {
        "coreset_far_wrong_decoy": 0.72,
        "noise_decoy": 0.10,
        "uncertainty_badge_decoy": 0.08,
    },
    "typiclust_dense": {
        "typiclust_dense_decoy": 0.72,
        "kmeans_center_decoy": 0.10,
        "cosine_decoy": 0.08,
    },
    "kmeans_center": {
        "kmeans_center_decoy": 0.72,
        "typiclust_dense_decoy": 0.10,
        "coreset_far_wrong_decoy": 0.08,
    },
    "uncertainty_badge": {
        "uncertainty_badge_decoy": 0.72,
        "coreset_far_wrong_decoy": 0.10,
        "cosine_decoy": 0.08,
    },
    "cosine": {
        "cosine_decoy": 0.72,
        "typiclust_dense_decoy": 0.10,
        "uncertainty_badge_decoy": 0.08,
    },
    "all_average": {
        "noise_decoy": 1.0,
        "coreset_far_wrong_decoy": 1.0,
        "typiclust_dense_decoy": 1.0,
        "kmeans_center_decoy": 1.0,
        "uncertainty_badge_decoy": 1.0,
        "cosine_decoy": 1.0,
    },
}
METHODS = [
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
    "oracle_downstream_gain",
]

PACKAGE_METHODS = {
    "ours_sample_package_direct",
    "ours_sample_package_influence_only",
    "ours_sample_package_loss_reduction_only",
    "ours_sample_package_task_operator",
    "ours_sample_package_krr_influence_only",
    "ours_sample_package_krr_loss_reduction_only",
    "ours_sample_package_krr",
}

TEACHER_REFERENCE_METHODS = {
    "ours_downstream_direct",
    "ours_influence_only",
    "ours_loss_reduction_only",
    "ours_sample_package_direct",
    "ours_sample_package_influence_only",
    "ours_sample_package_loss_reduction_only",
    "oracle_downstream_gain",
}

ONLINE_STUDENT_METHODS = {
    "ours_kernel_ridge_student",
    "ours_krr_influence_only",
    "ours_krr_loss_reduction_only",
    "ours_sample_package_krr",
    "ours_sample_package_krr_influence_only",
    "ours_sample_package_krr_loss_reduction_only",
}

ONLINE_TASK_OPERATOR_METHODS = {
    "ours_task_operator",
    "ours_sample_package_task_operator",
}

MODEL_BASED_BASELINES = {
    "market_uncertainty_select",
    "market_badge_select",
}


@dataclass
class CandidateUniverse:
    rows: Dict[str, np.ndarray]
    buyer_phi: np.ndarray
    phi: np.ndarray
    downstream_gain: np.ndarray
    loss_reduction: np.ndarray
    influence_gain: np.ndarray
    harm_score: np.ndarray
    coreset_score: np.ndarray
    cosine_score: np.ndarray
    uncertainty_score: np.ndarray
    badge_score: np.ndarray
    kmeans_center_score: np.ndarray
    typiclust_score: np.ndarray
    task_operator_score: np.ndarray
    krr_score: np.ndarray
    krr_score_exact: np.ndarray
    krr_influence_score: np.ndarray
    krr_loss_score: np.ndarray
    package_students: Dict[str, RbfKernelRidgeStudent]
    package_calibration_count: int
    student_calibration_size: int
    student_landmark_count: int
    student_sigma2: float


@dataclass
class RbfKernelRidgeStudent:
    landmarks: np.ndarray
    beta: np.ndarray
    sigma2: float
    train_rows: int


KRR_SIGMA2_FLOOR = 0.5
KRR_EXP_INTERVAL = (-4.0, 0.0)
# Degree-4 Chebyshev interpolant of exp(u) on [-4, 0].
KRR_EXP_POLY4 = np.asarray(
    [0.9963358096138180, 0.9534382874063090, 0.3987763197687612,
     0.0819512647685308, 0.0066488054692884],
    dtype=np.float64,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_csv(text: str, allowed: Iterable[str] | None = None) -> List[str]:
    vals = [x.strip() for x in str(text).split(",") if x.strip()]
    if allowed is not None:
        allowed_set = set(allowed)
        bad = [x for x in vals if x not in allowed_set]
        if bad:
            raise ValueError(f"Unknown values: {bad}; allowed={sorted(allowed_set)}")
    return vals


def l2(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return hm.l2_normalize_np(np.asarray(x, dtype=np.float32), eps=eps)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return hm.sigmoid_np(np.asarray(x, dtype=np.float32))


def bce_logits(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return (np.maximum(logits, 0.0) - logits * y + np.log1p(np.exp(-np.abs(logits)))).astype(np.float32)


def zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return ((x - float(np.mean(x))) / max(float(np.std(x)), eps)).astype(np.float32)


def rank_ascending(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(x), kind="mergesort")
    rank = np.empty(len(order), dtype=np.float32)
    rank[order] = np.arange(len(order), dtype=np.float32)
    return rank


def allocate_counts(weights: Dict[str, float], total: int) -> Dict[str, int]:
    weights = {k: max(0.0, float(v)) for k, v in weights.items()}
    denom = float(sum(weights.values()))
    if denom <= 0.0:
        raise ValueError("Profile weights sum to zero.")
    raw = {k: weights[k] / denom * int(total) for k in weights}
    counts = {k: int(math.floor(v)) for k, v in raw.items()}
    remain = int(total) - int(sum(counts.values()))
    order = sorted(raw.keys(), key=lambda k: raw[k] - math.floor(raw[k]), reverse=True)
    for key in order[:remain]:
        counts[key] += 1
    return counts


def class_centroids(txt: np.ndarray, y: np.ndarray) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], np.ndarray]:
    txt = np.asarray(txt, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    global_mean = l2(np.mean(txt, axis=0, keepdims=True))[0].astype(np.float32)
    centroids: Dict[int, np.ndarray] = {}
    neg_centroids: Dict[int, np.ndarray] = {}
    for val in np.unique(y).tolist():
        mask = y == int(val)
        centroids[int(val)] = l2(np.mean(txt[mask], axis=0, keepdims=True))[0].astype(np.float32)
        neg = txt[~mask]
        if len(neg):
            neg_centroids[int(val)] = l2(np.mean(neg, axis=0, keepdims=True))[0].astype(np.float32)
        else:
            neg_centroids[int(val)] = global_mean
    return centroids, neg_centroids, global_mean


def pair_kernel_features(img: np.ndarray, txt: np.ndarray, out_dim: int, seed: int) -> np.ndarray:
    feat = np.concatenate(
        [
            img.astype(np.float32),
            txt.astype(np.float32),
            (img * txt).astype(np.float32),
            np.abs(img - txt).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    feat = l2(feat)
    if out_dim > 0 and out_dim < feat.shape[1]:
        rng = np.random.default_rng(seed + 7789)
        proj = rng.normal(0.0, 1.0 / math.sqrt(out_dim), size=(feat.shape[1], out_dim)).astype(np.float32)
        feat = feat @ proj
        feat = l2(feat)
    return feat.astype(np.float32)


def stratified_indices(y: np.ndarray, n: int, rng: np.random.Generator, exclude: np.ndarray | None = None) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    excluded = set(np.asarray(exclude, dtype=np.int64).tolist()) if exclude is not None else set()
    vals, counts = np.unique(y, return_counts=True)
    parts: List[np.ndarray] = []
    remaining = int(n)
    for i, val in enumerate(vals.tolist()):
        pool = np.flatnonzero(y == int(val)).astype(np.int64)
        if excluded:
            pool = np.asarray([p for p in pool.tolist() if p not in excluded], dtype=np.int64)
        if len(pool) == 0:
            continue
        if i == len(vals) - 1:
            k = remaining
        else:
            k = int(round(n * float(counts[i]) / float(len(y))))
            k = min(k, remaining)
        take = rng.choice(pool, size=k, replace=k > len(pool)).astype(np.int64)
        parts.append(take)
        remaining -= k
    if remaining > 0:
        pool = np.arange(len(y), dtype=np.int64)
        if excluded:
            pool = np.asarray([p for p in pool.tolist() if p not in excluded], dtype=np.int64)
        parts.append(rng.choice(pool, size=remaining, replace=remaining > len(pool)).astype(np.int64))
    out = np.concatenate(parts, axis=0).astype(np.int64)
    rng.shuffle(out)
    return out[:n]


def degrade_text(txt: np.ndarray, y: np.ndarray, seed: int, strength: float, swap_prob: float) -> np.ndarray:
    return hm.degrade_local_text(
        txt.astype(np.float32),
        y.astype(np.int64),
        seed=int(seed),
        strength=float(strength),
        wrong_swap_prob=float(swap_prob),
    ).astype(np.float32)


def choose_wrong_high_sim(
    img: np.ndarray,
    txt: np.ndarray,
    y: np.ndarray,
    base_idx: np.ndarray,
    search_idx: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    out = np.zeros((len(base_idx), txt.shape[1]), dtype=np.float32)
    txt_search = txt[search_idx].astype(np.float32)
    y_search = y[search_idx].astype(np.int64)
    for start in range(0, len(base_idx), batch_size):
        end = min(start + batch_size, len(base_idx))
        cur = base_idx[start:end]
        sim = img[cur].astype(np.float32) @ txt_search.T
        for j, bi in enumerate(cur.tolist()):
            mask = y_search != int(y[int(bi)])
            scores = sim[j].copy()
            scores[~mask] = -np.inf
            best = int(np.argmax(scores))
            if not np.isfinite(scores[best]):
                best = int(np.argmax(sim[j]))
            out[start + j] = txt_search[best]
    return l2(out)


def build_args_namespace(cli: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        device=str(cli.device),
        batch_size=int(cli.batch_size),
        pair_model="bopa",
        hidden_dim=256,
        bopa_combined_dim=64,
        dropout=0.2,
        lr=float(cli.lr),
        weight_decay=1e-4,
        val_ratio=0.15,
        init_max_epochs=int(cli.anchor_epochs),
        init_patience=int(cli.anchor_patience),
        update_max_epochs=2,
        update_patience=1,
        downstream_max_epochs=int(cli.downstream_epochs),
        downstream_patience=int(cli.downstream_patience),
        class_balanced_teacher_loss=bool(cli.class_balanced),
    )


def train_anchor_and_validation(
    train_pack: Dict[str, np.ndarray],
    args_ns: argparse.Namespace,
    cli: argparse.Namespace,
    rng: np.random.Generator,
    dataset_seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], object, Dict[str, object]]:
    all_y = train_pack["y"].astype(np.int64)
    initial_idx = stratified_indices(all_y, int(cli.initial_noisy_size), rng)
    val_idx = stratified_indices(all_y, int(cli.validation_size), rng, exclude=initial_idx)
    initial_clean, initial_noisy = hm.build_teacher_local_packs(
        train_pack,
        initial_idx,
        dataset_seed,
        local_text_mode=str(cli.local_text_mode),
        local_noise_strength=float(cli.local_noise_strength),
        local_wrong_swap_prob=float(cli.local_wrong_swap_prob),
        local_clean_ratio=float(cli.local_clean_ratio),
    )
    val_pack = {
        "img": train_pack["img"][val_idx].astype(np.float32),
        "txt": train_pack["txt"][val_idx].astype(np.float32),
        "y": train_pack["y"][val_idx].astype(np.int64),
    }
    t0 = time.perf_counter()
    anchor, _, anchor_val_auc = hm.train_pair_model(initial_noisy, args_ns, dataset_seed + 17)
    elapsed = time.perf_counter() - t0
    info = {
        "initial_idx": initial_idx,
        "validation_idx": val_idx,
        "anchor_internal_val_auc": float(anchor_val_auc),
        "anchor_train_time": float(elapsed),
    }
    return initial_clean, initial_noisy, val_pack, anchor, info


def compute_validation_gradient(
    anchor: object,
    val_pack: Dict[str, np.ndarray],
    args_ns: argparse.Namespace,
    feature_dim: int,
    seed: int,
) -> np.ndarray:
    logits = hm.predict_pair_logits(anchor, val_pack["img"], val_pack["txt"], args_ns.device, args_ns.batch_size)
    residual = (sigmoid(logits) - val_pack["y"].astype(np.float32)).astype(np.float32)
    phi = pair_kernel_features(val_pack["img"], val_pack["txt"], feature_dim, seed)
    grad = np.mean(residual.reshape(-1, 1) * phi, axis=0)
    return grad.astype(np.float32)


def assemble_operator_rows(
    train_pack: Dict[str, np.ndarray],
    base_idx: np.ndarray,
    search_idx: np.ndarray,
    cli: argparse.Namespace,
    seed: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    y = train_pack["y"].astype(np.int64)
    base_idx = np.asarray(base_idx, dtype=np.int64)
    search_idx = np.asarray(search_idx, dtype=np.int64)
    img_base = train_pack["img"][base_idx].astype(np.float32)
    txt_base = train_pack["txt"][base_idx].astype(np.float32)
    y_base = train_pack["y"][base_idx].astype(np.int64)
    weak_base = degrade_text(txt_base, y_base, seed + 201, float(cli.local_noise_strength), float(cli.local_wrong_swap_prob))
    wrong_high = choose_wrong_high_sim(
        train_pack["img"],
        train_pack["txt"],
        y,
        base_idx,
        search_idx,
        int(cli.batch_size),
    )
    pos_cent, neg_cent, global_mean = class_centroids(train_pack["txt"], y)
    pos_arr = np.stack([pos_cent.get(int(v), global_mean) for v in y_base.tolist()], axis=0).astype(np.float32)
    neg_arr = np.stack([neg_cent.get(int(v), global_mean) for v in y_base.tolist()], axis=0).astype(np.float32)
    far = rng.normal(0.0, 1.0, size=txt_base.shape).astype(np.float32)
    far = far - np.sum(far * img_base, axis=1, keepdims=True) * img_base
    far = far - np.sum(far * global_mean.reshape(1, -1), axis=1, keepdims=True) * global_mean.reshape(1, -1)
    far = l2(far)

    useful_clean = l2(0.75 * img_base + 0.25 * txt_base)
    noise_decoy = far
    coreset_far_wrong = l2(0.58 * far + 0.34 * wrong_high + 0.08 * neg_arr)
    typiclust_dense = l2(0.72 * neg_arr + 0.20 * wrong_high + 0.08 * global_mean.reshape(1, -1))
    kmeans_center = l2(0.70 * neg_arr + 0.30 * wrong_high)
    uncertainty_badge = l2(0.46 * txt_base + 0.46 * wrong_high + 0.08 * neg_arr)
    cosine_decoy = l2(0.68 * img_base + 0.26 * wrong_high - 0.20 * (pos_arr - neg_arr))

    parts = [
        ("useful_clean", useful_clean, np.ones(len(base_idx), dtype=np.int64)),
        ("noise_decoy", noise_decoy, np.zeros(len(base_idx), dtype=np.int64)),
        ("coreset_far_wrong_decoy", coreset_far_wrong, np.zeros(len(base_idx), dtype=np.int64)),
        ("typiclust_dense_decoy", typiclust_dense, np.zeros(len(base_idx), dtype=np.int64)),
        ("kmeans_center_decoy", kmeans_center, np.zeros(len(base_idx), dtype=np.int64)),
        ("uncertainty_badge_decoy", uncertainty_badge, np.zeros(len(base_idx), dtype=np.int64)),
        ("cosine_decoy", cosine_decoy, np.zeros(len(base_idx), dtype=np.int64)),
    ]

    rows: Dict[str, List[np.ndarray]] = {
        "img": [],
        "weak": [],
        "cand": [],
        "y": [],
        "is_good_seed": [],
        "row_type": [],
        "base_source_idx": [],
    }
    for name, cand, seed_good in parts:
        rows["img"].append(img_base)
        rows["weak"].append(weak_base)
        rows["cand"].append(cand.astype(np.float32))
        rows["y"].append(y_base)
        rows["is_good_seed"].append(seed_good.astype(np.int64))
        rows["row_type"].append(np.array([name] * len(base_idx), dtype=object))
        rows["base_source_idx"].append(base_idx.astype(np.int64))

    return {
        "img": np.concatenate(rows["img"], axis=0).astype(np.float32),
        "weak": np.concatenate(rows["weak"], axis=0).astype(np.float32),
        "cand": np.concatenate(rows["cand"], axis=0).astype(np.float32),
        "y": np.concatenate(rows["y"], axis=0).astype(np.int64),
        "is_good_seed": np.concatenate(rows["is_good_seed"], axis=0).astype(np.int64),
        "row_type": np.concatenate(rows["row_type"], axis=0).astype(object),
        "base_source_idx": np.concatenate(rows["base_source_idx"], axis=0).astype(np.int64),
    }


def compute_downstream_teacher_signals(
    rows: Dict[str, np.ndarray],
    anchor: object,
    val_grad: np.ndarray,
    args_ns: argparse.Namespace,
    feature_dim: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    cand_logits = hm.predict_pair_logits(anchor, rows["img"], rows["cand"], args_ns.device, args_ns.batch_size)
    weak_logits = hm.predict_pair_logits(anchor, rows["img"], rows["weak"], args_ns.device, args_ns.batch_size)
    yy = rows["y"].astype(np.float32)
    cand_loss = bce_logits(cand_logits, yy)
    weak_loss = bce_logits(weak_logits, yy)
    loss_reduction = (weak_loss - cand_loss).astype(np.float32)
    phi = pair_kernel_features(rows["img"], rows["cand"], int(feature_dim), seed)
    residual = (sigmoid(cand_logits) - yy).astype(np.float32)
    influence_gain = (residual.reshape(-1, 1) * phi @ val_grad).astype(np.float32)
    downstream_gain = (zscore(influence_gain) + zscore(loss_reduction)).astype(np.float32)
    return {
        "cand_logits": cand_logits.astype(np.float32),
        "phi": phi.astype(np.float32),
        "loss_reduction": loss_reduction.astype(np.float32),
        "influence_gain": influence_gain.astype(np.float32),
        "downstream_gain": downstream_gain.astype(np.float32),
        "harm_score": (-downstream_gain).astype(np.float32),
    }


def compute_task_operator_score(
    rows: Dict[str, np.ndarray],
    val_pack: Dict[str, np.ndarray],
    cand_phi: np.ndarray,
    feature_dim: int,
    seed: int,
) -> np.ndarray:
    val_phi = pair_kernel_features(val_pack["img"], val_pack["txt"], int(feature_dim), seed)
    val_sign = (2.0 * val_pack["y"].astype(np.float32) - 1.0).reshape(-1, 1)
    task_vec = np.mean(val_sign * val_phi.astype(np.float32), axis=0).astype(np.float32)
    norm = float(np.linalg.norm(task_vec))
    if norm > 1e-8:
        task_vec = (task_vec / norm).astype(np.float32)
    weak_phi = pair_kernel_features(rows["img"], rows["weak"], int(feature_dim), seed)
    row_sign = (2.0 * rows["y"].astype(np.float32) - 1.0)
    cand_margin = row_sign * (cand_phi.astype(np.float32) @ task_vec)
    weak_margin = row_sign * (weak_phi.astype(np.float32) @ task_vec)
    return (cand_margin - weak_margin).astype(np.float32)


def student_supervision_target(score: np.ndarray, cli: argparse.Namespace) -> np.ndarray:
    score = np.asarray(score, dtype=np.float32)
    mode = str(getattr(cli, "student_supervision", "top_quantile"))
    if mode == "regression":
        return zscore(score).astype(np.float32)
    if mode != "top_quantile":
        raise ValueError(f"Unknown student supervision mode: {mode}")
    n = int(len(score))
    if n == 0:
        return score
    pos = max(1, min(n, int(round(n * float(cli.good_ratio)))))
    order = np.argsort(-score, kind="mergesort")
    target = np.zeros((n,), dtype=np.float32)
    target[order[:pos]] = 1.0
    return target


def fit_rbf_kernel_ridge_student(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    train_size: int,
    ridge: float,
    batch_size: int,
) -> RbfKernelRidgeStudent:
    rng = np.random.default_rng(seed + 991)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    n = len(x)
    if n == 0:
        raise ValueError("Cannot fit KRR student with an empty calibration set.")
    m = min(int(train_size), n)
    landmark_idx = rng.choice(np.arange(n, dtype=np.int64), size=m, replace=False)
    xlm = x[landmark_idx].astype(np.float32)
    probe_n = min(m, 512)
    probe_a = rng.choice(np.arange(m, dtype=np.int64), size=probe_n, replace=False)
    probe_b = rng.choice(np.arange(m, dtype=np.int64), size=probe_n, replace=False)
    dist_probe = np.sum((xlm[probe_a] - xlm[probe_b]) ** 2, axis=1)
    dist_probe = dist_probe[dist_probe > 1e-8]
    sigma2 = float(np.median(dist_probe)) if len(dist_probe) else 1.0
    # Unit-normalized rows and their package means have norm at most one, so
    # ||x-l||^2 <= 4. Registering this public bandwidth floor certifies
    # u=-||x-l||^2/(2 sigma^2) in [-4, 0],
    # which is the fixed interval used by the encrypted degree-4 exp circuit.
    sigma2 = max(sigma2, KRR_SIGMA2_FLOOR)
    xlm_norm = np.sum(xlm * xlm, axis=1, keepdims=True)
    d2_mm = xlm_norm + xlm_norm.T - 2.0 * (xlm @ xlm.T)
    w = np.exp(-np.maximum(d2_mm, 0.0) / (2.0 * sigma2)).astype(np.float64)
    ctc = np.zeros((m, m), dtype=np.float64)
    cty = np.zeros((m,), dtype=np.float64)
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, n)
        xb = x[start:end].astype(np.float32)
        d2b = np.sum(xb * xb, axis=1, keepdims=True) + xlm_norm.T - 2.0 * (xb @ xlm.T)
        c = np.exp(-np.maximum(d2b, 0.0) / (2.0 * sigma2)).astype(np.float64)
        ctc += c.T @ c
        cty += c.T @ y[start:end].astype(np.float64)
    lhs = ctc + float(ridge) * w + 1e-6 * np.eye(m, dtype=np.float64)
    try:
        beta = np.linalg.solve(lhs, cty).astype(np.float32)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(lhs, cty, rcond=None)[0].astype(np.float32)
    return RbfKernelRidgeStudent(landmarks=xlm, beta=beta, sigma2=float(sigma2), train_rows=int(n))


def predict_rbf_kernel_ridge_student(
    model: RbfKernelRidgeStudent,
    x: np.ndarray,
    batch_size: int,
    kernel_eval: str = "poly4",
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    xlm = model.landmarks.astype(np.float32)
    beta = model.beta.astype(np.float32)
    sigma2 = max(float(model.sigma2), KRR_SIGMA2_FLOOR)
    xlm_norm = np.sum(xlm * xlm, axis=1, keepdims=True)
    pred = np.zeros((len(x),), dtype=np.float32)
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        xb = x[start:end].astype(np.float32)
        d2b = np.sum(xb * xb, axis=1, keepdims=True) + xlm_norm.T - 2.0 * (xb @ xlm.T)
        u = -np.maximum(d2b, 0.0) / (2.0 * sigma2)
        if float(np.min(u)) < KRR_EXP_INTERVAL[0] - 1e-5 or float(np.max(u)) > KRR_EXP_INTERVAL[1] + 1e-5:
            raise ValueError(
                f"KRR exponent input outside certified interval {KRR_EXP_INTERVAL}: "
                f"[{float(np.min(u))}, {float(np.max(u))}]"
            )
        if kernel_eval == "exact":
            kb = np.exp(u).astype(np.float32)
        elif kernel_eval == "poly4":
            c0, c1, c2, c3, c4 = KRR_EXP_POLY4
            u64 = u.astype(np.float64)
            u2 = u64 * u64
            kb = (c0 + c1 * u64 + c2 * u2 + c3 * u64 * u2 + c4 * u2 * u2).astype(np.float32)
        else:
            raise ValueError(f"Unknown KRR kernel evaluation mode: {kernel_eval}")
        pred[start:end] = (kb @ beta).astype(np.float32)
    return pred


def pca_balanced_packages(x: np.ndarray, package_size: int) -> List[np.ndarray]:
    """Create score-independent balanced packages along the principal axis."""
    x = np.asarray(x, dtype=np.float32)
    n = int(len(x))
    package_size = max(1, int(package_size))
    if n == 0:
        return []
    if package_size == 1:
        return [np.asarray([i], dtype=np.int64) for i in range(n)]
    centered = x - np.mean(x, axis=0, keepdims=True)
    if n == 1 or float(np.linalg.norm(centered)) <= 1e-8:
        order = np.arange(n, dtype=np.int64)
    else:
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            axis = vt[0].astype(np.float32)
            projection = centered @ axis
            order = np.argsort(projection, kind="mergesort").astype(np.int64)
        except np.linalg.LinAlgError:
            order = np.arange(n, dtype=np.int64)
    return [
        order[start : start + package_size].astype(np.int64)
        for start in range(0, n, package_size)
    ]


def summarize_packages(
    x: np.ndarray,
    packages: List[np.ndarray],
    normalize: bool = False,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if not packages:
        return np.zeros((0, x.shape[1]), dtype=np.float32)
    summary = np.stack(
        [np.mean(x[idx], axis=0).astype(np.float32) for idx in packages],
        axis=0,
    ).astype(np.float32)
    return l2(summary).astype(np.float32) if normalize else summary


def aggregate_package_targets(target: np.ndarray, packages: List[np.ndarray]) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32)
    return np.asarray(
        [float(np.mean(target[idx])) for idx in packages],
        dtype=np.float32,
    )


def kmeans_center_score(x: np.ndarray, seed: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = int(len(x))
    if n <= 1:
        return np.zeros((n,), dtype=np.float32)
    n_clusters = min(max(2, n // 256), 128, n)
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=int(seed) + 3301,
        batch_size=min(4096, max(256, n)),
        n_init=3,
        max_iter=80,
    )
    labels = km.fit_predict(x.astype(np.float32)).astype(np.int64)
    centers = km.cluster_centers_.astype(np.float32)
    dist = np.linalg.norm(x - centers[labels], axis=1).astype(np.float32)
    counts = np.bincount(labels, minlength=n_clusters).astype(np.float32)
    return (-dist + 0.05 * np.log1p(counts[labels])).astype(np.float32)


def typiclust_density_score(x: np.ndarray, k: int = 8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = int(len(x))
    if n <= 1:
        return np.zeros((n,), dtype=np.float32)
    nn_k = min(max(2, int(k) + 1), n)
    nn = NearestNeighbors(n_neighbors=nn_k, metric="euclidean", algorithm="auto")
    nn.fit(x.astype(np.float32))
    dist, _ = nn.kneighbors(x.astype(np.float32), return_distance=True)
    return (-np.mean(dist[:, 1:], axis=1)).astype(np.float32)


def read_ckks_summary(cli: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    paths = [
        Path(cli.ckks_student_summary),
        Path(cli.ckks_dcc_summary),
    ]
    out: Dict[str, Dict[str, float]] = {}
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            scheme = str(row.get("scheme", ""))
            logical_rows = int(float(row.get("logical_rows", 0) or 0))
            feature_dim = int(float(row.get("feature_dim", 0) or 0))
            package_size = int(float(row.get("package_size", 1) or 1))
            if logical_rows != int(cli.ckks_reference_rows):
                continue
            if feature_dim != int(cli.ckks_reference_dim):
                continue
            if package_size != int(cli.package_size):
                continue
            vals: Dict[str, float] = {}
            for col in [
                "input_prepare_encrypt_ms_mean",
                "encrypted_compute_ms_mean",
                "decrypt_decode_ms_mean",
                "total_full_flow_ms_mean",
                "rows_per_second_mean",
                "scored_objects_per_second_mean",
                "output_ciphertexts_mean",
                "decoded_values_mean",
                "input_ciphertexts_mean",
                "input_ciphertext_bytes_mean",
                "output_ciphertext_bytes_mean",
                "total_communication_bytes_mean",
                "ct_ct_mults_mean",
                "ct_pt_mults_mean",
                "rotations_mean",
                "additions_mean",
                "relinearizations_mean",
                "rescales_mean",
                "reference_count_mean",
                "poly_degree_mean",
                "ctct_nonlinear_depth_mean",
                "scored_objects",
                "raw_rows_per_scored_object",
            ]:
                if col in row and str(row[col]) != "":
                    vals[col] = float(row[col])
            vals["ckks_reference_rows"] = float(logical_rows)
            vals["ckks_reference_dim"] = float(feature_dim)
            vals["ckks_reference_package_size"] = float(package_size)
            out[scheme] = vals
    return out


def ckks_scheme_for_method(method: str) -> str:
    if method in {"ours_task_operator", "ours_kernel_ridge_student", "ours_krr_influence_only", "ours_krr_loss_reduction_only"}:
        return "ours_krr_row_exp_poly4_ctpt"
    if method in {
        "ours_sample_package_task_operator",
        "ours_sample_package_krr",
        "ours_sample_package_krr_influence_only",
        "ours_sample_package_krr_loss_reduction_only",
    }:
        return "ours_krr_pkg_exp_poly4_ctpt"
    if method in {
        "ours_sample_package_direct",
        "ours_sample_package_influence_only",
        "ours_sample_package_loss_reduction_only",
    }:
        return "ours_dcc_pkg_simd_ctpt"
    baseline_schemes = {
        "market_random_select": "baseline_random_noop",
        "market_cosine_select": "baseline_cosine_ctpt",
        "market_uncertainty_select": "baseline_uncertainty_poly4_ctpt",
        "market_coreset_select": "baseline_coreset_all_distances_ctpt",
        "market_badge_select": "baseline_badge_components_poly4_ctpt",
        "market_kmeans_center_select": "baseline_kmeans_all_distances_ctpt",
        "market_typiclust_select": "baseline_typiclust_sqrt_poly4_ctpt",
    }
    return baseline_schemes.get(method, "")


def attach_ckks_stats(row: Dict[str, object], method: str, market_size: int, ckks: Dict[str, Dict[str, float]]) -> None:
    scheme = ckks_scheme_for_method(method)
    row["ckks_simd_enabled"] = int(bool(scheme))
    row["ckks_scheme"] = scheme
    if not scheme or scheme not in ckks:
        return
    ref = ckks[scheme]
    rps = float(ref.get("rows_per_second_mean", 0.0))
    ref_rows = float(ref.get("ckks_reference_rows", 0.0))
    row["ckks_reference_rows"] = ref_rows
    row["ckks_reference_dim"] = float(ref.get("ckks_reference_dim", np.nan))
    row["ckks_reference_package_size"] = float(ref.get("ckks_reference_package_size", np.nan))
    row["ckks_rows_per_second_mean"] = rps
    row["ckks_est_encrypted_compute_ms"] = float(market_size) / max(rps, 1e-9) * 1000.0
    if "encrypted_compute_ms_mean" in ref and ref_rows > 0:
        row["ckks_scaled_from_reference_ms"] = float(ref["encrypted_compute_ms_mean"]) * float(market_size) / ref_rows
    for part in [
        "input_prepare_encrypt_ms_mean",
        "decrypt_decode_ms_mean",
        "total_full_flow_ms_mean",
    ]:
        if part in ref and ref_rows > 0:
            out_col = "ckks_est_" + part.replace("_mean", "")
            row[out_col] = float(ref[part]) * float(market_size) / ref_rows
    for col, val in ref.items():
        if col.startswith("ckks_reference"):
            continue
        row["ckks_ref_" + col] = float(val)


def score_for_ours_method(market: Dict[str, np.ndarray], method: str) -> np.ndarray:
    if method in {"ours_downstream_direct", "ours_sample_package_direct", "oracle_downstream_gain"}:
        return market["downstream_gain"].astype(np.float32)
    if method in {"ours_influence_only", "ours_sample_package_influence_only"}:
        return market["influence_gain"].astype(np.float32)
    if method in {"ours_loss_reduction_only", "ours_sample_package_loss_reduction_only"}:
        return market["loss_reduction"].astype(np.float32)
    if method in {"ours_task_operator", "ours_sample_package_task_operator"}:
        return market["task_operator_score"].astype(np.float32)
    if method in {"ours_kernel_ridge_student", "ours_sample_package_krr"}:
        return market["krr_score"].astype(np.float32)
    if method in {"ours_krr_influence_only", "ours_sample_package_krr_influence_only"}:
        return market["krr_influence_score"].astype(np.float32)
    if method in {"ours_krr_loss_reduction_only", "ours_sample_package_krr_loss_reduction_only"}:
        return market["krr_loss_score"].astype(np.float32)
    raise ValueError(method)


def method_scoring_semantics(method: str) -> str:
    if method in ONLINE_TASK_OPERATOR_METHODS:
        return "online_task_operator"
    if method in ONLINE_STUDENT_METHODS:
        return "online_student"
    if method in TEACHER_REFERENCE_METHODS:
        return "teacher_reference"
    if method in MODEL_BASED_BASELINES:
        return "baseline_model_based"
    return "baseline_feature"


def method_uses_downstream_model_at_online_scoring(method: str) -> int:
    return int(method in TEACHER_REFERENCE_METHODS or method in MODEL_BASED_BASELINES)


def build_candidate_universe(
    train_pack: Dict[str, np.ndarray],
    initial_noisy: Dict[str, np.ndarray],
    val_pack: Dict[str, np.ndarray],
    anchor: object,
    args_ns: argparse.Namespace,
    cli: argparse.Namespace,
    seed: int,
    rng: np.random.Generator,
    calibration_exclude: np.ndarray | None = None,
) -> CandidateUniverse:
    y = train_pack["y"].astype(np.int64)
    base_idx = stratified_indices(y, int(cli.candidate_pool), rng)
    search_idx = stratified_indices(y, min(int(cli.search_pool), len(y)), rng)
    out = assemble_operator_rows(train_pack, base_idx, search_idx, cli, seed, rng)
    val_grad = compute_validation_gradient(anchor, val_pack, args_ns, int(cli.operator_feature_dim), seed)
    # Experiment-only teacher signals: used for market labels and reference diagnostics,
    # never as the online score for deployable Ours methods.
    market_teacher = compute_downstream_teacher_signals(
        out,
        anchor,
        val_grad,
        args_ns,
        int(cli.operator_feature_dim),
        seed,
    )
    phi = market_teacher["phi"]
    downstream_gain = market_teacher["downstream_gain"]
    loss_reduction = market_teacher["loss_reduction"]
    influence_gain = market_teacher["influence_gain"]
    harm_score = market_teacher["harm_score"]

    coreset_score = hm.nearest_train_distance(out["cand"], initial_noisy["txt"], k=1).astype(np.float32)
    cosine_score = np.sum(out["img"] * out["cand"], axis=1).astype(np.float32)
    cand_prob = sigmoid(market_teacher["cand_logits"])
    uncertainty = (cand_prob * (1.0 - cand_prob)).astype(np.float32)
    badge_score = (uncertainty * (1.0 + coreset_score)).astype(np.float32)
    kmeans_score = kmeans_center_score(phi, seed).astype(np.float32)
    typiclust_score = typiclust_density_score(phi, k=8).astype(np.float32)
    task_operator_score = compute_task_operator_score(
        out,
        val_pack,
        phi,
        int(cli.operator_feature_dim),
        seed,
    )

    extra_exclude = np.asarray(calibration_exclude, dtype=np.int64) if calibration_exclude is not None else np.zeros((0,), dtype=np.int64)
    calibration_exclude_idx = np.unique(np.concatenate([base_idx.astype(np.int64), extra_exclude], axis=0)).astype(np.int64)
    calib_base_idx = stratified_indices(
        y,
        int(cli.student_calibration_pool),
        rng,
        exclude=calibration_exclude_idx,
    )
    calib_search_idx = stratified_indices(y, min(int(cli.search_pool), len(y)), rng, exclude=calibration_exclude_idx)
    calib_rows = assemble_operator_rows(train_pack, calib_base_idx, calib_search_idx, cli, seed + 1701, rng)
    calib_teacher = compute_downstream_teacher_signals(
        calib_rows,
        anchor,
        val_grad,
        args_ns,
        int(cli.operator_feature_dim),
        seed,
    )
    # The online student is fitted only on the independent calibration bank.
    # Market candidates are scored later from phi -> KRR score without a
    # downstream-model forward pass.
    student = fit_rbf_kernel_ridge_student(
        calib_teacher["phi"],
        student_supervision_target(calib_teacher["downstream_gain"], cli),
        seed,
        int(cli.krr_train_size),
        float(cli.krr_ridge),
        int(cli.batch_size),
    )
    krr_score_exact = predict_rbf_kernel_ridge_student(
        student,
        phi,
        int(cli.batch_size),
        kernel_eval="exact",
    ).astype(np.float32)
    krr_score = predict_rbf_kernel_ridge_student(
        student,
        phi,
        int(cli.batch_size),
        kernel_eval="poly4",
    ).astype(np.float32)
    requested_methods = set(parse_csv(cli.methods, METHODS))
    calib_buyer_phi = pair_kernel_features(
        calib_rows["img"],
        calib_rows["weak"],
        int(cli.operator_feature_dim),
        seed,
    )
    calibration_packages = pca_balanced_packages(
        calib_buyer_phi,
        int(cli.package_size),
    )
    calibration_package_phi = summarize_packages(
        calib_teacher["phi"],
        calibration_packages,
    )
    package_students: Dict[str, RbfKernelRidgeStudent] = {}
    if requested_methods & {
        "ours_sample_package_krr",
        "ours_sample_package_krr_influence_only",
        "ours_sample_package_krr_loss_reduction_only",
    }:
        package_students["full"] = fit_rbf_kernel_ridge_student(
            calibration_package_phi,
            aggregate_package_targets(
                student_supervision_target(calib_teacher["downstream_gain"], cli),
                calibration_packages,
            ),
            seed + 17011,
            int(cli.krr_train_size),
            float(cli.krr_ridge),
            int(cli.batch_size),
        )
    if requested_methods & {"ours_krr_influence_only", "ours_sample_package_krr_influence_only"}:
        influence_student = fit_rbf_kernel_ridge_student(
            calib_teacher["phi"],
            student_supervision_target(calib_teacher["influence_gain"], cli),
            seed + 19,
            int(cli.krr_train_size),
            float(cli.krr_ridge),
            int(cli.batch_size),
        )
        krr_influence_score = predict_rbf_kernel_ridge_student(
            influence_student,
            phi,
            int(cli.batch_size),
            kernel_eval="poly4",
        ).astype(np.float32)
        if "ours_sample_package_krr_influence_only" in requested_methods:
            package_students["influence"] = fit_rbf_kernel_ridge_student(
                calibration_package_phi,
                aggregate_package_targets(
                    student_supervision_target(calib_teacher["influence_gain"], cli),
                    calibration_packages,
                ),
                seed + 17019,
                int(cli.krr_train_size),
                float(cli.krr_ridge),
                int(cli.batch_size),
            )
    else:
        krr_influence_score = np.zeros_like(krr_score, dtype=np.float32)
    if requested_methods & {"ours_krr_loss_reduction_only", "ours_sample_package_krr_loss_reduction_only"}:
        loss_student = fit_rbf_kernel_ridge_student(
            calib_teacher["phi"],
            student_supervision_target(calib_teacher["loss_reduction"], cli),
            seed + 23,
            int(cli.krr_train_size),
            float(cli.krr_ridge),
            int(cli.batch_size),
        )
        krr_loss_score = predict_rbf_kernel_ridge_student(
            loss_student,
            phi,
            int(cli.batch_size),
            kernel_eval="poly4",
        ).astype(np.float32)
        if "ours_sample_package_krr_loss_reduction_only" in requested_methods:
            package_students["loss"] = fit_rbf_kernel_ridge_student(
                calibration_package_phi,
                aggregate_package_targets(
                    student_supervision_target(calib_teacher["loss_reduction"], cli),
                    calibration_packages,
                ),
                seed + 17023,
                int(cli.krr_train_size),
                float(cli.krr_ridge),
                int(cli.batch_size),
            )
    else:
        krr_loss_score = np.zeros_like(krr_score, dtype=np.float32)
    return CandidateUniverse(
        rows=out,
        buyer_phi=pair_kernel_features(
            out["img"],
            out["weak"],
            int(cli.operator_feature_dim),
            seed,
        ),
        phi=phi,
        downstream_gain=downstream_gain,
        loss_reduction=loss_reduction,
        influence_gain=influence_gain,
        harm_score=harm_score,
        coreset_score=coreset_score,
        cosine_score=cosine_score,
        uncertainty_score=uncertainty,
        badge_score=badge_score,
        kmeans_center_score=kmeans_score,
        typiclust_score=typiclust_score,
        task_operator_score=task_operator_score,
        krr_score=krr_score,
        krr_score_exact=krr_score_exact,
        krr_influence_score=krr_influence_score,
        krr_loss_score=krr_loss_score,
        package_students=package_students,
        package_calibration_count=int(len(calibration_packages)),
        student_calibration_size=int(len(calib_teacher["phi"])),
        student_landmark_count=int(len(student.landmarks)),
        student_sigma2=float(student.sigma2),
    )


def make_market(
    universe: CandidateUniverse,
    profile: str,
    market_size: int,
    good_count: int,
    seed: int,
    package_size: int,
    good_source: str = "downstream_any",
    good_target_avoid_weight: float = 0.0,
) -> Dict[str, np.ndarray]:
    if profile not in PROFILE_DECOY_MIXES:
        raise ValueError(profile)
    if profile in {"noise", "coreset_far_wrong"}:
        target_score = universe.coreset_score
    elif profile == "typiclust_dense":
        target_score = universe.typiclust_score
    elif profile == "kmeans_center":
        target_score = universe.kmeans_center_score
    elif profile == "uncertainty_badge":
        target_score = universe.badge_score
    elif profile == "cosine":
        target_score = universe.cosine_score
    elif profile == "all_average":
        target_score = np.mean(
            np.stack(
                [
                    zscore(universe.coreset_score),
                    zscore(universe.typiclust_score),
                    zscore(universe.kmeans_center_score),
                    zscore(universe.badge_score),
                    zscore(universe.cosine_score),
                ],
                axis=0,
            ),
            axis=0,
        ).astype(np.float32)
    else:
        raise ValueError(profile)

    # Good rows are ranked by the downstream operator.  The optional target
    # avoidance term keeps the target selector's high-score region malicious.
    row_type = universe.rows["row_type"].astype(str)
    if str(good_source) == "useful_clean":
        useful_pool = np.flatnonzero(row_type == "useful_clean").astype(np.int64)
    elif str(good_source) == "downstream_any":
        useful_pool = np.arange(len(universe.downstream_gain), dtype=np.int64)
    else:
        raise ValueError(f"Unknown good_source: {good_source}")
    utility_rank = rank_ascending(-universe.downstream_gain[useful_pool])
    if float(good_target_avoid_weight) > 0.0:
        target_rank = rank_ascending(target_score[useful_pool])
        good_rank = utility_rank + float(good_target_avoid_weight) * target_rank
        useful_order = useful_pool[np.argsort(good_rank, kind="mergesort")].astype(np.int64)
    else:
        useful_order = useful_pool[np.argsort(utility_rank, kind="mergesort")].astype(np.int64)
    if len(useful_order) < good_count:
        useful_idx = np.resize(useful_order, good_count).astype(np.int64)
    else:
        useful_idx = useful_order[:good_count].astype(np.int64)
    useful_set = set(useful_idx.tolist())
    decoy_pool = np.asarray([i for i in range(len(universe.downstream_gain)) if i not in useful_set], dtype=np.int64)

    bad_count = int(market_size) - int(good_count)
    decoy_counts = allocate_counts(PROFILE_DECOY_MIXES[profile], bad_count)
    decoy_parts: List[np.ndarray] = []
    for decoy_type, count in decoy_counts.items():
        if int(count) <= 0:
            continue
        pool = np.asarray(
            [i for i in np.flatnonzero(row_type == decoy_type).astype(np.int64).tolist() if i not in useful_set],
            dtype=np.int64,
        )
        if len(pool) == 0:
            pool = decoy_pool
        # Rank aggregation: high target-selector score and low downstream gain.
        b_rank = rank_ascending(-target_score[pool])
        g_rank = rank_ascending(universe.downstream_gain[pool])
        aggregate_rank = b_rank + g_rank
        order = pool[np.argsort(aggregate_rank, kind="mergesort")]
        take = np.resize(order, int(count)).astype(np.int64) if len(order) < int(count) else order[: int(count)].astype(np.int64)
        decoy_parts.append(take)
    decoy_idx = np.concatenate(decoy_parts, axis=0).astype(np.int64) if decoy_parts else np.zeros((0,), dtype=np.int64)
    if len(decoy_idx) < bad_count:
        b_rank = rank_ascending(-target_score[decoy_pool])
        g_rank = rank_ascending(universe.downstream_gain[decoy_pool])
        aggregate_rank = b_rank + g_rank
        fill_order = decoy_pool[np.argsort(aggregate_rank, kind="mergesort")]
        fill = np.resize(fill_order, bad_count - len(decoy_idx)).astype(np.int64)
        decoy_idx = np.concatenate([decoy_idx, fill], axis=0).astype(np.int64)
    elif len(decoy_idx) > bad_count:
        decoy_idx = decoy_idx[:bad_count].astype(np.int64)

    idx = np.concatenate([useful_idx, decoy_idx], axis=0).astype(np.int64)
    rng = np.random.default_rng(seed + 5003)
    good_flags = np.concatenate([np.ones(len(useful_idx), dtype=np.int64), np.zeros(len(decoy_idx), dtype=np.int64)], axis=0)
    order = rng.permutation(len(idx))
    idx = idx[order]
    good_flags = good_flags[order]
    out = {
        "img": universe.rows["img"][idx].astype(np.float32),
        "weak": universe.rows["weak"][idx].astype(np.float32),
        "cand": universe.rows["cand"][idx].astype(np.float32),
        "y": universe.rows["y"][idx].astype(np.int64),
        "is_good": good_flags.astype(np.int64),
        "row_type": universe.rows["row_type"][idx].astype(object),
        "base_source_idx": universe.rows["base_source_idx"][idx].astype(np.int64),
        "buyer_phi": universe.buyer_phi[idx].astype(np.float32),
        "phi": universe.phi[idx].astype(np.float32),
        "downstream_gain": universe.downstream_gain[idx].astype(np.float32),
        "loss_reduction": universe.loss_reduction[idx].astype(np.float32),
        "influence_gain": universe.influence_gain[idx].astype(np.float32),
        "harm_score": universe.harm_score[idx].astype(np.float32),
        "coreset_score": universe.coreset_score[idx].astype(np.float32),
        "cosine_score": universe.cosine_score[idx].astype(np.float32),
        "uncertainty_score": universe.uncertainty_score[idx].astype(np.float32),
        "badge_score": universe.badge_score[idx].astype(np.float32),
        "kmeans_center_score": universe.kmeans_center_score[idx].astype(np.float32),
        "typiclust_score": universe.typiclust_score[idx].astype(np.float32),
        "task_operator_score": universe.task_operator_score[idx].astype(np.float32),
        "krr_score": universe.krr_score[idx].astype(np.float32),
        "krr_score_exact": universe.krr_score_exact[idx].astype(np.float32),
        "krr_influence_score": universe.krr_influence_score[idx].astype(np.float32),
        "krr_loss_score": universe.krr_loss_score[idx].astype(np.float32),
        "student_calibration_size": np.asarray(universe.student_calibration_size, dtype=np.int64),
        "student_landmark_count": np.asarray(universe.student_landmark_count, dtype=np.int64),
        "student_sigma2": np.asarray(universe.student_sigma2, dtype=np.float32),
    }
    # Buyer-side PCA determines package membership once. The seller applies the
    # same row-index sets to its corresponding private candidate features.
    packages = pca_balanced_packages(out["buyer_phi"], int(package_size))
    package_phi = summarize_packages(out["phi"], packages)
    package_membership = np.full((len(out["phi"]),), -1, dtype=np.int64)
    package_sizes = np.zeros((len(packages),), dtype=np.int64)
    package_offsets = np.zeros((len(packages) + 1,), dtype=np.int64)
    flat_members: List[int] = []
    for package_id, members in enumerate(packages):
        package_membership[members] = int(package_id)
        package_sizes[package_id] = int(len(members))
        flat_members.extend(members.tolist())
        package_offsets[package_id + 1] = len(flat_members)
    out["sample_package_membership"] = package_membership
    out["sample_package_sizes"] = package_sizes
    out["sample_package_offsets"] = package_offsets
    out["sample_package_members"] = np.asarray(flat_members, dtype=np.int64)
    out["sample_package_phi"] = package_phi.astype(np.float32)
    for name, model in universe.package_students.items():
        out[f"sample_package_krr_{name}_sigma2"] = np.asarray(model.sigma2, dtype=np.float32)
        out[f"sample_package_krr_{name}_score"] = predict_rbf_kernel_ridge_student(
            model,
            package_phi,
            512,
            kernel_eval="poly4",
        ).astype(np.float32)
        out[f"sample_package_krr_{name}_score_exact"] = predict_rbf_kernel_ridge_student(
            model,
            package_phi,
            512,
            kernel_eval="exact",
        ).astype(np.float32)
    return out


def select_indices(
    market: Dict[str, np.ndarray],
    method: str,
    initial_noisy: Dict[str, np.ndarray],
    purchase_total: int,
    round_budget: int,
    package_size: int = 2,
) -> np.ndarray:
    if method in PACKAGE_METHODS:
        package_score = package_score_for_method(market, method)
        return select_indices_by_precomputed_packages(
            market,
            package_score,
            int(purchase_total),
        )
    available = np.arange(len(market["y"]), dtype=np.int64)
    selected: List[int] = []
    current_txt = initial_noisy["txt"].astype(np.float32)
    while len(selected) < purchase_total and len(available) > 0:
        budget = min(round_budget, purchase_total - len(selected))
        if method == "market_random_select":
            rng = np.random.default_rng(9001 + len(selected))
            order = available.copy()
            rng.shuffle(order)
        elif method == "market_cosine_select":
            score = market["cosine_score"]
            order = available[np.argsort(-score[available], kind="mergesort")]
        elif method == "market_uncertainty_select":
            score = market["uncertainty_score"]
            order = available[np.argsort(-score[available], kind="mergesort")]
        elif method == "market_coreset_select":
            score_local = hm.nearest_train_distance(market["cand"][available], current_txt, k=1)
            order = available[np.argsort(-score_local, kind="mergesort")]
        elif method == "market_badge_select":
            score = market["badge_score"]
            order = available[np.argsort(-score[available], kind="mergesort")]
        elif method == "market_kmeans_center_select":
            score = market["kmeans_center_score"]
            order = available[np.argsort(-score[available], kind="mergesort")]
        elif method == "market_typiclust_select":
            score = market["typiclust_score"]
            order = available[np.argsort(-score[available], kind="mergesort")]
        elif method in {"ours_downstream_direct", "ours_influence_only", "ours_loss_reduction_only"}:
            score = score_for_ours_method(market, method)
            order = available[np.argsort(-score[available], kind="mergesort")]
        elif method in ONLINE_STUDENT_METHODS or method in ONLINE_TASK_OPERATOR_METHODS:
            score = score_for_ours_method(market, method)
            order = available[np.argsort(-score[available], kind="mergesort")]
        elif method == "oracle_downstream_gain":
            score = score_for_ours_method(market, method)
            order = available[np.argsort(-score[available], kind="mergesort")]
        else:
            raise ValueError(method)
        chosen = order[:budget].astype(np.int64)
        selected.extend(chosen.tolist())
        current_txt = np.concatenate([current_txt, market["cand"][chosen].astype(np.float32)], axis=0).astype(np.float32)
        chosen_set = set(chosen.tolist())
        available = np.asarray([i for i in available.tolist() if i not in chosen_set], dtype=np.int64)
    return np.asarray(selected, dtype=np.int64)


def market_packages(market: Dict[str, np.ndarray]) -> List[np.ndarray]:
    offsets = np.asarray(market["sample_package_offsets"], dtype=np.int64)
    members = np.asarray(market["sample_package_members"], dtype=np.int64)
    return [
        members[offsets[i] : offsets[i + 1]].astype(np.int64)
        for i in range(max(0, len(offsets) - 1))
    ]


def package_score_for_method(market: Dict[str, np.ndarray], method: str) -> np.ndarray:
    packages = market_packages(market)
    if method == "ours_sample_package_krr":
        return np.asarray(market["sample_package_krr_full_score"], dtype=np.float32)
    if method == "ours_sample_package_krr_influence_only":
        return np.asarray(market["sample_package_krr_influence_score"], dtype=np.float32)
    if method == "ours_sample_package_krr_loss_reduction_only":
        return np.asarray(market["sample_package_krr_loss_score"], dtype=np.float32)
    row_score = score_for_ours_method(market, method)
    return aggregate_package_targets(row_score, packages)


def select_indices_by_precomputed_packages(
    market: Dict[str, np.ndarray],
    package_score: np.ndarray,
    purchase_total: int,
) -> np.ndarray:
    packages = market_packages(market)
    package_score = np.asarray(package_score, dtype=np.float32)
    if len(package_score) != len(packages):
        raise ValueError(
            f"Package score count {len(package_score)} does not match package count {len(packages)}"
        )
    pkg_order = np.argsort(-package_score, kind="mergesort")
    chosen: List[int] = []
    for pi in pkg_order.tolist():
        members = packages[int(pi)].tolist()
        if len(chosen) + len(members) > int(purchase_total):
            continue
        chosen.extend(int(ridx) for ridx in members)
        if len(chosen) == int(purchase_total):
            break
    return np.asarray(chosen, dtype=np.int64)


def extend_pack(initial_noisy: Dict[str, np.ndarray], market: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "img": np.concatenate([initial_noisy["img"], market["img"][idx]], axis=0).astype(np.float32),
        "txt": np.concatenate([initial_noisy["txt"], market["cand"][idx]], axis=0).astype(np.float32),
        "y": np.concatenate([initial_noisy["y"], market["y"][idx]], axis=0).astype(np.int64),
    }


def row_type_counts(row_type: np.ndarray) -> Dict[str, int]:
    vals, counts = np.unique(row_type.astype(str), return_counts=True)
    return {f"type_{v}": int(c) for v, c in zip(vals.tolist(), counts.tolist())}


def static_target_score_for_profile(profile: str, market: Dict[str, np.ndarray]) -> np.ndarray | None:
    if profile in {"noise", "coreset_far_wrong"}:
        return market["coreset_score"].astype(np.float32)
    if profile == "typiclust_dense":
        return market["typiclust_score"].astype(np.float32)
    if profile == "kmeans_center":
        return market["kmeans_center_score"].astype(np.float32)
    if profile == "uncertainty_badge":
        return market["badge_score"].astype(np.float32)
    if profile == "cosine":
        return market["cosine_score"].astype(np.float32)
    if profile == "all_average":
        return np.mean(
            np.stack(
                [
                    zscore(market["coreset_score"]),
                    zscore(market["typiclust_score"]),
                    zscore(market["kmeans_center_score"]),
                    zscore(market["badge_score"]),
                    zscore(market["cosine_score"]),
                ],
                axis=0,
            ),
            axis=0,
        ).astype(np.float32)
    return None


def market_label_diagnostics(profile: str, market: Dict[str, np.ndarray], purchase_total: int) -> Dict[str, object]:
    good = market["is_good"].astype(np.int64) == 1
    row_type = market["row_type"].astype(str)
    good_counts = row_type_counts(row_type[good])
    out: Dict[str, object] = {("good_" + k): v for k, v in good_counts.items()}
    useful_clean_good = int(np.sum(good & (row_type == "useful_clean")))
    total_good = int(np.sum(good))
    out["useful_clean_good_count"] = useful_clean_good
    out["decoy_type_good_count"] = int(total_good - useful_clean_good)
    out["decoy_type_good_ratio"] = float((total_good - useful_clean_good) / max(total_good, 1))
    score = static_target_score_for_profile(profile, market)
    if score is not None:
        top_n = min(int(purchase_total), len(score))
        top = np.argsort(-score, kind="mergesort")[:top_n].astype(np.int64)
        out["static_target_top_good_count"] = int(np.sum(market["is_good"][top]))
        out["static_target_top_good_ratio"] = float(np.mean(market["is_good"][top])) if len(top) else 0.0
        out["static_target_top_downstream_gain_mean"] = float(np.mean(market["downstream_gain"][top])) if len(top) else 0.0
        out.update({("static_target_" + k): v for k, v in row_type_counts(row_type[top]).items()})
    return out


def evaluate_method(
    dataset: str,
    profile: str,
    method: str,
    market: Dict[str, np.ndarray],
    initial_noisy: Dict[str, np.ndarray],
    test_pack: Dict[str, np.ndarray],
    args_ns: argparse.Namespace,
    cli: argparse.Namespace,
    seed: int,
    ckks: Dict[str, Dict[str, float]],
) -> Dict[str, object]:
    idx = select_indices(
        market,
        method,
        initial_noisy,
        int(cli.purchase_total),
        int(cli.round_budget),
        package_size=int(cli.package_size),
    )
    metrics: Dict[str, float] = {"test_auroc": np.nan, "test_macro_f1": np.nan, "test_acc": np.nan}
    eval_time = 0.0
    if not bool(cli.score_only):
        cur = extend_pack(initial_noisy, market, idx)
        t0 = time.perf_counter()
        model, _, _ = hm.train_pair_model(
            cur,
            args_ns,
            seed + 70001 + abs(hash(method)) % 1000,
            max_epochs=int(cli.downstream_epochs),
            patience=int(cli.downstream_patience),
        )
        metrics = hm.evaluate_pair_model(model, test_pack, args_ns)
        eval_time = time.perf_counter() - t0
    selected_good = market["is_good"][idx].astype(np.int64)
    row = {
        "dataset": dataset,
        "profile": profile,
        "method": method,
        "seed": int(seed),
        "selected_rows": int(len(idx)),
        "good_count": int(np.sum(selected_good)),
        "good_ratio": float(np.mean(selected_good)) if len(selected_good) else 0.0,
        "downstream_gain_mean": float(np.mean(market["downstream_gain"][idx])) if len(idx) else 0.0,
        "harm_score_mean": float(np.mean(market["harm_score"][idx])) if len(idx) else 0.0,
        "loss_reduction_mean": float(np.mean(market["loss_reduction"][idx])) if len(idx) else 0.0,
        "influence_gain_mean": float(np.mean(market["influence_gain"][idx])) if len(idx) else 0.0,
        "collapse_to_single_digits": int(np.sum(selected_good) < 10),
        "scoring_semantics": method_scoring_semantics(method),
        "online_scoring_uses_downstream_model": method_uses_downstream_model_at_online_scoring(method),
        "sample_packaging_enabled": int(method in PACKAGE_METHODS),
        "package_size": int(cli.package_size) if method in PACKAGE_METHODS else 1,
        "sample_packaging_stage": "buyer_pca_synchronized_indices" if method in PACKAGE_METHODS else "none",
        "sample_package_summary": "mean_seller_phi_by_buyer_membership" if method in PACKAGE_METHODS else "none",
        "seller_packaging_rule": "reuse_buyer_package_membership" if method in PACKAGE_METHODS else "none",
        "package_count": int(len(market.get("sample_package_sizes", []))) if method in PACKAGE_METHODS else int(len(market["y"])),
        "package_size_min": int(np.min(market["sample_package_sizes"])) if method in PACKAGE_METHODS and len(market.get("sample_package_sizes", [])) else 1,
        "package_size_max": int(np.max(market["sample_package_sizes"])) if method in PACKAGE_METHODS and len(market.get("sample_package_sizes", [])) else 1,
        "scored_objects": int(len(market.get("sample_package_sizes", []))) if method in PACKAGE_METHODS else int(len(market["y"])),
        "student_calibration_size": int(market.get("student_calibration_size", 0)),
        "student_landmark_count": int(market.get("student_landmark_count", 0)),
        "student_sigma2": float(
            market.get("sample_package_krr_full_sigma2", np.nan)
            if method == "ours_sample_package_krr"
            else market.get("student_sigma2", np.nan)
        ),
        "student_exp_interval_min": float(KRR_EXP_INTERVAL[0]),
        "student_exp_interval_max": float(KRR_EXP_INTERVAL[1]),
        "student_exp_poly_degree": 4,
        "student_supervision": str(cli.student_supervision),
        "eval_time": float(eval_time),
    }
    if method == "ours_kernel_ridge_student":
        exact_score = np.asarray(market["krr_score_exact"], dtype=np.float32)
        poly_score = np.asarray(market["krr_score"], dtype=np.float32)
        exact_idx = np.argsort(-exact_score, kind="mergesort")[: int(cli.purchase_total)]
        row["krr_poly4_score_max_abs_error"] = float(np.max(np.abs(poly_score - exact_score)))
        row["krr_poly4_top_budget_overlap"] = float(
            len(set(idx.tolist()) & set(exact_idx.tolist())) / max(1, len(exact_idx))
        )
    elif method == "ours_sample_package_krr":
        exact_score = np.asarray(market["sample_package_krr_full_score_exact"], dtype=np.float32)
        poly_score = np.asarray(market["sample_package_krr_full_score"], dtype=np.float32)
        exact_idx = select_indices_by_precomputed_packages(
            market,
            exact_score,
            int(cli.purchase_total),
        )
        row["krr_poly4_score_max_abs_error"] = float(np.max(np.abs(poly_score - exact_score)))
        row["krr_poly4_top_budget_overlap"] = float(
            len(set(idx.tolist()) & set(exact_idx.tolist())) / max(1, len(exact_idx))
        )
    row.update(metrics)
    row.update(row_type_counts(market["row_type"][idx]))
    attach_ckks_stats(row, method, len(market["y"]), ckks)
    return row


def save_market_npz(out_dir: Path, profile: str, market: Dict[str, np.ndarray]) -> None:
    ensure_dir(out_dir / "assembled")
    np.savez_compressed(out_dir / "assembled" / f"{profile}.npz", **market)


def run_dataset(dataset: str, cli: argparse.Namespace, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = DATASETS[dataset]
    args_ns = build_args_namespace(cli)
    ckks = read_ckks_summary(cli)
    dataset_offsets = {"hateful_memes": 0, "hatespeech": 100, "mscoco": 200}
    dataset_seed = int(cli.seed) + int(dataset_offsets.get(dataset, 0))
    rng = np.random.default_rng(dataset_seed)
    print(f"[dataset] {dataset} loading base cache", flush=True)
    packs = hm.load_base_cache(Path(cfg["base_dir"]))
    train_pack = packs["train"]
    test_pack = packs["test"]
    initial_clean, initial_noisy, val_pack, anchor, anchor_info = train_anchor_and_validation(
        train_pack,
        args_ns,
        cli,
        rng,
        dataset_seed,
    )
    anchor_metrics = hm.evaluate_pair_model(anchor, test_pack, args_ns)
    print(f"[dataset] {dataset} anchor={anchor_metrics}", flush=True)
    calibration_exclude = np.unique(
        np.concatenate(
            [
                np.asarray(anchor_info.get("initial_idx", []), dtype=np.int64),
                np.asarray(anchor_info.get("validation_idx", []), dtype=np.int64),
            ],
            axis=0,
        )
    ).astype(np.int64)
    universe = build_candidate_universe(
        train_pack,
        initial_noisy,
        val_pack,
        anchor,
        args_ns,
        cli,
        dataset_seed,
        rng,
        calibration_exclude=calibration_exclude,
    )
    profiles = parse_csv(cli.profiles, PROFILES)
    methods = parse_csv(cli.methods, METHODS)
    good_count = int(cli.good_count)
    if good_count <= 0:
        good_count = int(round(int(cli.market_size) * float(cli.good_ratio)))
    rows: List[Dict[str, object]] = []
    market_rows: List[Dict[str, object]] = []
    for profile in profiles:
        market = make_market(
            universe,
            profile,
            int(cli.market_size),
            good_count,
            dataset_seed,
            int(cli.package_size),
            str(cli.good_source),
            float(cli.good_target_avoid_weight),
        )
        market_dir = out_dir / "markets" / dataset / f"score_only_targeted_seed{int(cli.seed)}"
        save_market_npz(market_dir, profile, market)
        market_rows.append(
            {
                "dataset": dataset,
                "profile": profile,
                "market_size": int(len(market["y"])),
                "good_count": int(np.sum(market["is_good"])),
                "good_ratio": float(np.mean(market["is_good"])),
                "downstream_gain_mean_all": float(np.mean(market["downstream_gain"])),
                "downstream_gain_mean_good": float(np.mean(market["downstream_gain"][market["is_good"] == 1])),
                "downstream_gain_mean_bad": float(np.mean(market["downstream_gain"][market["is_good"] == 0])),
                "anchor_test_auroc": float(anchor_metrics["test_auroc"]),
                "anchor_test_macro_f1": float(anchor_metrics["test_macro_f1"]),
                "anchor_test_acc": float(anchor_metrics["test_acc"]),
                "student_calibration_size": int(universe.student_calibration_size),
                "student_landmark_count": int(universe.student_landmark_count),
                "student_sigma2": float(universe.student_sigma2),
                "package_calibration_count": int(universe.package_calibration_count),
                "sample_package_count": int(len(market["sample_package_sizes"])),
                "sample_package_size_min": int(np.min(market["sample_package_sizes"])),
                "sample_package_size_max": int(np.max(market["sample_package_sizes"])),
                "sample_packaging_stage": "buyer_pca_synchronized_indices",
                "sample_package_summary": "mean_seller_phi_by_buyer_membership",
                "seller_packaging_rule": "reuse_buyer_package_membership",
                "student_supervision": str(cli.student_supervision),
                **anchor_info,
                **market_label_diagnostics(profile, market, int(cli.purchase_total)),
            }
        )
        print(f"[market] {dataset}/{profile} good={int(np.sum(market['is_good']))}/{len(market['y'])}", flush=True)
        for method in methods:
            print(f"  [method] {method}", flush=True)
            row = evaluate_method(dataset, profile, method, market, initial_noisy, test_pack, args_ns, cli, dataset_seed, ckks)
            row["primary_metric"] = cfg["primary_metric"]
            row["primary_metric_value"] = float(row[cfg["primary_metric"]])
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_dir / "results_partial.csv", index=False)
    return pd.DataFrame(rows), pd.DataFrame(market_rows)


def build_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="hateful_memes,hatespeech,mscoco")
    ap.add_argument("--profiles", default=",".join(PROFILES))
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--output-dir", type=Path, default=Path("quick_downstream_collapse_outputs"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--initial-noisy-size", type=int, default=1000)
    ap.add_argument("--validation-size", type=int, default=1000)
    ap.add_argument("--candidate-pool", type=int, default=5000)
    ap.add_argument("--search-pool", type=int, default=8000)
    ap.add_argument("--student-calibration-pool", type=int, default=4000)
    ap.add_argument("--market-size", type=int, default=20000)
    ap.add_argument("--good-count", type=int, default=2000)
    ap.add_argument("--good-ratio", type=float, default=0.10)
    ap.add_argument("--good-source", choices=["useful_clean", "downstream_any"], default="downstream_any")
    ap.add_argument("--good-target-avoid-weight", type=float, default=0.05)
    ap.add_argument("--purchase-total", type=int, default=50)
    ap.add_argument("--round-budget", type=int, default=5)
    ap.add_argument("--package-size", type=int, default=2)
    ap.add_argument("--local-text-mode", default="severe_noisy")
    ap.add_argument("--local-noise-strength", type=float, default=0.9)
    ap.add_argument("--local-wrong-swap-prob", type=float, default=0.6)
    ap.add_argument("--local-clean-ratio", type=float, default=0.0)
    ap.add_argument("--operator-feature-dim", type=int, default=128)
    ap.add_argument("--krr-train-size", type=int, default=1500)
    ap.add_argument("--krr-ridge", type=float, default=1e-2)
    ap.add_argument("--student-supervision", choices=["top_quantile", "regression"], default="top_quantile")
    ap.add_argument("--anchor-epochs", type=int, default=20)
    ap.add_argument("--anchor-patience", type=int, default=5)
    ap.add_argument("--downstream-epochs", type=int, default=10)
    ap.add_argument("--downstream-patience", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--class-balanced", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument(
        "--ckks-student-summary",
        default=str(
            Path(__file__).resolve().parent
            / "main_text_experiments"
            / "runs"
            / "ckks_poly4_all_methods"
            / "ckks_seal_summary.csv"
        ),
    )
    ap.add_argument(
        "--ckks-dcc-summary",
        default=str(REPO_DIR / "outputs" / "ckks_dcc_fullflow_10k" / "ckks_seal_summary.csv"),
    )
    ap.add_argument("--ckks-reference-rows", type=int, default=10000)
    ap.add_argument("--ckks-reference-dim", type=int, default=64)
    return ap.parse_args()


def main() -> None:
    cli = build_cli()
    out_dir = Path(cli.output_dir)
    ensure_dir(out_dir)
    datasets = parse_csv(cli.datasets, DATASETS.keys())
    all_results: List[pd.DataFrame] = []
    all_markets: List[pd.DataFrame] = []
    (out_dir / "config.json").write_text(json.dumps(vars(cli), default=str, indent=2), encoding="utf-8")
    for dataset in datasets:
        t0 = time.perf_counter()
        res, market_df = run_dataset(dataset, cli, out_dir)
        all_results.append(res)
        all_markets.append(market_df)
        print(f"[dataset done] {dataset} elapsed={time.perf_counter() - t0:.1f}s", flush=True)
    results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    markets = pd.concat(all_markets, ignore_index=True) if all_markets else pd.DataFrame()
    results.to_csv(out_dir / "results.csv", index=False)
    markets.to_csv(out_dir / "market_summary.csv", index=False)
    if len(results):
        summary = results.sort_values(["dataset", "profile", "primary_metric_value"], ascending=[True, True, False])
        summary.to_csv(out_dir / "results_ranked.csv", index=False)
        cols = [
            "dataset",
            "profile",
            "method",
            "good_count",
            "downstream_gain_mean",
            "harm_score_mean",
            "sample_packaging_enabled",
            "ckks_scheme",
            "ckks_est_encrypted_compute_ms",
            "test_auroc",
            "test_macro_f1",
            "test_acc",
            "primary_metric_value",
        ]
        cols = [c for c in cols if c in summary.columns]
        print(summary[cols].to_string(index=False), flush=True)
    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
