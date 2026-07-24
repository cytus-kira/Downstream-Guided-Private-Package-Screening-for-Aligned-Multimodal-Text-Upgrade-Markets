#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset


# =========================================================
# Utilities
# =========================================================


def clean_cli_path(x: str) -> str:
    return str(x).strip().strip('"').strip("'").strip()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_list(text: str) -> List[str]:
    return [t.strip() for t in str(text).split(',') if t.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(t.strip()) for t in str(text).split(',') if t.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        d = max(float(np.linalg.norm(x)), eps)
        return (x / d).astype(np.float32)
    d = np.linalg.norm(x, axis=1, keepdims=True)
    d = np.maximum(d, eps)
    return (x / d).astype(np.float32)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))).astype(np.float32)


def zscore_np(x: np.ndarray, clip: float | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x.astype(np.float32)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-8:
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        mu = med
        sd = 1.4826 * mad
    if sd < 1e-8:
        out = np.zeros_like(x, dtype=np.float32)
    else:
        out = ((x - mu) / sd).astype(np.float32)
    if clip is not None and float(clip) > 0:
        out = np.clip(out, -float(clip), float(clip)).astype(np.float32)
    return out.astype(np.float32)


def mean_std_columns(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=group_cols)
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing group columns: {missing}")
    num_cols = [c for c in df.columns if c not in group_cols and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return df[group_cols].drop_duplicates().reset_index(drop=True)
    out = df.groupby(group_cols, dropna=False)[num_cols].agg(['mean', 'std']).reset_index()
    out.columns = [
        c if isinstance(c, str) else (c[0] if c[1] == '' else f"{c[0]}_{c[1]}")
        for c in out.columns.to_flat_index()
    ]
    return out


def safe_auroc(y: np.ndarray, prob: np.ndarray) -> float:
    y = np.asarray(y).astype(np.int64)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, prob))


# =========================================================
# Cached loading
# =========================================================


def load_base_cache(base_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    base_dir = Path(base_dir)
    train = {
        'img': np.load(base_dir / 'train_img.npy').astype(np.float32),
        'txt': np.load(base_dir / 'train_txt.npy').astype(np.float32),
        'y': np.load(base_dir / 'train_y.npy').astype(np.int64),
    }
    test = {
        'img': np.load(base_dir / 'test_img.npy').astype(np.float32),
        'txt': np.load(base_dir / 'test_txt.npy').astype(np.float32),
        'y': np.load(base_dir / 'test_y.npy').astype(np.int64),
    }
    return {'train': train, 'test': test}


def load_market_npz(markets_root: Path, market_name: str) -> Dict[str, np.ndarray]:
    markets_root = Path(markets_root)
    candidates = [
        markets_root / 'assembled' / (market_name + '.npz'),
        markets_root / (market_name + '.npz'),
    ]
    src = None
    checked = []
    for p in candidates:
        checked.append(str(p.resolve()))
        if p.exists():
            src = p
            break

    if src is None:
        script_dir = Path(__file__).resolve().parent
        search_roots = [
            Path.cwd(),
            script_dir,
            script_dir.parent,
            script_dir / 'feature_cache',
            script_dir.parent / 'feature_cache',
        ]
        for root in search_roots:
            if root.exists():
                hits = list(root.rglob(market_name + '.npz'))
                if hits:
                    src = hits[0]
                    print('[market auto-found]', str(src.resolve()))
                    break

    if src is None:
        msg = [
            'Cannot find assembled market npz for ' + str(market_name),
            'cwd = ' + str(Path.cwd()),
            'script_dir = ' + str(Path(__file__).resolve().parent),
            'markets_root = ' + str(markets_root),
            'markets_root_abs = ' + str(markets_root.resolve()),
            'checked candidates:',
        ]
        msg.extend(['  ' + x for x in checked])
        raise FileNotFoundError('\n'.join(msg))

    print('[market load]', str(src.resolve()))
    z = np.load(src, allow_pickle=True)
    out: Dict[str, np.ndarray] = {key: z[key] for key in z.files}
    expected = ['img', 'weak', 'cand', 'y', 'is_good']
    for key in expected:
        if key not in out:
            raise KeyError('Missing key in market cache: ' + key + ' from ' + str(src.resolve()))
    out['img'] = out['img'].astype(np.float32)
    out['weak'] = out['weak'].astype(np.float32)
    out['cand'] = out['cand'].astype(np.float32)
    out['y'] = out['y'].astype(np.int64)
    out['is_good'] = out['is_good'].astype(np.int64)
    if 'row_type' not in out:
        out['row_type'] = np.array(['unknown'] * len(out['y']), dtype=object)
    else:
        out['row_type'] = np.asarray(out['row_type'], dtype=object)
    return out


# =========================================================
# Built-in sparse market builder
# =========================================================


SPARSE_MARKET_PROFILES: Dict[str, Dict[str, float]] = {
    'bcs_sparse_hard_conflict_v2': {
        'useful_good': 0.10,
        'high_sim_conflict': 0.42,
        'decision_wrong_detail': 0.33,
        'adversarial_paraphrase': 0.15,
    },
    'bcs_sparse_decision_boundary_v2': {
        'useful_good': 0.10,
        'decision_wrong_detail': 0.50,
        'high_sim_conflict': 0.30,
        'adversarial_paraphrase': 0.10,
    },
    'bcs_sparse_adversarial_mixed_v2': {
        'useful_good': 0.10,
        'adversarial_paraphrase': 0.40,
        'high_sim_conflict': 0.30,
        'decision_wrong_detail': 0.20,
    },
    # Malicious stress-test markets. These are still assembled only from the
    # cached feature components below; the difference is the sampler used inside
    # each component. Useful rows are concentrated in high-structural pockets,
    # while malicious rows are selected to look attractive to coverage,
    # representative, or shortcut-based baselines.
    'bcs_sparse_coverage_trap_v2': {
        'useful_good': 0.10,
        'high_sim_conflict': 0.30,
        'decision_wrong_detail': 0.30,
        'adversarial_paraphrase': 0.30,
    },
    'bcs_sparse_boundary_decoy_v2': {
        'useful_good': 0.10,
        'decision_wrong_detail': 0.56,
        'high_sim_conflict': 0.24,
        'adversarial_paraphrase': 0.10,
    },
    'bcs_sparse_package_trap_v2': {
        'useful_good': 0.10,
        'high_sim_conflict': 0.40,
        'adversarial_paraphrase': 0.30,
        'decision_wrong_detail': 0.20,
    },
    'bcs_coreset_far_noise_attack_v2': {
        'useful_good': 0.10,
        'high_sim_conflict': 0.72,
        'decision_wrong_detail': 0.10,
        'adversarial_paraphrase': 0.08,
    },
    'bcs_coreset_far_wrong_attack_v2': {
        'useful_good': 0.10,
        'decision_wrong_detail': 0.72,
        'high_sim_conflict': 0.10,
        'adversarial_paraphrase': 0.08,
    },
    'bcs_coreset_shell_attack_v2': {
        'useful_good': 0.10,
        'adversarial_paraphrase': 0.72,
        'high_sim_conflict': 0.10,
        'decision_wrong_detail': 0.08,
    },
    'bcs_typiclust_dense_attack_v2': {
        'useful_good': 0.10,
        'high_sim_conflict': 0.72,
        'decision_wrong_detail': 0.10,
        'adversarial_paraphrase': 0.08,
    },
    'bcs_kmeans_center_attack_v2': {
        'useful_good': 0.10,
        'decision_wrong_detail': 0.72,
        'high_sim_conflict': 0.10,
        'adversarial_paraphrase': 0.08,
    },
    'bcs_uncertainty_badge_attack_v2': {
        'useful_good': 0.10,
        'decision_wrong_detail': 0.72,
        'high_sim_conflict': 0.10,
        'adversarial_paraphrase': 0.08,
    },
    'bcs_cosine_attack_v2': {
        'useful_good': 0.10,
        'high_sim_conflict': 0.72,
        'decision_wrong_detail': 0.10,
        'adversarial_paraphrase': 0.08,
    },
    'bcs_hybrid_selector_attack_v2': {
        'useful_good': 0.10,
        'high_sim_conflict': 0.34,
        'decision_wrong_detail': 0.26,
        'adversarial_paraphrase': 0.30,
    },
}


STRATEGIC_MARKET_PROFILE_MODES: Dict[str, str] = {
    'bcs_sparse_coverage_trap_v2': 'coverage_trap',
    'bcs_sparse_boundary_decoy_v2': 'boundary_decoy',
    'bcs_sparse_package_trap_v2': 'package_trap',
    'bcs_coreset_far_noise_attack_v2': 'coreset_far_noise_attack',
    'bcs_coreset_far_wrong_attack_v2': 'coreset_far_wrong_attack',
    'bcs_coreset_shell_attack_v2': 'coreset_shell_attack',
    'bcs_typiclust_dense_attack_v2': 'typiclust_dense_attack',
    'bcs_kmeans_center_attack_v2': 'kmeans_center_attack',
    'bcs_uncertainty_badge_attack_v2': 'uncertainty_badge_attack',
    'bcs_cosine_attack_v2': 'cosine_attack',
    'bcs_hybrid_selector_attack_v2': 'hybrid_selector_attack',
}

TARGETED_SELECTOR_ATTACK_MODES = {
    'coreset_far_noise_attack',
    'coreset_far_wrong_attack',
    'coreset_shell_attack',
    'typiclust_dense_attack',
    'kmeans_center_attack',
    'uncertainty_badge_attack',
    'cosine_attack',
    'hybrid_selector_attack',
}

BASE_SPARSE_MARKET_PROFILE_NAMES = [
    'bcs_sparse_hard_conflict_v2',
    'bcs_sparse_decision_boundary_v2',
    'bcs_sparse_adversarial_mixed_v2',
]


def slug_number(value: float) -> str:
    return ("%.4g" % float(value)).replace("-", "m").replace(".", "p")


def scaled_sparse_profiles(
    good_ratio: float,
    prefix: str = "",
    malicious_mix: str = "none",
) -> Dict[str, Dict[str, float]]:
    good = min(max(float(good_ratio), 0.0), 0.95)
    prefix = str(prefix or "").strip()
    if prefix and not prefix.endswith("_"):
        prefix += "_"
    mix = str(malicious_mix or "none").lower()
    malicious_negative_mixes: Dict[str, Dict[str, float]] = {
        "adversarial_heavy": {
            "adversarial_paraphrase": 0.60,
            "high_sim_conflict": 0.25,
            "decision_wrong_detail": 0.15,
        },
        "conflict_heavy": {
            "high_sim_conflict": 0.60,
            "decision_wrong_detail": 0.25,
            "adversarial_paraphrase": 0.15,
        },
        "decision_heavy": {
            "decision_wrong_detail": 0.60,
            "high_sim_conflict": 0.25,
            "adversarial_paraphrase": 0.15,
        },
        "balanced_hard": {
            "adversarial_paraphrase": 1.0 / 3.0,
            "high_sim_conflict": 1.0 / 3.0,
            "decision_wrong_detail": 1.0 / 3.0,
        },
    }
    out: Dict[str, Dict[str, float]] = {}
    for base_name in BASE_SPARSE_MARKET_PROFILE_NAMES:
        base_weights = SPARSE_MARKET_PROFILES[base_name]
        if mix in malicious_negative_mixes:
            neg_weights = malicious_negative_mixes[mix]
        else:
            neg_weights = {k: v for k, v in base_weights.items() if k != "useful_good"}
        neg_weights = normalize_weights(neg_weights)
        profile = {"useful_good": good}
        for comp, weight in neg_weights.items():
            profile[comp] = (1.0 - good) * float(weight)
        out[prefix + base_name] = profile
    return out


def get_sparse_market_profiles(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    if bool(getattr(args, "use_dynamic_market_profiles", False)):
        return scaled_sparse_profiles(
            float(getattr(args, "market_good_ratio_override", 0.10)),
            prefix=str(getattr(args, "dynamic_profile_prefix", "dynamic")),
            malicious_mix=str(getattr(args, "market_malicious_mix", "none")),
        )
    return SPARSE_MARKET_PROFILES

COMPONENT_ALIASES: Dict[str, List[str]] = {
    'useful_good': ['useful_good', 'good', 'oracle_good', 'clean_good', 'aligned_good', 'true_good', 'helpful_good', 'matched_good'],
    'high_sim_conflict': ['high_sim_conflict', 'conflict_high_sim', 'high_similarity_conflict', 'label_conflict', 'label_mismatch', 'high_sim_label_conflict'],
    'decision_wrong_detail': ['decision_wrong_detail', 'decision_wrong', 'wrong_detail', 'decision_conflict', 'wrong_decision_detail'],
    'adversarial_paraphrase': ['adversarial_paraphrase', 'adv_paraphrase', 'paraphrase_adv', 'adversarial_text', 'adversarial'],
    'pure_noise': ['pure_noise', 'noise', 'random_noise', 'ood', 'irrelevant', 'irrelevant_text'],
    'duplicate': ['duplicate', 'dup', 'near_duplicate', 'repeated', 'copy'],
}

KEY_ALIASES: Dict[str, List[str]] = {
    'img': ['img', 'image', 'image_feat', 'img_feat', 'x_img', 'vision'],
    'weak': ['weak', 'weak_txt', 'weak_text', 'base_txt', 'buyer_txt', 'noisy_txt', 'local_txt'],
    'cand': ['cand', 'candidate', 'candidate_txt', 'seller_txt', 'txt', 'text', 'text_feat', 'x_txt'],
    'y': ['y', 'label', 'labels', 'target'],
    'is_good': ['is_good', 'good', 'is_useful', 'useful', 'oracle_good'],
    'row_type': ['row_type', 'type', 'component', 'source_type', 'category'],
}


def normalize_weights(d: Dict[str, float]) -> Dict[str, float]:
    s = float(sum(max(0.0, v) for v in d.values()))
    if s <= 0:
        raise ValueError('Profile weights sum to zero.')
    return {k: float(max(0.0, v) / s) for k, v in d.items()}


def allocate_counts(weights: Dict[str, float], total: int) -> Dict[str, int]:
    weights = normalize_weights(weights)
    raw = {k: weights[k] * int(total) for k in weights}
    counts = {k: int(math.floor(v)) for k, v in raw.items()}
    remain = int(total) - sum(counts.values())
    frac_order = sorted(raw.keys(), key=lambda k: raw[k] - math.floor(raw[k]), reverse=True)
    for k in frac_order[:remain]:
        counts[k] += 1
    return counts


def find_first_key(raw: Dict[str, np.ndarray], aliases: List[str]) -> str | None:
    for a in aliases:
        if a in raw:
            return a
    lower_map = {str(k).lower(): k for k in raw.keys()}
    for a in aliases:
        if a.lower() in lower_map:
            return lower_map[a.lower()]
    return None


def standardize_component_rows(raw: Dict[str, np.ndarray], component_name: str) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for canonical in ['img', 'weak', 'cand', 'y']:
        key = find_first_key(raw, KEY_ALIASES[canonical])
        if key is None:
            raise KeyError(f"Component {component_name} missing key {canonical}. Available keys: {list(raw.keys())}")
        out[canonical] = raw[key]
    out['img'] = np.asarray(out['img'], dtype=np.float32)
    out['weak'] = np.asarray(out['weak'], dtype=np.float32)
    out['cand'] = np.asarray(out['cand'], dtype=np.float32)
    out['y'] = np.asarray(out['y'], dtype=np.int64).reshape(-1)
    n = len(out['y'])
    if len(out['img']) != n or len(out['weak']) != n or len(out['cand']) != n:
        raise ValueError(f"Length mismatch in component {component_name}")
    out['is_good'] = np.full((n,), 1 if component_name == 'useful_good' else 0, dtype=np.int64)
    out['row_type'] = np.array([component_name] * n, dtype=object)
    return out


def discover_component_files(source_markets_dir: Path) -> Dict[str, Path]:
    search_dirs = [source_markets_dir / 'components', source_markets_dir / 'assembled', source_markets_dir]
    files: Dict[str, Path] = {}
    for d in search_dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob('*.npz')):
            files.setdefault(p.stem, p)
    return files


def resolve_component_name(component_key: str, files: Dict[str, Path]) -> Tuple[str, Path]:
    candidates = COMPONENT_ALIASES.get(component_key, [component_key])
    lower_to_name = {k.lower(): k for k in files.keys()}
    for c in candidates:
        if c in files:
            return c, files[c]
        if c.lower() in lower_to_name:
            real = lower_to_name[c.lower()]
            return real, files[real]
    matches: List[str] = []
    for real in files:
        rl = real.lower()
        for c in candidates:
            cl = c.lower()
            if cl in rl or rl in cl:
                matches.append(real)
                break
    matches = sorted(set(matches))
    if len(matches) == 1:
        real = matches[0]
        return real, files[real]
    raise FileNotFoundError(f"Cannot resolve component {component_key}. Available npz: {sorted(files.keys())}")


def load_component_bank(
    source_markets_dir: Path,
    allow_missing_components: bool,
    profiles: Dict[str, Dict[str, float]] | None = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    files = discover_component_files(source_markets_dir)
    if not files:
        raise FileNotFoundError(f'No .npz files found under {source_markets_dir}')
    bank: Dict[str, Dict[str, np.ndarray]] = {}
    profile_source = profiles if profiles is not None else SPARSE_MARKET_PROFILES
    needed = sorted({k for profile in profile_source.values() for k in profile.keys()})
    for comp_key in needed:
        try:
            _, path = resolve_component_name(comp_key, files)
            raw = {k: v for k, v in np.load(path, allow_pickle=True).items()}
            bank[comp_key] = standardize_component_rows(raw, comp_key)
            print(f"[component] {comp_key:24s} <- {path} n={len(bank[comp_key]['y'])}")
        except Exception as e:
            if not allow_missing_components:
                raise
            print('[warn] missing component', comp_key, repr(e))
    if 'useful_good' not in bank:
        raise FileNotFoundError('Missing useful_good component.')
    return bank


def sample_component(rows: Dict[str, np.ndarray], n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    total = len(rows['y'])
    replace = n > total
    idx = rng.choice(total, size=int(n), replace=replace)
    return {k: np.asarray(v)[idx] for k, v in rows.items()}


def take_component_rows(rows: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
    idx = np.asarray(idx, dtype=np.int64)
    return {k: np.asarray(v)[idx] for k, v in rows.items()}


def topk_indices(score: np.ndarray, k: int, largest: bool = True) -> np.ndarray:
    score = np.asarray(score, dtype=np.float32)
    n = len(score)
    if n == 0 or int(k) <= 0:
        return np.zeros((0,), dtype=np.int64)
    k = min(int(k), n)
    rank_score = -score if largest else score
    part = np.argpartition(rank_score, k - 1)[:k]
    part = part[np.argsort(rank_score[part])]
    return part.astype(np.int64)


def build_market_sampling_context(args: argparse.Namespace) -> Dict[str, np.ndarray]:
    base_dir = Path(args.base_dir)
    train_txt = np.load(base_dir / 'train_txt.npy').astype(np.float32)
    feat_dim = int(train_txt.shape[1])
    proj_dim = min(32, feat_dim)
    rng = np.random.default_rng(int(getattr(args, 'market_seed', 42)) + 52013)
    proj = (rng.normal(0.0, 1.0 / math.sqrt(max(1, feat_dim)), size=(feat_dim, proj_dim))).astype(np.float32)
    train_proj = l2_normalize_np(train_txt @ proj)
    n_clusters = min(64, max(2, len(train_proj) // 64), len(train_proj))
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=int(getattr(args, 'market_seed', 42)) + 52017,
        batch_size=min(2048, max(256, len(train_proj))),
        n_init=3,
        max_iter=80,
    )
    km.fit(train_proj.astype(np.float32))
    return {
        'projection': proj,
        'train_centers': l2_normalize_np(km.cluster_centers_.astype(np.float32)),
    }


def project_sampling_features(x: np.ndarray, context: Dict[str, np.ndarray]) -> np.ndarray:
    return l2_normalize_np(np.asarray(x, dtype=np.float32) @ context['projection'])


def novelty_to_train_centers(cand: np.ndarray, context: Dict[str, np.ndarray]) -> np.ndarray:
    feat = project_sampling_features(cand, context)
    centers = np.asarray(context['train_centers'], dtype=np.float32)
    best = np.full((len(feat),), np.inf, dtype=np.float32)
    for start in range(0, len(feat), 8192):
        chunk = feat[start:start + 8192]
        dist = np.linalg.norm(chunk[:, None, :] - centers[None, :, :], axis=2)
        best[start:start + len(chunk)] = np.min(dist, axis=1).astype(np.float32)
    return best.astype(np.float32)


def component_density_score(cand: np.ndarray, context: Dict[str, np.ndarray], k: int = 8) -> np.ndarray:
    feat = project_sampling_features(cand, context)
    n = int(len(feat))
    if n <= 1:
        return np.zeros((n,), dtype=np.float32)
    nn_k = min(max(2, int(k) + 1), n)
    nn = NearestNeighbors(n_neighbors=nn_k, metric='euclidean', algorithm='auto')
    nn.fit(feat.astype(np.float32))
    dist, _ = nn.kneighbors(feat.astype(np.float32), return_distance=True)
    return (-np.mean(dist[:, 1:], axis=1)).astype(np.float32)


def component_center_score(cand: np.ndarray, context: Dict[str, np.ndarray], seed: int) -> np.ndarray:
    feat = project_sampling_features(cand, context)
    n = int(len(feat))
    if n <= 1:
        return np.zeros((n,), dtype=np.float32)
    n_clusters = min(max(2, n // 256), 128, n)
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=int(seed),
        batch_size=min(4096, max(256, n)),
        n_init=3,
        max_iter=80,
    )
    labels = km.fit_predict(feat.astype(np.float32))
    centers = km.cluster_centers_.astype(np.float32)
    dist = np.linalg.norm(feat - centers[labels], axis=1).astype(np.float32)
    counts = np.bincount(labels, minlength=n_clusters).astype(np.float32)
    cluster_mass = np.log1p(counts[labels]).astype(np.float32)
    return (-dist + 0.05 * cluster_mass).astype(np.float32)


def component_structural_prior_for_sampling(rows: Dict[str, np.ndarray], args: argparse.Namespace) -> np.ndarray:
    img = np.asarray(rows['img'], dtype=np.float32)
    weak = np.asarray(rows['weak'], dtype=np.float32)
    cand = np.asarray(rows['cand'], dtype=np.float32)
    cos_img_cand = np.sum(img * cand, axis=1).astype(np.float32)
    cos_img_weak = np.sum(img * weak, axis=1).astype(np.float32)
    cos_cand_weak = np.sum(cand * weak, axis=1).astype(np.float32)
    return compute_structural_good_prior_from_cos(cos_img_cand, cos_img_weak, cos_cand_weak, args)


def compact_good_indices(
    rows: Dict[str, np.ndarray],
    n: int,
    rng: np.random.Generator,
    args: argparse.Namespace,
    context: Dict[str, np.ndarray],
    novelty_weight: float,
) -> np.ndarray:
    total = len(rows['y'])
    if int(n) >= total:
        return np.arange(total, dtype=np.int64)
    structural = component_structural_prior_for_sampling(rows, args)
    novelty = novelty_to_train_centers(rows['cand'], context)
    score = zscore_np(structural, float(getattr(args, 'structural_prior_clip', 4.0))) - float(novelty_weight) * zscore_np(novelty, 4.0)
    pool_size = min(total, max(int(n) * 8, int(n) + 1024, 2048))
    pool = topk_indices(score, pool_size, largest=True)
    if len(pool) <= int(n):
        return pool.astype(np.int64)

    feat = project_sampling_features(rows['cand'][pool], context)
    cluster_k = min(8, max(2, len(pool) // max(int(n), 1)), len(pool))
    try:
        km = MiniBatchKMeans(
            n_clusters=cluster_k,
            random_state=int(rng.integers(0, 2**31 - 1)),
            batch_size=min(2048, max(256, len(pool))),
            n_init=3,
            max_iter=80,
        )
        labels = km.fit_predict(feat.astype(np.float32))
        best_cluster = None
        best_value = -float('inf')
        for cid in np.unique(labels).tolist():
            local = np.flatnonzero(labels == cid)
            if len(local) == 0:
                continue
            radius = cluster_radius(feat[local])
            value = float(np.mean(score[pool[local]])) - 0.25 * float(radius) + 0.02 * math.log1p(len(local))
            if value > best_value:
                best_value = value
                best_cluster = local
        chosen_pool = pool[best_cluster] if best_cluster is not None else pool
        chosen_pool = chosen_pool[np.argsort(-score[chosen_pool])]
        if len(chosen_pool) >= int(n):
            return chosen_pool[: int(n)].astype(np.int64)
        chosen_set = set(chosen_pool.astype(int).tolist())
        rest = np.array([int(i) for i in pool.tolist() if int(i) not in chosen_set], dtype=np.int64)
        rest = rest[np.argsort(-score[rest])]
        return np.concatenate([chosen_pool, rest[: int(n) - len(chosen_pool)]], axis=0).astype(np.int64)
    except Exception:
        return pool[np.argsort(-score[pool])][: int(n)].astype(np.int64)


def diverse_bad_indices(
    rows: Dict[str, np.ndarray],
    n: int,
    rng: np.random.Generator,
    args: argparse.Namespace,
    context: Dict[str, np.ndarray],
    mode: str,
) -> np.ndarray:
    total = len(rows['y'])
    if int(n) >= total:
        return np.arange(total, dtype=np.int64)

    structural = component_structural_prior_for_sampling(rows, args)
    novelty = novelty_to_train_centers(rows['cand'], context)
    img = np.asarray(rows['img'], dtype=np.float32)
    weak = np.asarray(rows['weak'], dtype=np.float32)
    cand = np.asarray(rows['cand'], dtype=np.float32)
    cos_img_cand = np.sum(img * cand, axis=1).astype(np.float32)
    cos_cand_weak = np.sum(cand * weak, axis=1).astype(np.float32)
    boundary_uncertainty = (-np.abs(cos_img_cand - cos_cand_weak)).astype(np.float32)

    if mode == 'boundary_decoy':
        score = (
            1.20 * zscore_np(cos_img_cand, 4.0)
            + 0.40 * zscore_np(cos_cand_weak, 4.0)
            + 0.25 * zscore_np(novelty, 4.0)
            - 0.70 * zscore_np(structural, 4.0)
        )
    elif mode == 'package_trap':
        score = 0.90 * zscore_np(novelty, 4.0) - 1.10 * zscore_np(structural, 4.0)
    elif mode in ['coreset_far_noise_attack', 'coreset_far_wrong_attack', 'coreset_shell_attack']:
        score = 1.25 * zscore_np(novelty, 4.0) + 0.25 * zscore_np(cos_cand_weak, 4.0) - 0.90 * zscore_np(structural, 4.0)
    elif mode == 'typiclust_dense_attack':
        density = component_density_score(rows['cand'], context, k=8)
        score = 1.35 * zscore_np(density, 4.0) + 0.15 * zscore_np(cos_cand_weak, 4.0) - 0.95 * zscore_np(structural, 4.0)
    elif mode == 'kmeans_center_attack':
        center = component_center_score(rows['cand'], context, int(rng.integers(0, 2**31 - 1)))
        score = 1.25 * zscore_np(center, 4.0) + 0.20 * zscore_np(cos_cand_weak, 4.0) - 0.90 * zscore_np(structural, 4.0)
    elif mode == 'uncertainty_badge_attack':
        score = (
            1.20 * zscore_np(boundary_uncertainty, 4.0)
            + 0.45 * zscore_np(novelty, 4.0)
            + 0.20 * zscore_np(cos_img_cand, 4.0)
            - 0.95 * zscore_np(structural, 4.0)
        )
    elif mode == 'cosine_attack':
        score = 1.35 * zscore_np(cos_img_cand, 4.0) + 0.20 * zscore_np(cos_cand_weak, 4.0) - 1.00 * zscore_np(structural, 4.0)
    elif mode == 'hybrid_selector_attack':
        density = component_density_score(rows['cand'], context, k=8)
        center = component_center_score(rows['cand'], context, int(rng.integers(0, 2**31 - 1)))
        score = (
            0.55 * zscore_np(novelty, 4.0)
            + 0.45 * zscore_np(density, 4.0)
            + 0.35 * zscore_np(center, 4.0)
            + 0.35 * zscore_np(boundary_uncertainty, 4.0)
            + 0.40 * zscore_np(cos_img_cand, 4.0)
            - 1.10 * zscore_np(structural, 4.0)
        )
    else:
        score = 1.15 * zscore_np(novelty, 4.0) - 0.85 * zscore_np(structural, 4.0)

    pool_size = min(total, max(int(n) * 6, int(n) + 2048, 4096))
    pool = topk_indices(score, pool_size, largest=True)
    if len(pool) <= int(n):
        return pool.astype(np.int64)

    feat = project_sampling_features(rows['cand'][pool], context)
    cluster_k = min(len(pool), max(16, min(512, int(n))))
    chosen: List[int] = []
    try:
        km = MiniBatchKMeans(
            n_clusters=cluster_k,
            random_state=int(rng.integers(0, 2**31 - 1)),
            batch_size=min(4096, max(256, len(pool))),
            n_init=3,
            max_iter=80,
        )
        labels = km.fit_predict(feat.astype(np.float32))
        representatives: List[Tuple[float, int]] = []
        for cid in np.unique(labels).tolist():
            local = np.flatnonzero(labels == cid)
            if len(local) == 0:
                continue
            best_local = int(local[int(np.argmax(score[pool[local]]))])
            representatives.append((float(score[pool[best_local]]), int(pool[best_local])))
        representatives.sort(key=lambda x: (-x[0], x[1]))
        chosen = [idx for _, idx in representatives[: int(n)]]
    except Exception:
        chosen = []

    chosen_set = set(int(i) for i in chosen)
    if len(chosen) < int(n):
        for idx in pool[np.argsort(-score[pool])].tolist():
            ii = int(idx)
            if ii in chosen_set:
                continue
            chosen.append(ii)
            chosen_set.add(ii)
            if len(chosen) >= int(n):
                break
    return np.array(chosen[: int(n)], dtype=np.int64)


def strategic_sample_component(
    rows: Dict[str, np.ndarray],
    n: int,
    rng: np.random.Generator,
    comp: str,
    profile_name: str,
    args: argparse.Namespace,
    context: Dict[str, np.ndarray] | None,
) -> Dict[str, np.ndarray]:
    mode = STRATEGIC_MARKET_PROFILE_MODES.get(str(profile_name), '')
    if not mode or context is None or bool(getattr(args, 'disable_strategic_market_sampling', False)):
        return sample_component(rows, n, rng)
    if comp == 'useful_good':
        novelty_weight = 0.20 if mode == 'coverage_trap' else 0.10
        idx = compact_good_indices(rows, int(n), rng, args, context, novelty_weight=novelty_weight)
    else:
        idx = diverse_bad_indices(rows, int(n), rng, args, context, mode=mode)
    return take_component_rows(rows, idx)


def concat_market_parts(parts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = ['img', 'weak', 'cand', 'y', 'is_good', 'row_type']
    out: Dict[str, np.ndarray] = {}
    for k in keys:
        if k == 'row_type':
            out[k] = np.concatenate([np.asarray(p[k], dtype=object) for p in parts], axis=0).astype(object)
        elif k in ['y', 'is_good']:
            out[k] = np.concatenate([np.asarray(p[k], dtype=np.int64).reshape(-1) for p in parts], axis=0).astype(np.int64)
        else:
            out[k] = np.concatenate([np.asarray(p[k], dtype=np.float32) for p in parts], axis=0).astype(np.float32)
    return out


def build_sparse_markets(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_markets_dir)
    out_dir = Path(args.markets_dir)
    out_assembled = out_dir / 'assembled'
    ensure_dir(out_assembled)
    profiles = get_sparse_market_profiles(args)
    bank = load_component_bank(source_dir, allow_missing_components=bool(args.allow_missing_components), profiles=profiles)
    use_strategic = (
        not bool(getattr(args, 'disable_strategic_market_sampling', False))
        and any(name in STRATEGIC_MARKET_PROFILE_MODES for name in profiles)
    )
    sampling_context = build_market_sampling_context(args) if use_strategic else None
    summaries = []
    for pi, (profile_name, raw_weights) in enumerate(profiles.items()):
        sampling_mode = STRATEGIC_MARKET_PROFILE_MODES.get(profile_name, 'random')
        weights = {k: v for k, v in raw_weights.items() if k in bank}
        missing = [k for k in raw_weights if k not in bank]
        if missing:
            if not args.allow_missing_components:
                raise FileNotFoundError(f'{profile_name} missing components {missing}')
            print(f'[warn] {profile_name} missing {missing}; redistributing weights over {list(weights.keys())}')
        weights = normalize_weights(weights)
        counts = allocate_counts(weights, int(args.market_size))
        rng = np.random.default_rng(int(args.market_seed) + 1009 * pi)
        parts = []
        for comp, n in counts.items():
            part = strategic_sample_component(
                bank[comp],
                int(n),
                rng,
                comp=comp,
                profile_name=profile_name,
                args=args,
                context=sampling_context,
            )
            row_type = comp
            if comp != 'useful_good' and sampling_mode in TARGETED_SELECTOR_ATTACK_MODES:
                row_type = sampling_mode + '_' + comp
            elif bool(getattr(args, 'mark_non_good_as_malicious', False)) and comp != 'useful_good':
                row_type = 'malicious_' + comp
            part['row_type'] = np.array([row_type] * len(part['y']), dtype=object)
            part['is_good'] = np.full((len(part['y']),), 1 if comp == 'useful_good' else 0, dtype=np.int64)
            parts.append(part)
        market = concat_market_parts(parts)
        order = rng.permutation(len(market['y']))
        market = {k: np.asarray(v)[order] for k, v in market.items()}
        path = out_assembled / f'{profile_name}.npz'
        np.savez_compressed(path, **market)
        print('[market built]', path, 'n=', len(market['y']), 'good_ratio=', float(np.mean(market['is_good'])))
        rt = np.asarray(market['row_type'], dtype=object)
        for comp in sorted(set(rt.tolist())):
            mask = rt == comp
            summaries.append({
                'market_profile': profile_name,
                'sampling_mode': sampling_mode,
                'component': comp,
                'count': int(np.sum(mask)),
                'ratio': float(np.mean(mask)),
                'good_ratio': float(np.mean(market['is_good'][mask])) if np.any(mask) else np.nan,
                'label_1_ratio': float(np.mean(market['y'][mask])) if np.any(mask) else np.nan,
            })
    pd.DataFrame(summaries).to_csv(out_dir / 'market_build_summary.csv', index=False)
    (out_dir / 'market_build_profiles.json').write_text(json.dumps(profiles, indent=2), encoding='utf-8')




def required_market_paths(args: argparse.Namespace) -> List[Path]:
    root = Path(args.markets_dir)
    return [root / 'assembled' / (str(name) + '.npz') for name in args.market_profiles]


def ensure_sparse_markets_exist(args: argparse.Namespace) -> None:
    missing = [p for p in required_market_paths(args) if not p.exists()]
    if not missing:
        return

    print('[market check] missing assembled markets:')
    for p in missing:
        print('  ' + str(p.resolve()))
    print('[market check] build_sparse_markets will be called automatically.')
    print('[market check] source_markets_dir = ' + str(Path(args.source_markets_dir).resolve()))
    print('[market check] markets_dir = ' + str(Path(args.markets_dir).resolve()))

    build_sparse_markets(args)

    still_missing = [p for p in required_market_paths(args) if not p.exists()]
    if still_missing:
        msg = [
            'Sparse market construction finished, but required market files are still missing.',
            'This usually means the source component directory does not contain the required components.',
            'source_markets_dir = ' + str(Path(args.source_markets_dir).resolve()),
            'markets_dir = ' + str(Path(args.markets_dir).resolve()),
            'missing files:',
        ]
        msg.extend(['  ' + str(p.resolve()) for p in still_missing])
        raise FileNotFoundError('\n'.join(msg))


# =========================================================
# Models
# =========================================================


class PairClassifier(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([img, txt, img * txt, torch.abs(img - txt)], dim=1)
        return self.net(feat).squeeze(1)


class TestOptCosCatModel(nn.Module):
    def __init__(self, num_classes: int = 512, combined_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.query_dim = int(num_classes)
        self.key_value_dim = int(num_classes)
        self.query_layer = nn.Linear(self.query_dim, self.query_dim)
        self.key_layer = nn.Linear(self.key_value_dim, self.key_value_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.output_fc = nn.Linear(self.query_dim * 2, self.query_dim)
        self.output_fc_2 = nn.Linear(self.query_dim, int(combined_dim))
        self.output_fc_3 = nn.Linear(int(combined_dim), 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, img_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        img_map = self.query_layer(img_features)
        text_map = self.key_layer(text_features)
        query = self.dropout(img_map)
        key = self.dropout(text_map)
        query_norm = torch.norm(query, p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        key_norm = torch.norm(key, p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        denom = (query_norm * key_norm).clamp_min(1e-8)
        img_kv_ = torch.einsum("nm,nm->n", key, img_features).unsqueeze(1)
        text_qv_ = torch.einsum("nm,nm->n", query, img_features).unsqueeze(1)
        img = query * img_kv_ / denom
        text = key * text_qv_ / denom
        concat = torch.cat((img, text), dim=-1)
        concat = concat * concat
        output = self.output_fc(concat)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_2(output)
        output = self.dropout(output)
        output = self.gelu(output)
        output = self.output_fc_3(output)
        output = self.dropout(output)
        return output.squeeze(1)


def build_pair_model(feat_dim: int, args: argparse.Namespace) -> nn.Module:
    pair_model = str(getattr(args, 'pair_model', 'bopa')).lower()
    if pair_model == 'mlp':
        return PairClassifier(feat_dim=feat_dim, hidden_dim=int(args.hidden_dim), dropout=float(args.dropout))
    if pair_model == 'bopa':
        return TestOptCosCatModel(
            num_classes=int(feat_dim),
            combined_dim=int(getattr(args, 'bopa_combined_dim', 64)),
            dropout=float(args.dropout),
        )
    raise ValueError('Unknown pair_model: ' + str(pair_model))


class RowStudent(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32, dropout: float = 0.05):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.base_margin_head = nn.Linear(hidden_dim, 1)
        self.cand_margin_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.base_margin_head(h).squeeze(1), self.cand_margin_head(h).squeeze(1)


class PackageStudent(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32, dropout: float = 0.05):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.base_margin_head = nn.Linear(hidden_dim, 1)
        self.cand_margin_head = nn.Linear(hidden_dim, 1)
        self.final_score_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return (
            self.base_margin_head(h).squeeze(1),
            self.cand_margin_head(h).squeeze(1),
            self.final_score_head(h).squeeze(1),
        )


@dataclass
class RowStudentTrainPack:
    features: np.ndarray
    base_margin_target: np.ndarray
    cand_margin_target: np.ndarray
    final_target: np.ndarray
    indices: np.ndarray


@dataclass
class PackageBundle:
    features: np.ndarray
    size: np.ndarray
    radius: np.ndarray
    row_ptr: np.ndarray
    teacher_base_margin: np.ndarray
    teacher_cand_margin: np.ndarray
    teacher_final_target: np.ndarray
    cos_raw: np.ndarray
    structural_prior: np.ndarray
    good_ratio: np.ndarray
    package_method: str


# =========================================================
# Training / evaluation helpers
# =========================================================


@torch.no_grad()
def predict_pair_logits(model: nn.Module, img: np.ndarray, txt: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    if len(img) == 0:
        return np.zeros((0,), dtype=np.float32)
    ds = TensorDataset(torch.from_numpy(img.astype(np.float32)), torch.from_numpy(txt.astype(np.float32)))
    dl = DataLoader(ds, batch_size=min(max(16, int(batch_size)), len(ds)), shuffle=False)
    model.eval()
    device_t = torch.device(device)
    outs: List[np.ndarray] = []
    for bi, bt in dl:
        pred = model(bi.to(device_t), bt.to(device_t))
        outs.append(pred.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0).astype(np.float32)


def make_train_val_split(y_int: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(y_int), dtype=np.int64)
    if len(idx) < 4:
        return idx, idx
    vals, counts = np.unique(y_int, return_counts=True)
    strat = y_int if len(vals) > 1 and np.min(counts) >= 2 else None
    try:
        tr_idx, va_idx = train_test_split(idx, test_size=float(val_ratio), random_state=seed, stratify=strat)
    except ValueError:
        tr_idx, va_idx = train_test_split(idx, test_size=float(val_ratio), random_state=seed, stratify=None)
    return tr_idx.astype(np.int64), va_idx.astype(np.int64)


def train_pair_model(
    pack: Dict[str, np.ndarray],
    args: argparse.Namespace,
    seed: int,
    init_state: Dict[str, torch.Tensor] | None = None,
    max_epochs: int | None = None,
    patience: int | None = None,
) -> Tuple[nn.Module, Dict[str, torch.Tensor], float]:
    set_seed(seed)
    img = pack['img'].astype(np.float32)
    txt = pack['txt'].astype(np.float32)
    if img.shape[1] != txt.shape[1]:
        raise ValueError('Pair model expects image/text features with the same dimension.')
    y_float = pack['y'].astype(np.float32)
    y_int = pack['y'].astype(np.int64)
    tr_idx, va_idx = make_train_val_split(y_int, float(args.val_ratio), seed)
    ds_tr = TensorDataset(torch.from_numpy(img[tr_idx]), torch.from_numpy(txt[tr_idx]), torch.from_numpy(y_float[tr_idx]))
    dl_tr = DataLoader(ds_tr, batch_size=min(max(16, int(args.batch_size)), len(tr_idx)), shuffle=True)
    model = build_pair_model(feat_dim=img.shape[1], args=args).to(torch.device(args.device))
    if init_state is not None:
        model.load_state_dict(init_state, strict=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    pos_weight = None
    if bool(getattr(args, 'class_balanced_teacher_loss', False)):
        pos = float(np.sum(y_float[tr_idx] > 0.5))
        neg = float(len(tr_idx) - pos)
        if pos > 0 and neg > 0:
            pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=torch.device(args.device))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_score = -1e18
    best_state = None
    bad = 0
    use_epochs = int(args.init_max_epochs if max_epochs is None else max_epochs)
    use_patience = int(args.init_patience if patience is None else patience)
    device_t = torch.device(args.device)
    for _ in range(use_epochs):
        model.train()
        for bi, bt, by in dl_tr:
            bi = bi.to(device_t)
            bt = bt.to(device_t)
            by = by.to(device_t)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(bi, bt), by)
            loss.backward()
            opt.step()
        va_logits = predict_pair_logits(model, img[va_idx], txt[va_idx], args.device, args.batch_size)
        va_prob = sigmoid_np(va_logits)
        score = safe_auroc(y_int[va_idx], va_prob)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= use_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return model, final_state, float(best_score)


def evaluate_pair_model(model: nn.Module, test_pack: Dict[str, np.ndarray], args: argparse.Namespace) -> Dict[str, float]:
    logits = predict_pair_logits(model, test_pack['img'], test_pack['txt'], args.device, args.batch_size)
    prob = sigmoid_np(logits)
    y = test_pack['y'].astype(np.int64)
    pred = (prob >= 0.5).astype(np.int64)
    return {
        'test_auroc': safe_auroc(y, prob),
        'test_macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
        'test_acc': float(accuracy_score(y, pred)),
    }


def train_row_student(pack: RowStudentTrainPack, args: argparse.Namespace, seed: int) -> RowStudent:
    set_seed(seed)
    device_t = torch.device(args.device)
    features = torch.from_numpy(pack.features.astype(np.float32)).to(device_t)
    m0_t = torch.from_numpy(pack.base_margin_target.astype(np.float32)).to(device_t)
    mc_t = torch.from_numpy(pack.cand_margin_target.astype(np.float32)).to(device_t)
    final_t = torch.from_numpy(pack.final_target.astype(np.float32)).to(device_t)
    model = RowStudent(in_dim=pack.features.shape[1], hidden_dim=int(args.student_hidden_dim), dropout=float(args.student_dropout)).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.student_lr), weight_decay=float(args.student_weight_decay))
    n = features.shape[0]
    rng = np.random.default_rng(seed + 17)
    for _ in range(int(args.student_epochs)):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred_m0, pred_mc = model(features)
        base_score = pred_mc - pred_m0 - float(args.neg_margin_lambda) * F.relu(-pred_mc)
        loss_m0 = F.mse_loss(pred_m0, m0_t)
        loss_mc = F.mse_loss(pred_mc, mc_t)
        num_pairs = min(int(args.student_rank_pairs), max(0, n // 2))
        rank_loss = torch.tensor(0.0, device=device_t)
        if num_pairs > 0:
            i_idx = rng.integers(0, n, size=(num_pairs,), dtype=np.int64)
            j_idx = rng.integers(0, n, size=(num_pairs,), dtype=np.int64)
            i_t = torch.from_numpy(i_idx).to(device_t)
            j_t = torch.from_numpy(j_idx).to(device_t)
            target_diff = final_t[i_t] - final_t[j_t]
            valid = torch.abs(target_diff) > float(args.rank_eps)
            if torch.any(valid):
                sign = torch.sign(target_diff[valid])
                pred_diff = base_score[i_t][valid] - base_score[j_t][valid]
                rank_loss = torch.mean(F.relu(float(args.rank_margin) - sign * pred_diff))
        loss = loss_m0 + float(args.student_margin_weight) * loss_mc + float(args.student_rank_weight) * rank_loss
        loss.backward()
        opt.step()
    return model


def train_package_student(pack: RowStudentTrainPack, args: argparse.Namespace, seed: int) -> PackageStudent:
    set_seed(seed)
    device_t = torch.device(args.device)
    features = torch.from_numpy(pack.features.astype(np.float32)).to(device_t)
    m0_t = torch.from_numpy(pack.base_margin_target.astype(np.float32)).to(device_t)
    mc_t = torch.from_numpy(pack.cand_margin_target.astype(np.float32)).to(device_t)
    final_t = torch.from_numpy(pack.final_target.astype(np.float32)).to(device_t)
    model = PackageStudent(in_dim=pack.features.shape[1], hidden_dim=int(args.student_hidden_dim), dropout=float(args.student_dropout)).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.student_lr), weight_decay=float(args.student_weight_decay))
    n = features.shape[0]
    rng = np.random.default_rng(seed + 37)
    for _ in range(int(args.student_epochs)):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred_m0, pred_mc, pred_final = model(features)
        base_score = pred_mc - pred_m0 - float(args.neg_margin_lambda) * F.relu(-pred_mc)
        score_for_rank = pred_final if str(getattr(args, 'package_student_score_mode', 'margin')) == 'final' else base_score
        loss_m0 = F.mse_loss(pred_m0, m0_t)
        loss_mc = F.mse_loss(pred_mc, mc_t)
        loss_final = F.mse_loss(pred_final, final_t)
        num_pairs = min(int(args.student_rank_pairs), max(0, n // 2))
        rank_loss = torch.tensor(0.0, device=device_t)
        if num_pairs > 0:
            i_idx = rng.integers(0, n, size=(num_pairs,), dtype=np.int64)
            j_idx = rng.integers(0, n, size=(num_pairs,), dtype=np.int64)
            i_t = torch.from_numpy(i_idx).to(device_t)
            j_t = torch.from_numpy(j_idx).to(device_t)
            target_diff = final_t[i_t] - final_t[j_t]
            valid = torch.abs(target_diff) > float(args.rank_eps)
            if torch.any(valid):
                sign = torch.sign(target_diff[valid])
                pred_diff = score_for_rank[i_t][valid] - score_for_rank[j_t][valid]
                rank_loss = torch.mean(F.relu(float(args.rank_margin) - sign * pred_diff))
        loss = (
            float(getattr(args, 'package_student_base_weight', 1.0)) * loss_m0
            + float(args.student_margin_weight) * loss_mc
            + float(getattr(args, 'package_student_final_weight', 0.0)) * loss_final
            + float(args.student_rank_weight) * rank_loss
        )
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def predict_row_student(model: RowStudent, features: np.ndarray, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    x = torch.from_numpy(features.astype(np.float32)).to(torch.device(args.device))
    model.eval()
    pred_m0, pred_mc = model(x)
    delta = pred_mc - pred_m0
    score = delta - float(args.neg_margin_lambda) * F.relu(-pred_mc)
    return {
        'base_margin': pred_m0.detach().cpu().numpy().astype(np.float32),
        'cand_margin': pred_mc.detach().cpu().numpy().astype(np.float32),
        'delta': delta.detach().cpu().numpy().astype(np.float32),
        'score': score.detach().cpu().numpy().astype(np.float32),
    }


@torch.no_grad()
def predict_package_student(model: PackageStudent, features: np.ndarray, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    x = torch.from_numpy(features.astype(np.float32)).to(torch.device(args.device))
    model.eval()
    pred_m0, pred_mc, pred_final = model(x)
    delta = pred_mc - pred_m0
    score = delta - float(args.neg_margin_lambda) * F.relu(-pred_mc)
    return {
        'base_margin': pred_m0.detach().cpu().numpy().astype(np.float32),
        'cand_margin': pred_mc.detach().cpu().numpy().astype(np.float32),
        'final_score': pred_final.detach().cpu().numpy().astype(np.float32),
        'delta': delta.detach().cpu().numpy().astype(np.float32),
        'score': score.detach().cpu().numpy().astype(np.float32),
    }


# =========================================================
# Feature and package helpers
# =========================================================


def degrade_local_text(
    txt: np.ndarray,
    y: np.ndarray,
    seed: int,
    strength: float = 0.9,
    wrong_swap_prob: float = 0.6,
) -> np.ndarray:
    rng = np.random.default_rng(seed + 91)
    out = txt.copy().astype(np.float32)
    labels = y.astype(np.int64)
    global_mean = l2_normalize_np(out.mean(axis=0).astype(np.float32))
    for i in range(len(out)):
        same_idx = np.where(labels == labels[i])[0]
        wrong_idx = np.where(labels != labels[i])[0]
        if len(wrong_idx) > 0 and float(rng.random()) < float(wrong_swap_prob):
            base = out[int(rng.choice(wrong_idx))]
        elif len(same_idx) > 0:
            base = out[int(rng.choice(same_idx))]
        else:
            base = global_mean
        out[i] = (base + rng.normal(0.0, strength * 0.03, size=(out.shape[1],)).astype(np.float32)).astype(np.float32)
    return l2_normalize_np(out)


def degrade_market_style_text(
    txt: np.ndarray,
    y: np.ndarray,
    seed: int,
    noise: float = 0.03,
    wrong_swap_prob: float = 0.65,
) -> np.ndarray:
    """Match the weak-text construction used by the scalability market builders."""
    rng = np.random.default_rng(seed + 91)
    txt = np.asarray(txt, dtype=np.float32)
    labels = np.asarray(y, dtype=np.int64)
    out = np.zeros_like(txt, dtype=np.float32)
    global_mean = l2_normalize_np(txt.mean(axis=0).astype(np.float32))
    by_label = {
        int(lbl): np.flatnonzero(labels == int(lbl)).astype(np.int64)
        for lbl in np.unique(labels).tolist()
    }
    all_idx = np.arange(len(labels), dtype=np.int64)
    for i in range(len(txt)):
        lbl = int(labels[i])
        same_idx = by_label.get(lbl, all_idx)
        wrong_idx = all_idx[labels != lbl]
        if len(wrong_idx) > 0 and float(rng.random()) < float(wrong_swap_prob):
            base = txt[int(rng.choice(wrong_idx))]
        elif len(same_idx) > 0:
            base = txt[int(rng.choice(same_idx))]
        else:
            base = global_mean
        out[i] = base + rng.normal(0.0, float(noise), size=(txt.shape[1],)).astype(np.float32)
    return l2_normalize_np(out.astype(np.float32))


def add_local_text_noise(txt: np.ndarray, seed: int, strength: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(seed + 1091)
    x = txt.copy().astype(np.float32)
    noise = rng.normal(0.0, float(strength) * 0.03, size=x.shape).astype(np.float32)
    return l2_normalize_np((x + noise).astype(np.float32))


def make_projection_matrix(feat_dim: int, lowdim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1003)
    return rng.normal(0.0, 1.0 / math.sqrt(lowdim), size=(feat_dim, lowdim)).astype(np.float32)


def project_features(x: np.ndarray, proj: np.ndarray) -> np.ndarray:
    return l2_normalize_np((x.astype(np.float32) @ proj.astype(np.float32)).astype(np.float32))


def build_row_student_features(u: np.ndarray, v: np.ndarray, w: np.ndarray, ysign: np.ndarray) -> np.ndarray:
    cos_uv = np.sum(u * v, axis=1, keepdims=True).astype(np.float32)
    cos_uw = np.sum(u * w, axis=1, keepdims=True).astype(np.float32)
    cos_vw = np.sum(v * w, axis=1, keepdims=True).astype(np.float32)
    return np.concatenate([
        u.astype(np.float32),
        v.astype(np.float32),
        (u * v).astype(np.float32),
        np.abs(u - v).astype(np.float32),
        (v - w).astype(np.float32),
        (u * w).astype(np.float32),
        cos_uv,
        cos_uw,
        cos_vw,
        ysign.reshape(-1, 1).astype(np.float32),
    ], axis=1).astype(np.float32)


def compute_structural_good_prior_from_cos(
    cos_img_cand: np.ndarray,
    cos_img_weak: np.ndarray,
    cos_cand_weak: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """Feature-only prior for sparse-good markets.

    The hard negative components in these caches often look deceptively image
    similar. Useful rows tend to be less driven by that shortcut, so this prior
    rewards low image-candidate similarity with small penalties for weak/candidate
    redundancy and weak-to-candidate shortcut drift.
    """
    ic = np.asarray(cos_img_cand, dtype=np.float32)
    iw = np.asarray(cos_img_weak, dtype=np.float32)
    cw = np.asarray(cos_cand_weak, dtype=np.float32)
    if len(ic) == 0:
        return np.zeros((0,), dtype=np.float32)

    clip = float(getattr(args, 'structural_prior_clip', 4.0))
    mode = str(getattr(args, 'structural_prior_mode', 'hybrid_anti_sim')).lower()
    delta = (ic - iw).astype(np.float32)

    if mode == 'anti_image_sim':
        raw = -zscore_np(ic, clip)
    elif mode == 'hybrid_anti_sim':
        raw = -zscore_np(ic, clip) - 0.10 * zscore_np(cw, clip) - 0.10 * zscore_np(delta, clip)
    elif mode == 'low_triad':
        triad = (ic + cw - iw).astype(np.float32)
        raw = -zscore_np(triad, clip) - 0.50 * zscore_np(ic, clip)
    elif mode == 'dist_minus_image':
        # Embeddings are L2-normalized in the cache, so distance is determined by cosine.
        cand_weak_dist = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * cw)).astype(np.float32)
        raw = zscore_np(cand_weak_dist, clip) - zscore_np(ic, clip)
    else:
        raise ValueError('Unknown structural_prior_mode: ' + str(mode))

    return zscore_np(np.asarray(raw, dtype=np.float32), clip).astype(np.float32)


def compute_row_structural_good_prior(
    market_rows: Dict[str, np.ndarray],
    idx: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    if len(idx) == 0:
        return np.zeros((0,), dtype=np.float32)
    img = market_rows['img'][idx].astype(np.float32)
    weak = market_rows['weak'][idx].astype(np.float32)
    cand = market_rows['cand'][idx].astype(np.float32)
    cos_img_cand = np.sum(img * cand, axis=1).astype(np.float32)
    cos_img_weak = np.sum(img * weak, axis=1).astype(np.float32)
    cos_cand_weak = np.sum(cand * weak, axis=1).astype(np.float32)
    return compute_structural_good_prior_from_cos(cos_img_cand, cos_img_weak, cos_cand_weak, args)


def aggregate_package_structural_prior(row_prior: np.ndarray, args: argparse.Namespace) -> float:
    row_prior = np.asarray(row_prior, dtype=np.float32)
    if len(row_prior) == 0:
        return 0.0
    mode = str(getattr(args, 'package_structural_agg', 'topmean')).lower()
    if mode == 'mean':
        return float(np.mean(row_prior))
    if mode == 'max':
        return float(np.max(row_prior))
    if mode == 'topmean':
        frac = min(max(float(getattr(args, 'package_structural_top_frac', 0.5)), 1e-6), 1.0)
        k = max(1, int(math.ceil(len(row_prior) * frac)))
        return float(np.mean(np.sort(row_prior)[-k:]))
    raise ValueError('Unknown package_structural_agg: ' + str(mode))


def mean_top_fraction(values: np.ndarray, frac: float) -> float:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return 0.0
    frac = min(max(float(frac), 1e-6), 1.0)
    k = max(1, int(math.ceil(len(values) * frac)))
    return float(np.mean(np.sort(values)[-k:]))


def safe_pearson_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(mask)) < 2:
        return float('nan')
    a = a[mask]
    b = b[mask]
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std <= 1e-12 or b_std <= 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def safe_spearman_np(a: np.ndarray, b: np.ndarray) -> float:
    a_rank = pd.Series(np.asarray(a, dtype=np.float64)).rank(method='average').to_numpy(dtype=np.float64)
    b_rank = pd.Series(np.asarray(b, dtype=np.float64)).rank(method='average').to_numpy(dtype=np.float64)
    return safe_pearson_np(a_rank, b_rank)


def topk_overlap_ratio(a: np.ndarray, b: np.ndarray, k: int) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    n = min(len(a), len(b))
    if n == 0:
        return float('nan')
    k = max(1, min(int(k), n))
    top_a = set(np.argsort(-a[:n])[:k].astype(np.int64).tolist())
    top_b = set(np.argsort(-b[:n])[:k].astype(np.int64).tolist())
    return float(len(top_a.intersection(top_b)) / max(1, k))


def package_distillation_metrics(pred_score: np.ndarray, target_score: np.ndarray, top_k: int = 50) -> Dict[str, float]:
    pred = np.asarray(pred_score, dtype=np.float32)
    target = np.asarray(target_score, dtype=np.float32)
    n = min(len(pred), len(target))
    if n == 0:
        return {
            'distill_pearson': float('nan'),
            'distill_spearman': float('nan'),
            'distill_mse': float('nan'),
            'distill_top50_overlap': float('nan'),
        }
    pred = pred[:n]
    target = target[:n]
    return {
        'distill_pearson': safe_pearson_np(pred, target),
        'distill_spearman': safe_spearman_np(pred, target),
        'distill_mse': float(np.mean((pred - target) ** 2)),
        'distill_top50_overlap': topk_overlap_ratio(pred, target, top_k),
    }


def is_pure_structural_raw_selector(args: argparse.Namespace) -> bool:
    return str(getattr(args, 'raw_score_variant', 'base')) == 'structural'



def compute_raw_final_target(
    m0: np.ndarray,
    mc: np.ndarray,
    cos_raw: np.ndarray,
    args: argparse.Namespace,
    structural_prior: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the raw utility target used to train the row student.

    base:
        original margin-improvement target.
    gated:
        requires the candidate itself to be plausible and the weak text to need correction.
    gated_need:
        stronger variant that removes the positive cosine bonus and focuses on teacher utility.
    structural:
        feature-only shortcut-resistant prior, used for high-precision top-k selection.
    gated_need_structural:
        teacher utility plus the shortcut-resistant structural prior.
    """
    variant = str(getattr(args, 'raw_score_variant', 'base'))
    neg_margin = np.maximum(0.0, -mc).astype(np.float32)
    delta = (mc - m0).astype(np.float32)
    if structural_prior is None:
        structural = np.zeros_like(delta, dtype=np.float32)
    else:
        structural = np.asarray(structural_prior, dtype=np.float32)
    structural_alpha = float(getattr(args, 'structural_prior_alpha', 1.0))

    if variant == 'base':
        final = (
            delta
            + float(args.cos_alpha) * cos_raw
            - float(args.neg_margin_lambda) * neg_margin
        )
    elif variant in ['gated', 'gated_need', 'gated_need_structural']:
        tau_c = max(float(getattr(args, 'candidate_gate_tau', 1.0)), 1e-6)
        tau_0 = max(float(getattr(args, 'need_gate_tau', 1.0)), 1e-6)
        candidate_gate = sigmoid_np(mc / tau_c)
        need_gate = sigmoid_np((-m0) / tau_0)
        gate = (candidate_gate * need_gate).astype(np.float32)
        if variant == 'gated':
            final = (
                gate * delta
                + float(args.cos_alpha) * cos_raw
                - float(args.neg_margin_lambda) * neg_margin
            )
        else:
            final = (
                gate * delta
                - float(args.neg_margin_lambda) * neg_margin
            )
            if variant == 'gated_need_structural':
                final = final + structural_alpha * structural
    elif variant == 'structural':
        final = structural_alpha * structural
    else:
        raise ValueError('Unknown raw_score_variant: ' + str(variant))

    return final.astype(np.float32)


def build_row_student_train_pack(market_rows: Dict[str, np.ndarray], teacher_model: nn.Module, proj: np.ndarray, args: argparse.Namespace, available_idx: np.ndarray) -> RowStudentTrainPack:
    idx = np.asarray(available_idx, dtype=np.int64)
    u = project_features(market_rows['img'][idx], proj)
    v = project_features(market_rows['cand'][idx], proj)
    w = project_features(market_rows['weak'][idx], proj)
    ysign = (2 * market_rows['y'][idx].astype(np.float32) - 1.0).astype(np.float32)
    features = build_row_student_features(u, v, w, ysign)
    weak_logits = predict_pair_logits(teacher_model, market_rows['img'][idx], market_rows['weak'][idx], args.device, args.batch_size)
    cand_logits = predict_pair_logits(teacher_model, market_rows['img'][idx], market_rows['cand'][idx], args.device, args.batch_size)
    m0 = (ysign * weak_logits).astype(np.float32)
    mc = (ysign * cand_logits).astype(np.float32)
    cos_raw = np.sum(market_rows['img'][idx] * market_rows['cand'][idx], axis=1).astype(np.float32)
    structural_prior = compute_row_structural_good_prior(market_rows, idx, args)
    final = compute_raw_final_target(m0, mc, cos_raw, args, structural_prior=structural_prior)
    return RowStudentTrainPack(features=features, base_margin_target=m0, cand_margin_target=mc, final_target=final, indices=idx)


def build_row_score_dict(market_rows: Dict[str, np.ndarray], student_model: RowStudent | None, proj: np.ndarray, args: argparse.Namespace, available_idx: np.ndarray) -> Dict[str, np.ndarray]:
    """Build only the score needed by our row-level selector."""
    n = len(market_rows['y'])
    scores = {'market_ours_row_select': np.full((n,), -1e9, dtype=np.float32)}
    idx = np.asarray(available_idx, dtype=np.int64)
    if len(idx) == 0:
        return scores
    structural_prior = compute_row_structural_good_prior(market_rows, idx, args)
    if is_pure_structural_raw_selector(args):
        scores['market_ours_row_select'][idx] = structural_prior.astype(np.float32)
        return scores
    if student_model is None:
        raise RuntimeError('Row student is required unless raw_score_variant=structural.')

    u = project_features(market_rows['img'][idx], proj)
    v = project_features(market_rows['cand'][idx], proj)
    w = project_features(market_rows['weak'][idx], proj)
    ysign = (2 * market_rows['y'][idx].astype(np.float32) - 1.0).astype(np.float32)
    features = build_row_student_features(u, v, w, ysign)
    pred = predict_row_student(student_model, features, args)
    cos_raw = np.sum(market_rows['img'][idx] * market_rows['cand'][idx], axis=1).astype(np.float32)
    row_score = compute_raw_final_target(pred['base_margin'], pred['cand_margin'], cos_raw, args, structural_prior=structural_prior)
    scores['market_ours_row_select'][idx] = row_score
    return scores


def select_rows_by_budget_label_balanced(
    score_dict: Dict[str, np.ndarray],
    available_idx: np.ndarray,
    y: np.ndarray,
    round_budget: int,
    method: str,
    seed: int,
) -> np.ndarray:
    """Top-k selection with a simple per-label budget split.

    This is used only for controlled raw-method ablations. It prevents the selector
    from improving useful-good count while collapsing to a single label.
    """
    available_idx = np.asarray(available_idx, dtype=np.int64)
    if len(available_idx) == 0 or int(round_budget) <= 0:
        return np.zeros((0,), dtype=np.int64)

    labels = np.unique(y[available_idx])
    labels = np.asarray(labels, dtype=np.int64)
    per_label = max(1, int(round_budget) // max(1, len(labels)))
    chosen_parts: List[np.ndarray] = []

    if method == 'market_random_select':
        rng = np.random.default_rng(seed + 901)
        for lab in labels.tolist():
            sub = available_idx[y[available_idx] == lab].copy()
            rng.shuffle(sub)
            chosen_parts.append(sub[:per_label])
        chosen = np.concatenate(chosen_parts, axis=0) if chosen_parts else np.zeros((0,), dtype=np.int64)
        if len(chosen) < int(round_budget):
            chosen_set = set(chosen.tolist())
            rest = np.array([i for i in available_idx.tolist() if int(i) not in chosen_set], dtype=np.int64)
            rng.shuffle(rest)
            chosen = np.concatenate([chosen, rest[: int(round_budget) - len(chosen)]], axis=0)
        return chosen[: int(round_budget)].astype(np.int64)

    scores = score_dict[method].astype(np.float32)
    for lab in labels.tolist():
        sub = available_idx[y[available_idx] == lab]
        if len(sub) == 0:
            continue
        order = sub[np.argsort(-scores[sub])]
        chosen_parts.append(order[:per_label])

    chosen = np.concatenate(chosen_parts, axis=0) if chosen_parts else np.zeros((0,), dtype=np.int64)
    if len(chosen) < int(round_budget):
        chosen_set = set(chosen.tolist())
        rest = np.array([i for i in available_idx.tolist() if int(i) not in chosen_set], dtype=np.int64)
        rest = rest[np.argsort(-scores[rest])]
        chosen = np.concatenate([chosen, rest[: int(round_budget) - len(chosen)]], axis=0)
    return chosen[: int(round_budget)].astype(np.int64)

def build_joint_row_cluster_features(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.concatenate([u, v, u * v, np.abs(u - v), v - w], axis=1).astype(np.float32)


def cluster_radius(feat: np.ndarray) -> float:
    if len(feat) <= 1:
        return 0.0
    mu = feat.mean(axis=0, keepdims=True)
    return float(np.mean(np.linalg.norm(feat - mu, axis=1)))


def farthest_first_seeds(feat: np.ndarray, n_seeds: int) -> np.ndarray:
    n = len(feat)
    if n_seeds >= n:
        return np.arange(n, dtype=np.int64)
    seeds = [0]
    dist = np.linalg.norm(feat - feat[[0]], axis=1)
    for _ in range(1, n_seeds):
        nxt = int(np.argmax(dist))
        seeds.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(feat - feat[[nxt]], axis=1))
    return np.array(seeds, dtype=np.int64)


def build_groups_balanced_greedy(feat: np.ndarray, idx: np.ndarray, target_size: int) -> List[np.ndarray]:
    idx = np.asarray(idx, dtype=np.int64)
    n = len(idx)
    if n <= target_size:
        return [idx.copy()]
    n_groups = int(math.ceil(n / target_size))
    local_feat = feat[idx]
    seeds_local = farthest_first_seeds(local_feat, n_groups)
    centers = local_feat[seeds_local].copy()
    base = n // n_groups
    rem = n % n_groups
    cap = np.array([base + (1 if i < rem else 0) for i in range(n_groups)], dtype=np.int64)
    order = np.argsort(-np.min(np.linalg.norm(local_feat[:, None, :] - centers[None, :, :], axis=2), axis=1))
    groups_local: List[List[int]] = [[] for _ in range(n_groups)]
    for li in order.tolist():
        d = np.linalg.norm(local_feat[li:li + 1] - centers, axis=1)
        cand = np.argsort(d)
        placed = False
        for c in cand.tolist():
            if len(groups_local[c]) < int(cap[c]):
                groups_local[c].append(li)
                placed = True
                break
        if not placed:
            groups_local[int(cand[0])].append(li)
    return [idx[np.array(g, dtype=np.int64)].astype(np.int64) for g in groups_local if len(g) > 0]


def recursive_split(feat: np.ndarray, idx: np.ndarray, target_size: int, radius_thr: float, seed: int) -> List[np.ndarray]:
    idx = np.asarray(idx, dtype=np.int64)
    if len(idx) <= target_size:
        return [idx.copy()]
    cur_rad = cluster_radius(feat[idx])
    if cur_rad <= radius_thr and len(idx) <= 2 * target_size:
        return [idx.copy()]
    kmeans = MiniBatchKMeans(n_clusters=2, random_state=seed, batch_size=min(max(32, len(idx) // 2), len(idx)), n_init=3, max_iter=60)
    lab = kmeans.fit_predict(feat[idx]).astype(np.int64)
    out: List[np.ndarray] = []
    for gid in [0, 1]:
        sub = idx[lab == gid].astype(np.int64)
        if len(sub) > 0:
            out.extend(recursive_split(feat, sub, target_size, radius_thr, seed + gid + len(sub)))
    return out


def build_package_groups(feat: np.ndarray, method: str, target_size: int, radius_thr: float, seed: int, anchor_target_size: int) -> List[np.ndarray]:
    n = len(feat)
    if n == 0:
        return []
    anchor_k = max(1, int(math.ceil(n / max(1, anchor_target_size))))
    anchor_model = MiniBatchKMeans(n_clusters=anchor_k, random_state=seed + 71, batch_size=min(max(64, n // 8), n), n_init=3, max_iter=80)
    anchor_ids = anchor_model.fit_predict(feat).astype(np.int64)
    all_idx = np.arange(n, dtype=np.int64)
    groups: List[np.ndarray] = []
    for aid in np.unique(anchor_ids).tolist():
        cur = all_idx[anchor_ids == int(aid)].astype(np.int64)
        if len(cur) == 0:
            continue
        if method == 'anchor_kmeans':
            if len(cur) <= target_size:
                groups.append(cur)
            else:
                n_groups = int(math.ceil(len(cur) / target_size))
                kmeans = MiniBatchKMeans(n_clusters=n_groups, random_state=seed + int(aid), batch_size=min(max(32, len(cur) // 2), len(cur)), n_init=3, max_iter=80)
                lab = kmeans.fit_predict(feat[cur]).astype(np.int64)
                for gid in range(n_groups):
                    sub = cur[lab == gid].astype(np.int64)
                    if len(sub) > 0:
                        groups.append(sub)
        elif method == 'anchor_recursive':
            groups.extend(recursive_split(feat, cur, target_size, radius_thr, seed + int(aid)))
        elif method == 'anchor_greedy':
            groups.extend(build_groups_balanced_greedy(feat, cur, target_size))
        else:
            raise ValueError(method)
    return groups


def build_package_student_features(u_pkg: np.ndarray, v_pkg: np.ndarray, w_pkg: np.ndarray, size: np.ndarray, radius: np.ndarray, ysign_pkg: np.ndarray) -> np.ndarray:
    cos_uv = np.sum(u_pkg * v_pkg, axis=1, keepdims=True).astype(np.float32)
    cos_uw = np.sum(u_pkg * w_pkg, axis=1, keepdims=True).astype(np.float32)
    cos_vw = np.sum(v_pkg * w_pkg, axis=1, keepdims=True).astype(np.float32)
    return np.concatenate([
        u_pkg.astype(np.float32),
        v_pkg.astype(np.float32),
        (u_pkg * v_pkg).astype(np.float32),
        np.abs(u_pkg - v_pkg).astype(np.float32),
        (v_pkg - w_pkg).astype(np.float32),
        (u_pkg * w_pkg).astype(np.float32),
        cos_uv, cos_uw, cos_vw,
        np.log1p(size.astype(np.float32)).reshape(-1, 1),
        radius.astype(np.float32).reshape(-1, 1),
        ysign_pkg.reshape(-1, 1).astype(np.float32),
    ], axis=1).astype(np.float32)


def build_package_bundle(market_rows: Dict[str, np.ndarray], teacher_model: nn.Module | None, proj: np.ndarray, args: argparse.Namespace, package_method: str, seed: int, available_idx: np.ndarray | None = None) -> PackageBundle:
    idx = np.arange(len(market_rows['y']), dtype=np.int64) if available_idx is None else np.asarray(available_idx, dtype=np.int64)
    u_row = project_features(market_rows['img'][idx], proj)
    v_row = project_features(market_rows['cand'][idx], proj)
    w_row = project_features(market_rows['weak'][idx], proj)
    structural_row = compute_row_structural_good_prior(market_rows, idx, args)
    joint_feat = build_joint_row_cluster_features(u_row, v_row, w_row)
    cluster_weight = float(getattr(args, 'package_structural_cluster_weight', 1.0))
    if cluster_weight > 0:
        joint_feat = np.concatenate(
            [joint_feat, (cluster_weight * structural_row).reshape(-1, 1).astype(np.float32)],
            axis=1,
        ).astype(np.float32)
    groups_local = build_package_groups(joint_feat, package_method, int(args.package_target_size), float(args.package_radius_threshold), seed, int(args.anchor_target_size))
    if len(groups_local) == 0:
        raise RuntimeError('No packages were built. Check available rows and package settings.')

    ysign = (2 * market_rows['y'][idx].astype(np.float32) - 1.0).astype(np.float32)
    if teacher_model is None:
        if not is_pure_structural_raw_selector(args):
            raise RuntimeError('Teacher model is required unless raw_score_variant=structural.')
        m0_row = np.zeros((len(idx),), dtype=np.float32)
        mc_row = np.zeros((len(idx),), dtype=np.float32)
    else:
        weak_logits = predict_pair_logits(teacher_model, market_rows['img'][idx], market_rows['weak'][idx], args.device, args.batch_size)
        cand_logits = predict_pair_logits(teacher_model, market_rows['img'][idx], market_rows['cand'][idx], args.device, args.batch_size)
        m0_row = (ysign * weak_logits).astype(np.float32)
        mc_row = (ysign * cand_logits).astype(np.float32)
    cos_raw_row = np.sum(market_rows['img'][idx] * market_rows['cand'][idx], axis=1).astype(np.float32)
    row_final_target = compute_raw_final_target(m0_row, mc_row, cos_raw_row, args, structural_prior=structural_row)
    teacher_target_mode = str(getattr(args, 'package_teacher_target_mode', 'aggregate')).lower()
    teacher_top_frac = float(getattr(args, 'package_teacher_top_frac', getattr(args, 'package_structural_top_frac', 0.5)))

    row_ptr = np.empty((len(groups_local),), dtype=object)
    u_pkg=[]; v_pkg=[]; w_pkg=[]; size=[]; radius=[]; m0_pkg=[]; mc_pkg=[]; final_pkg=[]; cos_pkg_list=[]; structural_pkg_list=[]; g_pkg=[]; ysign_pkg=[]
    for pi, g_local in enumerate(groups_local):
        g_local = np.asarray(g_local, dtype=np.int64)
        if bool(getattr(args, 'package_sort_rows_by_structural', True)):
            order = np.argsort(-structural_row[g_local])
            g_local = g_local[order].astype(np.int64)
        row_ptr[pi] = idx[g_local].astype(np.int64)
        u_mean = l2_normalize_np(u_row[g_local].mean(axis=0).astype(np.float32))
        v_mean = l2_normalize_np(v_row[g_local].mean(axis=0).astype(np.float32))
        w_mean = l2_normalize_np(w_row[g_local].mean(axis=0).astype(np.float32))
        rad = cluster_radius(joint_feat[g_local])
        m0 = float(np.mean(m0_row[g_local]))
        mc = float(np.mean(mc_row[g_local]))
        cos_pkg = float(np.mean(cos_raw_row[g_local]))
        structural_pkg = aggregate_package_structural_prior(structural_row[g_local], args)
        aggregate_final = float(compute_raw_final_target(
            np.array([m0], dtype=np.float32),
            np.array([mc], dtype=np.float32),
            np.array([cos_pkg], dtype=np.float32),
            args,
            structural_prior=np.array([structural_pkg], dtype=np.float32),
        )[0])
        if teacher_target_mode == 'aggregate':
            final = aggregate_final
        elif teacher_target_mode in ['topmean', 'shuffled']:
            final = mean_top_fraction(row_final_target[g_local], teacher_top_frac)
        elif teacher_target_mode == 'mean':
            final = float(np.mean(row_final_target[g_local]))
        elif teacher_target_mode == 'structural':
            final = float(structural_pkg)
        else:
            raise ValueError('Unknown package_teacher_target_mode: ' + str(teacher_target_mode))
        final = float(final - float(args.package_radius_penalty) * float(rad))
        u_pkg.append(u_mean); v_pkg.append(v_mean); w_pkg.append(w_mean)
        size.append(int(len(g_local))); radius.append(float(rad)); m0_pkg.append(m0); mc_pkg.append(mc); final_pkg.append(float(final)); cos_pkg_list.append(float(cos_pkg)); structural_pkg_list.append(float(structural_pkg))
        g_pkg.append(float(np.mean(market_rows['is_good'][idx[g_local]])))
        mean_sign = float(np.mean(ysign[g_local]))
        ysign_pkg.append(float(np.sign(mean_sign) if mean_sign != 0 else 1.0))

    u_pkg = np.stack(u_pkg, axis=0).astype(np.float32)
    v_pkg = np.stack(v_pkg, axis=0).astype(np.float32)
    w_pkg = np.stack(w_pkg, axis=0).astype(np.float32)
    size_arr = np.array(size, dtype=np.int64)
    radius_arr = np.array(radius, dtype=np.float32)
    ysign_pkg_arr = np.array(ysign_pkg, dtype=np.float32)
    final_arr = np.array(final_pkg, dtype=np.float32)
    if teacher_target_mode == 'shuffled' and len(final_arr) > 1:
        rng = np.random.default_rng(seed + 99173)
        final_arr = final_arr[rng.permutation(len(final_arr))].astype(np.float32)
    features = build_package_student_features(u_pkg, v_pkg, w_pkg, size_arr, radius_arr, ysign_pkg_arr)
    return PackageBundle(
        features=features,
        size=size_arr,
        radius=radius_arr,
        row_ptr=row_ptr,
        teacher_base_margin=np.array(m0_pkg, dtype=np.float32),
        teacher_cand_margin=np.array(mc_pkg, dtype=np.float32),
        teacher_final_target=final_arr,
        cos_raw=np.array(cos_pkg_list, dtype=np.float32),
        structural_prior=np.array(structural_pkg_list, dtype=np.float32),
        good_ratio=np.array(g_pkg, dtype=np.float32),
        package_method=package_method,
    )


def build_package_score_dict(bundle: PackageBundle, student_model: PackageStudent | None, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    if student_model is None:
        zeros = np.zeros_like(bundle.structural_prior, dtype=np.float32)
        score = compute_raw_final_target(
            zeros,
            zeros,
            bundle.cos_raw.astype(np.float32),
            args,
            structural_prior=bundle.structural_prior.astype(np.float32),
        )
    else:
        pred = predict_package_student(student_model, bundle.features, args)
        if str(getattr(args, 'package_student_score_mode', 'margin')).lower() == 'final':
            score = pred['final_score'].astype(np.float32)
        else:
            score = compute_raw_final_target(
                pred['base_margin'].astype(np.float32),
                pred['cand_margin'].astype(np.float32),
                bundle.cos_raw.astype(np.float32),
                args,
                structural_prior=bundle.structural_prior.astype(np.float32),
            )
    score = score - float(args.package_radius_penalty) * bundle.radius.astype(np.float32)
    score = score - float(getattr(args, 'package_size_penalty', 0.0)) * np.log1p(bundle.size.astype(np.float32))
    return {'market_ours_package_select': score.astype(np.float32)}


# =========================================================
# Selection and target-good protocol
# =========================================================


def build_teacher_local_packs(
    train_pack: Dict[str, np.ndarray],
    initial_idx: np.ndarray,
    seed: int,
    local_text_mode: str = 'severe_noisy',
    local_noise_strength: float = 0.9,
    local_wrong_swap_prob: float = 0.6,
    local_clean_ratio: float = 0.0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    initial_clean = {
        'img': train_pack['img'][initial_idx].astype(np.float32),
        'txt': train_pack['txt'][initial_idx].astype(np.float32),
        'y': train_pack['y'][initial_idx].astype(np.int64),
    }
    mode = str(local_text_mode).lower()
    if mode in ['clean', 'none']:
        noisy_txt = initial_clean['txt'].copy()
    elif mode in ['light_noisy', 'gaussian', 'general_noisy']:
        noisy_txt = add_local_text_noise(initial_clean['txt'], seed=seed, strength=float(local_noise_strength))
    elif mode in ['mixed_good_noisy', 'mixed_clean_noisy', 'mixed_market_noisy']:
        degraded = degrade_market_style_text(
            initial_clean['txt'],
            initial_clean['y'],
            seed=seed,
            noise=float(local_noise_strength),
            wrong_swap_prob=float(local_wrong_swap_prob),
        )
        clean_ratio = min(max(float(local_clean_ratio), 0.0), 1.0)
        rng = np.random.default_rng(seed + 41017)
        keep_clean = rng.random(len(initial_clean['y'])) < clean_ratio
        noisy_txt = degraded.astype(np.float32)
        noisy_txt[keep_clean] = initial_clean['txt'][keep_clean].astype(np.float32)
    elif mode in ['mixed_noisy', 'weak_noisy']:
        noisy_txt = degrade_local_text(
            initial_clean['txt'],
            initial_clean['y'],
            seed=seed,
            strength=float(local_noise_strength),
            wrong_swap_prob=float(local_wrong_swap_prob),
        )
    elif mode in ['severe_noisy', 'legacy']:
        noisy_txt = degrade_local_text(
            initial_clean['txt'],
            initial_clean['y'],
            seed=seed,
            strength=float(local_noise_strength),
            wrong_swap_prob=float(local_wrong_swap_prob),
        )
    else:
        raise ValueError('Unknown local_text_mode: ' + str(local_text_mode))
    initial_noisy = {
        'img': initial_clean['img'].copy(),
        'txt': noisy_txt.astype(np.float32),
        'y': initial_clean['y'].copy(),
    }
    return initial_clean, initial_noisy


def extend_train_pack(current_pack: Dict[str, np.ndarray], market_rows: Dict[str, np.ndarray], chosen_idx: np.ndarray) -> Dict[str, np.ndarray]:
    if len(chosen_idx) == 0:
        return {k: v.copy() for k, v in current_pack.items()}
    return {
        'img': np.concatenate([current_pack['img'], market_rows['img'][chosen_idx].astype(np.float32)], axis=0).astype(np.float32),
        'txt': np.concatenate([current_pack['txt'], market_rows['cand'][chosen_idx].astype(np.float32)], axis=0).astype(np.float32),
        'y': np.concatenate([current_pack['y'], market_rows['y'][chosen_idx].astype(np.int64)], axis=0).astype(np.int64),
    }


def select_rows_by_budget(score_dict: Dict[str, np.ndarray], available_idx: np.ndarray, round_budget: int, method: str, seed: int) -> np.ndarray:
    available_idx = np.asarray(available_idx, dtype=np.int64)
    if len(available_idx) == 0 or int(round_budget) <= 0:
        return np.zeros((0,), dtype=np.int64)
    if method == 'market_random_select':
        rng = np.random.default_rng(seed + 901)
        order = available_idx.copy()
        rng.shuffle(order)
    else:
        scores = score_dict[method].astype(np.float32)
        order = available_idx[np.argsort(-scores[available_idx])]
    return order[: max(0, int(round_budget))].astype(np.int64)


def select_packages_strict_budget(bundle: PackageBundle, score_dict: Dict[str, np.ndarray], available_pkg_idx: np.ndarray, row_budget: int) -> np.ndarray:
    available_pkg_idx = np.asarray(available_pkg_idx, dtype=np.int64)
    if len(available_pkg_idx) == 0 or int(row_budget) <= 0:
        return np.zeros((0,), dtype=np.int64)
    scores = score_dict['market_ours_package_select'].astype(np.float32)
    order = available_pkg_idx[np.argsort(-scores[available_pkg_idx])]
    chosen: List[int] = []
    remain = int(row_budget)
    for pid in order.tolist():
        sz = int(bundle.size[int(pid)])
        if sz <= remain:
            chosen.append(int(pid))
            remain -= sz
        if remain <= 0:
            break
    return np.array(chosen, dtype=np.int64)


def select_package_rows_by_raw_budget(bundle: PackageBundle, score_dict: Dict[str, np.ndarray], available_pkg_idx: np.ndarray, row_budget: int) -> np.ndarray:
    # In the oracle-matched protocol, every method must buy exactly the same raw
    # data count as Oracle Good in the current round. The package selector ranks
    # packages, but the purchased object after selection is raw data. Therefore we
    # take rows from top-ranked packages until the round raw budget is filled.
    available_pkg_idx = np.asarray(available_pkg_idx, dtype=np.int64)
    if len(available_pkg_idx) == 0 or int(row_budget) <= 0:
        return np.zeros((0,), dtype=np.int64)
    scores = score_dict['market_ours_package_select'].astype(np.float32)
    order = available_pkg_idx[np.argsort(-scores[available_pkg_idx])]
    rows: List[int] = []
    remain = int(row_budget)
    for pid in order.tolist():
        ptr = np.asarray(bundle.row_ptr[int(pid)], dtype=np.int64)
        if len(ptr) == 0:
            continue
        take = min(remain, len(ptr))
        rows.extend(ptr[:take].astype(np.int64).tolist())
        remain -= take
        if remain <= 0:
            break
    return np.array(rows, dtype=np.int64)


def run_downstream(current_train: Dict[str, np.ndarray], test_pack: Dict[str, np.ndarray], args: argparse.Namespace, seed: int) -> Tuple[Dict[str, float], float]:
    _, _, metrics, elapsed = train_downstream_model(current_train, test_pack, args, seed, init_state=None)
    return metrics, elapsed


def train_downstream_model(
    current_train: Dict[str, np.ndarray],
    test_pack: Dict[str, np.ndarray],
    args: argparse.Namespace,
    seed: int,
    init_state: Dict[str, torch.Tensor] | None = None,
) -> Tuple[nn.Module, Dict[str, torch.Tensor], Dict[str, float], float]:
    t0 = time.perf_counter()
    model, state, _ = train_pair_model(
        current_train,
        args,
        seed,
        init_state=init_state,
        max_epochs=int(args.downstream_max_epochs),
        patience=int(args.downstream_patience),
    )
    metrics = evaluate_pair_model(model, test_pack, args)
    return model, state, metrics, float(time.perf_counter() - t0)


def downstream_eval_seed(base_seed: int, method_key: str, round_idx: int, args: argparse.Namespace) -> int:
    if bool(getattr(args, 'downstream_eval_fixed_seed', False)):
        # Use the same downstream initialization for every method and acquisition
        # round within a data seed. This makes smoke-test curves reflect the
        # purchased data rather than random model initialization noise.
        return int(base_seed) + 424242
    token = int(hashlib.md5(str(method_key).encode('utf-8')).hexdigest()[:6], 16)
    return int(base_seed) + 5000 * int(round_idx) + token % 997


def is_target_good_protocol(args: argparse.Namespace) -> bool:
    return str(getattr(args, 'acquisition_protocol', 'fixed_budget')) == 'target_good'


def is_oracle_matched_protocol(args: argparse.Namespace) -> bool:
    return str(getattr(args, 'acquisition_protocol', 'fixed_budget')) == 'oracle_matched'


def get_oracle_raw_target(market_rows: Dict[str, np.ndarray], args: argparse.Namespace) -> int:
    # Oracle Good buys only useful samples. Therefore the raw-data budget of the
    # oracle-matched protocol is the number of useful samples Oracle Good can buy,
    # capped by target_good_count. All non-oracle methods must buy the same number
    # of raw rows under the same per-round schedule.
    available_good = int(np.sum(np.asarray(market_rows['is_good'], dtype=np.int64)))
    return int(min(int(args.target_good_count), available_good))


def get_good_count(market_rows: Dict[str, np.ndarray], purchased: List[int]) -> int:
    if not purchased:
        return 0
    idx = np.asarray(purchased, dtype=np.int64)
    return int(np.sum(market_rows['is_good'][idx].astype(np.int64)))


def should_stop_acquisition(market_rows: Dict[str, np.ndarray], purchased: List[int], available: np.ndarray, args: argparse.Namespace) -> bool:
    if len(available) == 0:
        return True
    if is_target_good_protocol(args):
        if get_good_count(market_rows, purchased) >= int(args.target_good_count):
            return True
        if len(purchased) >= int(args.max_purchase_rows):
            return True
        return False
    if is_oracle_matched_protocol(args):
        # Stop by the raw-data count used by Oracle Good, not by the number of good
        # samples acquired by this method. This is the fair setting requested here:
        # every method receives the same raw-data budget in every round.
        return len(purchased) >= get_oracle_raw_target(market_rows, args)
    return len(purchased) >= int(args.purchase_total)


def get_round_budget(args: argparse.Namespace, method: str, market_rows: Dict[str, np.ndarray], purchased: List[int]) -> int:
    rb = int(args.round_budget)
    if is_target_good_protocol(args):
        remain_rows = int(args.max_purchase_rows) - len(purchased)
        rb = min(rb, remain_rows)
        if method == 'oracle_good_select':
            remain_good = int(args.target_good_count) - get_good_count(market_rows, purchased)
            rb = min(rb, max(0, remain_good))
        return max(0, rb)
    if is_oracle_matched_protocol(args):
        target_raw = get_oracle_raw_target(market_rows, args)
        remain_raw = target_raw - len(purchased)
        return max(0, min(rb, remain_raw))
    remain = int(args.purchase_total) - len(purchased)
    return max(0, min(rb, remain))


def row_type_composition(market_rows: Dict[str, np.ndarray], purchased: List[int], prefix: str = 'selected_type') -> Dict[str, float]:
    if 'row_type' not in market_rows or len(purchased) == 0:
        return {}
    idx = np.asarray(purchased, dtype=np.int64)
    rt = np.asarray(market_rows['row_type'], dtype=object)[idx]
    out: Dict[str, float] = {}
    for t in sorted(set(rt.tolist())):
        safe_t = str(t).replace(' ', '_').replace('/', '_')
        out[f'{prefix}_{safe_t}_ratio'] = float(np.mean(rt == t))
        out[f'{prefix}_{safe_t}_count'] = int(np.sum(rt == t))
    return out


def summarize_acquisition_result(
    seed: int,
    market_name: str,
    setting: str,
    package_method: str,
    method: str,
    market_rows: Dict[str, np.ndarray],
    purchased: List[int],
    refresh_count: int,
    teacher_time_total: float,
    student_time_total: float,
    score_time_total: float,
    selection_time_total: float,
    downstream_time_total: float,
    final_metrics: Dict[str, float],
    rounds_used: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    purchased_arr = np.asarray(purchased, dtype=np.int64)
    good = market_rows['is_good'][purchased_arr].astype(np.int64) if len(purchased_arr) else np.zeros((0,), dtype=np.int64)
    first = good[: min(1000, len(good))]
    good_count = int(np.sum(good))
    selected_rows = int(len(purchased_arr))
    market_good_ratio = float(np.mean(market_rows['is_good']))
    if is_target_good_protocol(args):
        target_good_count = int(args.target_good_count)
        raw_target_count = int(args.max_purchase_rows)
        reached_target = bool(good_count >= target_good_count)
        reached_raw_budget = bool(selected_rows >= raw_target_count)
    elif is_oracle_matched_protocol(args):
        target_good_count = int(args.target_good_count)
        raw_target_count = get_oracle_raw_target(market_rows, args)
        reached_target = bool(good_count >= target_good_count)
        reached_raw_budget = bool(selected_rows >= raw_target_count)
    else:
        target_good_count = int(args.purchase_total)
        raw_target_count = int(args.purchase_total)
        reached_target = bool(selected_rows >= int(args.purchase_total))
        reached_raw_budget = bool(selected_rows >= raw_target_count)
    res: Dict[str, object] = {
        'seed': int(seed),
        'market_profile': market_name,
        'setting': setting,
        'package_method': package_method,
        'package_target_size': int(getattr(args, 'package_target_size', -1)) if setting == 'packaged' else -1,
        'anchor_target_size': int(getattr(args, 'anchor_target_size', -1)) if setting == 'packaged' else -1,
        'package_radius_penalty': float(getattr(args, 'package_radius_penalty', 0.0)) if setting == 'packaged' else 0.0,
        'package_radius_threshold': float(getattr(args, 'package_radius_threshold', 0.0)) if setting == 'packaged' else 0.0,
        'method': method,
        'acquisition_protocol': str(getattr(args, 'acquisition_protocol', 'fixed_budget')),
        'pair_model': str(getattr(args, 'pair_model', 'bopa')),
        'bopa_combined_dim': int(getattr(args, 'bopa_combined_dim', 64)),
        'target_good_count': int(target_good_count),
        'oracle_matched_raw_target': int(raw_target_count),
        'reached_target_good': int(reached_target),
        'reached_raw_budget': int(reached_raw_budget),
        'good_shortfall_vs_target': int(max(0, int(target_good_count) - int(good_count))),
        'raw_budget_gap': int(int(raw_target_count) - int(selected_rows)),
        'rounds_used': int(rounds_used),
        'market_good_ratio': market_good_ratio,
        'selected_rows': selected_rows,
        'good_count': good_count,
        'bad_count': int(selected_rows - good_count),
        'purchase_efficiency': float(good_count / max(1, selected_rows)),
        'extra_rows_vs_oracle_good': int(selected_rows - raw_target_count),
        'selected_good_ratio': float(np.mean(good)) if len(good) else 0.0,
        'selected_first1000_good_ratio': float(np.mean(first)) if len(first) else 0.0,
        'selection_lift_vs_market': float(np.mean(good) / max(1e-8, market_good_ratio)) if len(good) else 0.0,
        'teacher_refresh_count': int(refresh_count),
        'teacher_time_total': float(teacher_time_total),
        'student_time_total': float(student_time_total),
        'score_time_total': float(score_time_total),
        'selection_time_total': float(selection_time_total),
        'downstream_train_time_total': float(downstream_time_total),
        'test_auroc': float(final_metrics['test_auroc']),
        'test_macro_f1': float(final_metrics['test_macro_f1']),
        'test_acc': float(final_metrics['test_acc']),
    }
    res.update(row_type_composition(market_rows, purchased))
    if bool(getattr(args, 'save_selected_indices', False)):
        res['purchased_idx_json'] = json.dumps([int(i) for i in purchased], separators=(',', ':'))
    return res


def fill_available_scores(n: int, available_idx: np.ndarray, values: np.ndarray) -> np.ndarray:
    full = np.full((int(n),), -1e9, dtype=np.float32)
    full[np.asarray(available_idx, dtype=np.int64)] = np.asarray(values, dtype=np.float32)
    return full


def nearest_train_distance(candidate_feat: np.ndarray, train_feat: np.ndarray, k: int = 1) -> np.ndarray:
    if len(candidate_feat) == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(train_feat) == 0:
        return np.ones((len(candidate_feat),), dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=min(max(1, int(k)), len(train_feat)), metric='euclidean', algorithm='auto')
    nn.fit(train_feat.astype(np.float32))
    dist, _ = nn.kneighbors(candidate_feat.astype(np.float32), return_distance=True)
    return np.mean(dist, axis=1).astype(np.float32)


def kmeans_center_scores(candidate_feat: np.ndarray, round_budget: int, seed: int) -> np.ndarray:
    if len(candidate_feat) == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(candidate_feat) == 1:
        return np.zeros((1,), dtype=np.float32)
    k = min(max(1, int(round_budget)), len(candidate_feat))
    km = MiniBatchKMeans(n_clusters=k, random_state=int(seed), batch_size=min(4096, max(256, len(candidate_feat))), n_init=3)
    labels = km.fit_predict(candidate_feat.astype(np.float32))
    centers = km.cluster_centers_.astype(np.float32)
    diff = candidate_feat.astype(np.float32) - centers[labels]
    return (-np.linalg.norm(diff, axis=1)).astype(np.float32)


def typiclust_scores(candidate_feat: np.ndarray, round_budget: int, typiclust_k: int, seed: int) -> np.ndarray:
    """TypiClust-style representative scoring for the current candidate pool.

    The previous implementation ranked all rows by global kNN density. That is
    only a typicality score; TypiClust also needs a diversity step so a round
    does not spend the whole budget inside one dense cluster. We therefore run
    MiniBatchKMeans on the available pool and promote the most typical row from
    each large cluster, with density-based fallback if fewer representatives are
    produced than the requested budget.
    """
    candidate_feat = np.nan_to_num(np.asarray(candidate_feat, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    n = int(len(candidate_feat))
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    if n == 1:
        return np.zeros((1,), dtype=np.float32)

    budget = max(1, min(int(round_budget), n))
    if n <= budget:
        return np.arange(n, 0, -1, dtype=np.float32)

    feat = l2_normalize_np(candidate_feat)

    n_neighbors = min(max(2, int(typiclust_k) + 1), n)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean', algorithm='auto')
    nn.fit(feat)
    dist, _ = nn.kneighbors(feat, return_distance=True)
    typicality = (-np.mean(dist[:, 1:], axis=1)).astype(np.float32)

    n_clusters = min(budget, n)
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=int(seed),
        batch_size=min(4096, max(256, n)),
        n_init=3,
    )
    labels = km.fit_predict(feat)

    representatives: List[Tuple[int, float, int]] = []
    for cluster_id in np.unique(labels).tolist():
        local = np.flatnonzero(labels == cluster_id)
        if len(local) == 0:
            continue
        best = int(local[int(np.argmax(typicality[local]))])
        representatives.append((int(len(local)), float(typicality[best]), best))

    # Large clusters first, then the most typical row inside the cluster.
    representatives.sort(key=lambda item: (-item[0], -item[1], item[2]))
    selected = [idx for _, _, idx in representatives[:budget]]

    if len(selected) < budget:
        selected_set = set(selected)
        for idx in np.argsort(-typicality).tolist():
            if int(idx) not in selected_set:
                selected.append(int(idx))
                selected_set.add(int(idx))
                if len(selected) >= budget:
                    break

    # Give representatives an explicit top-rank score, while preserving density
    # as a low-priority tie breaker for all other rows.
    scores = (typicality - 1e6).astype(np.float32)
    for rank, idx in enumerate(selected):
        scores[int(idx)] = np.float32(1e6 - rank)
    return scores


def run_row_method(seed: int, market_name: str, market_rows: Dict[str, np.ndarray], initial_noisy: Dict[str, np.ndarray], test_pack: Dict[str, np.ndarray], args: argparse.Namespace, method: str, method_label: str | None = None) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    method_label = str(method_label) if method_label is not None else str(method)
    feat_dim = market_rows['img'].shape[1]
    proj = make_projection_matrix(feat_dim, int(args.row_dim), seed + 10000 + 97 * (args.market_profiles.index(market_name) + 1))
    current_train = {k: v.copy() for k, v in initial_noisy.items()}
    teacher_time_total = 0.0
    student_time_total = 0.0
    score_time_total = 0.0
    selection_time_total = 0.0
    downstream_time_total = 0.0
    refresh_count = 0
    teacher = None
    teacher_state = None
    student = None
    ours_needs_student = method == 'market_ours_row_select' and not is_pure_structural_raw_selector(args)
    model_based = ours_needs_student or method in ['market_entropy_select', 'market_margin_select', 'market_badge_select']
    closed_loop_downstream_teacher = (
        bool(getattr(args, 'closed_loop_downstream_teacher', False))
        and bool(getattr(args, 'downstream_eval_each_round', False))
        and model_based
    )
    convergence_enabled = (
        bool(getattr(args, 'acquisition_convergence_stop', False))
        and bool(getattr(args, 'downstream_eval_each_round', False))
    )
    convergence_metric = str(getattr(args, 'convergence_metric', 'test_macro_f1'))
    convergence_best = -float('inf')
    convergence_bad_rounds = 0
    convergence_stop_triggered = False

    if model_based:
        t0 = time.perf_counter()
        teacher, teacher_state, _ = train_pair_model(current_train, args, seed + 11)
        teacher_time_total += float(time.perf_counter() - t0)
        refresh_count += 1
        if ours_needs_student:
            avail0 = np.arange(len(market_rows['y']), dtype=np.int64)
            t1 = time.perf_counter()
            student_pack = build_row_student_train_pack(market_rows, teacher, proj, args, avail0)
            student = train_row_student(student_pack, args, seed + 23)
            student_time_total += float(time.perf_counter() - t1)

    available = np.arange(len(market_rows['y']), dtype=np.int64)
    purchased: List[int] = []
    round_rows: List[Dict[str, object]] = []
    round_idx = 0
    while not should_stop_acquisition(market_rows, purchased, available, args):
        round_idx += 1
        round_budget = get_round_budget(args, method, market_rows, purchased)
        if round_budget <= 0:
            break
        t_score = time.perf_counter()
        if method == 'market_ours_row_select':
            score_dict = build_row_score_dict(market_rows, student, proj, args, available)
        elif method == 'market_random_select':
            score_dict = {}
        elif method == 'market_cosine_select':
            score_dict = {'market_cosine_select': np.sum(market_rows['img'] * market_rows['cand'], axis=1).astype(np.float32)}
        elif method == 'market_entropy_select':
            cand_logits = predict_pair_logits(teacher, market_rows['img'][available], market_rows['cand'][available], args.device, args.batch_size)
            prob = sigmoid_np(cand_logits)
            entropy = -(prob * np.log(np.clip(prob, 1e-6, 1.0)) + (1.0 - prob) * np.log(np.clip(1.0 - prob, 1e-6, 1.0)))
            full = np.full((len(market_rows['y']),), -1e9, dtype=np.float32)
            full[available] = entropy.astype(np.float32)
            score_dict = {'market_entropy_select': full}
        elif method == 'market_margin_select':
            cand_logits = predict_pair_logits(teacher, market_rows['img'][available], market_rows['cand'][available], args.device, args.batch_size)
            prob = sigmoid_np(cand_logits)
            margin_uncertainty = -np.abs(prob - 0.5).astype(np.float32)
            score_dict = {'market_margin_select': fill_available_scores(len(market_rows['y']), available, margin_uncertainty)}
        elif method == 'market_badge_select':
            cand_logits = predict_pair_logits(teacher, market_rows['img'][available], market_rows['cand'][available], args.device, args.batch_size)
            prob = sigmoid_np(cand_logits)
            uncertainty = (prob * (1.0 - prob)).astype(np.float32)
            novelty = nearest_train_distance(market_rows['cand'][available], current_train['txt'], k=1)
            score = (uncertainty * (1.0 + novelty)).astype(np.float32)
            score_dict = {'market_badge_select': fill_available_scores(len(market_rows['y']), available, score)}
        elif method == 'market_coreset_select':
            novelty = nearest_train_distance(market_rows['cand'][available], current_train['txt'], k=1)
            score_dict = {'market_coreset_select': fill_available_scores(len(market_rows['y']), available, novelty)}
        elif method == 'market_kmeans_center_select':
            score = kmeans_center_scores(market_rows['cand'][available], round_budget, seed + 3037 * round_idx)
            score_dict = {'market_kmeans_center_select': fill_available_scores(len(market_rows['y']), available, score)}
        elif method == 'oracle_good_select':
            score_dict = {'oracle_good_select': market_rows['is_good'].astype(np.float32)}
        elif method == 'market_typiclust_select':
            u = project_features(market_rows['img'][available], proj)
            v = project_features(market_rows['cand'][available], proj)
            w = project_features(market_rows['weak'][available], proj)
            typi_feat = build_joint_row_cluster_features(u, v, w)
            score = typiclust_scores(typi_feat, round_budget, int(args.typiclust_k), seed + 3039 * round_idx)
            full = fill_available_scores(len(market_rows['y']), available, score)
            score_dict = {'market_typiclust_select': full}
        else:
            raise ValueError(method)
        score_time = float(time.perf_counter() - t_score)
        score_time_total += score_time

        t_sel = time.perf_counter()
        if bool(getattr(args, 'label_balanced_select', False)):
            chosen = select_rows_by_budget_label_balanced(score_dict, available, market_rows['y'], round_budget, method, seed + 1009 * round_idx)
        else:
            chosen = select_rows_by_budget(score_dict, available, round_budget, method, seed + 1009 * round_idx)
        selection_time = float(time.perf_counter() - t_sel)
        selection_time_total += selection_time
        if len(chosen) == 0:
            break

        purchased.extend(chosen.tolist())
        chosen_set = set(chosen.tolist())
        available = np.array([i for i in available.tolist() if int(i) not in chosen_set], dtype=np.int64)
        current_train = extend_train_pack(current_train, market_rows, chosen)
        cur_sel = market_rows['is_good'][chosen].astype(np.int64)
        cum_sel = market_rows['is_good'][np.array(purchased, dtype=np.int64)].astype(np.int64)
        round_metrics = {'test_auroc': np.nan, 'test_macro_f1': np.nan, 'test_acc': np.nan}
        downstream_time = 0.0
        downstream_teacher_refreshed = False
        teacher_update_source = 'none'
        stop_for_convergence = False
        if args.downstream_eval_each_round:
            eval_seed = downstream_eval_seed(seed, method, round_idx, args)
            if closed_loop_downstream_teacher:
                init_state = None if bool(getattr(args, 'closed_loop_retrain_from_scratch', False)) else teacher_state
                teacher, teacher_state, round_metrics, downstream_time = train_downstream_model(
                    current_train, test_pack, args, eval_seed, init_state=init_state
                )
                downstream_teacher_refreshed = True
                teacher_update_source = 'downstream_converged'
                teacher_time_total += downstream_time
                refresh_count += 1
            else:
                round_metrics, downstream_time = run_downstream(current_train, test_pack, args, eval_seed)
            downstream_time_total += downstream_time
            metric_val = float(round_metrics.get(convergence_metric, np.nan))
            if convergence_enabled and np.isfinite(metric_val):
                if metric_val > convergence_best + float(getattr(args, 'convergence_min_delta', 0.0)):
                    convergence_best = metric_val
                    convergence_bad_rounds = 0
                else:
                    convergence_bad_rounds += 1
                stop_for_convergence = (
                    round_idx >= int(getattr(args, 'convergence_min_rounds', 1))
                    and convergence_bad_rounds >= int(getattr(args, 'convergence_patience', 1))
                )
                convergence_stop_triggered = convergence_stop_triggered or stop_for_convergence
        rr = {
            'seed': int(seed), 'market_profile': market_name, 'setting': 'row_level', 'package_method': 'none', 'method': method_label,
            'acquisition_protocol': str(args.acquisition_protocol), 'pair_model': str(getattr(args, 'pair_model', 'bopa')),
            'bopa_combined_dim': int(getattr(args, 'bopa_combined_dim', 64)), 'round_idx': int(round_idx), 'selected_rows': int(len(chosen)),
            'selected_good_ratio': float(np.mean(cur_sel)) if len(cur_sel) else 0.0,
            'cumulative_selected_rows': int(len(cum_sel)), 'cumulative_good_count': int(np.sum(cum_sel)),
            'cumulative_good_ratio': float(np.mean(cum_sel)) if len(cum_sel) else 0.0,
            'score_time': score_time, 'selection_time': selection_time,
            'teacher_update_time': downstream_time if downstream_teacher_refreshed else 0.0,
            'teacher_update_source': teacher_update_source,
            'closed_loop_downstream_teacher': int(closed_loop_downstream_teacher),
            'acquisition_convergence_stop': int(convergence_enabled),
            'convergence_metric_value': float(round_metrics.get(convergence_metric, np.nan)),
            'convergence_best': float(convergence_best) if np.isfinite(convergence_best) else np.nan,
            'convergence_bad_rounds': int(convergence_bad_rounds),
            'convergence_stop_triggered': int(stop_for_convergence),
            'student_train_time': 0.0, 'downstream_train_time': downstream_time,
            'test_auroc': float(round_metrics['test_auroc']), 'test_macro_f1': float(round_metrics['test_macro_f1']), 'test_acc': float(round_metrics['test_acc']),
        }
        if bool(getattr(args, 'save_selected_indices', False)):
            rr['selected_idx_json'] = json.dumps([int(i) for i in chosen.tolist()], separators=(',', ':'))
            rr['cumulative_idx_json'] = json.dumps([int(i) for i in purchased], separators=(',', ':'))
        rr.update(row_type_composition(market_rows, chosen.tolist(), prefix='round_type'))
        round_rows.append(rr)

        if stop_for_convergence:
            break

        if model_based and not should_stop_acquisition(market_rows, purchased, available, args):
            if downstream_teacher_refreshed:
                teacher_update_time = downstream_time
                round_rows[-1]['teacher_update_source'] = 'downstream_converged'
            else:
                t_up = time.perf_counter()
                teacher, teacher_state, _ = train_pair_model(current_train, args, seed + 101 * round_idx + 7, init_state=teacher_state, max_epochs=int(args.update_max_epochs), patience=int(args.update_patience))
                teacher_update_time = float(time.perf_counter() - t_up)
                teacher_time_total += teacher_update_time
                refresh_count += 1
                round_rows[-1]['teacher_update_time'] = teacher_update_time
                round_rows[-1]['teacher_update_source'] = 'separate_teacher'
            if ours_needs_student:
                t_st = time.perf_counter()
                student_pack = build_row_student_train_pack(market_rows, teacher, proj, args, available)
                student = train_row_student(student_pack, args, seed + 211 * round_idx + 5)
                student_time = float(time.perf_counter() - t_st)
                student_time_total += student_time
                round_rows[-1]['student_train_time'] = student_time

    final_metrics = {'test_auroc': np.nan, 'test_macro_f1': np.nan, 'test_acc': np.nan}
    if bool(getattr(args, 'skip_downstream_eval', False)):
        pass
    elif not args.downstream_eval_each_round or len(round_rows) == 0:
        final_metrics, dt = run_downstream(current_train, test_pack, args, seed + 999 + len(method))
        downstream_time_total += dt
    else:
        final_metrics = {k: float(round_rows[-1][k]) for k in ['test_auroc', 'test_macro_f1', 'test_acc']}

    res = summarize_acquisition_result(seed, market_name, 'row_level', 'none', method_label, market_rows, purchased, refresh_count, teacher_time_total, student_time_total, score_time_total, selection_time_total, downstream_time_total, final_metrics, round_idx, args)
    res['raw_score_variant'] = str(getattr(args, 'raw_score_variant', 'base'))
    res['label_balanced_select'] = int(bool(getattr(args, 'label_balanced_select', False)))
    res['structural_prior_mode'] = str(getattr(args, 'structural_prior_mode', 'hybrid_anti_sim'))
    res['structural_prior_alpha'] = float(getattr(args, 'structural_prior_alpha', 1.0))
    res['neg_margin_lambda'] = float(getattr(args, 'neg_margin_lambda', 1.0))
    res['closed_loop_downstream_teacher'] = int(closed_loop_downstream_teacher)
    res['closed_loop_retrain_from_scratch'] = int(closed_loop_downstream_teacher and bool(getattr(args, 'closed_loop_retrain_from_scratch', False)))
    res['acquisition_convergence_stop'] = int(convergence_enabled)
    res['convergence_stop_triggered'] = int(convergence_stop_triggered)
    res['convergence_metric'] = convergence_metric
    res['convergence_best'] = float(convergence_best) if np.isfinite(convergence_best) else np.nan
    res['convergence_bad_rounds'] = int(convergence_bad_rounds)
    for rr in round_rows:
        rr['raw_score_variant'] = str(getattr(args, 'raw_score_variant', 'base'))
        rr['label_balanced_select'] = int(bool(getattr(args, 'label_balanced_select', False)))
        rr['structural_prior_mode'] = str(getattr(args, 'structural_prior_mode', 'hybrid_anti_sim'))
        rr['structural_prior_alpha'] = float(getattr(args, 'structural_prior_alpha', 1.0))
        rr['neg_margin_lambda'] = float(getattr(args, 'neg_margin_lambda', 1.0))
    return res, round_rows


def run_packaged_ours(seed: int, market_name: str, market_rows: Dict[str, np.ndarray], initial_noisy: Dict[str, np.ndarray], test_pack: Dict[str, np.ndarray], args: argparse.Namespace, package_method: str, package_label: str | None = None) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    package_label = str(package_label) if package_label is not None else str(package_method)
    feat_dim = market_rows['img'].shape[1]
    proj = make_projection_matrix(feat_dim, int(args.row_dim), seed + 20000 + 97 * (args.market_profiles.index(market_name) + 1))
    current_train = {k: v.copy() for k, v in initial_noisy.items()}
    teacher_time_total = 0.0
    student_time_total = 0.0
    score_time_total = 0.0
    selection_time_total = 0.0
    downstream_time_total = 0.0
    refresh_count = 0
    teacher_needs_model = not is_pure_structural_raw_selector(args)
    package_needs_student = bool(getattr(args, 'force_package_student', False)) or teacher_needs_model
    teacher = None
    teacher_state = None

    if teacher_needs_model:
        t0 = time.perf_counter()
        teacher, teacher_state, _ = train_pair_model(current_train, args, seed + 31)
        teacher_time_total += float(time.perf_counter() - t0)
        refresh_count += 1

    available_rows = np.arange(len(market_rows['y']), dtype=np.int64)
    purchased: List[int] = []
    round_rows: List[Dict[str, object]] = []
    round_idx = 0
    while not should_stop_acquisition(market_rows, purchased, available_rows, args):
        round_idx += 1
        round_budget = get_round_budget(args, 'market_ours_package_select', market_rows, purchased)
        if round_budget <= 0:
            break
        t_pkg = time.perf_counter()
        bundle = build_package_bundle(market_rows, teacher, proj, args, package_method, seed + 71 + round_idx, available_rows)
        student = None
        if package_needs_student:
            student_pack = RowStudentTrainPack(
                features=bundle.features,
                base_margin_target=bundle.teacher_base_margin,
                cand_margin_target=bundle.teacher_cand_margin,
                final_target=bundle.teacher_final_target,
                indices=np.arange(len(bundle.size), dtype=np.int64),
            )
            student = train_package_student(student_pack, args, seed + 83 + round_idx)
        student_time = float(time.perf_counter() - t_pkg)
        student_time_total += student_time
        t_score = time.perf_counter()
        score_dict = build_package_score_dict(bundle, student, args)
        score_time = float(time.perf_counter() - t_score)
        score_time_total += score_time
        distill_metrics = {
            'distill_pearson': float('nan'),
            'distill_spearman': float('nan'),
            'distill_mse': float('nan'),
            'distill_top50_overlap': float('nan'),
        }
        if student is not None:
            pred_for_distill = predict_package_student(student, bundle.features, args)
            if str(getattr(args, 'package_student_score_mode', 'margin')).lower() == 'final':
                pred_pkg_score = pred_for_distill['final_score'].astype(np.float32)
            else:
                pred_pkg_score = compute_raw_final_target(
                    pred_for_distill['base_margin'].astype(np.float32),
                    pred_for_distill['cand_margin'].astype(np.float32),
                    bundle.cos_raw.astype(np.float32),
                    args,
                    structural_prior=bundle.structural_prior.astype(np.float32),
                )
            distill_metrics = package_distillation_metrics(pred_pkg_score, bundle.teacher_final_target, top_k=50)
        t_sel = time.perf_counter()
        package_purchase_mode = str(getattr(args, 'package_purchase_mode', 'raw_budget'))
        if package_purchase_mode == 'raw_budget' or is_oracle_matched_protocol(args):
            chosen_rows = select_package_rows_by_raw_budget(bundle, score_dict, np.arange(len(bundle.size), dtype=np.int64), round_budget)
        elif package_purchase_mode == 'strict_package':
            chosen_pkg = select_packages_strict_budget(bundle, score_dict, np.arange(len(bundle.size), dtype=np.int64), round_budget)
            if len(chosen_pkg) == 0:
                chosen_rows = np.zeros((0,), dtype=np.int64)
            else:
                chosen_rows = np.concatenate([np.asarray(bundle.row_ptr[int(i)], dtype=np.int64) for i in chosen_pkg], axis=0).astype(np.int64)
        else:
            raise ValueError('Unknown package_purchase_mode: ' + package_purchase_mode)
        selection_time = float(time.perf_counter() - t_sel)
        selection_time_total += selection_time
        if len(chosen_rows) == 0:
            break
        purchased.extend(chosen_rows.tolist())
        chosen_set = set(chosen_rows.tolist())
        available_rows = np.array([i for i in available_rows.tolist() if int(i) not in chosen_set], dtype=np.int64)
        current_train = extend_train_pack(current_train, market_rows, chosen_rows)
        cur_sel = market_rows['is_good'][chosen_rows].astype(np.int64)
        cum_sel = market_rows['is_good'][np.array(purchased, dtype=np.int64)].astype(np.int64)
        round_metrics = {'test_auroc': np.nan, 'test_macro_f1': np.nan, 'test_acc': np.nan}
        downstream_time = 0.0
        if args.downstream_eval_each_round:
            eval_seed = downstream_eval_seed(seed, 'market_ours_package_select', round_idx, args)
            round_metrics, downstream_time = run_downstream(current_train, test_pack, args, eval_seed)
            downstream_time_total += downstream_time
        rr = {
            'seed': int(seed), 'market_profile': market_name, 'setting': 'packaged', 'package_method': package_label,
            'package_target_size': int(getattr(args, 'package_target_size', -1)),
            'anchor_target_size': int(getattr(args, 'anchor_target_size', -1)),
            'package_radius_penalty': float(getattr(args, 'package_radius_penalty', 0.0)),
            'package_radius_threshold': float(getattr(args, 'package_radius_threshold', 0.0)),
            'package_purchase_mode': str(getattr(args, 'package_purchase_mode', 'raw_budget')),
            'raw_score_variant': str(getattr(args, 'raw_score_variant', 'base')),
            'structural_prior_mode': str(getattr(args, 'structural_prior_mode', 'hybrid_anti_sim')),
            'structural_prior_alpha': float(getattr(args, 'structural_prior_alpha', 1.0)),
            'package_structural_agg': str(getattr(args, 'package_structural_agg', 'topmean')),
            'package_structural_cluster_weight': float(getattr(args, 'package_structural_cluster_weight', 1.0)),
            'package_teacher_target_mode': str(getattr(args, 'package_teacher_target_mode', 'aggregate')),
            'package_teacher_top_frac': float(getattr(args, 'package_teacher_top_frac', getattr(args, 'package_structural_top_frac', 0.5))),
            'package_student_score_mode': str(getattr(args, 'package_student_score_mode', 'margin')),
            'package_student_base_weight': float(getattr(args, 'package_student_base_weight', 1.0)),
            'package_student_final_weight': float(getattr(args, 'package_student_final_weight', 0.0)),
            'force_package_student': int(bool(getattr(args, 'force_package_student', False))),
            'method': 'market_ours_package_select', 'acquisition_protocol': str(args.acquisition_protocol),
            'pair_model': str(getattr(args, 'pair_model', 'bopa')),
            'bopa_combined_dim': int(getattr(args, 'bopa_combined_dim', 64)),
            'round_idx': int(round_idx), 'selected_rows': int(len(chosen_rows)),
            'selected_good_ratio': float(np.mean(cur_sel)) if len(cur_sel) else 0.0,
            'cumulative_selected_rows': int(len(cum_sel)), 'cumulative_good_count': int(np.sum(cum_sel)),
            'cumulative_good_ratio': float(np.mean(cum_sel)) if len(cum_sel) else 0.0,
            'score_time': score_time, 'selection_time': selection_time, 'teacher_update_time': 0.0,
            'student_train_time': student_time, 'downstream_train_time': downstream_time,
            'test_auroc': float(round_metrics['test_auroc']), 'test_macro_f1': float(round_metrics['test_macro_f1']), 'test_acc': float(round_metrics['test_acc']),
        }
        rr.update(distill_metrics)
        if bool(getattr(args, 'save_selected_indices', False)):
            rr['selected_idx_json'] = json.dumps([int(i) for i in chosen_rows.tolist()], separators=(',', ':'))
            rr['cumulative_idx_json'] = json.dumps([int(i) for i in purchased], separators=(',', ':'))
        rr.update(row_type_composition(market_rows, chosen_rows.tolist(), prefix='round_type'))
        round_rows.append(rr)

        if teacher_needs_model and not should_stop_acquisition(market_rows, purchased, available_rows, args):
            t_up = time.perf_counter()
            teacher, teacher_state, _ = train_pair_model(current_train, args, seed + 301 * round_idx + 7, init_state=teacher_state, max_epochs=int(args.update_max_epochs), patience=int(args.update_patience))
            teacher_update_time = float(time.perf_counter() - t_up)
            teacher_time_total += teacher_update_time
            refresh_count += 1
            round_rows[-1]['teacher_update_time'] = teacher_update_time

    final_metrics = {'test_auroc': np.nan, 'test_macro_f1': np.nan, 'test_acc': np.nan}
    if bool(getattr(args, 'skip_downstream_eval', False)):
        pass
    elif not args.downstream_eval_each_round or len(round_rows) == 0:
        final_metrics, dt = run_downstream(current_train, test_pack, args, seed + 1888 + len(package_method))
        downstream_time_total += dt
    else:
        final_metrics = {k: float(round_rows[-1][k]) for k in ['test_auroc', 'test_macro_f1', 'test_acc']}

    res = summarize_acquisition_result(seed, market_name, 'packaged', package_label, 'market_ours_package_select', market_rows, purchased, refresh_count, teacher_time_total, student_time_total, score_time_total, selection_time_total, downstream_time_total, final_metrics, round_idx, args)
    res['package_purchase_mode'] = str(getattr(args, 'package_purchase_mode', 'raw_budget'))
    res['raw_score_variant'] = str(getattr(args, 'raw_score_variant', 'base'))
    res['structural_prior_mode'] = str(getattr(args, 'structural_prior_mode', 'hybrid_anti_sim'))
    res['structural_prior_alpha'] = float(getattr(args, 'structural_prior_alpha', 1.0))
    res['package_structural_agg'] = str(getattr(args, 'package_structural_agg', 'topmean'))
    res['package_structural_cluster_weight'] = float(getattr(args, 'package_structural_cluster_weight', 1.0))
    res['package_teacher_target_mode'] = str(getattr(args, 'package_teacher_target_mode', 'aggregate'))
    res['package_teacher_top_frac'] = float(getattr(args, 'package_teacher_top_frac', getattr(args, 'package_structural_top_frac', 0.5)))
    res['package_student_score_mode'] = str(getattr(args, 'package_student_score_mode', 'margin'))
    res['package_student_base_weight'] = float(getattr(args, 'package_student_base_weight', 1.0))
    res['package_student_final_weight'] = float(getattr(args, 'package_student_final_weight', 0.0))
    res['force_package_student'] = int(bool(getattr(args, 'force_package_student', False)))
    for metric_name in ['distill_pearson', 'distill_spearman', 'distill_mse', 'distill_top50_overlap']:
        vals = [float(rr.get(metric_name, np.nan)) for rr in round_rows if np.isfinite(float(rr.get(metric_name, np.nan)))]
        res[metric_name + '_mean'] = float(np.mean(vals)) if len(vals) else float('nan')
    res['package_rebuilt_each_round'] = 1
    return res, round_rows




# =========================================================
# Package sweep helpers
# =========================================================


def parse_float_list(text: str) -> List[float]:
    return [float(t.strip()) for t in str(text).split(',') if t.strip()]


def _sweep_label(method: str, package_size: int, anchor_size: int, radius_penalty: float) -> str:
    # This label is used only for result tables. The actual package construction
    # method remains one of anchor_greedy / anchor_kmeans / anchor_recursive.
    rp = ("%.4g" % float(radius_penalty)).replace('.', 'p')
    return f"{method}_p{int(package_size)}_a{int(anchor_size)}_rp{rp}"


def build_package_sweep_configs(args: argparse.Namespace) -> List[Dict[str, object]]:
    methods = list(args.package_methods)
    if bool(getattr(args, 'no_package_sweep', False)):
        return [{
            'method': m,
            'package_target_size': int(args.package_target_size),
            'anchor_target_size': int(args.anchor_target_size),
            'package_radius_penalty': float(args.package_radius_penalty),
            'label': _sweep_label(m, int(args.package_target_size), int(args.anchor_target_size), float(args.package_radius_penalty)),
        } for m in methods]

    package_sizes = parse_int_list(getattr(args, 'package_target_sizes', ''))
    anchor_sizes = parse_int_list(getattr(args, 'anchor_target_sizes', ''))
    penalties = parse_float_list(getattr(args, 'package_radius_penalties', ''))

    if len(package_sizes) == 0:
        package_sizes = [int(args.package_target_size)]
    if len(anchor_sizes) == 0:
        anchor_sizes = [int(args.anchor_target_size)]
    if len(penalties) == 0:
        penalties = [float(args.package_radius_penalty)]

    mode = str(getattr(args, 'package_sweep_mode', 'grid')).lower()
    configs: List[Dict[str, object]] = []

    if mode == 'paired':
        max_len = max(len(package_sizes), len(anchor_sizes), len(penalties))
        if len(package_sizes) not in {1, max_len} or len(anchor_sizes) not in {1, max_len} or len(penalties) not in {1, max_len}:
            raise ValueError('For package_sweep_mode=paired, each list length must be 1 or the same maximum length.')
        for m in methods:
            for i in range(max_len):
                psize = package_sizes[i if len(package_sizes) > 1 else 0]
                asize = anchor_sizes[i if len(anchor_sizes) > 1 else 0]
                pen = penalties[i if len(penalties) > 1 else 0]
                configs.append({'method': m, 'package_target_size': int(psize), 'anchor_target_size': int(asize), 'package_radius_penalty': float(pen), 'label': _sweep_label(m, int(psize), int(asize), float(pen))})
    elif mode == 'grid':
        for m in methods:
            for psize in package_sizes:
                for asize in anchor_sizes:
                    for pen in penalties:
                        configs.append({'method': m, 'package_target_size': int(psize), 'anchor_target_size': int(asize), 'package_radius_penalty': float(pen), 'label': _sweep_label(m, int(psize), int(asize), float(pen))})
    else:
        raise ValueError('package_sweep_mode must be grid or paired.')

    # Deduplicate while preserving order.
    seen = set()
    dedup: List[Dict[str, object]] = []
    for c in configs:
        key = (c['method'], c['package_target_size'], c['anchor_target_size'], c['package_radius_penalty'])
        if key not in seen:
            seen.add(key)
            dedup.append(c)
    return dedup


def make_args_for_package_config(args: argparse.Namespace, cfg: Dict[str, object]) -> argparse.Namespace:
    # Shallow-copy argparse namespace so row-level settings are unchanged.
    new_args = argparse.Namespace(**vars(args))
    new_args.package_target_size = int(cfg['package_target_size'])
    new_args.anchor_target_size = int(cfg['anchor_target_size'])
    new_args.package_radius_penalty = float(cfg['package_radius_penalty'])
    pkg_variant = str(getattr(args, 'package_raw_score_variant', 'same_as_global'))
    if pkg_variant != 'same_as_global':
        new_args.raw_score_variant = pkg_variant
    return new_args


# =========================================================
# Experiment orchestration
# =========================================================



def build_raw_variant_configs(args: argparse.Namespace) -> List[Dict[str, object]]:
    if bool(getattr(args, 'no_ours_raw', False)):
        return []
    variants = parse_list(getattr(args, 'raw_variants', 'base,gated,gated_balanced'))
    configs: List[Dict[str, object]] = []
    for v in variants:
        name = str(v).strip()
        if not name:
            continue
        if name == 'base':
            configs.append({'label': 'ours_raw_base', 'raw_score_variant': 'base', 'label_balanced_select': False})
        elif name == 'base_balanced':
            configs.append({'label': 'ours_raw_base_balanced', 'raw_score_variant': 'base', 'label_balanced_select': True})
        elif name == 'gated':
            configs.append({'label': 'ours_raw_gated', 'raw_score_variant': 'gated', 'label_balanced_select': False})
        elif name == 'gated_balanced':
            configs.append({'label': 'ours_raw_gated_balanced', 'raw_score_variant': 'gated', 'label_balanced_select': True})
        elif name == 'gated_need':
            configs.append({'label': 'ours_raw_gated_need', 'raw_score_variant': 'gated_need', 'label_balanced_select': False})
        elif name == 'gated_need_balanced':
            configs.append({'label': 'ours_raw_gated_need_balanced', 'raw_score_variant': 'gated_need', 'label_balanced_select': True})
        elif name in ['structural', 'structural_prior']:
            configs.append({'label': 'ours_raw_structural_prior', 'raw_score_variant': 'structural', 'label_balanced_select': False})
        elif name in ['structural_balanced', 'structural_prior_balanced']:
            configs.append({'label': 'ours_raw_structural_prior_balanced', 'raw_score_variant': 'structural', 'label_balanced_select': True})
        elif name == 'gated_need_structural':
            configs.append({'label': 'ours_raw_gated_need_structural', 'raw_score_variant': 'gated_need_structural', 'label_balanced_select': False})
        elif name == 'gated_need_structural_balanced':
            configs.append({'label': 'ours_raw_gated_need_structural_balanced', 'raw_score_variant': 'gated_need_structural', 'label_balanced_select': True})
        else:
            raise ValueError('Unknown raw variant: ' + name)
    # Deduplicate while preserving order.
    seen = set()
    out: List[Dict[str, object]] = []
    for c in configs:
        key = (c['label'], c['raw_score_variant'], c['label_balanced_select'])
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def make_args_for_raw_config(args: argparse.Namespace, cfg: Dict[str, object]) -> argparse.Namespace:
    new_args = argparse.Namespace(**vars(args))
    new_args.raw_score_variant = str(cfg['raw_score_variant'])
    new_args.label_balanced_select = bool(cfg['label_balanced_select'])
    return new_args


def run_one_seed(seed: int, train_pack: Dict[str, np.ndarray], test_pack: Dict[str, np.ndarray], markets_root: Path, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    rng = np.random.default_rng(seed + 1)
    idx = np.arange(len(train_pack['y']), dtype=np.int64)
    rng.shuffle(idx)
    initial_idx = idx[: int(args.initial_noisy_size)].astype(np.int64)
    _, initial_noisy = build_teacher_local_packs(train_pack, initial_idx, seed)
    results: List[Dict[str, object]] = []
    rounds: List[Dict[str, object]] = []
    drops: List[Dict[str, object]] = []

    raw_configs = build_raw_variant_configs(args)
    for mi, market_name in enumerate(args.market_profiles, start=1):
        print(f"  [market {mi}/{len(args.market_profiles)}] {market_name}")
        market_rows = load_market_npz(markets_root, market_name)
        print('    [market stats] n=', len(market_rows['y']), 'good_ratio=', float(np.mean(market_rows['is_good'])))
        if int(len(market_rows['y'])) != int(args.market_size):
            print('    [warn] loaded market size differs from args.market_size. Use a fresh --markets_dir when changing --market_size.')

        baseline_results: Dict[str, Dict[str, object]] = {}
        if bool(getattr(args, 'include_raw_references', True)):
            reference_methods = parse_list(getattr(
                args,
                'reference_methods',
                (
                    'oracle_good_select,market_random_select,market_cosine_select,'
                    'market_entropy_select,market_margin_select,market_badge_select,'
                    'market_typiclust_select'
                ),
            ))
            allowed_refs = {
                'oracle_good_select',
                'market_random_select',
                'market_cosine_select',
                'market_entropy_select',
                'market_typiclust_select',
                'market_margin_select',
                'market_badge_select',
                'market_coreset_select',
                'market_kmeans_center_select',
            }
            for method in reference_methods:
                if method not in allowed_refs:
                    raise ValueError('Unknown reference method: ' + str(method))
                print(f"    [reference] {method}")
                res, rd = run_row_method(seed, market_name, market_rows, initial_noisy, test_pack, args, method)
                results.append(res)
                rounds.extend(rd)
                baseline_results[method] = res

        raw_results: Dict[str, Dict[str, object]] = {}
        for cfg in raw_configs:
            label = str(cfg['label'])
            r_args = make_args_for_raw_config(args, cfg)
            print(f"    [raw] {label}  score={r_args.raw_score_variant} balanced={int(r_args.label_balanced_select)}")
            res, rd = run_row_method(seed, market_name, market_rows, initial_noisy, test_pack, r_args, 'market_ours_row_select', method_label=label)
            results.append(res)
            rounds.extend(rd)
            raw_results[label] = res

        packaged_results: Dict[str, Dict[str, object]] = {}
        if not bool(getattr(args, 'no_packaged', False)):
            pkg_configs = build_package_sweep_configs(args)
            print(f"    [packaged] {len(pkg_configs)} config(s)")
            for pcfg in pkg_configs:
                p_args = make_args_for_package_config(args, pcfg)
                label = str(pcfg['label'])
                print(
                    f"    [package] {label} score={p_args.raw_score_variant} "
                    f"window={p_args.package_target_size} anchor={p_args.anchor_target_size} "
                    f"rp={p_args.package_radius_penalty}"
                )
                res, rd = run_packaged_ours(
                    seed, market_name, market_rows, initial_noisy, test_pack,
                    p_args, str(pcfg['method']), package_label=label,
                )
                results.append(res)
                rounds.extend(rd)
                packaged_results[label] = res

        if raw_results:
            best_raw = max(raw_results.values(), key=lambda r: float(r['purchase_efficiency']))
            random_ref = baseline_results.get('market_random_select')
            oracle_ref = baseline_results.get('oracle_good_select')
            row = {
                'seed': int(seed),
                'market_profile': market_name,
                'comparison_type': 'best_raw',
                'package_method': 'none',
                'best_raw_method': str(best_raw['method']),
                'best_raw_good_count': int(best_raw['good_count']),
                'best_raw_purchase_efficiency': float(best_raw['purchase_efficiency']),
                'best_raw_test_auroc': float(best_raw['test_auroc']),
            }
            if random_ref is not None:
                row.update({
                    'random_good_count': int(random_ref['good_count']),
                    'random_purchase_efficiency': float(random_ref['purchase_efficiency']),
                    'raw_gain_good_count_vs_random': int(best_raw['good_count']) - int(random_ref['good_count']),
                    'raw_gain_efficiency_vs_random': float(best_raw['purchase_efficiency']) - float(random_ref['purchase_efficiency']),
                })
            if oracle_ref is not None:
                row.update({
                    'oracle_good_count': int(oracle_ref['good_count']),
                    'raw_shortfall_vs_oracle': int(oracle_ref['good_count']) - int(best_raw['good_count']),
                })
            drops.append(row)

            for pkg_label, pkg_res in packaged_results.items():
                prow = {
                    'seed': int(seed),
                    'market_profile': market_name,
                    'comparison_type': 'packaged_vs_best_raw',
                    'package_method': str(pkg_label),
                    'best_raw_method': str(best_raw['method']),
                    'best_raw_good_count': int(best_raw['good_count']),
                    'best_raw_purchase_efficiency': float(best_raw['purchase_efficiency']),
                    'packaged_good_count': int(pkg_res['good_count']),
                    'packaged_purchase_efficiency': float(pkg_res['purchase_efficiency']),
                    'packaged_minus_best_raw_good_count': int(pkg_res['good_count']) - int(best_raw['good_count']),
                    'packaged_minus_best_raw_efficiency': float(pkg_res['purchase_efficiency']) - float(best_raw['purchase_efficiency']),
                }
                if random_ref is not None:
                    prow.update({
                        'random_good_count': int(random_ref['good_count']),
                        'random_purchase_efficiency': float(random_ref['purchase_efficiency']),
                        'packaged_gain_good_count_vs_random': int(pkg_res['good_count']) - int(random_ref['good_count']),
                        'packaged_gain_efficiency_vs_random': float(pkg_res['purchase_efficiency']) - float(random_ref['purchase_efficiency']),
                    })
                drops.append(prow)
    return pd.DataFrame(results), pd.DataFrame(rounds), pd.DataFrame(drops)

def _flatten_groupby_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c if isinstance(c, str) else (c[0] if c[1] == '' else f"{c[0]}_{c[1]}") for c in df.columns.to_flat_index()]
    return df


def build_paper_tables(results_df: pd.DataFrame, out_dir: Path) -> None:
    table_dir = out_dir / 'paper_tables'
    ensure_dir(table_dir)
    if results_df.empty:
        return
    method_s = results_df['method'].astype(str)
    raw_ours_mask = method_s.str.startswith('ours_raw_')
    row_ref_mask = method_s.isin([
        'oracle_good_select', 'market_ours_row_select', 'market_cosine_select',
        'market_entropy_select', 'market_margin_select', 'market_badge_select',
        'market_coreset_select', 'market_kmeans_center_select',
        'market_typiclust_select', 'market_random_select',
    ])
    main_rows = results_df[
        (((results_df['setting'] == 'row_level') & (row_ref_mask | raw_ours_mask))
         | ((results_df['setting'] == 'packaged') & (results_df['method'] == 'market_ours_package_select')))
    ].copy()
    name_map = {
        'oracle_good_select': 'Oracle Good',
        'market_ours_row_select': 'Ours Row',
        'market_ours_package_select': 'Ours Packaged',
        'market_cosine_select': 'Cosine',
        'market_entropy_select': 'Entropy',
        'market_margin_select': 'Margin',
        'market_badge_select': 'BADGE approx.',
        'market_coreset_select': 'CoreSet',
        'market_kmeans_center_select': 'KMeans-center',
        'market_typiclust_select': 'TypiClust',
        'market_random_select': 'Random',
        'ours_raw_base': 'Ours Raw Base',
        'ours_raw_base_balanced': 'Ours Raw Base Balanced',
        'ours_raw_gated': 'Ours Raw Gated',
        'ours_raw_gated_balanced': 'Ours Raw Gated Balanced',
        'ours_raw_gated_need': 'Ours Raw Gated Need',
        'ours_raw_gated_need_balanced': 'Ours Raw Gated Need Balanced',
        'ours_raw_structural_prior': 'Ours Raw Structural',
        'ours_raw_structural_prior_balanced': 'Ours Raw Structural Balanced',
        'ours_raw_gated_need_structural': 'Ours Raw Gated Need Structural',
        'ours_raw_gated_need_structural_balanced': 'Ours Raw Gated Need Structural Balanced',
    }
    main_rows['paper_method'] = main_rows['method'].map(name_map).fillna(main_rows['method'])
    pkg_mask = (main_rows['setting'] == 'packaged') & (main_rows['method'] == 'market_ours_package_select')
    main_rows.loc[pkg_mask, 'paper_method'] = 'Ours Packaged (' + main_rows.loc[pkg_mask, 'package_method'].astype(str) + ')'
    metrics = [
        'package_target_size', 'anchor_target_size', 'package_radius_penalty', 'reached_target_good', 'reached_raw_budget', 'good_shortfall_vs_target', 'raw_budget_gap', 'oracle_matched_raw_target', 'rounds_used', 'selected_rows', 'good_count', 'bad_count',
        'purchase_efficiency', 'extra_rows_vs_oracle_good', 'selected_good_ratio',
        'selected_first1000_good_ratio', 'selection_lift_vs_market', 'score_time_total',
        'selection_time_total', 'teacher_time_total', 'student_time_total',
        'downstream_train_time_total', 'test_auroc', 'test_macro_f1', 'test_acc',
        'label_balanced_select', 'structural_prior_alpha',
        'package_structural_cluster_weight', 'package_rebuilt_each_round',
    ]
    metrics = [m for m in metrics if m in main_rows.columns]
    overall = main_rows.groupby('paper_method', dropna=False)[metrics].agg(['mean', 'std']).reset_index()
    overall = _flatten_groupby_columns(overall)
    overall.to_csv(table_dir / 'paper_main_table_overall.csv', index=False)
    per_market = main_rows.groupby(['market_profile', 'paper_method'], dropna=False)[metrics].agg(['mean', 'std']).reset_index()
    per_market = _flatten_groupby_columns(per_market)
    per_market.to_csv(table_dir / 'paper_main_table_per_market.csv', index=False)

    # Raw-only branch convenience table: rank all raw variants and references.
    sort_cols = [c for c in ['good_count_mean', 'purchase_efficiency_mean', 'test_auroc_mean'] if c in overall.columns]
    if sort_cols:
        ranked = overall.sort_values(sort_cols, ascending=[False] * len(sort_cols)).copy()
        ranked.to_csv(table_dir / 'paper_raw_branch_ranked.csv', index=False)
    pkg = results_df[(results_df['setting'] == 'packaged') & (results_df['method'] == 'market_ours_package_select')].copy()
    if len(pkg) > 0:
        pkg_metrics = [m for m in metrics if m in pkg.columns]
        pkg_table = pkg.groupby('package_method', dropna=False)[pkg_metrics].agg(['mean', 'std']).reset_index()
        pkg_table = _flatten_groupby_columns(pkg_table)
        pkg_table.to_csv(table_dir / 'paper_package_ablation_table.csv', index=False)
        # A sorted convenience table. Lower bad_count and higher good_count / purchase_efficiency are preferred.
        sort_cols = [c for c in ['good_count_mean', 'purchase_efficiency_mean', 'test_auroc_mean'] if c in pkg_table.columns]
        if sort_cols:
            ranked = pkg_table.sort_values(sort_cols, ascending=[False] * len(sort_cols)).copy()
            ranked.to_csv(table_dir / 'paper_package_sweep_ranked.csv', index=False)
    try:
        overall.to_latex(table_dir / 'paper_main_table_overall.tex', index=False, float_format='%.4f')
        per_market.to_latex(table_dir / 'paper_main_table_per_market.tex', index=False, float_format='%.4f')
        if len(pkg) > 0:
            pkg_table.to_latex(table_dir / 'paper_package_ablation_table.tex', index=False, float_format='%.4f')
    except Exception as e:
        print('[warn] LaTeX export failed:', repr(e))


def build_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_base_dir = script_dir / 'feature_cache' / 'hateful_memes' / 'clip_vit_base_patch32' / 'base'
    default_source_markets_dir = script_dir / 'feature_cache' / 'hateful_memes' / 'clip_vit_base_patch32' / 'markets' / 'seed42_ratio02_purchase3000'
    default_markets_dir = script_dir / 'feature_cache' / 'hateful_memes' / 'clip_vit_base_patch32' / 'markets' / 'article_sparse_good_v3'
    default_output_dir = script_dir / 'outputs' / 'hm_tdsc_nonpackage_ablations'
    if not default_base_dir.exists():
        alt = script_dir.parent / 'feature_cache' / 'hateful_memes' / 'clip_vit_base_patch32' / 'base'
        if alt.exists():
            default_base_dir = alt
    if not default_source_markets_dir.exists():
        alt = script_dir.parent / 'feature_cache' / 'hateful_memes' / 'clip_vit_base_patch32' / 'markets' / 'seed42_ratio02_purchase3000'
        if alt.exists():
            default_source_markets_dir = alt
    if not default_markets_dir.exists():
        alt = script_dir.parent / 'feature_cache' / 'hateful_memes' / 'clip_vit_base_patch32' / 'markets' / 'article_sparse_good_v3'
        if alt.exists():
            default_markets_dir = alt

    ap = argparse.ArgumentParser()
    ap.add_argument('--base_dir', type=str, default=str(default_base_dir))
    ap.add_argument('--source_markets_dir', type=str, default=str(default_source_markets_dir))
    ap.add_argument('--markets_dir', type=str, default=str(default_markets_dir))
    ap.add_argument('--output_dir', type=str, default=str(default_output_dir))
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch_size', type=int, default=256)

    ap.add_argument('--seeds', type=str, default='42')
    ap.add_argument('--market_profiles', type=str, default='bcs_sparse_hard_conflict_v2,bcs_sparse_decision_boundary_v2,bcs_sparse_adversarial_mixed_v2')
    ap.add_argument('--package_methods', type=str, default='anchor_greedy')

    ap.add_argument('--build_sparse_markets', action='store_true')
    ap.add_argument('--only_build_markets', action='store_true')
    ap.add_argument('--allow_missing_components', action='store_true')
    ap.add_argument('--disable_strategic_market_sampling', action='store_true',
                    help='Use plain random component sampling even for the malicious coverage/boundary/package trap profiles.')
    ap.add_argument('--market_size', type=int, default=100000 )
    ap.add_argument('--market_seed', type=int, default=42)

    ap.add_argument('--initial_noisy_size', type=int, default=1000)
    ap.add_argument('--purchase_total', type=int, default=50)
    ap.add_argument('--round_budget', type=int, default=5)
    ap.add_argument('--acquisition_protocol', type=str, default='fixed_budget', choices=['fixed_budget', 'target_good', 'oracle_matched'])
    ap.add_argument('--target_good_count', type=int, default=1000)
    ap.add_argument('--max_purchase_rows', type=int, default=15000)

    ap.add_argument('--pair_model', type=str, default='bopa', choices=['bopa', 'mlp'])
    ap.add_argument('--hidden_dim', type=int, default=256)
    ap.add_argument('--bopa_combined_dim', type=int, default=64)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--val_ratio', type=float, default=0.15)
    ap.add_argument('--init_max_epochs', type=int, default=5)
    ap.add_argument('--init_patience', type=int, default=2)
    ap.add_argument('--update_max_epochs', type=int, default=1)
    ap.add_argument('--update_patience', type=int, default=1)
    ap.add_argument('--downstream_max_epochs', type=int, default=5)
    ap.add_argument('--downstream_patience', type=int, default=2)
    ap.add_argument('--downstream_eval_each_round', action='store_true')
    ap.add_argument('--downstream_eval_fixed_seed', action='store_true')
    ap.add_argument('--skip_downstream_eval', action='store_true',
                    help='Skip downstream training/evaluation entirely; useful for selection-quality score ablations.')
    ap.add_argument('--closed_loop_downstream_teacher', action='store_true',
                    help='For model-based raw selectors, train the downstream model after each purchase round and reuse that converged model as the next-round teacher.')
    ap.add_argument('--closed_loop_retrain_from_scratch', action='store_true',
                    help='With --closed_loop_downstream_teacher, retrain each downstream/teacher model from scratch instead of warm-starting from the previous teacher state.')
    ap.add_argument('--acquisition_convergence_stop', action='store_true',
                    help='Stop row-level acquisition early when the per-round downstream metric has plateaued.')
    ap.add_argument('--convergence_metric', type=str, default='test_macro_f1',
                    choices=['test_auroc', 'test_macro_f1', 'test_acc'])
    ap.add_argument('--convergence_min_delta', type=float, default=0.001)
    ap.add_argument('--convergence_patience', type=int, default=3)
    ap.add_argument('--convergence_min_rounds', type=int, default=5)
    ap.add_argument('--save_selected_indices', action='store_true')

    ap.add_argument('--row_dim', type=int, default=32)
    ap.add_argument('--cos_alpha', type=float, default=0.10)
    ap.add_argument('--neg_margin_lambda', type=float, default=1.00)
    ap.add_argument('--student_hidden_dim', type=int, default=32)
    ap.add_argument('--student_dropout', type=float, default=0.05)
    ap.add_argument('--student_lr', type=float, default=2e-3)
    ap.add_argument('--student_weight_decay', type=float, default=1e-4)
    ap.add_argument('--student_epochs', type=int, default=5)
    ap.add_argument('--student_margin_weight', type=float, default=1.0)
    ap.add_argument('--student_rank_weight', type=float, default=0.5)
    ap.add_argument('--student_rank_pairs', type=int, default=512)
    ap.add_argument('--package_student_base_weight', type=float, default=1.0,
                    help='Weight for package-student base-margin regression. Defaults preserve the original margin-student objective.')
    ap.add_argument('--package_student_final_weight', type=float, default=0.0,
                    help='Weight for direct package teacher-target regression.')
    ap.add_argument('--package_student_score_mode', type=str, default='margin', choices=['margin', 'final'],
                    help='Use margin-derived score or the direct final-score head for package ranking.')
    ap.add_argument('--force_package_student', action='store_true',
                    help='Train a package student even when the package raw score is pure structural.')
    ap.add_argument('--rank_margin', type=float, default=0.05)
    ap.add_argument('--rank_eps', type=float, default=1e-4)
    ap.add_argument('--typiclust_k', type=int, default=8)

    # Raw and packaged branch controls
    ap.add_argument('--raw_variants', type=str, default='structural_prior_balanced')
    ap.add_argument('--no_ours_raw', action='store_true')
    ap.add_argument('--raw_score_variant', type=str, default='base', choices=['base', 'gated', 'gated_need', 'structural', 'gated_need_structural'])
    ap.add_argument('--candidate_gate_tau', type=float, default=1.0)
    ap.add_argument('--need_gate_tau', type=float, default=1.0)
    ap.add_argument('--structural_prior_mode', type=str, default='hybrid_anti_sim',
                    choices=['anti_image_sim', 'hybrid_anti_sim', 'low_triad', 'dist_minus_image'])
    ap.add_argument('--structural_prior_alpha', type=float, default=1.0)
    ap.add_argument('--structural_prior_clip', type=float, default=4.0)
    ap.add_argument('--label_balanced_select', action='store_true')
    ap.add_argument('--include_raw_references', action='store_true', default=True)
    ap.add_argument('--no_raw_references', action='store_true')
    ap.add_argument(
        '--reference_methods',
        type=str,
        default=(
            'oracle_good_select,market_random_select,market_cosine_select,'
            'market_entropy_select,market_margin_select,market_badge_select,'
            'market_typiclust_select'
        ),
    )

    ap.add_argument('--no_packaged', action='store_true')
    ap.add_argument('--package_purchase_mode', type=str, default='raw_budget', choices=['raw_budget', 'strict_package'])
    ap.add_argument('--anchor_target_size', type=int, default=16)
    ap.add_argument('--package_target_size', type=int, default=2)
    # Small packages are the default because they preserve the high-precision raw
    # structural signal while still letting packaged scoring group nearby rows.
    ap.add_argument('--package_target_sizes', type=str, default='2')
    ap.add_argument('--package_windows', type=str, default='', help='Alias of --package_target_sizes; e.g., 2,4,8.')
    ap.add_argument('--anchor_target_sizes', type=str, default='16')
    ap.add_argument('--anchor_windows', type=str, default='', help='Alias of --anchor_target_sizes; e.g., 16,32,64.')
    ap.add_argument('--package_sweep_mode', type=str, default='grid', choices=['grid', 'paired'])
    ap.add_argument('--no_package_sweep', action='store_true')
    ap.add_argument('--package_radius_threshold', type=float, default=1.10)
    ap.add_argument('--package_radius_penalty', type=float, default=0.0)
    ap.add_argument('--package_radius_penalties', type=str, default='0.0')
    ap.add_argument('--package_raw_score_variant', type=str, default='structural',
                    choices=['base', 'gated', 'gated_need', 'structural', 'gated_need_structural', 'same_as_global'])
    ap.add_argument('--package_structural_agg', type=str, default='topmean', choices=['mean', 'max', 'topmean'])
    ap.add_argument('--package_structural_top_frac', type=float, default=0.5)
    ap.add_argument('--package_structural_cluster_weight', type=float, default=1.0)
    ap.add_argument('--package_teacher_target_mode', type=str, default='aggregate',
                    choices=['aggregate', 'topmean', 'mean', 'shuffled', 'structural'])
    ap.add_argument('--package_teacher_top_frac', type=float, default=0.5)
    ap.add_argument('--package_size_penalty', type=float, default=0.0)
    ap.add_argument('--package_sort_rows_by_structural', action='store_true', default=True)
    ap.add_argument('--no_package_sort_rows_by_structural', action='store_true')

    # TDSC non-package ablation suite. Package scoring is fixed to the main
    # structural configuration above; the suite varies market, budget, scale,
    # student/teacher controls, crypto proxy cost, baselines, and malicious mix.
    ap.add_argument('--experiment_suite', type=str, default='sparsity,budget,scale',
                    help='Comma list from: single,sparsity,budget,scale,student_ablation,crypto,baselines,malicious,all.')
    ap.add_argument('--sweep_good_ratios', type=str, default='0.01,0.05,0.10,0.20')
    ap.add_argument('--sweep_market_sizes', type=str, default='10000,50000,100000')
    ap.add_argument('--sweep_purchase_totals', type=str, default='25,50,100,200')
    ap.add_argument('--sweep_round_budgets', type=str, default='5,5,10,20')
    ap.add_argument('--use_dynamic_market_profiles', action='store_true')
    ap.add_argument('--market_good_ratio_override', type=float, default=0.10)
    ap.add_argument('--dynamic_profile_prefix', type=str, default='dynamic', nargs='?', const='')
    ap.add_argument('--market_malicious_mix', type=str, default='none',
                    choices=['none', 'adversarial_heavy', 'conflict_heavy', 'decision_heavy', 'balanced_hard'])
    ap.add_argument('--mark_non_good_as_malicious', action='store_true')
    ap.add_argument('--crypto_package_counts', type=str, default='100,500,1000,5000,10000')
    ap.add_argument('--crypto_feature_dims', type=str, default='16,32,64')
    ap.add_argument('--crypto_repeats', type=int, default=5)

    ap.add_argument('--debug_small', action='store_true')
    ap.add_argument('--main_package_only', action='store_true', default=True)
    ap.add_argument('--full_package_ablation', action='store_true')
    return ap.parse_args(argv)


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.base_dir = clean_cli_path(args.base_dir)
    args.source_markets_dir = clean_cli_path(args.source_markets_dir)
    args.markets_dir = clean_cli_path(args.markets_dir)
    args.output_dir = clean_cli_path(args.output_dir)
    args.seeds = parse_int_list(args.seeds) if isinstance(args.seeds, str) else list(args.seeds)
    args.market_profiles = parse_list(args.market_profiles) if isinstance(args.market_profiles, str) else list(args.market_profiles)
    args.package_methods = parse_list(args.package_methods) if isinstance(args.package_methods, str) else list(args.package_methods)
    if bool(getattr(args, 'use_dynamic_market_profiles', False)):
        args.market_profiles = list(get_sparse_market_profiles(args).keys())
    if str(getattr(args, 'package_windows', '')).strip():
        args.package_target_sizes = str(args.package_windows)
    if str(getattr(args, 'anchor_windows', '')).strip():
        args.anchor_target_sizes = str(args.anchor_windows)
    if bool(getattr(args, 'no_package_sort_rows_by_structural', False)):
        args.package_sort_rows_by_structural = False
    if bool(getattr(args, 'no_raw_references', False)):
        args.include_raw_references = False
    if args.full_package_ablation:
        args.package_methods = ['anchor_greedy', 'anchor_kmeans', 'anchor_recursive']
    elif args.main_package_only:
        args.package_methods = ['anchor_greedy']
    if is_oracle_matched_protocol(args):
        # The requested protocol trains/evaluates after every matched raw-data batch.
        args.downstream_eval_each_round = True
    if bool(getattr(args, 'closed_loop_downstream_teacher', False)):
        # Closed-loop acquisition needs a converged downstream model at each round.
        args.downstream_eval_each_round = True
    if bool(getattr(args, 'acquisition_convergence_stop', False)):
        # Metric-plateau stopping is defined on the per-round downstream curve.
        args.downstream_eval_each_round = True
    if bool(getattr(args, 'skip_downstream_eval', False)):
        args.downstream_eval_each_round = False
        args.closed_loop_downstream_teacher = False
        args.acquisition_convergence_stop = False
    if args.debug_small:
        args.seeds = [42]
        if bool(getattr(args, 'use_dynamic_market_profiles', False)):
            args.market_profiles = list(args.market_profiles)[:1]
        else:
            args.market_profiles = ['bcs_sparse_hard_conflict_v2']
        args.package_methods = ['anchor_greedy']
        args.raw_variants = 'structural_prior_balanced,gated_need_balanced'
        args.package_target_sizes = '2'
        args.anchor_target_sizes = '16'
        args.package_radius_penalties = '0.0'
        args.target_good_count = 100
        args.max_purchase_rows = 2000
        args.round_budget = 50
        args.init_max_epochs = 2
        args.downstream_max_epochs = 2
        args.student_epochs = 2
        args.output_dir = str(Path(args.output_dir) / 'debug_small')
    return args


def validate_args(args: argparse.Namespace) -> None:
    if str(getattr(args, 'pair_model', 'bopa')).lower() not in {'bopa', 'mlp'}:
        raise ValueError('pair_model must be one of: bopa, mlp.')
    if int(getattr(args, 'bopa_combined_dim', 64)) <= 0:
        raise ValueError('bopa_combined_dim must be positive.')
    if int(args.round_budget) <= 0:
        raise ValueError('round_budget must be positive.')
    if is_target_good_protocol(args) or is_oracle_matched_protocol(args):
        if int(args.target_good_count) <= 0:
            raise ValueError('target_good_count must be positive.')
        if is_target_good_protocol(args) and int(args.max_purchase_rows) < int(args.target_good_count):
            raise ValueError('max_purchase_rows must be >= target_good_count under target_good.')
    else:
        if int(args.purchase_total) <= 0:
            raise ValueError('purchase_total must be positive under fixed_budget.')
    if int(args.package_target_size) <= 0 or int(args.anchor_target_size) <= 0:
        raise ValueError('package_target_size and anchor_target_size must be positive.')
    for x in parse_int_list(getattr(args, 'package_target_sizes', str(args.package_target_size))):
        if int(x) <= 0:
            raise ValueError('All package_target_sizes must be positive.')
    for x in parse_int_list(getattr(args, 'anchor_target_sizes', str(args.anchor_target_size))):
        if int(x) <= 0:
            raise ValueError('All anchor_target_sizes must be positive.')
    for x in parse_float_list(getattr(args, 'package_radius_penalties', str(args.package_radius_penalty))):
        if float(x) < 0:
            raise ValueError('All package_radius_penalties must be non-negative.')
    if float(getattr(args, 'structural_prior_alpha', 1.0)) < 0:
        raise ValueError('structural_prior_alpha must be non-negative.')
    if float(getattr(args, 'structural_prior_clip', 4.0)) < 0:
        raise ValueError('structural_prior_clip must be non-negative.')
    if float(getattr(args, 'neg_margin_lambda', 1.0)) < 0:
        raise ValueError('neg_margin_lambda must be non-negative.')
    if float(getattr(args, 'package_student_base_weight', 1.0)) < 0:
        raise ValueError('package_student_base_weight must be non-negative.')
    if float(getattr(args, 'package_student_final_weight', 0.0)) < 0:
        raise ValueError('package_student_final_weight must be non-negative.')
    if float(getattr(args, 'package_structural_top_frac', 0.5)) <= 0 or float(getattr(args, 'package_structural_top_frac', 0.5)) > 1:
        raise ValueError('package_structural_top_frac must be in (0, 1].')
    if float(getattr(args, 'package_teacher_top_frac', 0.5)) <= 0 or float(getattr(args, 'package_teacher_top_frac', 0.5)) > 1:
        raise ValueError('package_teacher_top_frac must be in (0, 1].')
    if float(getattr(args, 'package_structural_cluster_weight', 1.0)) < 0:
        raise ValueError('package_structural_cluster_weight must be non-negative.')
    if float(getattr(args, 'package_size_penalty', 0.0)) < 0:
        raise ValueError('package_size_penalty must be non-negative.')
    if int(getattr(args, 'convergence_patience', 3)) <= 0:
        raise ValueError('convergence_patience must be positive.')
    if int(getattr(args, 'convergence_min_rounds', 5)) <= 0:
        raise ValueError('convergence_min_rounds must be positive.')
    if float(getattr(args, 'convergence_min_delta', 0.001)) < 0:
        raise ValueError('convergence_min_delta must be non-negative.')


def run_all(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    args = normalize_args(args)
    validate_args(args)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    if args.build_sparse_markets:
        build_sparse_markets(args)
        if args.only_build_markets:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Do not let the experiment fail later only because the sparse markets were not built.
    # If the requested market files are absent, build them once automatically.
    ensure_sparse_markets_exist(args)

    cfg = {k: v for k, v in vars(args).items()}
    (out_dir / 'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[config] device=', args.device)
    print('[config] seeds=', args.seeds)
    print('[config] markets=', args.market_profiles)
    print('[config] raw_variants=', args.raw_variants)
    print('[config] reference_methods=', getattr(args, 'reference_methods', ''))
    print('[config] pair_model=', getattr(args, 'pair_model', 'bopa'), 'bopa_combined_dim=', getattr(args, 'bopa_combined_dim', 64), 'hidden_dim=', getattr(args, 'hidden_dim', None))
    print('[config] row_dim=', args.row_dim, 'student_epochs=', args.student_epochs, 'rank_pairs=', args.student_rank_pairs)
    print('[config] candidate_gate_tau=', getattr(args, 'candidate_gate_tau', None), 'need_gate_tau=', getattr(args, 'need_gate_tau', None))
    print('[config] structural_prior_mode=', getattr(args, 'structural_prior_mode', None), 'alpha=', getattr(args, 'structural_prior_alpha', None))
    print('[config] acquisition_protocol=', args.acquisition_protocol)
    print('[config] target_good_count=', args.target_good_count, 'max_purchase_rows=', args.max_purchase_rows)
    print('[config] round_budget=', args.round_budget)
    print('[config] closed_loop_downstream_teacher=', bool(getattr(args, 'closed_loop_downstream_teacher', False)),
          'retrain_from_scratch=', bool(getattr(args, 'closed_loop_retrain_from_scratch', False)))
    print('[config] acquisition_convergence_stop=', bool(getattr(args, 'acquisition_convergence_stop', False)),
          'metric=', getattr(args, 'convergence_metric', 'test_macro_f1'),
          'patience=', getattr(args, 'convergence_patience', 3),
          'min_delta=', getattr(args, 'convergence_min_delta', 0.001),
          'min_rounds=', getattr(args, 'convergence_min_rounds', 5))
    print('[config] packaged_enabled=', not bool(getattr(args, 'no_packaged', False)))
    print('[config] package_methods=', getattr(args, 'package_methods', []))
    print('[config] package_target_sizes/windows=', getattr(args, 'package_target_sizes', ''))
    print('[config] anchor_target_sizes/windows=', getattr(args, 'anchor_target_sizes', ''))
    print('[config] package_radius_penalties=', getattr(args, 'package_radius_penalties', ''))
    print('[config] package_raw_score_variant=', getattr(args, 'package_raw_score_variant', 'structural'))
    print('[config] package_purchase_mode=', getattr(args, 'package_purchase_mode', 'raw_budget'))
    print('[config] package_structural_agg=', getattr(args, 'package_structural_agg', 'topmean'), 'cluster_weight=', getattr(args, 'package_structural_cluster_weight', 1.0))
    print('[config] package_teacher_target_mode=', getattr(args, 'package_teacher_target_mode', 'aggregate'),
          'teacher_top_frac=', getattr(args, 'package_teacher_top_frac', 0.5),
          'student_score_mode=', getattr(args, 'package_student_score_mode', 'margin'))
    print('[config] dynamic_markets=', bool(getattr(args, 'use_dynamic_market_profiles', False)),
          'good_ratio=', getattr(args, 'market_good_ratio_override', ''),
          'malicious_mix=', getattr(args, 'market_malicious_mix', 'none'))
    print('[config] strategic_market_sampling=', not bool(getattr(args, 'disable_strategic_market_sampling', False)))
    print('[config] markets_dir=', args.markets_dir)
    print('[config] output_dir=', out_dir)

    packs = load_base_cache(Path(args.base_dir))
    train_pack = packs['train']
    test_pack = packs['test']
    print('[cache] loaded base train=' + str(len(train_pack['y'])) + ' test=' + str(len(test_pack['y'])))

    all_results: List[pd.DataFrame] = []
    all_rounds: List[pd.DataFrame] = []
    all_drops: List[pd.DataFrame] = []
    markets_root = Path(args.markets_dir)
    for si, seed in enumerate(args.seeds, start=1):
        print('\n[seed]', si, '/', len(args.seeds), 'seed=', seed)
        t_seed = time.perf_counter()
        res_df, rd_df, drop_df = run_one_seed(seed, train_pack, test_pack, markets_root, args)
        all_results.append(res_df)
        all_rounds.append(rd_df)
        all_drops.append(drop_df)
        partial_results = pd.concat(all_results, axis=0, ignore_index=True)
        partial_rounds = pd.concat(all_rounds, axis=0, ignore_index=True)
        partial_drops = pd.concat(all_drops, axis=0, ignore_index=True) if all_drops else pd.DataFrame()
        partial_results.to_csv(out_dir / 'results_per_seed_partial.csv', index=False)
        partial_rounds.to_csv(out_dir / 'round_logs_partial.csv', index=False)
        partial_drops.to_csv(out_dir / 'packaging_drop_per_seed_partial.csv', index=False)
        print('[seed_done] seed=', seed, 'elapsed=%.1fs' % (time.perf_counter() - t_seed))

    results_df = pd.concat(all_results, axis=0, ignore_index=True) if all_results else pd.DataFrame()
    round_df = pd.concat(all_rounds, axis=0, ignore_index=True) if all_rounds else pd.DataFrame()
    drop_df = pd.concat(all_drops, axis=0, ignore_index=True) if all_drops else pd.DataFrame()

    summary_df = mean_std_columns(results_df, ['market_profile', 'setting', 'package_method', 'method'])
    round_summary_df = mean_std_columns(round_df, ['market_profile', 'setting', 'package_method', 'method', 'round_idx'])
    if len(drop_df):
        drop_group_cols = [c for c in ['comparison_type', 'market_profile', 'package_method', 'best_raw_method'] if c in drop_df.columns]
        drop_summary_df = mean_std_columns(drop_df, drop_group_cols)
    else:
        drop_summary_df = pd.DataFrame()

    results_df.to_csv(out_dir / 'results_per_seed.csv', index=False)
    round_df.to_csv(out_dir / 'round_logs.csv', index=False)
    summary_df.to_csv(out_dir / 'summary_mean_std.csv', index=False)
    round_summary_df.to_csv(out_dir / 'round_summary_mean_std.csv', index=False)
    drop_df.to_csv(out_dir / 'packaging_drop_per_seed.csv', index=False)
    drop_summary_df.to_csv(out_dir / 'packaging_drop_summary_mean_std.csv', index=False)
    build_paper_tables(results_df, out_dir)

    print('\n[done] saved to ' + str(out_dir))
    print('[done] main csv: ' + str(out_dir / 'results_per_seed.csv'))
    print('[done] paper tables: ' + str(out_dir / 'paper_tables'))
    return results_df, round_df, drop_df


def clone_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**copy.deepcopy(vars(args)))


def parse_suite_list(text: str) -> List[str]:
    suites = parse_list(text)
    if len(suites) == 0:
        return ['single']
    if 'all' in suites:
        return ['sparsity', 'budget', 'scale', 'student_ablation', 'crypto', 'baselines', 'malicious']
    return suites


def paired_int_sweep(values_text: str, fallback_text: str) -> List[Tuple[int, int]]:
    values = parse_int_list(values_text)
    fallback = parse_int_list(fallback_text)
    if len(values) == 0:
        raise ValueError('sweep_purchase_totals must contain at least one value.')
    if len(fallback) == 0:
        fallback = [max(1, min(10, values[0]))]
    out: List[Tuple[int, int]] = []
    for i, value in enumerate(values):
        budget = fallback[i if i < len(fallback) else len(fallback) - 1]
        out.append((int(value), int(budget)))
    return out


def safe_variant_name(text: str) -> str:
    raw = str(text).strip().replace('\\', '_').replace('/', '_').replace(':', '_')
    raw = raw.replace(' ', '_').replace(',', '_')
    return ''.join(ch for ch in raw if ch.isalnum() or ch in ['_', '-', '.'])


def suite_variant_out(root: Path, suite: str, variant: str) -> Path:
    return Path(root) / safe_variant_name(suite) / safe_variant_name(variant)


def fixed_main_package_changes() -> Dict[str, object]:
    return {
        'main_package_only': True,
        'full_package_ablation': False,
        'package_methods': 'anchor_greedy',
        'no_package_sweep': True,
        'package_target_size': 2,
        'anchor_target_size': 16,
        'package_radius_penalty': 0.0,
        'package_target_sizes': '2',
        'anchor_target_sizes': '16',
        'package_radius_penalties': '0.0',
        'package_raw_score_variant': 'structural',
        'package_structural_agg': 'topmean',
        'package_structural_top_frac': 0.5,
        'package_structural_cluster_weight': 1.0,
        'package_purchase_mode': 'raw_budget',
    }


def apply_changes(args: argparse.Namespace, changes: Dict[str, object]) -> argparse.Namespace:
    for key, value in changes.items():
        setattr(args, key, value)
    return args


def configure_dynamic_market(
    args: argparse.Namespace,
    root: Path,
    suite: str,
    variant: str,
    good_ratio: float,
    market_size: int,
    malicious_mix: str = 'none',
    mark_malicious: bool = False,
) -> argparse.Namespace:
    args.use_dynamic_market_profiles = True
    args.market_good_ratio_override = float(good_ratio)
    args.market_size = int(market_size)
    args.dynamic_profile_prefix = safe_variant_name(variant)
    args.market_malicious_mix = str(malicious_mix)
    args.mark_non_good_as_malicious = bool(mark_malicious)
    profiles = scaled_sparse_profiles(
        float(good_ratio),
        prefix=str(args.dynamic_profile_prefix),
        malicious_mix=str(malicious_mix),
    )
    args.market_profiles = ','.join(profiles.keys())
    args.markets_dir = str(Path(root) / '_market_cache' / safe_variant_name(suite) / safe_variant_name(variant))
    args.build_sparse_markets = True
    args.only_build_markets = False
    return args


def run_suite_variant(
    base_args: argparse.Namespace,
    suite: str,
    variant: str,
    changes: Dict[str, object],
    combined: Dict[str, List[pd.DataFrame]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(clean_cli_path(str(base_args.output_dir)))
    args = clone_args(base_args)
    args.experiment_suite = 'single'
    apply_changes(args, fixed_main_package_changes())
    apply_changes(args, changes)
    args.output_dir = str(suite_variant_out(root, suite, variant))
    print('\n[suite]', suite, 'variant=', variant, 'output=', args.output_dir)
    results_df, round_df, drop_df = run_all(args)
    for df in [results_df, round_df, drop_df]:
        if len(df):
            df.insert(0, 'suite_variant', str(variant))
            df.insert(0, 'suite_name', str(suite))
    if len(results_df):
        combined.setdefault('results', []).append(results_df)
    if len(round_df):
        combined.setdefault('rounds', []).append(round_df)
    if len(drop_df):
        combined.setdefault('drops', []).append(drop_df)
    return results_df, round_df, drop_df


def write_combined_suite_outputs(root: Path, combined: Dict[str, List[pd.DataFrame]]) -> None:
    out_dir = Path(root) / 'suite_combined'
    ensure_dir(out_dir)

    if combined.get('results'):
        results_df = pd.concat(combined['results'], axis=0, ignore_index=True)
        results_df.to_csv(out_dir / 'results_per_seed.csv', index=False)
        group_cols = [c for c in ['suite_name', 'suite_variant', 'market_profile', 'setting', 'package_method', 'method'] if c in results_df.columns]
        mean_std_columns(results_df, group_cols).to_csv(out_dir / 'summary_mean_std.csv', index=False)
        build_paper_tables(results_df, out_dir)

    if combined.get('rounds'):
        round_df = pd.concat(combined['rounds'], axis=0, ignore_index=True)
        round_df.to_csv(out_dir / 'round_logs.csv', index=False)
        group_cols = [c for c in ['suite_name', 'suite_variant', 'market_profile', 'setting', 'package_method', 'method', 'round_idx'] if c in round_df.columns]
        mean_std_columns(round_df, group_cols).to_csv(out_dir / 'round_summary_mean_std.csv', index=False)

    if combined.get('drops'):
        drop_df = pd.concat(combined['drops'], axis=0, ignore_index=True)
        drop_df.to_csv(out_dir / 'packaging_drop_per_seed.csv', index=False)
        group_cols = [c for c in ['suite_name', 'suite_variant', 'comparison_type', 'market_profile', 'package_method', 'best_raw_method'] if c in drop_df.columns]
        mean_std_columns(drop_df, group_cols).to_csv(out_dir / 'packaging_drop_summary_mean_std.csv', index=False)

    print('[suite done] combined outputs saved to ' + str(out_dir))


def run_crypto_microbench(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    counts = parse_int_list(getattr(args, 'crypto_package_counts', ''))
    dims = parse_int_list(getattr(args, 'crypto_feature_dims', ''))
    repeats = max(1, int(getattr(args, 'crypto_repeats', 1)))
    if not counts:
        counts = [100, 500, 1000]
    if not dims:
        dims = [16, 32, 64]

    rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(int(getattr(args, 'market_seed', 42)) + 9091)
    for n in counts:
        for dim in dims:
            for rep in range(repeats):
                feat = rng.normal(size=(int(n), int(dim))).astype(np.float32)
                weight = rng.normal(size=(int(dim),)).astype(np.float32)

                t0 = time.perf_counter()
                encoded = feat + rng.normal(scale=1e-4, size=feat.shape).astype(np.float32)
                encode_s = float(time.perf_counter() - t0)

                t1 = time.perf_counter()
                score = encoded @ weight
                score = np.tanh(score).astype(np.float32) + (0.05 * score * score).astype(np.float32)
                eval_s = float(time.perf_counter() - t1)

                t2 = time.perf_counter()
                noise_a = rng.normal(scale=0.01, size=score.shape).astype(np.float32)
                noise_b = rng.normal(scale=0.01, size=score.shape).astype(np.float32)
                share_c = score.astype(np.float32) - noise_a - noise_b
                recon = noise_a + noise_b + share_c
                decode_s = float(time.perf_counter() - t2)

                t3 = time.perf_counter()
                digest = hashlib.sha256(recon[: min(len(recon), 4096)].astype(np.float32).tobytes()).hexdigest()
                verify_s = float(time.perf_counter() - t3)

                rows.append({
                    'benchmark_type': 'ckks_path_proxy_no_security',
                    'package_count': int(n),
                    'feature_dim': int(dim),
                    'repeat': int(rep),
                    'encode_s': encode_s,
                    'encrypted_eval_proxy_s': eval_s,
                    'decode_s': decode_s,
                    'verify_s': verify_s,
                    'total_s': encode_s + eval_s + decode_s + verify_s,
                    'rows_per_second': float(int(n) / max(1e-9, encode_s + eval_s + decode_s + verify_s)),
                    'digest_prefix': digest[:12],
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'crypto_microbench.csv', index=False)
    summary = mean_std_columns(df, ['benchmark_type', 'package_count', 'feature_dim'])
    summary.to_csv(out_dir / 'crypto_microbench_summary.csv', index=False)
    print('[crypto] proxy microbenchmark saved to ' + str(out_dir))
    return df


def run_experiment_suite(args: argparse.Namespace) -> None:
    suites = parse_suite_list(getattr(args, 'experiment_suite', 'single'))
    known = {'single', 'sparsity', 'budget', 'scale', 'student_ablation', 'crypto', 'baselines', 'malicious'}
    unknown = [s for s in suites if s not in known]
    if unknown:
        raise ValueError('Unknown experiment suite(s): ' + ','.join(unknown))
    if bool(getattr(args, 'build_sparse_markets', False)) and bool(getattr(args, 'only_build_markets', False)):
        # Market construction is a terminal action. Without this guard, the
        # default sparsity/budget/scale suite can unexpectedly run full
        # acquisition experiments after building markets.
        run_all(args)
        return
    if suites == ['single']:
        run_all(args)
        return

    root = Path(clean_cli_path(str(args.output_dir)))
    ensure_dir(root)
    combined: Dict[str, List[pd.DataFrame]] = {'results': [], 'rounds': [], 'drops': []}
    base_market_size = int(getattr(args, 'market_size', 100000))

    for suite in suites:
        if suite == 'sparsity':
            for good_ratio in parse_float_list(getattr(args, 'sweep_good_ratios', '0.01,0.05,0.10,0.20')):
                variant = 'good_ratio_' + slug_number(good_ratio)
                dyn_args = configure_dynamic_market(
                    clone_args(args), root, suite, variant,
                    good_ratio=float(good_ratio),
                    market_size=base_market_size,
                )
                run_suite_variant(dyn_args, suite, variant, {}, combined)

        elif suite == 'budget':
            for purchase_total, round_budget in paired_int_sweep(
                getattr(args, 'sweep_purchase_totals', '25,50,100,200'),
                getattr(args, 'sweep_round_budgets', '5,5,10,20'),
            ):
                variant = f'purchase_{int(purchase_total)}_round_{int(round_budget)}'
                run_suite_variant(args, suite, variant, {
                    'acquisition_protocol': 'fixed_budget',
                    'purchase_total': int(purchase_total),
                    'round_budget': int(round_budget),
                }, combined)

        elif suite == 'scale':
            for market_size in parse_int_list(getattr(args, 'sweep_market_sizes', '10000,50000,100000')):
                variant = 'market_size_' + str(int(market_size))
                dyn_args = configure_dynamic_market(
                    clone_args(args), root, suite, variant,
                    good_ratio=float(getattr(args, 'market_good_ratio_override', 0.10)),
                    market_size=int(market_size),
                )
                run_suite_variant(dyn_args, suite, variant, {}, combined)

        elif suite == 'student_ablation':
            variants = [
                ('structural_only_main', {
                    'raw_variants': 'structural_prior_balanced',
                    'package_raw_score_variant': 'structural',
                    'student_epochs': int(getattr(args, 'student_epochs', 5)),
                    'student_rank_weight': float(getattr(args, 'student_rank_weight', 0.5)),
                }),
                ('hybrid_teacher_student', {
                    'raw_variants': 'gated_need_structural_balanced',
                    'package_raw_score_variant': 'gated_need_structural',
                    'student_epochs': int(getattr(args, 'student_epochs', 5)),
                    'student_rank_weight': float(getattr(args, 'student_rank_weight', 0.5)),
                }),
                ('hybrid_no_rank_loss', {
                    'raw_variants': 'gated_need_structural_balanced',
                    'package_raw_score_variant': 'gated_need_structural',
                    'student_epochs': int(getattr(args, 'student_epochs', 5)),
                    'student_rank_weight': 0.0,
                }),
                ('hybrid_light_student', {
                    'raw_variants': 'gated_need_structural_balanced',
                    'package_raw_score_variant': 'gated_need_structural',
                    'student_epochs': 1,
                    'student_hidden_dim': 16,
                    'student_rank_weight': float(getattr(args, 'student_rank_weight', 0.5)),
                }),
                ('structural_no_label_balance', {
                    'raw_variants': 'structural_prior',
                    'package_raw_score_variant': 'structural',
                }),
            ]
            for variant, changes in variants:
                run_suite_variant(args, suite, variant, changes, combined)

        elif suite == 'crypto':
            out_dir = suite_variant_out(root, suite, 'proxy_microbench')
            crypto_df = run_crypto_microbench(args, out_dir)
            if len(crypto_df):
                crypto_tagged = crypto_df.copy()
                crypto_tagged.insert(0, 'suite_variant', 'proxy_microbench')
                crypto_tagged.insert(0, 'suite_name', 'crypto')
                ensure_dir(root / 'suite_combined')
                crypto_tagged.to_csv(root / 'suite_combined' / 'crypto_microbench.csv', index=False)

        elif suite == 'baselines':
            run_suite_variant(args, suite, 'extended_active_learning_baselines', {
                'reference_methods': (
                    'oracle_good_select,market_random_select,market_cosine_select,'
                    'market_entropy_select,market_margin_select,market_badge_select,'
                    'market_coreset_select,market_kmeans_center_select'
                ),
            }, combined)

        elif suite == 'malicious':
            mixes = ['adversarial_heavy', 'conflict_heavy', 'decision_heavy', 'balanced_hard']
            for mix in mixes:
                variant = 'malicious_' + str(mix)
                dyn_args = configure_dynamic_market(
                    clone_args(args), root, suite, variant,
                    good_ratio=float(getattr(args, 'market_good_ratio_override', 0.10)),
                    market_size=base_market_size,
                    malicious_mix=str(mix),
                    mark_malicious=True,
                )
                run_suite_variant(dyn_args, suite, variant, {}, combined)

    write_combined_suite_outputs(root, combined)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_args(argv)
    run_experiment_suite(args)


if __name__ == '__main__':
    main()
